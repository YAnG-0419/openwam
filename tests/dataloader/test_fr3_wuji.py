"""FR3/Wuji semantic mapping, time alignment, split isolation and normalization."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from openwam.dataloader import fr3_wuji as reader
from openwam.dataloader.registry import build_dataset
from openwam.dataloader.transforms.normalize import Normalizer, load_mode_stats


@pytest.fixture()
def root(tmp_path, monkeypatch):
    # Deliberately reverse field ordering to ensure the reader follows names.
    state_names = list(reversed(reader.STATE_NAMES)) + [f"velocity_{i}" for i in range(54)]
    action_names = list(reversed(reader.ACTION_NAMES))
    for name in ("A", "B"):
        bucket = tmp_path / name
        (bucket / "meta").mkdir(parents=True)
        (bucket / "data").mkdir()
        (bucket / "videos").mkdir()
        info = dict(
            codebase_version="v2.1",
            robot_type="dual_fr3_wuji",
            fps=30,
            data_path="data/episode_{episode_index:06d}.parquet",
            video_path="videos/{video_key}_{episode_index:06d}.mp4",
            features={
                "action": dict(shape=[54], names=action_names),
                "observation.state": dict(shape=[108], names=state_names),
                **{cam: dict(dtype="video") for cam in reader.CAMERAS},
            },
        )
        (bucket / "meta/info.json").write_text(json.dumps(info))
        (bucket / "meta/tasks.jsonl").write_text(json.dumps(dict(task_index=0, task="pick tomato")))
        episodes = []
        for ep in range(12):
            # A and B share recording IDs: they must never cross train/val.
            episodes.append(dict(episode_index=ep, length=6, source_recording_id=f"recording-{ep}"))
            rows = []
            for t in range(6):
                rows.append(
                    {
                        "action": (np.arange(54) + t * 100 + ep * 1000).astype(float).tolist(),
                        "observation.state": (np.arange(108) + t * 200 + ep * 2000).astype(float).tolist(),
                        "frame_index": t,
                        "timestamp": t / 30,
                        "task_index": 0,
                    }
                )
            pq.write_table(pa.Table.from_pylist(rows), bucket / f"data/episode_{ep:06d}.parquet")
            for cam in reader.CAMERAS:
                (bucket / f"videos/{cam}_{ep:06d}.mp4").touch()
        (bucket / "meta/episodes.jsonl").write_text("\n".join(json.dumps(e) for e in episodes))

    def decode(path, indices, height, width):
        return [Image.new("RGB", (width, height), (i, 0, 0)) for i in indices]

    monkeypatch.setattr(reader, "decode_video_frames", decode)
    reader._table.cache_clear()
    return tmp_path


def test_mapping_alignment_and_terminal_masks(root):
    ds = reader.Fr3WujiDataset(root, validation_fraction=0, normalize_mode=None, num_frames=5, video_stride=2)
    s = ds[0]
    np.testing.assert_array_equal(s["action"][0], np.arange(54)[::-1])
    np.testing.assert_array_equal(s["action"][1], np.arange(54)[::-1] + 100)
    np.testing.assert_array_equal(s["proprio"][0], np.arange(54)[::-1])
    assert s["action"].shape == (4, 54)
    assert s["video_mask"].tolist() == [True, True, True]
    assert [im.getpixel((0, 0))[0] for im in s["video"]] == [0, 2, 4]
    tail = ds[int(ds._ends[0]) - 1]
    assert tail["action_mask"][:, 0].tolist() == [True, False, False, False]
    assert tail["video_mask"].tolist() == [True, False, False]
    assert [im.getpixel((0, 0))[0] for im in tail["video"]] == [4, 4, 4]
    assert ds[-1]["action"].shape == (4, 54)
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_recording_split_and_train_only_stats(root):
    kwargs = dict(validation_fraction=0.5, normalize_mode=None)
    train = reader.Fr3WujiDataset(root, **kwargs)
    val = reader.Fr3WujiDataset(root, split="val", **kwargs)
    assert train.fingerprint == val.fingerprint
    assert not {e["group"] for e in train.episodes} & {e["group"] for e in val.episodes}
    out = root / "stats.npy"
    payload = reader.compute_stats(root, out, validation_fraction=0.5)
    train_ids = [e["record"]["episode_index"] for e in train.episodes]
    # Mean action is over times 0..4; terminal t=5 is excluded.
    np.testing.assert_allclose(payload["joint"]["mean"], np.arange(54)[::-1] + 200 + np.mean(train_ids) * 1000)
    ds = reader.Fr3WujiDataset(root, validation_fraction=0.5, normalization_stats_path=out)
    raw = train[0]
    normalized = ds[0]
    assert normalized["action"].abs().max() <= 1
    # The serving normalizer must restore action commands and use separate state stats.
    normalizer = Normalizer(mode="min_max", stats=load_mode_stats(str(out), "joint"))
    valid = normalized["action_mask"].numpy()
    np.testing.assert_allclose(
        normalizer.unnormalize(normalized["action"].numpy())[valid], raw["action"].numpy()[valid], atol=2e-3
    )
    np.testing.assert_allclose(normalizer.normalize(raw["proprio"].numpy()), normalized["proprio"].numpy())
    with pytest.raises(ValueError, match="do not match"):
        reader.Fr3WujiDataset(root, validation_fraction=0.25, normalization_stats_path=out)


def test_registry_and_single_view(root):
    ds = build_dataset(
        dict(
            type="fr3_wuji",
            dataset_dir=str(root),
            validation_fraction=0,
            normalize_mode=None,
            multiview=False,
            num_frames=5,
            video_stride=2,
            height=32,
            width=40,
        )
    )
    assert ds[0]["video"][0].size == (40, 32)
    single = reader.Fr3WujiDataset(root / "A", validation_fraction=0, normalize_mode=None)
    assert len(single.episodes) == 12


def test_reject_incompatible_stats_and_schema(root):
    stats = root / "openpi.npy"
    np.save(stats, {"norm_stats": {}})
    with pytest.raises(ValueError, match="do not match"):
        reader.Fr3WujiDataset(root, normalization_stats_path=stats)
    path = root / "A/meta/info.json"
    info = json.loads(path.read_text())
    info["features"]["action"]["names"][0] = "unknown_joint"
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="feature names"):
        reader.Fr3WujiDataset(root, normalize_mode=None)


def test_bad_timing_and_nonfinite_commands_fail(root):
    path = root / "A/data/episode_000000.parquet"
    table = pq.read_table(path).to_pylist()
    table[1]["timestamp"] = 10.0
    pq.write_table(pa.Table.from_pylist(table), path)
    ds = reader.Fr3WujiDataset(root, validation_fraction=0, normalize_mode=None)
    with pytest.raises(ValueError, match="timestamps"):
        ds[0]
    table[1]["timestamp"] = 1 / 30
    table[0]["action"][0] = float("nan")
    pq.write_table(pa.Table.from_pylist(table), path)
    reader._table.cache_clear()
    with pytest.raises(ValueError, match="non-finite"):
        ds[0]


def test_missing_camera_is_not_silently_hidden(root):
    (root / "A/videos/observation.images.cam1_000000.mp4").unlink()
    with pytest.raises(FileNotFoundError):
        reader.Fr3WujiDataset(root, validation_fraction=0, normalize_mode=None)


def test_spawn_worker_after_parent_video_preview(root, monkeypatch):
    import av
    from torch.utils.data import DataLoader

    from openwam.dataloader.utils.video_io import decode_video_frames

    monkeypatch.setattr(reader, "decode_video_frames", decode_video_frames)
    bucket = root / "A"
    episodes_path = bucket / "meta/episodes.jsonl"
    episodes_path.write_text(episodes_path.read_text().splitlines()[0])
    for camera in reader.CAMERAS:
        with av.open(str(bucket / f"videos/{camera}_000000.mp4"), "w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width = stream.height = 32
            stream.pix_fmt = "yuv420p"
            for i in range(6):
                frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 30, dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    ds = reader.Fr3WujiDataset(bucket, validation_fraction=0, normalize_mode=None, height=48, width=32)
    ds[0]  # Initialize Arrow and FFmpeg in parent before starting workers.
    loader = DataLoader(
        ds, batch_size=2, num_workers=1, collate_fn=list, multiprocessing_context=ds.multiprocessing_context
    )
    iterator = iter(loader)
    try:
        batch = next(iterator)
        assert len(batch) == 2
        assert batch[0]["action"].shape == (32, 54)
        assert len(batch[0]["video"]) == 9
    finally:
        del iterator
