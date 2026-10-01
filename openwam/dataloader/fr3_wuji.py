"""FR3/Wuji LeRobot v2.1 reader (single bucket or a root containing A/B).

Outputs 54 positions/commands in left-arm, left-hand, right-arm, right-hand
order. Actions are the recorded row-aligned commands: no temporal shift,
delta conversion, FK, or gripper conversion. State velocities are omitted.
All cameras are required. Data and metadata are never modified by this reader.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.transforms.multiview import assemble_multiview_layout
from openwam.dataloader.utils.normalization import apply_normalization
from openwam.dataloader.utils.video_io import decode_video_frames

DIM = 54
CAMERAS = tuple(f"observation.images.cam{i}" for i in range(3))
FINGERS = (
    "thumb_cmc_flex",
    "thumb_cmc_abd",
    "thumb_mcp",
    "thumb_ip",
    "index_finger_mcp_flex",
    "index_finger_mcp_abd",
    "index_finger_pip",
    "index_finger_dip",
    "middle_finger_mcp_flex",
    "middle_finger_mcp_abd",
    "middle_finger_pip",
    "middle_finger_dip",
    "ring_finger_mcp_flex",
    "ring_finger_mcp_abd",
    "ring_finger_pip",
    "ring_finger_dip",
    "pinky_mcp_flex",
    "pinky_mcp_abd",
    "pinky_pip",
    "pinky_dip",
)
STATE_NAMES = tuple(
    [f"left_fr3_joint{i}.position" for i in range(1, 8)]
    + [f"l_{name}.position" for name in FINGERS]
    + [f"right_fr3_joint{i}.position" for i in range(1, 8)]
    + [f"r_{name}.position" for name in FINGERS]
)
ACTION_NAMES = tuple(
    name.replace(".position", ".command") if name.startswith(("l_", "r_")) else name for name in STATE_NAMES
)
SCHEMA = "openwam.fr3_wuji.joint54.row_aligned.v1"
MODES = (None, "none", "null", "min-max", "z-score", "quantile")


def _jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


@functools.lru_cache(maxsize=4)
def _table(path, length, fps):
    table = pq.read_table(path, columns=["action", "observation.state", "task_index", "frame_index", "timestamp"])
    if table.num_rows != length:
        raise ValueError(f"{path}: rows do not match episode length")
    indices = np.asarray(table["frame_index"].to_pylist())
    stamps = np.asarray(table["timestamp"].to_pylist())
    if not np.array_equal(indices, np.arange(length)) or not np.allclose(
        stamps,
        np.arange(length) / fps,
        atol=1e-4,
        rtol=1e-5,
    ):
        raise ValueError(f"{path}: frame indices/timestamps do not match declared FPS")
    return table


def _indices(feature, names, width):
    actual = feature.get("names")
    if feature.get("shape") != [width] or not isinstance(actual, list) or len(actual) != width:
        raise ValueError(f"Expected named {width}-D feature, got {feature}")
    if len(set(actual)) != len(actual) or not set(names).issubset(actual):
        raise ValueError("Missing or duplicate FR3/Wuji feature names; refusing positional guesses")
    return np.array([actual.index(name) for name in names])


class Fr3WujiDataset(BaseDataset):
    """Exhaustive windows with deterministic recording-group train/val splits."""

    action_dim = DIM
    state_dim = DIM
    # Arrow/FFmpeg thread state must not be inherited after a parent preview.
    multiprocessing_context = "spawn"

    def __init__(
        self,
        dataset_dir,
        *,
        split="train",
        num_frames=33,
        video_stride=4,
        window_stride=1,
        height=384,
        width=320,
        multiview=True,
        camera_layout=CAMERAS,
        target_camera=CAMERAS[0],
        validation_fraction=0.1,
        seed=42,
        normalize_mode="min-max",
        normalization_stats_path=None,
        action_mode="joint",
        unify_action=False,
    ):
        if action_mode != "joint" or unify_action:
            raise ValueError("FR3/Wuji uses raw joint54: action_mode=joint, unify_action=false")
        if split not in ("train", "val") or not 0 <= validation_fraction < 1:
            raise ValueError("Expected train/val split and 0 <= validation_fraction < 1")
        if num_frames < 2 or min(video_stride, window_stride, height, width) < 1:
            raise ValueError("Invalid window, stride, or image size")
        if normalize_mode not in MODES:
            raise ValueError(f"Unsupported normalize_mode: {normalize_mode}")
        self.root = Path(dataset_dir).expanduser().resolve()
        self.split, self.seed = split, int(seed)
        self.validation_fraction = float(validation_fraction)
        self.num_frames, self.window_stride = int(num_frames), int(window_stride)
        self.video_indices = np.arange(0, self.num_frames, int(video_stride))
        self.height, self.width, self.multiview = int(height), int(width), bool(multiview)
        self.cameras = tuple(camera_layout) if multiview else (target_camera,)
        if multiview and (len(self.cameras) != 3 or len(set(self.cameras)) != 3):
            raise ValueError("multiview requires three distinct camera names")
        self.normalize_mode = normalize_mode
        self.normalization_stats_path = None
        self.stats = None
        buckets = (
            [self.root]
            if (self.root / "meta/info.json").is_file()
            else sorted(p for p in self.root.iterdir() if p.is_dir() and (p / "meta/info.json").is_file())
        )
        if not buckets:
            raise ValueError(f"No LeRobot v2.1 buckets under {self.root}")
        self.episodes = []
        population = []
        for bucket in buckets:
            info = json.loads((bucket / "meta/info.json").read_text())
            if info.get("codebase_version") != "v2.1" or info.get("robot_type") != "dual_fr3_wuji":
                raise ValueError(f"{bucket}: expected v2.1 dual_fr3_wuji data")
            features = info["features"]
            state_indices = _indices(features["observation.state"], STATE_NAMES, 108)
            action_indices = _indices(features["action"], ACTION_NAMES, DIM)
            for cam in self.cameras:
                if features.get(cam, {}).get("dtype") != "video":
                    raise ValueError(f"{bucket}: missing video feature {cam}")
            tasks_list = _jsonl(bucket / "meta/tasks.jsonl")
            tasks = {int(t["task_index"]): t["task"] for t in tasks_list}
            if len(tasks) != len(tasks_list) or any(not isinstance(t, str) or not t.strip() for t in tasks.values()):
                raise ValueError(f"{bucket}: duplicate or blank tasks")
            records = _jsonl(bucket / "meta/episodes.jsonl")
            if len({int(e["episode_index"]) for e in records}) != len(records):
                raise ValueError(f"{bucket}: duplicate episode indices")
            exclusion_path = bucket / "meta/excluded_episodes.json"
            excluded = set(json.loads(exclusion_path.read_text())) if exclusion_path.exists() else set()
            population.append({"bucket": str(bucket.relative_to(self.root)), "info": info, "tasks": tasks})
            for record in records:
                ep_idx = int(record["episode_index"])
                if ep_idx in excluded:
                    continue
                # The supplied data declares everything as train. This reader deliberately
                # replaces that declaration with one global recording-group split.
                group = str(record.get("source_recording_id") or f"{bucket.relative_to(self.root)}/{ep_idx}")
                score = int.from_bytes(hashlib.sha256(f"{self.seed}:{group}".encode()).digest()[:8], "big") / 2**64
                is_val = score < self.validation_fraction
                values = dict(episode_index=ep_idx, episode_chunk=ep_idx // int(info.get("chunks_size", 1000)))
                path = bucket / info["data_path"].format(**values)
                if not path.is_file():
                    raise FileNotFoundError(path)
                if int(record["length"]) < 2:
                    raise ValueError(f"{path}: at least two frames required")
                if not is_val:
                    st = path.stat()
                    population.append(
                        {
                            "path": str(path.relative_to(self.root)),
                            "episode": record,
                            "size": st.st_size,
                            "mtime_ns": st.st_mtime_ns,
                        }
                    )
                if is_val != (split == "val"):
                    continue
                videos = [bucket / info["video_path"].format(**values, video_key=cam) for cam in self.cameras]
                for video in videos:
                    if not video.is_file():
                        raise FileNotFoundError(video)
                self.episodes.append(
                    dict(
                        record=record,
                        group=group,
                        path=path,
                        videos=videos,
                        fps=float(info["fps"]),
                        tasks=tasks,
                        action_indices=action_indices,
                        state_indices=state_indices,
                    )
                )
        if not self.episodes:
            raise ValueError(f"Empty {split} split; adjust validation_fraction/seed or provide more recordings")
        if any(not np.isfinite(e["fps"]) or e["fps"] <= 0 for e in self.episodes):
            raise ValueError("Dataset FPS must be positive and finite")
        self.fingerprint = hashlib.sha256(
            json.dumps(
                dict(
                    schema=SCHEMA, seed=self.seed, validation_fraction=self.validation_fraction, population=population
                ),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        # Include short windows, but require at least two real frames. No cross-episode windows.
        counts = [(int(e["record"]["length"]) - 2) // self.window_stride + 1 for e in self.episodes]
        self._ends = np.cumsum(counts)
        if normalize_mode not in (None, "none", "null"):
            if normalization_stats_path is None or not Path(normalization_stats_path).is_file():
                raise FileNotFoundError(
                    "Generate FR3/Wuji stats first with scripts/compute_fr3_wuji_stats.py; "
                    "set dataloader.normalization_stats_path to its output"
                )
            self.normalization_stats_path = str(Path(normalization_stats_path).resolve())
            payload = np.load(self.normalization_stats_path, allow_pickle=True).item()
            if payload.get("schema") != SCHEMA or payload.get("fingerprint") != self.fingerprint:
                raise ValueError(
                    "FR3/Wuji stats do not match this data/split; regenerate stats (OpenPI stats are incompatible)"
                )
            self.stats = {}
            for kind in ("joint", "joint_state"):
                self.stats[kind] = {}
                for key in ("mean", "std", "min", "max", "q01", "q99"):
                    arr = np.asarray(payload[kind][key], dtype=np.float32)
                    if arr.shape != (DIM,) or not np.isfinite(arr).all():
                        raise ValueError(f"Invalid {kind}/{key} stats")
                    self.stats[kind][key] = arr
                s = self.stats[kind]
                if np.any(s["std"] < 0) or np.any(s["min"] > s["max"]) or np.any(s["q01"] > s["q99"]):
                    raise ValueError(f"Invalid {kind} stats ranges")

    @classmethod
    def from_config(cls, config, split="train"):
        keys = (
            "dataset_dir",
            "num_frames",
            "video_stride",
            "window_stride",
            "height",
            "width",
            "multiview",
            "camera_layout",
            "target_camera",
            "validation_fraction",
            "seed",
            "normalize_mode",
            "normalization_stats_path",
            "action_mode",
            "unify_action",
        )
        return cls(split=split, **{key: config[key] for key in keys if key in config})

    def __len__(self):
        return int(self._ends[-1])

    def _read(self, episode):
        return _table(str(episode["path"]), int(episode["record"]["length"]), episode["fps"])

    @staticmethod
    def _vectors(table, column, indices, width):
        if table[column].null_count:
            raise ValueError(f"Null values in {column}")
        values = np.asarray(table[column].to_pylist(), dtype=np.float32)
        if values.shape != (table.num_rows, width) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {column} shape or non-finite values")
        return values[:, indices]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        ep_pos = int(np.searchsorted(self._ends, index, side="right"))
        start = int(index - (self._ends[ep_pos - 1] if ep_pos else 0)) * self.window_stride
        episode = self.episodes[ep_pos]
        window = self._read(episode).slice(start, self.num_frames)
        n = window.num_rows
        raw_action = self._vectors(window, "action", episode["action_indices"], DIM)
        state = self._vectors(window.slice(0, 1), "observation.state", episode["state_indices"], 108)
        if self.stats is not None:
            raw_action = apply_normalization(raw_action, self.stats["joint"], self.normalize_mode)
            state = apply_normalization(state, self.stats["joint_state"], self.normalize_mode)
        action = np.zeros((self.num_frames - 1, DIM), dtype=np.float32)
        action_mask = np.zeros(action.shape, dtype=bool)
        # Last recorded row has no following observed frame: do not supervise it.
        valid = min(n - 1, self.num_frames - 1)
        action[:valid], action_mask[:valid] = raw_action[:valid], True
        task_ids = window["task_index"].to_pylist()
        if len(set(task_ids)) != 1 or task_ids[0] not in episode["tasks"]:
            raise ValueError(f"{episode['path']}: window has missing or changing task_index")
        real_video = self.video_indices[self.video_indices < n]
        frames = {}
        for i, (camera, path) in enumerate(zip(self.cameras, episode["videos"])):
            h, w = self.height, self.width
            if self.multiview:
                top = int(round(self.height * 2 / 3))
                h = top if i == 0 else self.height - top
                w = self.width if i == 0 else (self.width // 2 if i == 1 else self.width - self.width // 2)
            decoded = decode_video_frames(str(path), (start + real_video).tolist(), h, w)
            frames[camera] = decoded + [decoded[-1]] * (len(self.video_indices) - len(decoded))
        video = (
            [
                assemble_multiview_layout(
                    {cam: frames[cam][i] for cam in self.cameras},
                    list(self.cameras),
                    self.height,
                    self.width,
                )
                for i in range(len(self.video_indices))
            ]
            if self.multiview
            else frames[self.cameras[0]]
        )
        return dict(
            video=video,
            first_frame_image=[video[0]],
            vace_video=None,
            prompt=episode["tasks"][task_ids[0]],
            action=torch.from_numpy(action),
            action_mask=torch.from_numpy(action_mask),
            proprio=torch.from_numpy(state.astype(np.float32)),
            proprio_mask=torch.ones((1, DIM), dtype=torch.bool),
            video_mask=torch.from_numpy(self.video_indices < n),
        )


def compute_stats(dataset_dir, output, *, validation_fraction=0.1, seed=42):
    """Exact training-only per-row stats; actions exclude each terminal row.

    Uses about 140 MB for the supplied 328K-frame dataset's selected action and
    state arrays, plus concatenation/quantile scratch. Run once before torchrun.
    """
    dataset = Fr3WujiDataset(dataset_dir, normalize_mode=None, validation_fraction=validation_fraction, seed=seed)
    actions, states = [], []
    for episode in dataset.episodes:
        table = dataset._read(episode)
        actions.append(dataset._vectors(table, "action", episode["action_indices"], DIM)[:-1])
        states.append(dataset._vectors(table, "observation.state", episode["state_indices"], 108)[:-1])
    payload = dict(
        schema=SCHEMA,
        fingerprint=dataset.fingerprint,
        seed=seed,
        validation_fraction=validation_fraction,
        action_names=ACTION_NAMES,
        state_names=STATE_NAMES,
        train_episodes=len(dataset.episodes),
        action_alignment="row_aligned_terminal_excluded",
    )
    for key, parts in (("joint", actions), ("joint_state", states)):
        values = np.concatenate(parts)
        payload[key] = dict(
            mean=values.mean(0, dtype=np.float64).astype(np.float32),
            std=values.std(0, dtype=np.float64).astype(np.float32),
            min=values.min(0),
            max=values.max(0),
            q01=np.quantile(values, 0.01, axis=0).astype(np.float32),
            q99=np.quantile(values, 0.99, axis=0).astype(np.float32),
        )
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, payload)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return payload
