# FR3 + Wuji: tomato LeRobot v2.1 data

`dataloader=fr3_wuji` reads a single bucket or a parent with immediate A/B
buckets. Each bucket contains `meta/info.json`, `episodes.jsonl`, `tasks.jsonl`,
per-episode Parquet, and MP4 videos. Original data is read-only.

## Representation

- Robot type: `dual_fr3_wuji`, LeRobot `v2.1`.
- Output order: left arm (7), left hand (20), right arm (7), right hand (20).
- Actions: recorded row-aligned position/hand commands, without delta conversion
  or frame shifting. The last episode row is masked because it has no next frame.
- State: the corresponding 54 positions selected by feature names from the
  original 108-D position/velocity vector. Velocities are omitted.
- Feature names are checked; this is not a generic reader for arbitrary robots.
- `cam0` is the main view, `cam1` left wrist, `cam2` right wrist, following the
  existing OpenPI configuration. The default composite is 320 px wide × 384 px
  high. All three cameras are required; decode failures raise visibly.
- A 33-row window with video stride 4 emits 9 composite frames, 32 actions,
  and one current state. Short windows repeat the last sampled real image and
  mask padding. Windows never cross episodes. FPS stays at the recorded 30 Hz;
  video stride only subsamples images.
- The trainer uses spawned workers for this reader to avoid inheriting native
  Arrow/FFmpeg thread state after a parent-process sample preview. Custom
  PyTorch loaders should also use `multiprocessing_context="spawn"` with workers
  (and put their entry point behind `if __name__ == "__main__":`), or use
  `num_workers=0` for interactive inspection.
- Model dimensions must be `action_dim=54`, `state_dim=54`, `unify_action=false`.
  Existing 80-D full checkpoints need a separate compatible weight-loading plan;
  this reader does not migrate checkpoint weights or reinterpret EEF slots.

## Split and normalization

The reader intentionally replaces `info.json` split declarations with a global
deterministic split: SHA-256 of `seed:source_recording_id`, with a default 10%
validation threshold. Related recordings stay together across A/B. If the ID is
missing, the bucket-relative path and episode index define the group. The hash
threshold is approximate, not an exact percentage. For the supplied 250-episode
dataset and seed 42 the result is 230 train / 20 validation episodes. Set
`validation_fraction=0` explicitly to train on all episodes. The reader honors
each bucket's `meta/excluded_episodes.json` list. The current training entry point
builds only the training split; exposing a validation split here does not add an
automatic validation loop to the trainer.

Statistics are computed separately for commands and measured positions from
training episodes only, excluding terminal rows. They describe raw recorded
commands, not OpenPI's reordered delta-action horizon statistics. **Do not use
the existing `norm_stats.json`.** The saved artifact contains `joint` and
`joint_state` blocks compatible with OpenWAM's checkpoint normalizer. Its schema
and training-population fingerprint reject changed split settings, metadata, or
Parquet sizes/mtime; the fingerprint is not a full content checksum.

Run from the OpenWAM repository in an environment with its dependencies:

```bash
python scripts/compute_fr3_wuji_stats.py \
  --dataset-dir /home/descfly/lpy/openpi/data/tomato/0924-250ep \
  --out outputs/fr3_wuji/normalization_stats.npy
```

Use the same seed and validation fraction for stats and training. Recompute
stats after changing either or the data. Computation is an explicit preparation
step, so distributed workers never race to write statistics. It reads all
training Parquets but does not decode every video frame.

## Inspect a sample and start training

```python
from omegaconf import OmegaConf
from openwam.dataloader.registry import build_dataset

cfg = OmegaConf.load("configs/dataloader/fr3_wuji.yaml")
dataset = build_dataset(cfg, split="train")
sample = dataset[0]
print(len(dataset), sample["action"].shape, sample["proprio"].shape)
print(len(sample["video"]), sample["video"][0].size)
# action: (32, 54), proprio: (1, 54), 9 frames of size (320, 384)
```

Once model assets and the training environment are configured:

```bash
bash scripts/train.sh \
  dataloader=fr3_wuji \
  model.architecture.action_dim=54 \
  model.architecture.state_dim=54 \
  project.seed=42
```

The three-camera order, 54-D position/command order, and separate state/action
normalization must also be used by a future robot deployment client. This change
adds training data support; it does not add or execute a hardware controller.

Tests: `python -m pytest tests/dataloader/test_fr3_wuji.py -q`.

## 在服务器上拉取和训练

本仓库包含完整 OpenWAM 源码和 FR3/Wuji 适配。数据集、模型权重、虚拟环境和
本机生成的归一化文件不在 Git 中；服务器不需要本机的 OpenPI 仓库。

```bash
git clone https://github.com/YAnG-0419/openwam.git
cd openwam
```

按 README 的环境安装说明准备 Python、CUDA/PyTorch 和 `pip install -e .`。
Wan2.2 路径不依赖 Cosmos 子模块；若选择 Cosmos，请按 README 初始化对应子模块。

将完整的 `0924-250ep` 数据目录复制到服务器，保留 A、B 内的 `meta/`、`data/`、
`videos/` 结构。下面的路径请替换成服务器实际路径：

```bash
export FR3_WUJI_DATASET_DIR=/path/to/0924-250ep
python scripts/compute_fr3_wuji_stats.py \
  --dataset-dir "$FR3_WUJI_DATASET_DIR" \
  --out outputs/fr3_wuji/normalization_stats.npy
```

默认使用 Wan2.2-TI2V-5B 视频基础权重和新建的 54 维动作模型。可以运行官方
下载工具（会更新对应配置中的模型路径）：

```bash
python scripts/download_assets/download_video_backbone.py \
  --name Wan2.2-TI2V-5B --source huggingface --yes
```

先跑短训练，验证前向、反向、保存检查点及显存是否够用：

```bash
bash scripts/train.sh \
  dataloader=fr3_wuji \
  dataloader.dataset_dir="$FR3_WUJI_DATASET_DIR" \
  model.architecture.action_dim=54 \
  model.architecture.state_dim=54 \
  project.seed=42 \
  training.batch_size=1 \
  training.debug=true
```

脚本默认使用所有可见 GPU，可以通过 `CUDA_VISIBLE_DEVICES` 或 `NPROC_PER_NODE`
指定。`batch_size=1` 只是初始配置，不保证任意显存都能运行。短训练通过后移除
`training.debug=true`，再根据服务器显存调整每张卡的 batch size。
如果已有视频基础权重，可跳过下载，并在训练命令中传入
`model.video_backbone.model_path=/path/to/Wan2.2-TI2V-5B`。

这套命令不加载已有的 80 维完整 OpenWAM 检查点；若要微调此类检查点，需另行
适配动作和状态层。数据读取和多进程测试已在本机通过，完整 GPU 训练尚未运行。
