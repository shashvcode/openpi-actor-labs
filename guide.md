# OpenPI Fine-Tuning Guide

Complete guide to fine-tuning a pi0.5 LoRA model for a new robot using the OpenPI framework. Covers every step from raw repo to running inference on physical hardware.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Data Format](#2-data-format)
3. [Collecting Data](#3-collecting-data)
4. [Uploading to HuggingFace](#4-uploading-to-huggingface)
5. [Modifying the OpenPI Codebase](#5-modifying-the-openpi-codebase)
6. [GPU Setup](#6-gpu-setup)
7. [Computing Normalization Stats](#7-computing-normalization-stats)
8. [Training](#8-training)
9. [Evaluation](#9-evaluation)
10. [Serving & Inference](#10-serving--inference)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Prerequisites

- A fork of the [OpenPI repo](https://github.com/Physical-Intelligence/openpi)
- A HuggingFace account with a write token
- A cloud GPU (RunPod, Vast.ai, Lambda, etc.)
  - Minimum: 1x A100 40GB (slower) or 1x A100 80GB (recommended)
  - Faster: 8x H100 80GB (finishes 15k steps in ~50 min)
  - Disk: at least 50GB, ideally 100GB+
- Teleoperation data: 50-300 episodes of your robot performing tasks
- Python 3.11+

---

## 2. Data Format

OpenPI expects data in **LeRobot v2** format. Each training frame contains:

| Field | Type | Description |
|-------|------|-------------|
| `observation.state` | `float32[N]` | Control input vector (joystick axes, joint positions, etc.) |
| `action` | `float32[N]` | The state at time `t+1` (what the operator does next) |
| `observation.images.<cam>` | JPEG bytes | One column per camera, stored as `struct{bytes, path}` |
| `timestamp` | `float32` | `frame_index / fps` |
| `frame_index` | `int64` | 0-based within each episode |
| `episode_index` | `int64` | Episode number (0-based) |
| `index` | `int64` | Global frame counter across all episodes |
| `task_index` | `int64` | Index into tasks.jsonl |

### Action Convention

```
action[t] = state[t+1]
```

The model learns to predict "what should the controls do next." For the last frame of an episode, copy the current state as the action.

### Directory Layout

```
your_dataset/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── meta/
    ├── info.json
    ├── episodes.jsonl
    ├── episodes_stats.jsonl
    └── tasks.jsonl
```

### info.json

```json
{
  "codebase_version": "v2.1",
  "robot_type": "your_robot",
  "total_episodes": 150,
  "total_frames": 30718,
  "total_tasks": 1,
  "chunks_size": 1000,
  "fps": 10,
  "splits": {"train": "0:150"},
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "features": {
    "observation.state": {
      "dtype": "float32",
      "shape": [6],
      "names": [["dim1", "dim2", "dim3", "dim4", "dim5", "dim6"]]
    },
    "action": {
      "dtype": "float32",
      "shape": [6],
      "names": [["dim1", "dim2", "dim3", "dim4", "dim5", "dim6"]]
    },
    "observation.images.scene": {
      "dtype": "image",
      "shape": [3, 480, 640],
      "names": ["channels", "height", "width"]
    },
    "observation.images.wrist": {
      "dtype": "image",
      "shape": [3, 480, 640],
      "names": ["channels", "height", "width"]
    },
    "timestamp":      {"dtype": "float32", "shape": [1], "names": null},
    "frame_index":    {"dtype": "int64",   "shape": [1], "names": null},
    "episode_index":  {"dtype": "int64",   "shape": [1], "names": null},
    "index":          {"dtype": "int64",   "shape": [1], "names": null},
    "task_index":     {"dtype": "int64",   "shape": [1], "names": null}
  }
}
```

**Critical**: The `data_path` template must use `episode_chunk`, not `chunk_index`.

### episodes.jsonl

One JSON object per line, one per episode:

```json
{"episode_index": 0, "length": 203, "task_index": 0, "task": "Pick up the bottle"}
{"episode_index": 1, "length": 187, "task_index": 0, "task": "Pick up the bottle"}
```

### tasks.jsonl

```json
{"task_index": 0, "task": "Pick up the bottle and place it on the yellow square."}
```

### episodes_stats.jsonl

One JSON per line per episode with min/max/mean/std/count for numerical columns. The `count` field must be a **scalar** `[N]`, not a vector:

```json
{"episode_index": 0, "stats": {"observation.state": {"mean": [0.1, ...], "std": [0.2, ...], "min": [-1.0, ...], "max": [1.0, ...], "count": [203]}, "action": {"mean": [0.1, ...], "std": [0.2, ...], "min": [-1.0, ...], "max": [1.0, ...], "count": [203]}}}
```

### Data Quality Rules

1. **Consistent dimensions**: Every episode must have the exact same state/action dimensionality. No mixing 6-dim and 8-dim.
2. **Monotonic timestamps**: Must increase at exactly `1/fps`. Use `frame_index / fps`.
3. **Image compression**: JPEG quality 50-70 is fine. The model resizes to 224x224 during training anyway. High quality images just waste disk.
4. **Image format**: Images stored as `struct{bytes: binary, path: string}` in parquet. The `bytes` field contains raw JPEG data.

---

## 3. Collecting Data

Record teleoperation at a fixed FPS (10 Hz recommended):

```python
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset.create(
    repo_id="your_username/your_dataset",
    fps=10,
    features={
        "observation.state": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": [["dim1", "dim2", ...]],
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": [["dim1", "dim2", ...]],
        },
        "observation.images.scene": {
            "dtype": "image",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        },
    },
)

# During teleoperation, buffer frames at 10 Hz:
episode_buffer = []
# ... append {"state": np.array(...), "images": {"scene": rgb, "wrist": rgb}} each tick ...

# When episode ends, convert to training format:
for i in range(len(episode_buffer)):
    state = episode_buffer[i]["state"]
    action = episode_buffer[i + 1]["state"] if i < len(episode_buffer) - 1 else state

    dataset.add_frame({
        "observation.state": state,
        "action": action.copy(),
        "observation.images.scene": episode_buffer[i]["images"]["scene"],
        "observation.images.wrist": episode_buffer[i]["images"]["wrist"],
        "task": "Your task description here",
    })

dataset.save_episode()
```

### Tips

- **50-150 episodes** is enough for LoRA fine-tuning on a single task
- **Vary starting positions** -- don't always place objects in the same spot
- **Include recovery** -- if you make a mistake, correct it rather than restarting
- **Consistent task labels** -- use the exact same string for all episodes of the same task
- **Keep episodes under ~60 seconds** (600 frames at 10 Hz)

---

## 4. Uploading to HuggingFace

```bash
pip install huggingface-hub[cli]
huggingface-cli login
huggingface-cli upload your_username/your_dataset ./your_dataset --repo-type dataset
```

Then **tag it** (required for LeRobot compatibility):

```python
from huggingface_hub import HfApi
HfApi().create_tag("your_username/your_dataset", tag="v2.1", repo_type="dataset")
```

Without the `v2.1` tag, the data loader will fail silently.

For large datasets (>5 GB), upload in batches to avoid memory crashes.

---

## 5. Modifying the OpenPI Codebase

You need to edit 4 files:

### 5.1 Create: `src/openpi/policies/your_robot_policy.py`

This maps your dataset columns to what the model expects internally.

The model has these internal keys:

| Model Key | What It Is |
|-----------|------------|
| `state` | Your observation state vector |
| `image/base_0_rgb` | Primary camera (uint8 HxWx3) |
| `image/left_wrist_0_rgb` | Second camera |
| `image/right_wrist_0_rgb` | Third camera (zeros if unused) |
| `actions` | Action vector |
| `prompt` | Natural language task string |

```python
import dataclasses
import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

ACTION_DIM = 6  # your action dimensionality

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@dataclasses.dataclass(frozen=True)
class YourRobotInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image_scene"])
        wrist_image = _parse_image(data["observation/image_wrist"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs

@dataclasses.dataclass(frozen=True)
class YourRobotOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}
```

### 5.2 Edit: `src/openpi/training/config.py`

Add a data config class and a training config.

```python
# At the top, add import:
import openpi.policies.your_robot_policy as your_robot_policy

# Add data config class (inside the file, near other LeRobot configs):
@dataclasses.dataclass
class LeRobotYourRobotDataConfig:
    # ... define repack_transforms mapping LeRobot columns to your policy keys
    # ... define data_transforms including YourRobotInputs, ResizeImages, etc.
    # Key decision: include DeltaActions if your actions are absolute positions,
    # skip DeltaActions if your actions are already relative (joystick/velocity)
    pass

# Add training config (in the CONFIGS list):
TrainConfig(
    name="pi05_yourrobot_lora",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=ACTION_DIM,      # must match your state/action vector
        action_horizon=10,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotYourRobotDataConfig(
        repo_id="your_username/your_dataset",
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    freeze_filter=pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    ema_decay=None,
    num_train_steps=15_000,
    save_interval=5_000,
    keep_period=5_000,
    batch_size=32,
),
```

### 5.3 Edit: `src/openpi/training/weight_loaders.py`

The base pi0.5 checkpoint has 32-dim action layers. If your `action_dim` is different, loading will crash with a shape mismatch. Fix by skipping mismatched weights in `_merge_params`:

```python
result = {}
for k, v in flat_loaded.items():
    if k in flat_ref:
        ref = flat_ref[k]
        if hasattr(v, "shape") and hasattr(ref, "shape") and v.shape != ref.shape:
            logger.info(
                "Skipping weight %s: shape mismatch (loaded %s, expected %s)",
                k, v.shape, ref.shape,
            )
            result[k] = ref  # keep randomly initialized placeholder
            continue
        result[k] = v.astype(ref.dtype) if v.dtype != ref.dtype else v
```

This lets the action projection layers initialize randomly and learn from scratch, while preserving all pretrained vision/language weights.

### 5.4 Edit: `src/openpi/training/data_loader.py`

LeRobot v2 stores images as `{bytes, path}` dicts in parquet. The HuggingFace datasets library doesn't auto-decode these. Add a picklable transform class:

```python
class _ImageDecodingTransform:
    """Decode image columns stored as {bytes, path} structs."""
    def __init__(self, image_keys: list[str]):
        self.image_keys = image_keys

    def __call__(self, items_dict):
        import io
        from PIL import Image
        from torchvision import transforms as tv_transforms
        to_tensor = tv_transforms.ToTensor()
        for key in self.image_keys:
            if key in items_dict:
                decoded = []
                for item in items_dict[key]:
                    if isinstance(item, dict) and "bytes" in item and item["bytes"] is not None:
                        decoded.append(
                            to_tensor(Image.open(io.BytesIO(item["bytes"])).convert("RGB"))
                        )
                    elif isinstance(item, Image.Image):
                        decoded.append(to_tensor(item))
                    else:
                        decoded.append(item)
                items_dict[key] = decoded
        return items_dict
```

This must be a **top-level class**, not a closure, because PyTorch's DataLoader uses pickle for multiprocessing workers.

---

## 6. GPU Setup

### Provision a machine

Recommended specs:
- **GPU**: 1x A100 80GB or 8x H100 80GB
- **Disk**: 50GB+ (use `/workspace` on RunPod, not root `/`)
- **Template**: PyTorch (includes CUDA)

### Initial setup commands

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/YOUR_FORK.git openpi
cd openpi

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install dependencies
uv sync

# Login to HuggingFace
uv run huggingface-cli login
```

### RunPod-specific: survive pod restarts

Always clone into `/workspace` (persistent volume), not `/` (ephemeral):

```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/YOUR_FORK.git openpi
cd openpi

# Symlink checkpoints to /workspace so they survive restarts
rm -rf checkpoints
ln -s /workspace/checkpoints checkpoints
mkdir -p /workspace/checkpoints
```

### After a pod restart (re-setup)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
cd /workspace/openpi
uv sync
```

Everything in `/workspace` persists. Only `uv` (installed to root) needs reinstalling.

### Verify GPU access

```bash
nvidia-smi
uv run python -c "import jax; print(f'{len(jax.devices())} devices:', jax.devices())"
```

---

## 7. Computing Normalization Stats

```bash
HF_HOME=/workspace/.hf_home uv run scripts/compute_norm_stats.py \
    --config-name pi05_yourrobot_lora
```

This:
1. Downloads your dataset from HuggingFace (~2-10 min depending on size)
2. Iterates through every frame computing mean/std for state and actions
3. Saves `norm_stats.json` to `assets/<config_name>/<repo_id>/`

Must complete before training. Only needs to run once per dataset.

---

## 8. Training

### Start training (survives SSH disconnect)

```bash
nohup bash -c 'export HF_HOME=/workspace/.hf_home && \
  source $HOME/.local/bin/env && \
  cd /workspace/openpi && \
  WANDB_MODE=disabled XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_yourrobot_lora \
  --exp-name=run1 --overwrite' \
  > /workspace/train.log 2>&1 &
```

### Environment variables explained

| Variable | Value | Purpose |
|----------|-------|---------|
| `WANDB_MODE=disabled` | Skip Weights & Biases | Remove if you want experiment tracking |
| `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` | Use 90% of GPU VRAM | Prevents OOM for LoRA fine-tuning |
| `HF_HOME=/workspace/.hf_home` | HuggingFace cache on persistent disk | Prevents root filesystem from filling up |

### Monitor training

```bash
# Check latest progress
tail -20 /workspace/train.log

# Watch live (Ctrl+C to stop watching, training continues)
tail -f /workspace/train.log

# Check if process is still running
ps aux | grep train.py
```

### What to expect

```
Step    0: loss=1.5220
Step  500: loss=0.6338
Step 1000: loss=0.2998
Step 2000: loss=0.1987
Step 5000: loss=0.1430
Step 10000: loss=0.1106
Step 15000: loss=~0.09-0.10
```

- Loss drops steeply in the first 1-2k steps, then gradually
- If loss plateaus above 1.0, something is wrong with data or config
- The action projection layers start from random initialization, so the first few hundred steps show rapid improvement as they catch up

### Timing estimates

| Setup | 5k steps | 15k steps | 30k steps |
|-------|----------|-----------|-----------|
| 1x A100 80GB | ~2 hr | ~6 hr | ~12 hr |
| 8x H100 80GB | ~15 min | ~50 min | ~1.5 hr |

### Checkpoints

With `save_interval=5_000` and `keep_period=5_000`, checkpoints save every 5k steps. Old ones are deleted to save disk. Each checkpoint is ~10GB.

Checkpoints save to: `checkpoints/<config_name>/<exp_name>/<step>/`

---

## 9. Evaluation

### Setup

After training, copy norm stats into the checkpoint directory:

```bash
STEP=14999  # the final step number
CONFIG=pi05_yourrobot_lora
EXP=run1
REPO=your_username/your_dataset

mkdir -p checkpoints/$CONFIG/$EXP/$STEP/assets/$REPO/
cp assets/$CONFIG/$REPO/norm_stats.json \
   checkpoints/$CONFIG/$EXP/$STEP/assets/$REPO/
```

### Run evaluation

```bash
# Sanity check on training data
HF_HOME=/workspace/.hf_home uv run scripts/eval_offline.py \
    --checkpoint-dir checkpoints/$CONFIG/$EXP/$STEP \
    --source hf --num-episodes 30 --stride 5

# Generalization test on held-out data (if available on S3)
HF_HOME=/workspace/.hf_home uv run scripts/eval_offline.py \
    --checkpoint-dir checkpoints/$CONFIG/$EXP/$STEP \
    --source s3 --num-episodes 30 --stride 5
```

### Interpreting results

| Metric | Good | Okay | Bad |
|--------|------|------|-----|
| MSE (training) | < 0.03 | 0.03-0.07 | > 0.07 |
| MSE (held-out) | < 0.05 | 0.05-0.10 | > 0.10 |
| Cosine Similarity | > 0.7 | 0.4-0.7 | < 0.4 |

- **Small train/held-out gap** = underfitting, need more steps
- **Large train/held-out gap** = overfitting, need more data
- **Both high MSE** = data issue, config issue, or need way more training

### Upload best model to HuggingFace

```bash
HF_HOME=/workspace/.hf_home uv run huggingface-cli upload \
    your_username/your_model_name \
    checkpoints/$CONFIG/$EXP/$STEP \
    --repo-type model
```

Always upload immediately after training. Cloud instances are ephemeral.

---

## 10. Serving & Inference

### Start the policy server (on GPU machine)

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_yourrobot_lora \
    --policy.dir checkpoints/$CONFIG/$EXP/$STEP
```

This starts a WebSocket server on port 8000.

### SSH tunnel (from local machine to GPU)

```bash
ssh -L 8000:localhost:8000 your-gpu-ssh-command
```

### Run on physical robot (from local machine)

```bash
python examples/your_robot/run_policy.py \
    --host localhost --port 8000 \
    --arm-port /dev/tty.usbmodemXXX \
    --scene-cam 1 --wrist-cam 2 \
    --prompt "Your task description"
```

### Dry run (no physical robot)

```bash
python examples/your_robot/run_policy.py \
    --host localhost --port 8000 \
    --dry-run --scene-cam 1 --wrist-cam 2
```

---

## 11. Troubleshooting

### No space left on device

The most common issue. Prevention:

- Clone repo to `/workspace`, not `/`
- Symlink `checkpoints` to `/workspace`
- Set `HF_HOME=/workspace/.hf_home`
- Use `save_interval` and `keep_period` to limit checkpoint count
- JPEG quality 50-70 for images in dataset

Cleanup commands:

```bash
rm -rf ~/.cache/huggingface/datasets/
rm -rf /tmp/*
rm -rf checkpoints/*/run*/  # delete old runs
df -h / /workspace
```

### Shape mismatch loading checkpoint

```
ValueError: Shape mismatch at ['action_in_proj']['kernel']
```

The base model has 32-dim actions, yours has N-dim. Apply the weight_loaders.py fix from Section 5.3.

### KeyError: 'chunk_index'

Your `info.json` uses wrong template variable. Must be `episode_chunk`:

```json
"data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
```

### Timestamp violation

```
ValueError: timestamps unexpectedly violate the tolerance
```

Timestamps must be `frame_index / fps`. Replace raw timestamps:

```python
df["timestamp"] = df["frame_index"].astype(float) / fps
```

### Can't pickle local object

```
AttributeError: Can't pickle local object '_patch_image_transform.<locals>...'
```

The image decoding transform must be a top-level class, not a closure. See Section 5.4.

### SSH keeps disconnecting

Use `nohup` for all long-running commands:

```bash
nohup bash -c 'your_command_here' > /workspace/output.log 2>&1 &
tail -f /workspace/output.log
```

### Pod restarted, lost everything

If you cloned to `/workspace`, your code and checkpoints survive. Only `uv` needs reinstalling:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
cd /workspace/openpi
uv sync
```

---

## Summary: Files You Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/openpi/policies/your_robot_policy.py` | Create | Map your robot's I/O to model format |
| `src/openpi/training/config.py` | Edit | Register data config + training config |
| `src/openpi/training/weight_loaders.py` | Edit | Skip shape-mismatched weights |
| `src/openpi/training/data_loader.py` | Edit | Decode `{bytes,path}` images |
| HuggingFace dataset | Upload | Your data in LeRobot v2 format + `v2.1` tag |

## Summary: Commands to Run

```bash
# 1. Compute stats (once)
uv run scripts/compute_norm_stats.py --config-name pi05_yourrobot_lora

# 2. Train
WANDB_MODE=disabled XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_yourrobot_lora --exp-name=run1 --overwrite

# 3. Eval
uv run scripts/eval_offline.py --checkpoint-dir checkpoints/.../STEP \
  --source hf --num-episodes 30 --stride 5

# 4. Serve
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_yourrobot_lora --policy.dir checkpoints/.../STEP
```
