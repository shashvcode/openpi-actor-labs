# Collecting Data for VLA Fine-Tuning

How to collect, format, and store teleoperation data so it can be used to fine-tune a vision-language-action model (pi0.5) via OpenPI.

This guide is arm-agnostic. It applies to any robot controlled by joystick-style inputs with camera observations.

---

## What You Need to Record

Each training frame contains four things:

| Field | Type | Example |
|-------|------|---------|
| **State** | float32 vector | Joystick axes: `[left_x, left_y, right_x, right_y, l2_trigger, r2_trigger]` |
| **Action** | float32 vector (same dim as state) | The state at the *next* timestep |
| **Images** | RGB uint8 arrays | One per camera (e.g. scene overview, wrist close-up) |
| **Task label** | String | `"Pick up the bottle and place it on the yellow square"` |

### Key rule: action[t] = state[t+1]

The action at time `t` is defined as what the operator's inputs will be at time `t+1`. This means the model learns to predict "what should the controls do next given the current observation."

For the last frame in an episode, copy the current state as the action (there is no next frame).

---

## Recording Loop

Record at a fixed FPS (we use **10 Hz**). Each tick:

```python
import numpy as np

# 1. Read current controller inputs
state = np.array([left_x, left_y, right_x, right_y, l2, r2], dtype=np.float32)

# 2. Capture camera frames
scene_rgb = grab_frame(scene_camera)   # numpy uint8 (H, W, 3)
wrist_rgb = grab_frame(wrist_camera)   # numpy uint8 (H, W, 3)

# 3. Buffer this frame
episode_buffer.append({
    "state": state,
    "images": {"scene": scene_rgb, "wrist": wrist_rgb},
})
```

When the episode ends, convert the buffer to training format:

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset.create(
    repo_id="your_hf_username/your_dataset_name",
    fps=10,
    features={
        "observation.state": {"dtype": "float32", "shape": (N_DIMS,), "names": [dim_names]},
        "action":            {"dtype": "float32", "shape": (N_DIMS,), "names": [dim_names]},
        "observation.images.scene": {"dtype": "image", "shape": (3, H, W), "names": ["channels", "height", "width"]},
        "observation.images.wrist": {"dtype": "image", "shape": (3, H, W), "names": ["channels", "height", "width"]},
    },
)

for i in range(len(episode_buffer)):
    state = episode_buffer[i]["state"]
    action = episode_buffer[i + 1]["state"] if i < len(episode_buffer) - 1 else state

    dataset.add_frame({
        "observation.state": state,
        "action": action.copy(),
        "observation.images.scene": episode_buffer[i]["images"]["scene"],
        "observation.images.wrist": episode_buffer[i]["images"]["wrist"],
        "task": "your task description here",
    })

dataset.save_episode()
```

Repeat for each episode. The LeRobot library handles parquet files, metadata, and indexing automatically.

---

## Dataset Structure on Disk

After recording, the dataset looks like:

```
your_dataset/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── meta/
    ├── info.json              # Dataset metadata (features, fps, totals)
    ├── episodes.jsonl         # Per-episode info (length, task index)
    ├── episodes_stats.jsonl   # Per-episode normalization stats
    └── tasks.jsonl            # Task descriptions
```

Each parquet file contains all frames for one episode with columns:

| Column | Type | Description |
|--------|------|-------------|
| `observation.state` | `list<float32>` | Controller input vector |
| `action` | `list<float32>` | Next-step controller input |
| `observation.images.scene` | `struct{bytes, path}` | Scene camera JPEG (embedded) |
| `observation.images.wrist` | `struct{bytes, path}` | Wrist camera JPEG (embedded) |
| `timestamp` | `float32` | `frame_index / fps` |
| `frame_index` | `int64` | Frame number within episode (0-based) |
| `episode_index` | `int64` | Episode number (0-based) |
| `index` | `int64` | Global frame index across all episodes |
| `task_index` | `int64` | Index into tasks.jsonl |

---

## Uploading to HuggingFace

```python
dataset.push_to_hub()
```

Or manually:

```bash
huggingface-cli upload your_username/your_dataset ./your_dataset --repo-type dataset
```

Then tag it for LeRobot compatibility:

```python
from huggingface_hub import HfApi
HfApi().create_tag("your_username/your_dataset", tag="v2.1", repo_type="dataset")
```

---

## State/Action Dimensions

The state and action vectors must have the same dimensionality. Common setups:

| Setup | Dims | Vector |
|-------|------|--------|
| Dual-stick + 2 triggers | 6 | `[lx, ly, rx, ry, l2, r2]` |
| Dual-stick + 2 triggers + 2 buttons | 8 | `[lx, ly, rx, ry, l2, r2, l1, r1]` |
| Joint positions (6-DOF) | 6 | `[j1, j2, j3, j4, j5, j6]` |
| Joint positions + gripper | 7 | `[j1, j2, j3, j4, j5, j6, gripper]` |

The model config (`action_dim` in `TrainConfig`) must match this dimensionality.

---

## Cameras

- Any number of cameras works. Each gets its own `observation.images.<name>` column.
- Resolution at capture doesn't matter much -- the model resizes to 224x224 internally. We capture at 640x480 to keep file sizes reasonable.
- Use RGB format (convert from BGR if using OpenCV).
- Images are stored as JPEG bytes inside the parquet file. JPEG quality 75-8 is fine -- it saves disk space and the 224x224 resize during training dominates quality anyway.

---
