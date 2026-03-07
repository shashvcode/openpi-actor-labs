# LeRobot v2.1 Dataset Format — Exact Specification

This document defines the **exact** HuggingFace dataset format required by the OpenPI training pipeline. If your dataset matches this spec, `compute_norm_stats.py` and `train.py` will work with zero corrections.

**USE IMAGE FORMAT, NOT VIDEO.** The proven, battle-tested format stores camera frames directly inside parquet files as raw bytes — exactly how `verm11/runA` (SO-100) works. The video format (separate `.mp4` files) introduces timestamp validation, frame-count mismatches, FFmpeg dependencies, codec issues, and non-contiguous indexing bugs. Unless you have a very specific reason, always use image-in-parquet.

---

## Repository Structure (Image Format — Recommended)

```
your-hf-username/dataset-name/
├── .gitattributes
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   └── tasks.jsonl
└── data/
    └── chunk-000/
        ├── episode_000000.parquet
        ├── episode_000001.parquet
        ├── ...
        └── episode_000999.parquet
    └── chunk-001/              # only if >1000 episodes
        ├── episode_001000.parquet
        └── ...
```

No `videos/` directory. All camera frames are embedded in the parquet files.

### Chunking Rule
- Episodes are grouped into chunks of `chunks_size` (typically 1000).
- Episode N goes into `chunk-{N // chunks_size:03d}`.
- So episodes 0–999 go in `chunk-000/`, 1000–1999 in `chunk-001/`, etc.

---

## 1. `meta/info.json`

Every field below is **required**. Missing any one will cause a `KeyError` or `FileNotFoundError`.

```json
{
  "codebase_version": "v2.1",
  "robot_type": "your_robot_name",
  "total_episodes": 500,
  "total_frames": 150000,
  "total_tasks": 1,
  "chunks_size": 1000,
  "fps": 10,
  "splits": {
    "train": "0:500"
  },
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": null,
  "features": {
    ...
  }
}
```

### Field Details

| Field | Type | Description |
|---|---|---|
| `codebase_version` | string | Must be `"v2.1"` exactly. |
| `robot_type` | string | Arbitrary label, e.g. `"excavator"`, `"so100"`. |
| `total_episodes` | int | Total number of episodes in the dataset. |
| `total_frames` | int | Sum of all episode lengths (total rows across all parquets). |
| `total_tasks` | int | Number of distinct tasks. Usually `1`. |
| `chunks_size` | int | Episodes per chunk directory. Use `1000`. |
| `fps` | int | Frames per second the data was recorded at. |
| `splits` | object | `{"train": "0:N"}` where N = `total_episodes`. |
| `data_path` | string | **Use exactly** `"data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"`. |
| `video_path` | **null** | **Must be `null`** for image-in-parquet format. This is what makes it work like SO-100. |
| `features` | object | Schema of every column (see below). |

### Features Schema

Every column in your parquet files must be declared here. Missing a feature = crash.

#### Numeric features (state, action):

```json
"observation.state": {
  "dtype": "float32",
  "shape": [4],
  "names": [["left_x", "left_y", "right_x", "right_y"]]
}
```

```json
"action": {
  "dtype": "float32",
  "shape": [4],
  "names": [["left_x", "left_y", "right_x", "right_y"]]
}
```

- `shape` must match the actual array length in the parquet.
- `names` is a list-of-lists. The inner list has one name per dimension.

#### Image features (embedded in parquet):

```json
"observation.images.camera_name": {
  "dtype": "image",
  "shape": [3, 480, 640],
  "names": ["channels", "height", "width"]
}
```

- **dtype is `"image"`, NOT `"video"`.**
- Images are stored as `{"bytes": <raw JPEG/PNG bytes>, "path": null}` structs in each parquet row.
- Shape is `[C, H, W]` (channels first). Must match the actual image dimensions.

#### Required metadata columns:

These 5 columns must always be declared, even though they're auto-populated:

```json
"timestamp":     { "dtype": "float32", "shape": [1], "names": null },
"frame_index":   { "dtype": "int64",   "shape": [1], "names": null },
"episode_index": { "dtype": "int64",   "shape": [1], "names": null },
"index":         { "dtype": "int64",   "shape": [1], "names": null },
"task_index":    { "dtype": "int64",   "shape": [1], "names": null }
```

---

## 2. `meta/episodes.jsonl`

One JSON object per line. One line per episode. **No trailing comma, no array wrapper.**

```jsonl
{"episode_index": 0, "length": 300, "task_index": 0, "task": "Scoop peanuts from large pool and dump into small pool"}
{"episode_index": 1, "length": 250, "task_index": 0, "task": "Scoop peanuts from large pool and dump into small pool"}
{"episode_index": 2, "length": 280, "task_index": 0, "task": "Scoop peanuts from large pool and dump into small pool"}
```

| Field | Type | Description |
|---|---|---|
| `episode_index` | int | **Zero-based, sequential, no gaps.** |
| `length` | int | Number of frames in this episode. Must match rows in the parquet. |
| `task_index` | int | References `task_index` in `tasks.jsonl`. Usually `0`. |
| `task` | string | Human-readable task description. This becomes the model's text prompt during training. |

### Critical Rules:
- Episode indices must be **contiguous starting from 0**: `0, 1, 2, 3, ...`
- **No gaps** (e.g., skipping episode 5 or starting at 1 is NOT allowed).
- `length` must exactly match the number of rows in the corresponding parquet file.
- Number of lines must equal `total_episodes` in `info.json`.

---

## 3. `meta/tasks.jsonl`

One JSON object per line. Usually just one line for single-task datasets.

```jsonl
{"task_index": 0, "task": "Scoop peanuts from large pool and dump into small pool"}
```

| Field | Type | Description |
|---|---|---|
| `task_index` | int | Zero-based index. |
| `task` | string | Must match the `task` field used in `episodes.jsonl`. |

---

## 4. `meta/episodes_stats.jsonl`

One JSON object per line. One line per episode. Per-episode statistics for normalization.

```jsonl
{"episode_index": 0, "stats": {"observation.state": {"min": [-1.0, -1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0, 1.0], "mean": [0.01, -0.02, 0.03, 0.0], "std": [0.2, 0.3, 0.1, 0.3], "count": [300]}, "action": {"min": [-1.0, -1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0, 1.0], "mean": [0.01, -0.02, 0.03, 0.0], "std": [0.2, 0.3, 0.1, 0.3], "count": [300]}}}
```

### Structure:

```json
{
  "episode_index": 0,
  "stats": {
    "observation.state": {
      "min":   [float, float, ...],
      "max":   [float, float, ...],
      "mean":  [float, float, ...],
      "std":   [float, float, ...],
      "count": [int]
    },
    "action": {
      "min":   [float, float, ...],
      "max":   [float, float, ...],
      "mean":  [float, float, ...],
      "std":   [float, float, ...],
      "count": [int]
    }
  }
}
```

### Critical Rules:
- Stats must be nested under the `"stats"` key. **Not** at the top level.
- `min`, `max`, `mean`, `std` arrays must have length equal to the feature's dimension (e.g., 4 for 4-dim state).
- `count` must be a **single-element list** `[N]` where N = number of frames in the episode. **Not** `[N, N, N, N]`.
- `count` must match the `length` in `episodes.jsonl` for the same episode.
- Include stats for `observation.state` and `action` only. Do NOT include stats for image features.

---

## 5. Parquet Files (`data/chunk-XXX/episode_XXXXXX.parquet`)

Each episode gets one parquet file. Every parquet must contain these columns:

| Column | Type | Description |
|---|---|---|
| `observation.state` | `list<float>` | State vector. Length must match `shape` in `info.json`. |
| `action` | `list<float>` | Action vector. Length must match `shape` in `info.json`. |
| `observation.images.cam_name` | `struct{bytes: binary, path: string}` | Raw image bytes per camera. One column per camera. |
| `timestamp` | `float` | Time in seconds from episode start. |
| `frame_index` | `int64` | Zero-based index within the episode. `0, 1, 2, ...` |
| `episode_index` | `int64` | Same value for every row in the file. Matches the episode index. |
| `index` | `int64` | Global frame index across the entire dataset. |
| `task_index` | `int64` | References `tasks.jsonl`. Usually `0` for every row. |

### Image Column Format

Each image column stores frames as a struct with two fields:

```
{"bytes": <raw JPEG or PNG bytes>, "path": null}
```

In PyArrow, this is `struct<bytes: binary, path: string>`. The `path` field is always `null` — all image data goes in `bytes`.

When recording, encode each camera frame as JPEG:
```python
import cv2
_, jpeg_bytes = cv2.imencode('.jpg', frame)
row["observation.images.camera_name"] = {"bytes": jpeg_bytes.tobytes(), "path": None}
```

### Timestamp Rules

Timestamps must satisfy: `timestamp[i] = frame_index[i] / fps`

So for 10 FPS:
```
frame 0 → timestamp 0.0
frame 1 → timestamp 0.1
frame 2 → timestamp 0.2
...
```

Timestamps must be **monotonically increasing** within each episode, starting from `0.0`. LeRobot validates this with a tolerance of `1e-4` seconds. Violations cause `ValueError`.

### Global Index

The `index` column is a globally unique frame counter across the entire dataset:
- Episode 0 (300 frames): index 0–299
- Episode 1 (250 frames): index 300–549
- Episode 2 (280 frames): index 550–829
- etc.

### Frame Synchronization

**All cameras must be captured at the same instant for each frame.** The parquet row represents a single synchronized timestep. When recording:

1. Capture all cameras simultaneously (or as close to simultaneously as possible)
2. Read joystick state at the same instant
3. Bundle everything into one parquet row with one `timestamp`

This is critical — if cameras are asynchronous or at different FPS, the model will learn from misaligned visual/action pairs and perform poorly.

---

## 6. HuggingFace Tag

The dataset repository **must** have a git tag named `v2.1`. Without it, LeRobot raises `RevisionNotFoundError`.

Create it via:
```python
from huggingface_hub import HfApi
api = HfApi()
api.create_tag("your-username/dataset-name", tag="v2.1", repo_type="dataset")
```

---

## 7. Checklist Before Training

- [ ] `video_path` is `null` (images in parquet, NOT separate video files)
- [ ] `info.json` has all required fields (`chunks_size`, `data_path`, `splits`, etc.)
- [ ] Every camera used in training is declared in `features` with `"dtype": "image"`
- [ ] Feature `shape` values match actual image dimensions
- [ ] `episodes.jsonl` has contiguous 0-based indices: `0, 1, 2, ...` with NO gaps
- [ ] `episodes_stats.jsonl` uses `{"episode_index": N, "stats": {...}}` format
- [ ] `episodes_stats.jsonl` has `"count": [N]` (single-element list, not per-dimension)
- [ ] `episodes_stats.jsonl` count matches episode length
- [ ] Parquet columns include all 5 metadata columns + state + action + image columns
- [ ] Each parquet image column is `struct{bytes: binary, path: string}`
- [ ] Timestamps start at `0.0` per episode and increment by exactly `1/fps`
- [ ] `frame_index` is `0, 1, 2, ...` within each episode
- [ ] Global `index` column is contiguous across the whole dataset
- [ ] `tasks.jsonl` exists with at least one task
- [ ] Repository has a `v2.1` git tag
- [ ] `total_frames` = sum of all episode `length` values
- [ ] `total_episodes` = number of lines in `episodes.jsonl`

---

## 8. Example: Excavator Dataset (2 cameras, image format)

For an excavator with 2 cameras (`csi_0_imx219` cab-mounted + `usb_0` side-mounted), 4-dim joystick state, 500 episodes at 10 FPS.

This uses the **same image-in-parquet format as SO-100** — no video files, no video decoding, no FFmpeg dependency.

**`meta/info.json`:**
```json
{
  "codebase_version": "v2.1",
  "robot_type": "excavator",
  "total_episodes": 500,
  "total_frames": 125000,
  "total_tasks": 1,
  "chunks_size": 1000,
  "fps": 10,
  "splits": {"train": "0:500"},
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": null,
  "features": {
    "observation.state": {
      "dtype": "float32",
      "shape": [4],
      "names": [["left_x", "left_y", "right_x", "right_y"]]
    },
    "action": {
      "dtype": "float32",
      "shape": [4],
      "names": [["left_x", "left_y", "right_x", "right_y"]]
    },
    "observation.images.csi_0_imx219": {
      "dtype": "image",
      "shape": [3, 480, 640],
      "names": ["channels", "height", "width"]
    },
    "observation.images.usb_0": {
      "dtype": "image",
      "shape": [3, 480, 640],
      "names": ["channels", "height", "width"]
    },
    "timestamp":     {"dtype": "float32", "shape": [1], "names": null},
    "frame_index":   {"dtype": "int64",   "shape": [1], "names": null},
    "episode_index": {"dtype": "int64",   "shape": [1], "names": null},
    "index":         {"dtype": "int64",   "shape": [1], "names": null},
    "task_index":    {"dtype": "int64",   "shape": [1], "names": null}
  }
}
```

**Parquet schema (each row):**
```
observation.state          → [0.1, -0.3, 0.0, 0.5]           (list of 4 floats)
action                     → [0.1, -0.3, 0.0, 0.5]           (list of 4 floats)
observation.images.csi_0_imx219 → {"bytes": <JPEG>, "path": null}  (cab camera frame)
observation.images.usb_0        → {"bytes": <JPEG>, "path": null}  (side camera frame)
timestamp                  → 0.0                               (float)
frame_index                → 0                                 (int64)
episode_index              → 0                                 (int64)
index                      → 0                                 (int64, globally unique)
task_index                 → 0                                 (int64)
```

**`meta/tasks.jsonl`:**
```jsonl
{"task_index": 0, "task": "Scoop packing peanuts from large pool and dump into small pool"}
```

**`meta/episodes.jsonl`** (one line per episode):
```jsonl
{"episode_index": 0, "length": 250, "task_index": 0, "task": "Scoop packing peanuts from large pool and dump into small pool"}
{"episode_index": 1, "length": 230, "task_index": 0, "task": "Scoop packing peanuts from large pool and dump into small pool"}
{"episode_index": 2, "length": 280, "task_index": 0, "task": "Scoop packing peanuts from large pool and dump into small pool"}
...
```

**`meta/episodes_stats.jsonl`** (one line per episode):
```jsonl
{"episode_index": 0, "stats": {"observation.state": {"min": [-0.5, -0.8, -0.3, -0.7], "max": [0.6, 0.9, 0.4, 0.8], "mean": [0.01, -0.02, 0.03, 0.0], "std": [0.2, 0.3, 0.1, 0.3], "count": [250]}, "action": {"min": [-0.5, -0.8, -0.3, -0.7], "max": [0.6, 0.9, 0.4, 0.8], "mean": [0.01, -0.02, 0.03, 0.0], "std": [0.2, 0.3, 0.1, 0.3], "count": [250]}}}
{"episode_index": 1, "stats": {"observation.state": {"min": [-0.4, -0.7, -0.2, -0.6], "max": [0.5, 0.8, 0.3, 0.7], "mean": [0.02, -0.01, 0.02, 0.01], "std": [0.15, 0.25, 0.12, 0.28], "count": [230]}, "action": {"min": [-0.4, -0.7, -0.2, -0.6], "max": [0.5, 0.8, 0.3, 0.7], "mean": [0.02, -0.01, 0.02, 0.01], "std": [0.15, 0.25, 0.12, 0.28], "count": [230]}}}
...
```

### Recording Code Pattern

```python
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

episode_rows = []
global_index = 0  # tracks across ALL episodes

for frame_idx in range(num_frames):
    # Capture all cameras at the SAME instant
    _, cam_cab = cap_cab.read()
    _, cam_side = cap_side.read()

    # Read joystick state at the same instant
    state = get_joystick_state()  # [left_x, left_y, right_x, right_y]

    # Encode as JPEG
    _, cab_jpg = cv2.imencode('.jpg', cam_cab)
    _, side_jpg = cv2.imencode('.jpg', cam_side)

    episode_rows.append({
        "observation.state": state.tolist(),
        "action": state.tolist(),  # action = current state for teleop
        "observation.images.csi_0_imx219": {"bytes": cab_jpg.tobytes(), "path": None},
        "observation.images.usb_0": {"bytes": side_jpg.tobytes(), "path": None},
        "timestamp": frame_idx / FPS,
        "frame_index": frame_idx,
        "episode_index": current_episode,
        "index": global_index,
        "task_index": 0,
    })
    global_index += 1

# Write parquet
table = pa.Table.from_pylist(episode_rows)
pq.write_table(table, f"data/chunk-000/episode_{current_episode:06d}.parquet")
```
