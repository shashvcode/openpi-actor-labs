# LeRobot v2.1 Dataset Format — Exact Specification

This document defines the **exact** HuggingFace dataset format required by the openpi training pipeline. If your dataset matches this spec, `compute_norm_stats.py` and `train.py` will work with zero corrections.

---

## Repository Structure

```
your-hf-username/dataset-name/
├── .gitattributes
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   └── tasks.jsonl
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       ├── ...
│       └── episode_000999.parquet
│   └── chunk-001/              # only if >1000 episodes
│       ├── episode_001000.parquet
│       └── ...
└── videos/                     # only for video-based datasets
    └── chunk-000/
        └── observation.images.camera_name/
            ├── episode_000000.mp4
            ├── episode_000001.mp4
            └── ...
```

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
  "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
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
| `data_path` | string | Format string for parquet file paths. **Use exactly** `"data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"`. |
| `video_path` | string or null | Format string for video paths. Set to `null` if images are embedded in parquet. If using video: `"videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"`. |
| `features` | object | Schema of every column (see below). |

### Features Schema

Every column in your parquet files **and** every video camera must be declared here. Missing a feature = crash.

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

#### Image features (embedded in parquet, `video_path: null`):

```json
"observation.images.camera_name": {
  "dtype": "image",
  "shape": [3, 480, 640],
  "names": ["channels", "height", "width"]
}
```

- Images are stored as `{"bytes": <raw bytes>, "path": null}` structs in the parquet.
- Shape is `[C, H, W]` (channels first).

#### Video features (stored as .mp4 files):

```json
"observation.images.camera_name": {
  "dtype": "video",
  "shape": [3, 480, 640],
  "names": ["channels", "height", "width"],
  "video_info": {
    "video.fps": 10.0,
    "video.height": 480,
    "video.width": 640,
    "video.codec": "av1",
    "has_audio": false
  }
}
```

- The feature key (`observation.images.camera_name`) must exactly match the subdirectory name under `videos/chunk-XXX/`.
- Every episode listed in `episodes.jsonl` **must** have a corresponding `.mp4` file for every declared video feature. Missing files = crash.

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
| `episode_index` | int | Zero-based, sequential, no gaps. |
| `length` | int | Number of frames in this episode. Must match rows in the parquet. |
| `task_index` | int | References `task_index` in `tasks.jsonl`. Usually `0`. |
| `task` | string | Human-readable task description. This becomes the model's text prompt during training. |

### Critical Rules:
- Episode indices must be **contiguous** starting from 0: `0, 1, 2, 3, ...`
- No gaps (e.g., skipping episode 380 or 565 is NOT allowed).
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
- `count` must be a **single-element list** `[N]` where N = number of frames. **Not** `[N, N, N, N]`.
- Include stats for `observation.state` and `action` at minimum. Do NOT include stats for image/video features.

---

## 5. Parquet Files (`data/chunk-XXX/episode_XXXXXX.parquet`)

Each episode gets one parquet file. Every parquet must contain these columns:

| Column | Type | Description |
|---|---|---|
| `observation.state` | list of float32 | State vector. Length must match `shape` in `info.json`. |
| `action` | list of float32 | Action vector. Length must match `shape` in `info.json`. |
| `timestamp` | float32 | Time in seconds from episode start. First frame = `0.0`, second = `1/fps`, etc. |
| `frame_index` | int64 | Zero-based index within the episode. `0, 1, 2, ...` |
| `episode_index` | int64 | Same value for every row in the file. Matches the episode index. |
| `index` | int64 | Global frame index across the entire dataset. |
| `task_index` | int64 | References `tasks.jsonl`. Usually `0` for every row. |

If using **image** storage (not video):
| `observation.images.camera_name` | struct `{bytes, path}` | Raw image bytes. |

If using **video** storage:
- Image columns are **not** in the parquet.
- Videos are stored as `.mp4` files in the `videos/` directory.

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

---

## 6. Video Files (`videos/chunk-XXX/camera_key/episode_XXXXXX.mp4`)

Only relevant when `video_path` is not null.

- One `.mp4` per episode per camera.
- The directory name must **exactly** match the feature key in `info.json`.
  - Feature key: `observation.images.csi_0_imx219`
  - Directory: `videos/chunk-000/observation.images.csi_0_imx219/episode_000000.mp4`
- Video frame count must match the `length` in `episodes.jsonl`.
- Video FPS should match `fps` in `info.json`.
- Requires FFmpeg installed on the training machine (`apt-get install -y ffmpeg`).

---

## 7. HuggingFace Tag

The dataset repository **must** have a git tag named `v2.1`. Without it, LeRobot raises `RevisionNotFoundError`.

Create it via:
```python
from huggingface_hub import HfApi
api = HfApi()
api.create_tag("your-username/dataset-name", tag="v2.1", repo_type="dataset")
```

---

## 8. Checklist Before Training

- [ ] `info.json` has all required fields (`chunks_size`, `data_path`, `video_path`, `splits`, etc.)
- [ ] Every camera used in training is declared in `features`
- [ ] Feature `shape` values match actual data dimensions
- [ ] `episodes.jsonl` has contiguous indices `0, 1, 2, ...` with no gaps
- [ ] `episodes_stats.jsonl` uses `{"episode_index": N, "stats": {...}}` format
- [ ] `episodes_stats.jsonl` has `"count": [N]` (single-element list, not per-dimension)
- [ ] Parquet columns include all 5 metadata columns (`timestamp`, `frame_index`, `episode_index`, `index`, `task_index`)
- [ ] Timestamps start at `0.0` per episode and increment by `1/fps`
- [ ] Global `index` column is contiguous across the whole dataset
- [ ] Every declared video feature has a `.mp4` file for every episode
- [ ] `tasks.jsonl` exists with at least one task
- [ ] Repository has a `v2.1` git tag
- [ ] FFmpeg is installed on the training machine (for video datasets)

---

## 9. Example: Minimal Excavator Dataset

For an excavator with 1 camera (`csi_0_imx219`), 4-dim joystick state, 500 episodes at 10 FPS:

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
  "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
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
      "dtype": "video",
      "shape": [3, 480, 640],
      "names": ["channels", "height", "width"],
      "video_info": {
        "video.fps": 10.0,
        "video.height": 480,
        "video.width": 640,
        "video.codec": "av1",
        "has_audio": false
      }
    },
    "timestamp":     {"dtype": "float32", "shape": [1], "names": null},
    "frame_index":   {"dtype": "int64",   "shape": [1], "names": null},
    "episode_index": {"dtype": "int64",   "shape": [1], "names": null},
    "index":         {"dtype": "int64",   "shape": [1], "names": null},
    "task_index":    {"dtype": "int64",   "shape": [1], "names": null}
  }
}
```

**`meta/tasks.jsonl`:**
```jsonl
{"task_index": 0, "task": "Scoop packing peanuts from large pool and dump into small pool"}
```

**`meta/episodes.jsonl`** (one line per episode):
```jsonl
{"episode_index": 0, "length": 250, "task_index": 0, "task": "Scoop packing peanuts from large pool and dump into small pool"}
{"episode_index": 1, "length": 230, "task_index": 0, "task": "Scoop packing peanuts from large pool and dump into small pool"}
...
```

**`meta/episodes_stats.jsonl`** (one line per episode):
```jsonl
{"episode_index": 0, "stats": {"observation.state": {"min": [-0.5, -0.8, -0.3, -0.7], "max": [0.6, 0.9, 0.4, 0.8], "mean": [0.01, -0.02, 0.03, 0.0], "std": [0.2, 0.3, 0.1, 0.3], "count": [250]}, "action": {"min": [-0.5, -0.8, -0.3, -0.7], "max": [0.6, 0.9, 0.4, 0.8], "mean": [0.01, -0.02, 0.03, 0.0], "std": [0.2, 0.3, 0.1, 0.3], "count": [250]}}}
...
```
