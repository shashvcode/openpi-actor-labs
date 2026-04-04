# SO100 v2.2 Dataset Documentation

**HuggingFace Repo:** `verm11/so100_v2_2`  
**Codebase Version:** v2.1 (LeRobot format)  
**Robot Type:** SO100

---

## Overview

| Property | Value |
|---|---|
| Total Episodes (metadata) | 288 |
| Total Frames (metadata) | 103,296 |
| Parquet Files Present | 150 (episodes 0–149) |
| Missing Parquet Files | 138 (episodes 150–287) |
| FPS | 30 Hz |
| Cameras | 2 (`scene`, `wrist`) |
| Action Dimensions | 6 (gamepad axes + triggers) |
| State Dimensions | 6 (gamepad axes + triggers) |
| Task | "Pick up the bottle and place it on the yellow outlined square." |
| Estimated Storage | ~3.1 GB (150 parquet files) |
| Split | `train: 0:288` |

> **Note:** `meta/info.json` and `meta/episodes.jsonl` list 288 episodes, but only 150 parquet files (episodes 0–149) exist in `data/chunk-000/`. The remaining 138 episodes are missing from storage.

---

## Directory Structure

```
verm11/so100_v2_2/
├── .gitattributes              (LFS tracking rules)
├── meta/
│   ├── info.json               (dataset metadata & feature schema)
│   ├── tasks.jsonl             (task definitions)
│   ├── episodes.jsonl          (per-episode length & task mapping)
│   └── episodes_stats.jsonl    (per-episode min/max/mean/std statistics)
└── data/
    └── chunk-000/
        ├── episode_000000.parquet
        ├── episode_000001.parquet
        ├── ...
        └── episode_000149.parquet
```

**Path template:** `data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet`

All parquet files are stored via Git LFS. Average file size is ~21 MB.

---

## Data Schema (Parquet Columns)

Each parquet file contains one episode. Every row is one timestep (frame).

| Column | Arrow Type | Shape | Description |
|---|---|---|---|
| `observation.state` | `list<float>` | `[6]` | Gamepad state: `[left_x, left_y, right_x, right_y, l2_trigger, r2_trigger]` |
| `action` | `list<float>` | `[6]` | Action output: `[left_x, left_y, right_x, right_y, l2_trigger, r2_trigger]` |
| `observation.images.scene` | `struct<bytes: binary, path: string>` | `[3, 480, 640]` | Scene camera image (JPEG bytes embedded in struct) |
| `observation.images.wrist` | `struct<bytes: binary, path: string>` | `[3, 480, 640]` | Wrist camera image (JPEG bytes embedded in struct) |
| `timestamp` | `float` | `[1]` | Time in seconds from episode start |
| `frame_index` | `int64` | `[1]` | Frame index within the episode (0-based) |
| `episode_index` | `int64` | `[1]` | Episode index (constant per file) |
| `index` | `int64` | `[1]` | Global frame index (same as `frame_index` within an episode) |
| `task_index` | `int64` | `[1]` | Task index (always 0 in this dataset) |

---

## Feature Details

### Observation State & Action

Both `observation.state` and `action` are 6-dimensional float32 vectors representing gamepad inputs:

| Dim | Name | Range | Description |
|---|---|---|---|
| 0 | `left_x` | [-1.0, 1.0] | Left stick horizontal axis |
| 1 | `left_y` | [-1.0, 1.0] | Left stick vertical axis |
| 2 | `right_x` | [-1.0, 1.0] | Right stick horizontal axis |
| 3 | `right_y` | [-1.0, 1.0] | Right stick vertical axis |
| 4 | `l2_trigger` | [0.0, 1.0] | Left trigger (analog, 0=released, 1=fully pressed) |
| 5 | `r2_trigger` | [0.0, 1.0] | Right trigger (analog, 0=released, 1=fully pressed) |

**Value ranges (Episode 0 sample):**

| Dim | State Min | State Max | State Mean |
|---|---|---|---|
| `left_x` | -1.0000 | 1.0000 | 0.0063 |
| `left_y` | -1.0000 | 1.0000 | 0.0187 |
| `right_x` | -1.0000 | 0.2246 | -0.0441 |
| `right_y` | -1.0000 | 1.0000 | -0.0361 |
| `l2_trigger` | 0.0000 | 1.0000 | 0.0961 |
| `r2_trigger` | 0.0000 | 1.0000 | 0.0781 |

Action and state values are essentially identical — the action at frame `t` is the state at frame `t+1` (next-step prediction target).

### Images

| Camera | Resolution | Format | Avg Size |
|---|---|---|---|
| `scene` | 640x480 RGB | JPEG (bytes in parquet) | ~24 KB/frame |
| `wrist` | 640x480 RGB | JPEG (bytes in parquet) | ~35 KB/frame |

Images are stored as structs with two fields:
- `bytes`: Raw JPEG binary data
- `path`: Original filename (e.g., `ep0000_scene_000000.jpg`, `ep0000_wrist_000000.jpg`)

No external video files — `video_path` is `null` in `info.json`. All imagery is embedded directly in parquet.

### Timestamps

- Perfectly uniform at exactly 30 Hz (dt = 0.033333s, std = 0.0)
- Starts at 0.0 for each episode

---

## Episode Statistics

### Length Distribution

| Range (frames) | Count | Duration at 30fps |
|---|---|---|
| 200–300 | 59 episodes | 6.7–10.0 s |
| 300–400 | 152 episodes | 10.0–13.3 s |
| 400–500 | 66 episodes | 13.3–16.7 s |
| 500–600 | 10 episodes | 16.7–20.0 s |
| 600–700 | 1 episode | 20.0–23.3 s |

| Stat | Value |
|---|---|
| Min episode length | 207 frames (6.9 s) |
| Max episode length | 607 frames (20.2 s) |
| Mean episode length | 358.7 frames (12.0 s) |
| Total duration | ~57.4 minutes |

### Per-Episode Stats File (`episodes_stats.jsonl`)

Each line contains per-episode statistics with the following structure:

```json
{
  "episode_index": 0,
  "stats": {
    "observation.state": {
      "min": [-1.0, -0.9999, -1.0, -0.9999, 0.0, 0.0],
      "max": [0.9999, 1.0, 0.2245, 1.0, 0.9999, 0.9999],
      "mean": [0.0063, 0.0187, -0.0441, -0.0361, 0.0961, 0.0781],
      "std": [0.2559, 0.4542, 0.2198, 0.5629, 0.2939, 0.2680],
      "count": [472]
    },
    "action": {
      "min": [...],
      "max": [...],
      "mean": [...],
      "std": [...],
      "count": [472]
    }
  }
}
```

Stats are computed per-dimension for `observation.state` and `action` only (not for images).

---

## Tasks

Single task dataset:

```json
{"task_index": 0, "task": "Pick up the bottle and place it on the yellow outlined square."}
```

All 288 episodes share the same task.

---

## How to Load

### Using LeRobot (recommended)

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("verm11/so100_v2_2")
sample = dataset[0]
# sample keys: observation.state, action, observation.images.scene, observation.images.wrist, etc.
```

### Using PyArrow directly

```python
import pyarrow.parquet as pq
from PIL import Image
import io

table = pq.read_table("data/chunk-000/episode_000000.parquet")

# Get state/action as numpy
states = [row.as_py() for row in table.column("observation.state")]

# Decode an image
img_struct = table.column("observation.images.scene")[0].as_py()
img = Image.open(io.BytesIO(img_struct["bytes"]))
img.show()  # 640x480 RGB JPEG
```

### Using HuggingFace Hub

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    "verm11/so100_v2_2",
    "data/chunk-000/episode_000000.parquet",
    repo_type="dataset",
)
```

---

## Data Integrity Notes

1. **Episode count mismatch:** `info.json` declares 288 total episodes and `episodes.jsonl` lists 288 entries, but only 150 parquet files exist (episodes 0–149). Episodes 150–287 are referenced in metadata but have no corresponding data files.

2. **No video files:** `video_path` is `null`. All image data is stored as JPEG bytes inside parquet structs, not as separate video files.

3. **LFS-tracked:** All parquet files and media formats are tracked via Git LFS (see `.gitattributes`).

4. **Single chunk:** All 150 episodes reside in `data/chunk-000/` (chunk size configured as 1000 episodes per chunk).
