# pi_teleopv2v3 Dataset Format

**HuggingFace repo:** `verm11/pi_teleopv2v3`
**Format:** LeRobot v3.0 + Physical Intelligence metadata
**Total episodes:** 655 (503 from v2 + 152 from v3)
**Total frames:** 194,121
**FPS:** 10

---

## Directory Structure

```
verm11/pi_teleopv2v3/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       ├── ...
│       └── episode_000654.parquet
└── meta/
    ├── info.json                  (1.5 KB)   Global dataset metadata
    ├── episodes.jsonl             (87 KB)    Episode index (one JSON per line)
    ├── episodes.parquet           (13 KB)    Same as above, parquet format
    ├── tasks.jsonl                (496 B)    Task definitions
    ├── tasks.parquet              (2.1 KB)   Same as above, parquet format
    ├── custom_metadata.csv        (58 KB)    PI-required episode metadata
    └── custom_annotation.json     (277 KB)   PI-required per-episode annotations
```

---

## Data Files (Parquet)

Each episode is one parquet file: `data/chunk-000/episode_{index:06d}.parquet`

### Parquet Schema

| Column | Type | Shape | Description |
|--------|------|-------|-------------|
| `observation.state` | `list<float>` | `[4]` | Joystick state: `[-LX, -RY, RX, -LY]` |
| `action` | `list<float>` | `[4]` | Joystick action: `[-LX, -RY, RX, -LY]` |
| `observation.images.csi_0_imx219` | `struct{bytes: binary, path: string}` | `[480, 640, 3]` | Front camera (CSI IMX219 sensor), JPEG bytes inline |
| `observation.images.usb_0` | `struct{bytes: binary, path: string}` | `[480, 640, 3]` | Wrist camera (USB), JPEG bytes inline |
| `timestamp` | `float` | scalar | Relative time in seconds from episode start (0.0, 0.1, 0.2, ...) |
| `frame_index` | `int64` | scalar | Frame number within episode (0, 1, 2, ...) |
| `episode_index` | `int64` | scalar | Global episode index (same value for all rows in a file) |
| `index` | `int64` | scalar | Global frame index |
| `task_index` | `int64` | scalar | Index into tasks table |

### Joystick Transform

The raw HuggingFace data had joystick values as `[LX, LY, RX, RY]`. This dataset applies:

```
Original:  [LX,  LY, RX,  RY]
Output:    [-LX, -RY, RX, -LY]
```

Steps: swap LY ↔ RY, then negate LX, LY, and RY. Values are float32, range approximately `[-1.0, 1.0]`.

### Image Format

Camera images are stored inline in each parquet row as a struct with two fields:
- `bytes`: Raw JPEG image data (binary, ~30-50 KB per frame)
- `path`: Always `null` (images are embedded, not referenced by path)

To read an image: extract the `bytes` field and decode as JPEG.

---

## Meta Files

### `info.json`

```json
{
  "codebase_version": "v3.0",
  "robot_type": "excavator",
  "fps": 10,
  "total_episodes": 655,
  "total_frames": 194121,
  "data_path": "data/chunk-{chunk_index:03d}/episode_{file_index:06d}.parquet",
  "features": {
    "observation.state": {
      "dtype": "float32",
      "shape": [4],
      "description": "Joystick state [-LX, -RY, RX, -LY] (swapped LY<->RY, inverted LX/LY/RY)"
    },
    "action": {
      "dtype": "float32",
      "shape": [4],
      "description": "Joystick action [-LX, -RY, RX, -LY] (swapped LY<->RY, inverted LX/LY/RY)"
    },
    "observation.images.csi_0_imx219": {
      "dtype": "image_bytes",
      "shape": [480, 640, 3],
      "description": "Front camera (CSI IMX219), stored inline in parquet as {bytes, path}"
    },
    "observation.images.usb_0": {
      "dtype": "image_bytes",
      "shape": [480, 640, 3],
      "description": "Wrist camera (USB), stored inline in parquet as {bytes, path}"
    }
  },
  "joystick_transform": "Original [LX,LY,RX,RY] -> [-LX,-RY,RX,-LY]: swap LY<->RY then negate LX,LY,RY",
  "source_datasets": {
    "v2": { "hf_repo": "verm11/excavator_v2", "num_episodes": 503 },
    "v3": { "hf_repo": "verm11/excavator_v3", "num_episodes": 152,
             "note": "v3 episodes 0-49 removed (duplicates of v2)" }
  }
}
```

### `episodes.jsonl`

One JSON object per line. Each episode has:

```json
{"episode_index": 0, "length": 299, "task_index": 0, "task": "Scoop packing peanuts from large pool and dump into small pool"}
{"episode_index": 1, "length": 492, "task_index": 0, "task": "Scoop packing peanuts from large pool and dump into small pool"}
...
```

| Field | Type | Description |
|-------|------|-------------|
| `episode_index` | int | 0–654 |
| `length` | int | Number of frames in the episode |
| `task_index` | int | Index into tasks table |
| `task` | string | Human-readable task description |

### `episodes.parquet`

Same data as `episodes.jsonl` in parquet format, with additional columns:

| Field | Type | Description |
|-------|------|-------------|
| `episode_index` | int | 0–654 |
| `length` | int | Number of frames |
| `tasks` | list[string] | Task descriptions |
| `data/chunk_index` | int | Always 0 (single chunk) |
| `data/file_index` | int | Same as `episode_index` |

### `tasks.jsonl`

```json
{"task_index": 0, "task": "Scoop packing peanuts from large pool and dump into pool on the left"}
{"task_index": 1, "task": "Scoop packing peanuts from large pool and dump into pool on the right"}
{"task_index": 2, "task": "Scoop packing peanuts from large pool and dump into small pool"}
{"task_index": 3, "task": "Scoop packing peanuts from large pool and dump into the medium sized pool"}
{"task_index": 4, "task": "Scoop packing peanuts from large pool and dump into the smallest pool"}
```

### `custom_metadata.csv`

PI-required per-episode metadata. CSV with header row + 655 data rows.

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `episode_index` | int | `0` | Global episode index |
| `operator_id` | string | `operator_v2` | Operator identifier |
| `is_eval_episode` | bool | `False` | All episodes are training data |
| `episode_id` | string | `v2_ep_000000` | Unique episode identifier |
| `start_timestamp` | float | `1773141480.0` | Unix epoch (UTC seconds) |
| `checkpoint_path` | string | *(empty)* | Not applicable |
| `success` | bool | `True` | Episode completed successfully |
| `station_id` | string | `excavator_station_v2` | Collection station |
| `robot_id` | string | `excavator_v2` | Robot identifier |

### `custom_annotation.json`

PI-required per-episode annotation with time spans.

```json
{
  "episodes": [
    {
      "episode_id": "v2_ep_000000",
      "spans": [
        {
          "start_time": 0.0,
          "end_time": 29.9,
          "label": "teleop_execution"
        }
      ],
      "extras": {
        "source": "verm11/excavator_v2",
        "original_episode_index": 0,
        "task": "Scoop packing peanuts from large pool and dump into small pool",
        "num_frames": 299
      }
    }
  ]
}
```

---

## Episode Index Mapping

| Source | Original indices | Output indices | Count |
|--------|-----------------|----------------|-------|
| `verm11/excavator_v2` | 0–502 | 0–502 | 503 |
| `verm11/excavator_v3` | 50–201 | 503–654 | 152 |

v3 episodes 0–49 were removed because they are duplicates of v2 episodes.

---

## Reading the Data (Python)

```python
import pyarrow.parquet as pq
from PIL import Image
import io

# Load one episode
table = pq.read_table("data/chunk-000/episode_000100.parquet")

# Joystick data
obs_state = table.column("observation.state").to_pylist()   # List of [4] floats
actions = table.column("action").to_pylist()                 # List of [4] floats

# Camera image for frame 0
csi_struct = table.column("observation.images.csi_0_imx219")[0].as_py()
img = Image.open(io.BytesIO(csi_struct["bytes"]))

usb_struct = table.column("observation.images.usb_0")[0].as_py()
img2 = Image.open(io.BytesIO(usb_struct["bytes"]))
```
