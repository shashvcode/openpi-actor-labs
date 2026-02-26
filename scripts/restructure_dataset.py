"""Restructure the merged SO-100 dataset into proper LeRobot v2 format.

LeRobot v2 requires:
  - One parquet file per episode: data/chunk-000/episode_000000.parquet
  - data_path template: data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
  - episodes_stats.jsonl with min/max/mean/std/count (count as scalar)

This script reads our existing multi-episode files, splits them into per-episode files,
updates info.json, and regenerates episodes_stats.jsonl.
"""

import json
import pathlib
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DATASET_DIR = pathlib.Path("./so100_merged")
DATA_DIR = DATASET_DIR / "data" / "chunk-000"
META_DIR = DATASET_DIR / "meta"
CHUNKS_SIZE = 1000

OUTPUT_SCHEMA = pa.schema([
    ("observation.state", pa.list_(pa.float32())),
    ("action", pa.list_(pa.float32())),
    ("observation.images.scene", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("observation.images.wrist", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("timestamp", pa.float32()),
    ("frame_index", pa.int64()),
    ("episode_index", pa.int64()),
    ("index", pa.int64()),
    ("task_index", pa.int64()),
])

TASK = "Pick up the bottle and place it on the yellow outlined square."


def compute_stats(arr: np.ndarray) -> dict:
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(len(arr))],
    }


def main():
    # Step 1: Read ALL existing data into per-episode batches
    print("Step 1: Reading all existing parquet files...")
    old_files = sorted(DATA_DIR.glob("file-*.parquet"))
    print(f"  Found {len(old_files)} old-format files")

    ep_batches: dict[int, list[pa.RecordBatch]] = {}
    for i, fpath in enumerate(old_files):
        for batch in pq.ParquetFile(fpath).iter_batches(batch_size=5000):
            ep_col = batch.column("episode_index")
            unique_eps = set(ep_col.to_pylist())
            for ep_idx in unique_eps:
                mask = pc.equal(ep_col, ep_idx)
                filtered = batch.filter(mask)
                if ep_idx not in ep_batches:
                    ep_batches[ep_idx] = []
                ep_batches[ep_idx].append(filtered)
        if (i + 1) % 20 == 0:
            print(f"  Read {i + 1}/{len(old_files)} files...")

    print(f"  Found {len(ep_batches)} episodes")

    # Step 2: Write per-episode files and compute stats
    print("\nStep 2: Writing per-episode parquet files and computing stats...")

    # Remove old files first
    for f in old_files:
        f.unlink()

    episodes_meta = []
    episodes_stats = []
    total_frames = 0

    for ep_idx in sorted(ep_batches.keys()):
        ep_table = pa.Table.from_batches(ep_batches[ep_idx])

        # Ensure columns match schema (truncate 8-dim to 6 if needed)
        state_dim = len(ep_table.column("observation.state").to_pylist()[0])
        if state_dim > 6:
            states = [row[:6] for row in ep_table.column("observation.state").to_pylist()]
            actions = [row[:6] for row in ep_table.column("action").to_pylist()]
            ep_table = ep_table.set_column(
                ep_table.schema.get_field_index("observation.state"),
                "observation.state", pa.array(states, type=pa.list_(pa.float32()))
            )
            ep_table = ep_table.set_column(
                ep_table.schema.get_field_index("action"),
                "action", pa.array(actions, type=pa.list_(pa.float32()))
            )

        # Normalize schema
        arrays = {}
        for col_name in OUTPUT_SCHEMA.names:
            col = ep_table.column(col_name)
            target_type = OUTPUT_SCHEMA.field(col_name).type
            if col.type != target_type:
                col = col.cast(target_type)
            arrays[col_name] = col
        ep_table = pa.table(arrays, schema=OUTPUT_SCHEMA)

        n_rows = len(ep_table)
        ep_chunk = ep_idx // CHUNKS_SIZE
        chunk_dir = DATA_DIR.parent / f"chunk-{ep_chunk:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        out_path = chunk_dir / f"episode_{ep_idx:06d}.parquet"
        pq.write_table(ep_table, out_path)

        # Episode metadata
        episodes_meta.append({
            "episode_index": ep_idx,
            "length": n_rows,
            "task_index": 0,
            "task": TASK,
        })

        # Episode stats
        states = np.array(ep_table.column("observation.state").to_pylist(), dtype=np.float32)
        actions = np.array(ep_table.column("action").to_pylist(), dtype=np.float32)
        episodes_stats.append({
            "episode_index": ep_idx,
            "stats": {
                "observation.state": compute_stats(states),
                "action": compute_stats(actions),
            },
        })

        total_frames += n_rows
        if (ep_idx + 1) % 25 == 0:
            print(f"  Wrote {ep_idx + 1} episodes...")

        del ep_table

    del ep_batches

    total_episodes = len(episodes_meta)
    print(f"  Done: {total_episodes} episodes, {total_frames} frames")

    # Step 3: Write metadata
    print("\nStep 3: Writing metadata...")

    # episodes.jsonl
    with open(META_DIR / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    # episodes_stats.jsonl
    with open(META_DIR / "episodes_stats.jsonl", "w") as f:
        for s in episodes_stats:
            f.write(json.dumps(s) + "\n")

    # tasks.jsonl
    with open(META_DIR / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": TASK}) + "\n")

    # info.json - with correct data_path template
    info = {
        "codebase_version": "v2.1",
        "robot_type": "so100",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": CHUNKS_SIZE,
        "fps": 10,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [6],
                "names": [["left_x", "left_y", "right_x", "right_y", "l2_trigger", "r2_trigger"]],
            },
            "action": {
                "dtype": "float32",
                "shape": [6],
                "names": [["left_x", "left_y", "right_x", "right_y", "l2_trigger", "r2_trigger"]],
            },
            "observation.images.scene": {
                "dtype": "image",
                "shape": [3, 480, 640],
                "names": ["channels", "height", "width"],
            },
            "observation.images.wrist": {
                "dtype": "image",
                "shape": [3, 480, 640],
                "names": ["channels", "height", "width"],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    with open(META_DIR / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Verify
    new_files = sorted((DATA_DIR.parent / "chunk-000").glob("episode_*.parquet"))
    total_size = sum(f.stat().st_size for f in new_files) / 1e9
    print(f"\n=== Restructure complete ===")
    print(f"  {len(new_files)} per-episode parquet files")
    print(f"  {total_size:.1f} GB total")
    print(f"  {total_episodes} episodes, {total_frames} frames")
    print(f"  data_path: data/chunk-{{episode_chunk:03d}}/episode_{{episode_index:06d}}.parquet")


if __name__ == "__main__":
    main()
