"""Merge all SO-100 joystick datasets from S3 into a single LeRobot-format dataset.

Reads from s3://lerobot-trajectories/non-joint/ and merges 4 sub-datasets:
  - nojoint_parquet/         (134 episodes)
  - datasets_parquet/        (46 episodes)
  - so100-joystick-pickup_20260223_161151/  (23 episodes)
  - so100-joystick-pickup_20260224_164431_nojoints/  (46 episodes)

Outputs a single merged dataset in LeRobot v2 format, ready for HuggingFace upload.

Usage:
    python scripts/prepare_so100_data.py --output-dir ./so100_merged
    # Then upload:
    huggingface-cli upload <your_hf_username>/so100_joystick_pickup ./so100_merged --repo-type dataset
"""

import argparse
import io
import json
import os
import pathlib
import sys

import boto3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BUCKET = "lerobot-trajectories"

DATASETS = [
    {
        "prefix": "non-joint/nojoint_parquet/",
        "data_files": ["data/chunk-000/file-000.parquet"],
        "meta_format": "jsonl",
        "expected_episodes": 134,
    },
    {
        "prefix": "non-joint/datasets_parquet/",
        "data_files": ["data/chunk-000/file-000.parquet"],
        "meta_format": "jsonl",
        "expected_episodes": 46,
    },
    {
        "prefix": "non-joint/so100-joystick-pickup_20260223_161151/so100-joystick-pickup_20260223_161151/",
        "data_files": [f"data/chunk-000/file-{i:03d}.parquet" for i in range(23)],
        "meta_format": "parquet",
        "expected_episodes": 23,
    },
    {
        "prefix": "non-joint/so100-joystick-pickup_20260224_164431_nojoints/",
        "data_files": [f"data/chunk-000/file-{i:03d}.parquet" for i in range(46)],
        "meta_format": "parquet",
        "expected_episodes": 46,
    },
]

OUTPUT_SCHEMA = pa.schema(
    [
        ("observation.state", pa.list_(pa.float32())),
        ("action", pa.list_(pa.float32())),
        ("observation.images.scene", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("observation.images.wrist", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("timestamp", pa.float32()),
        ("frame_index", pa.int64()),
        ("episode_index", pa.int64()),
        ("index", pa.int64()),
        ("task_index", pa.int64()),
    ]
)

TASK = "Pick up the bottle and place it on the yellow outlined square."


def get_s3_client():
    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_REGION", "us-west-1"),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def read_parquet_from_s3(s3, key: str) -> pa.Table:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read()))


def normalize_table(table: pa.Table) -> pa.Table:
    """Ensure consistent column types across datasets."""
    arrays = {}
    for col_name in OUTPUT_SCHEMA.names:
        if col_name not in table.column_names:
            raise ValueError(f"Missing column {col_name} in table with columns {table.column_names}")
        col = table.column(col_name)
        target_type = OUTPUT_SCHEMA.field(col_name).type
        if col.type != target_type:
            col = col.cast(target_type)
        arrays[col_name] = col

    return pa.table(arrays, schema=OUTPUT_SCHEMA)


def process_dataset(s3, dataset_cfg: dict, episode_offset: int, index_offset: int):
    """Read and re-index a single dataset. Yields (table, episode_lengths) to keep memory bounded."""
    prefix = dataset_cfg["prefix"]
    episode_lengths = []
    total_rows = 0

    for file_path in dataset_cfg["data_files"]:
        key = prefix + file_path
        print(f"  Reading {key} ...")
        try:
            table = read_parquet_from_s3(s3, key)
        except Exception as e:
            print(f"  WARNING: Failed to read {key}: {e}")
            continue

        # Keep only the columns we need
        cols_to_keep = [c for c in OUTPUT_SCHEMA.names if c in table.column_names]
        table = table.select(cols_to_keep)
        table = normalize_table(table)

        # Re-index episodes and global index
        ep_col = table.column("episode_index").to_pylist()
        frame_col = table.column("frame_index").to_pylist()
        idx_col = table.column("index").to_pylist()

        new_ep = [e + episode_offset for e in ep_col]
        new_idx = [i + index_offset for i in range(len(idx_col))]

        table = table.set_column(
            table.schema.get_field_index("episode_index"), "episode_index", pa.array(new_ep, type=pa.int64())
        )
        table = table.set_column(table.schema.get_field_index("index"), "index", pa.array(new_idx, type=pa.int64()))
        table = table.set_column(
            table.schema.get_field_index("task_index"),
            "task_index",
            pa.array([0] * len(new_idx), type=pa.int64()),
        )

        # Track episode lengths
        unique_eps = sorted(set(ep_col))
        for ep in unique_eps:
            ep_mask = [e == ep for e in ep_col]
            ep_len = sum(ep_mask)
            episode_lengths.append({"original_ep": ep, "new_ep": ep + episode_offset, "length": ep_len})

        total_rows += len(table)
        yield table

    print(f"  Dataset total: {total_rows} rows, {len(episode_lengths)} episodes")


def main():
    parser = argparse.ArgumentParser(description="Merge SO-100 datasets from S3")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./so100_merged",
        help="Output directory for the merged dataset",
    )
    args = parser.parse_args()

    # Load AWS credentials from .env if present
    env_path = pathlib.Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

    s3 = get_s3_client()
    output_dir = pathlib.Path(args.output_dir)
    data_dir = output_dir / "data" / "chunk-000"
    meta_dir = output_dir / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    episode_offset = 0
    index_offset = 0
    all_episodes = []
    output_file_idx = 0

    for ds_idx, ds_cfg in enumerate(DATASETS):
        print(f"\n[{ds_idx + 1}/{len(DATASETS)}] Processing {ds_cfg['prefix']} ({ds_cfg['expected_episodes']} episodes)")

        ds_episode_start = episode_offset
        ds_rows = 0

        for table in process_dataset(s3, ds_cfg, episode_offset, index_offset):
            # Write each table as a separate output file to avoid memory issues
            out_path = data_dir / f"file-{output_file_idx:03d}.parquet"
            pq.write_table(table, out_path)
            print(f"  Wrote {out_path} ({len(table)} rows)")
            ds_rows += len(table)
            output_file_idx += 1

            # Track episodes from this chunk
            ep_col = table.column("episode_index").to_pylist()
            frame_col = table.column("frame_index").to_pylist()
            unique_eps = sorted(set(ep_col))
            for ep in unique_eps:
                ep_frames = [f for e, f in zip(ep_col, frame_col) if e == ep]
                if not any(existing["episode_index"] == ep for existing in all_episodes):
                    all_episodes.append(
                        {
                            "episode_index": ep,
                            "length": len(ep_frames),
                            "task_index": 0,
                            "task": TASK,
                        }
                    )

            index_offset += len(table)
            del table

        # Update episode_offset for next dataset
        if all_episodes:
            episode_offset = max(ep["episode_index"] for ep in all_episodes) + 1

    # Sort episodes
    all_episodes.sort(key=lambda e: e["episode_index"])
    total_episodes = len(all_episodes)
    total_frames = sum(e["length"] for e in all_episodes)

    print(f"\n=== Merge complete ===")
    print(f"Total episodes: {total_episodes}")
    print(f"Total frames: {total_frames}")

    # Write metadata
    # tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": TASK}) + "\n")

    # episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in all_episodes:
            f.write(json.dumps(ep) + "\n")

    # info.json
    total_data_mb = sum(f.stat().st_size for f in data_dir.glob("*.parquet")) / (1024 * 1024)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "so100",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
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

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nDataset saved to: {output_dir}")
    print(f"\nTo upload to HuggingFace:")
    print(f"  huggingface-cli upload <your_hf_username>/so100_joystick_pickup {output_dir} --repo-type dataset")


if __name__ == "__main__":
    main()
