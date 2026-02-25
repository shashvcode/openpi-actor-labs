"""Fetch 35 random episodes from the nojoint_parquet dataset (134 eps, 4GB single file)
and append them to the existing so100_merged dataset to reach ~150 episodes.

Streams the large file to disk first, then reads specific episodes to avoid memory issues.
"""

import io
import json
import os
import pathlib
import random
import sys
import tempfile

import boto3
import pyarrow as pa
import pyarrow.parquet as pq

BUCKET = "lerobot-trajectories"
KEY = "non-joint/nojoint_parquet/data/chunk-000/file-000.parquet"
TASK = "Pick up the bottle and place it on the yellow outlined square."
NUM_EPISODES_TO_SAMPLE = 35

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


def main():
    output_dir = pathlib.Path("./so100_merged")
    data_dir = output_dir / "data" / "chunk-000"
    meta_dir = output_dir / "meta"

    # Load .env
    env_path = pathlib.Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

    s3 = boto3.client(
        "s3",
        region_name=os.environ.get("AWS_REGION", "us-west-1"),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    # Load existing episode info
    episodes_path = meta_dir / "episodes.jsonl"
    existing_episodes = []
    with open(episodes_path) as f:
        for line in f:
            existing_episodes.append(json.loads(line))

    existing_count = len(existing_episodes)
    existing_frames = sum(e["length"] for e in existing_episodes)
    print(f"Existing dataset: {existing_count} episodes, {existing_frames} frames")

    # Find next episode_index and global index offset
    next_ep = max(e["episode_index"] for e in existing_episodes) + 1
    # Count existing data files to know which file index to start from
    existing_files = sorted(data_dir.glob("*.parquet"))
    next_file_idx = len(existing_files)

    # Compute existing global index max
    last_file = existing_files[-1]
    last_table = pq.read_table(last_file, columns=["index"])
    index_offset = last_table.column("index").to_pylist()[-1] + 1
    del last_table

    print(f"Next episode_index: {next_ep}, next file_idx: {next_file_idx}, index_offset: {index_offset}")

    # Stream the 4GB file to a temp file on disk (skip if already present)
    tmp_path = output_dir / "_tmp_nojoint.parquet"
    if tmp_path.exists() and tmp_path.stat().st_size > 1_000_000_000:
        print(f"\nUsing existing download at {tmp_path} ({tmp_path.stat().st_size / 1e9:.1f} GB)")
    else:
        print(f"\nDownloading {KEY} to {tmp_path} ...")
        print("(This is a ~4 GB file, will take a few minutes)")
        s3.download_file(BUCKET, KEY, str(tmp_path))
        print("Download complete.")

    # Read the file and pick 35 random episodes
    print(f"\nReading parquet and sampling {NUM_EPISODES_TO_SAMPLE} episodes...")
    pf = pq.ParquetFile(tmp_path)
    table = pf.read(columns=["episode_index"])
    all_ep_indices = sorted(set(table.column("episode_index").to_pylist()))
    del table

    print(f"Source has {len(all_ep_indices)} episodes (0-{all_ep_indices[-1]})")

    random.seed(42)
    sampled_eps = sorted(random.sample(all_ep_indices, NUM_EPISODES_TO_SAMPLE))
    print(f"Sampled episodes: {sampled_eps}")

    # Use iter_batches to avoid "Nested data conversions not implemented for chunked array outputs"
    # RecordBatch uses flat arrays, not chunked arrays, so nested structs work fine.
    import pyarrow.compute as pc

    pf = pq.ParquetFile(tmp_path)
    new_episodes = []
    sampled_set = set(sampled_eps)
    ep_batches: dict[int, list[pa.RecordBatch]] = {ep: [] for ep in sampled_eps}

    print(f"Scanning file in batches of 2000 rows for sampled episodes...")
    batch_count = 0
    for batch in pf.iter_batches(batch_size=2000):
        batch_count += 1
        ep_col = batch.column("episode_index")
        ep_vals = set(ep_col.to_pylist())
        matching = sampled_set & ep_vals
        if matching:
            for src_ep in matching:
                mask = pc.equal(ep_col, src_ep)
                filtered = batch.filter(mask)
                ep_batches[src_ep].append(filtered)
        if batch_count % 50 == 0:
            print(f"  Processed {batch_count} batches...")

    print(f"  Done scanning ({batch_count} batches total)")
    print("Writing sampled episodes...")

    for i, src_ep in enumerate(sampled_eps):
        batches = ep_batches[src_ep]
        if not batches:
            print(f"  WARNING: Episode {src_ep} had no data, skipping")
            continue
        ep_table = pa.Table.from_batches(batches)
        n_rows = len(ep_table)
        new_ep_idx = next_ep + i

        ep_table = ep_table.set_column(
            ep_table.schema.get_field_index("episode_index"),
            "episode_index",
            pa.array([new_ep_idx] * n_rows, type=pa.int64()),
        )
        ep_table = ep_table.set_column(
            ep_table.schema.get_field_index("index"),
            "index",
            pa.array(list(range(index_offset, index_offset + n_rows)), type=pa.int64()),
        )
        ep_table = ep_table.set_column(
            ep_table.schema.get_field_index("task_index"),
            "task_index",
            pa.array([0] * n_rows, type=pa.int64()),
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

        out_path = data_dir / f"file-{next_file_idx + i:03d}.parquet"
        pq.write_table(ep_table, out_path)

        new_episodes.append({
            "episode_index": new_ep_idx,
            "length": n_rows,
            "task_index": 0,
            "task": TASK,
        })

        index_offset += n_rows
        print(f"  Episode {src_ep} -> {new_ep_idx} ({n_rows} frames) -> {out_path.name}")
        del ep_table

    del ep_batches

    # Clean up temp file
    tmp_path.unlink()
    print(f"\nCleaned up temp file.")

    # Update metadata
    all_episodes = existing_episodes + new_episodes
    total_episodes = len(all_episodes)
    total_frames = sum(e["length"] for e in all_episodes)

    with open(episodes_path, "w") as f:
        for ep in all_episodes:
            f.write(json.dumps(ep) + "\n")

    info_path = meta_dir / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{total_episodes}"}
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n=== Done ===")
    print(f"Total episodes: {total_episodes}")
    print(f"Total frames: {total_frames}")


if __name__ == "__main__":
    main()
