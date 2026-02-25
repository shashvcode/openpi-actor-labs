"""Upload the merged SO-100 dataset to HuggingFace in small batches to avoid memory issues.

Usage:
    python3 scripts/upload_to_hf.py --repo-id YOUR_USERNAME/so100_joystick_pickup
"""

import argparse
import os
import pathlib

from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID, e.g. username/so100_joystick_pickup")
    parser.add_argument("--dataset-dir", default="./so100_merged", help="Local dataset directory")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of parquet files to upload per commit")
    args = parser.parse_args()

    dataset_dir = pathlib.Path(args.dataset_dir)
    api = HfApi()

    # Create the repo if it doesn't exist
    print(f"Creating/verifying repo: {args.repo_id}")
    create_repo(args.repo_id, repo_type="dataset", exist_ok=True)

    # 1. Upload metadata files first (tiny, safe)
    meta_dir = dataset_dir / "meta"
    meta_files = list(meta_dir.glob("*"))
    print(f"\n[1/2] Uploading {len(meta_files)} metadata files...")
    for f in meta_files:
        path_in_repo = f"meta/{f.name}"
        print(f"  {path_in_repo} ({f.stat().st_size / 1024:.0f} KB)")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=path_in_repo,
            repo_id=args.repo_id,
            repo_type="dataset",
        )
    print("  Metadata done.")

    # 2. Upload parquet files in small batches
    data_dir = dataset_dir / "data" / "chunk-000"
    parquet_files = sorted(data_dir.glob("*.parquet"))
    total = len(parquet_files)
    print(f"\n[2/2] Uploading {total} parquet files in batches of {args.batch_size}...")

    for batch_start in range(0, total, args.batch_size):
        batch = parquet_files[batch_start : batch_start + args.batch_size]
        batch_end = batch_start + len(batch)
        batch_size_mb = sum(f.stat().st_size for f in batch) / (1024 * 1024)
        print(f"\n  Batch {batch_start + 1}-{batch_end} of {total} ({batch_size_mb:.0f} MB)...")

        operations = []
        from huggingface_hub import CommitOperationAdd

        for f in batch:
            path_in_repo = f"data/chunk-000/{f.name}"
            operations.append(CommitOperationAdd(path_or_fileobj=str(f), path_in_repo=path_in_repo))

        api.create_commit(
            repo_id=args.repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Add data files {batch_start}-{batch_end - 1}",
        )
        print(f"  Batch uploaded.")

    print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
