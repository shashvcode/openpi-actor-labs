"""Download D01 (front-camera) videos from FlywheelAI/excavator-dataset and upload to S3.

Renames each video as fwh_<recording_timestamp>.mp4 and places them in
s3://actorgemma-raw-data/misc/.

Processes one file at a time to avoid saturating bandwidth while other
uploads (doug_*, hitatchi225) are running in parallel.
"""

import glob
import os
import shutil
import subprocess
import sys
import tempfile

from huggingface_hub import HfApi

HF_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "datasets--FlywheelAI--excavator-dataset",
)

HF_TOKEN = os.environ["HF_TOKEN"]
REPO_ID = "FlywheelAI/excavator-dataset"

AWS_ENV = {
    **os.environ,
    "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
    "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
    "AWS_REGION": "us-west-1",
}
DST_BUCKET = "actorgemma-raw-data"
DST_PREFIX = "misc"


def s3_upload(local_path: str, s3_key: str) -> bool:
    r = subprocess.run(
        ["aws", "s3", "cp", local_path, f"s3://{DST_BUCKET}/{s3_key}", "--region", "us-west-1"],
        env=AWS_ENV,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(f"  S3 UPLOAD FAILED: {r.stderr.strip()[:300]}", flush=True)
        return False
    return True


def s3_list_existing() -> set[str]:
    """List files already in s3://actorgemma-raw-data/misc/ to allow resume."""
    r = subprocess.run(
        ["aws", "s3", "ls", f"s3://{DST_BUCKET}/{DST_PREFIX}/", "--region", "us-west-1"],
        env=AWS_ENV,
        capture_output=True,
        text=True,
    )
    existing = set()
    for line in r.stdout.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) >= 4:
            existing.add(parts[-1])
    return existing


def _purge_hf_cache():
    """Remove cached blobs to avoid filling the disk."""
    blobs_dir = os.path.join(HF_CACHE_DIR, "blobs")
    if os.path.isdir(blobs_dir):
        for f in os.listdir(blobs_dir):
            fp = os.path.join(blobs_dir, f)
            try:
                os.remove(fp)
            except OSError:
                pass


def main():
    api = HfApi(token=HF_TOKEN)

    print("[1/3] Listing D01 (front) videos in FlywheelAI/excavator-dataset...", flush=True)
    all_files = list(api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True, token=HF_TOKEN))

    d01_files = []
    for f in all_files:
        if not hasattr(f, "size") or f.size is None:
            continue
        parts = f.path.split("/")
        if len(parts) == 2 and parts[1].startswith("D01_") and parts[1].endswith(".mp4"):
            ts = parts[0]
            size_mb = f.size / 1048576
            dst_name = f"fwh_{ts}.mp4"
            d01_files.append((f.path, dst_name, size_mb))

    d01_files.sort()
    print(f"  Found {len(d01_files)} front-camera (D01) videos", flush=True)

    print("\n[2/3] Checking what's already uploaded...", flush=True)
    existing = s3_list_existing()
    todo = [(src, dst, sz) for src, dst, sz in d01_files if dst not in existing]
    skipped = len(d01_files) - len(todo)
    if skipped:
        print(f"  Skipping {skipped} already uploaded", flush=True)
    print(f"  {len(todo)} videos to upload", flush=True)

    total_mb = sum(sz for _, _, sz in todo)
    print(f"  Total download size: {total_mb:.0f} MB ({total_mb/1024:.1f} GB)", flush=True)

    start_from = sys.argv[1] if len(sys.argv) > 1 else None

    print(f"\n[3/3] Downloading + uploading {len(todo)} videos...", flush=True)
    failed = []
    with tempfile.TemporaryDirectory(prefix="fwh_") as tmpdir:
        for i, (src_path, dst_name, size_mb) in enumerate(todo):
            if start_from and dst_name < start_from:
                continue

            print(f"\n  [{i+1}/{len(todo)}] {src_path} -> {dst_name} ({size_mb:.1f} MB)", flush=True)

            try:
                local = api.hf_hub_download(
                    repo_id=REPO_ID,
                    filename=src_path,
                    repo_type="dataset",
                    token=HF_TOKEN,
                )

                s3_key = f"{DST_PREFIX}/{dst_name}"
                ok = s3_upload(local, s3_key)
                if ok:
                    print(f"    -> s3://{DST_BUCKET}/{s3_key}", flush=True)
                else:
                    failed.append(src_path)
            except Exception as e:
                print(f"    FAILED: {e}", flush=True)
                failed.append(src_path)
            finally:
                _purge_hf_cache()

    print(f"\n=== DONE ===", flush=True)
    print(f"  Uploaded: {len(todo) - len(failed)}/{len(todo)}", flush=True)
    if failed:
        print(f"  Failed: {failed}", flush=True)


if __name__ == "__main__":
    main()
