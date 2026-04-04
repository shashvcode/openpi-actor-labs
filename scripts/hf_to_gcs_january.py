"""Transfer actorlabs/january-kansas-data from HuggingFace to GCS as actor_deployment_ks_v0."""
import os, shutil, tempfile
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi

TOKEN = os.environ["HF_TOKEN"]
GCS_BUCKET = "gs://actorlabs-raw-data"
GCS_NAME = "actor_deployment_ks_v0"
REPO = "actorlabs/january-kansas-data"
BATCH_SIZE = 25

def run(cmd):
    import subprocess
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED: {r.stderr.strip()[:200]}", flush=True)
    return r

def main():
    api = HfApi(token=TOKEN)
    gcs_prefix = f"{GCS_BUCKET}/{GCS_NAME}"

    print(f"TRANSFER: {REPO} -> {gcs_prefix}/", flush=True)

    all_files = list(api.list_repo_tree(REPO, repo_type="dataset", recursive=True, token=TOKEN))
    file_paths = [f.path for f in all_files if hasattr(f, "size") and f.size is not None]
    meta_files = sorted([f for f in file_paths if f.startswith("meta/")])
    data_files = sorted([f for f in file_paths if f.startswith("data/")])
    print(f"  Meta: {len(meta_files)}, Data: {len(data_files)}", flush=True)

    # Upload meta files one by one
    print(f"\nUploading meta files...", flush=True)
    for mf in meta_files:
        local = hf_hub_download(REPO, mf, repo_type="dataset", token=TOKEN)
        gcs_path = f"{gcs_prefix}/{mf}"
        r = run(f"gsutil cp '{local}' '{gcs_path}'")
        status = "OK" if r.returncode == 0 else "FAIL"
        print(f"  {mf} [{status}]", flush=True)

    # Upload data files in batches
    print(f"\nUploading {len(data_files)} parquets in batches of {BATCH_SIZE}...", flush=True)
    tmpdir = Path(tempfile.mkdtemp())

    for batch_start in range(0, len(data_files), BATCH_SIZE):
        batch = data_files[batch_start:batch_start + BATCH_SIZE]
        batch_dir = tmpdir / "batch"
        batch_dir.mkdir(exist_ok=True)

        for df in batch:
            local = hf_hub_download(REPO, df, repo_type="dataset", token=TOKEN)
            dest = batch_dir / Path(df).name
            shutil.copy2(local, dest)

        gcs_dest = f"{gcs_prefix}/data/chunk-000/"
        r = run(f"gsutil -m cp '{batch_dir}'/*.parquet '{gcs_dest}'")
        batch_end = min(batch_start + BATCH_SIZE, len(data_files))
        status = "OK" if r.returncode == 0 else "FAIL"
        print(f"  batch {batch_start}-{batch_end-1} ({len(batch)} files) [{status}]", flush=True)

        shutil.rmtree(batch_dir)

        # Clear HF cache after each batch
        cache = Path(os.path.expanduser("~/.cache/huggingface/hub"))
        cache_dir = cache / f"datasets--{REPO.replace('/', '--')}"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    shutil.rmtree(tmpdir, ignore_errors=True)

    # Verify
    print(f"\nVerifying...", flush=True)
    r = run(f"gsutil ls '{gcs_prefix}/meta/' | wc -l")
    meta_count = r.stdout.strip()
    r = run(f"gsutil ls '{gcs_prefix}/data/chunk-000/' | wc -l")
    data_count = r.stdout.strip()
    print(f"  Meta files on GCS: {meta_count} (expected {len(meta_files)})", flush=True)
    print(f"  Data files on GCS: {data_count} (expected {len(data_files)})", flush=True)

    if meta_count == str(len(meta_files)) and data_count == str(len(data_files)):
        print(f"\nDONE. All files verified at {gcs_prefix}/", flush=True)
    else:
        print(f"\nWARNING: File count mismatch! Check manually.", flush=True)

if __name__ == "__main__":
    main()
