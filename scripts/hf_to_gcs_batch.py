"""Transfer HF datasets to GCS bucket in batches to manage disk space."""
import os, json, tempfile, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi

TOKEN = os.environ["HF_TOKEN"]
GCS_BUCKET = "gs://actorlabs-raw-data"
BATCH_SIZE = 50

DATASETS = [
    "verm11/pi_kansas_data_1cam",
    "verm11/pi_kansasdata",
]

def run(cmd):
    import subprocess
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED: {r.stderr.strip()[:200]}", flush=True)
    return r

def transfer_dataset(repo):
    api = HfApi(token=TOKEN)
    ds_name = repo.split("/")[-1]
    gcs_prefix = f"{GCS_BUCKET}/{ds_name}"

    print(f"\n{'='*60}", flush=True)
    print(f"TRANSFER: {repo} -> {gcs_prefix}/", flush=True)
    print(f"{'='*60}", flush=True)

    # List files
    all_files = list(api.list_repo_tree(repo, repo_type="dataset", recursive=True, token=TOKEN))
    file_paths = [f.path for f in all_files if hasattr(f, "size") and f.size is not None]
    meta_files = [f for f in file_paths if f.startswith("meta/")]
    data_files = sorted([f for f in file_paths if f.startswith("data/")])
    print(f"  Meta: {len(meta_files)}, Data: {len(data_files)}", flush=True)

    # Upload meta files
    print(f"\n  Uploading meta files...", flush=True)
    for mf in meta_files:
        local = hf_hub_download(repo, mf, repo_type="dataset", token=TOKEN)
        gcs_path = f"{gcs_prefix}/{mf}"
        r = run(f"gsutil cp '{local}' '{gcs_path}'")
        status = "OK" if r.returncode == 0 else "FAIL"
        print(f"    {mf} [{status}]", flush=True)

    # Upload data files in batches
    print(f"\n  Uploading {len(data_files)} parquets in batches of {BATCH_SIZE}...", flush=True)
    tmpdir = Path(tempfile.mkdtemp())
    batch_dir = tmpdir / "batch"

    for batch_start in range(0, len(data_files), BATCH_SIZE):
        batch = data_files[batch_start:batch_start + BATCH_SIZE]
        batch_dir.mkdir(exist_ok=True)

        for df in batch:
            local = hf_hub_download(repo, df, repo_type="dataset", token=TOKEN)
            dest = batch_dir / Path(df).name
            shutil.copy2(local, dest)

        chunk_dir = "data/chunk-000/"
        gcs_dest = f"{gcs_prefix}/{chunk_dir}"
        r = run(f"gsutil -m cp '{batch_dir}'/*.parquet '{gcs_dest}'")
        batch_end = min(batch_start + BATCH_SIZE, len(data_files))
        status = "OK" if r.returncode == 0 else "FAIL"
        print(f"    batch {batch_start}-{batch_end-1} ({len(batch)} files) [{status}]", flush=True)

        shutil.rmtree(batch_dir)

    # Clear HF cache for this dataset
    import pathlib
    cache = pathlib.Path(os.path.expanduser("~/.cache/huggingface/hub"))
    cache_name = f"datasets--{repo.replace('/', '--')}"
    cache_dir = cache / cache_name
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"\n  Cleared HF cache for {repo}", flush=True)

    shutil.rmtree(tmpdir, ignore_errors=True)

    # Verify
    print(f"\n  Verifying...", flush=True)
    r = run(f"gsutil ls '{gcs_prefix}/meta/' | wc -l")
    print(f"    Meta files: {r.stdout.strip()}", flush=True)
    r = run(f"gsutil ls '{gcs_prefix}/data/chunk-000/' | wc -l")
    print(f"    Data files: {r.stdout.strip()}", flush=True)

    print(f"\n  DONE: {gcs_prefix}/", flush=True)

def main():
    for repo in DATASETS:
        transfer_dataset(repo)
    print("\nALL DONE.", flush=True)

if __name__ == "__main__":
    main()
