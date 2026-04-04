"""Transfer actor_teleop_300 from HuggingFace to GCS bucket."""
import os, json, tempfile, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi

TOKEN = os.environ["HF_TOKEN"]
REPO = "verm11/actor_teleop_300"
GCS_BUCKET = "gs://actorlabs-raw-data"
GCS_PREFIX = "actor_teleop_300"
BATCH_SIZE = 25

def run(cmd):
    import subprocess
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED: {cmd}\n  {r.stderr.strip()}", flush=True)
    return r

def main():
    api = HfApi(token=TOKEN)

    print("[1/4] Listing all files in HF repo...", flush=True)
    all_files = list(api.list_repo_tree(REPO, repo_type="dataset", recursive=True, token=TOKEN))
    file_paths = [f.path for f in all_files if hasattr(f, 'size') and f.size is not None]
    print(f"    {len(file_paths)} files found", flush=True)

    meta_files = [f for f in file_paths if f.startswith("meta/")]
    data_files = sorted([f for f in file_paths if f.startswith("data/")])
    print(f"    Meta: {len(meta_files)}, Data: {len(data_files)}", flush=True)

    print("\n[2/4] Uploading meta files...", flush=True)
    tmpdir = Path(tempfile.mkdtemp())
    for mf in meta_files:
        local = hf_hub_download(REPO, mf, repo_type="dataset", token=TOKEN)
        gcs_path = f"{GCS_BUCKET}/{GCS_PREFIX}/{mf}"
        r = run(f"gsutil cp '{local}' '{gcs_path}'")
        print(f"    {mf} -> {gcs_path}  {'OK' if r.returncode == 0 else 'FAIL'}", flush=True)

    print(f"\n[3/4] Uploading {len(data_files)} parquet files in batches of {BATCH_SIZE}...", flush=True)
    for batch_start in range(0, len(data_files), BATCH_SIZE):
        batch = data_files[batch_start:batch_start + BATCH_SIZE]
        batch_dir = tmpdir / "batch"
        batch_dir.mkdir(exist_ok=True)

        for df in batch:
            local = hf_hub_download(REPO, df, repo_type="dataset", token=TOKEN)
            dest = batch_dir / Path(df).name
            shutil.copy2(local, dest)

        gcs_dest = f"{GCS_BUCKET}/{GCS_PREFIX}/data/chunk-000/"
        r = run(f"gsutil -m cp '{batch_dir}'/*.parquet '{gcs_dest}'")
        batch_end = min(batch_start + BATCH_SIZE, len(data_files))
        print(f"    batch {batch_start}-{batch_end-1} ({len(batch)} files)  {'OK' if r.returncode == 0 else 'FAIL'}", flush=True)

        shutil.rmtree(batch_dir)

    shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n[4/4] Verifying upload...", flush=True)
    r = run(f"gsutil ls '{GCS_BUCKET}/{GCS_PREFIX}/meta/'")
    print(f"    Meta files: {r.stdout.strip()}", flush=True)
    r = run(f"gsutil ls '{GCS_BUCKET}/{GCS_PREFIX}/data/chunk-000/' | wc -l")
    print(f"    Data files: {r.stdout.strip()}", flush=True)

    print(f"\nDONE. Dataset at {GCS_BUCKET}/{GCS_PREFIX}/", flush=True)

if __name__ == "__main__":
    main()
