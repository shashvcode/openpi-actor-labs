"""Resume transferring pi_kansasdata to GCS from episode 500."""
import os, shutil, tempfile
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi

TOKEN = os.environ["HF_TOKEN"]
GCS_BUCKET = "gs://actorlabs-raw-data"
REPO = "verm11/pi_kansasdata"
BATCH_SIZE = 25
START_EP = 500
END_EP = 974

def run(cmd):
    import subprocess
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED: {r.stderr.strip()[:200]}", flush=True)
    return r

def main():
    api = HfApi(token=TOKEN)
    ds_name = REPO.split("/")[-1]
    gcs_prefix = f"{GCS_BUCKET}/{ds_name}"

    print(f"RESUME: {REPO} -> {gcs_prefix}/", flush=True)
    print(f"  Episodes {START_EP}-{END_EP-1} ({END_EP - START_EP} parquets remaining)", flush=True)

    tmpdir = Path(tempfile.mkdtemp())
    batch_dir = tmpdir / "batch"

    for batch_start in range(START_EP, END_EP, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, END_EP)
        batch_dir.mkdir(exist_ok=True)

        for ep_idx in range(batch_start, batch_end):
            fname = f"data/chunk-000/episode_{ep_idx:06d}.parquet"
            local = hf_hub_download(REPO, fname, repo_type="dataset", token=TOKEN)
            dest = batch_dir / f"episode_{ep_idx:06d}.parquet"
            shutil.copy2(local, dest)

        gcs_dest = f"{gcs_prefix}/data/chunk-000/"
        r = run(f"gsutil -m cp '{batch_dir}'/*.parquet '{gcs_dest}'")
        status = "OK" if r.returncode == 0 else "FAIL"
        print(f"  batch {batch_start}-{batch_end-1} ({batch_end - batch_start} files) [{status}]", flush=True)

        shutil.rmtree(batch_dir)

        # Clear HF cache after each batch to avoid disk fill
        cache = Path(os.path.expanduser("~/.cache/huggingface/hub"))
        cache_dir = cache / f"datasets--{REPO.replace('/', '--')}"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    shutil.rmtree(tmpdir, ignore_errors=True)

    # Final verify
    print(f"\n  Verifying...", flush=True)
    r = run(f"gsutil ls '{gcs_prefix}/data/chunk-000/' | wc -l")
    print(f"  Total data files on GCS: {r.stdout.strip()} (expected {END_EP})", flush=True)
    r = run(f"gsutil ls '{gcs_prefix}/meta/' | wc -l")
    print(f"  Meta files: {r.stdout.strip()}", flush=True)
    print(f"\nDONE.", flush=True)

if __name__ == "__main__":
    main()
