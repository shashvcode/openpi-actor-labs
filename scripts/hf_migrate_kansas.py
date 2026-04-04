"""Migrate verm11/pi_kansasdata -> actorlabs/actor_kansas_3-28 with rate-limit handling."""

import os
import re
import shutil
import time
from huggingface_hub import HfApi, hf_hub_download

SRC_TOKEN = os.environ["HF_TOKEN"]
DST_TOKEN = os.environ["MIGRATION_HF_TOKEN"]

src_api = HfApi(token=SRC_TOKEN)
dst_api = HfApi(token=DST_TOKEN)

SRC_REPO = "verm11/pi_kansasdata"
DST_REPO = "actorlabs/actor_kansas_3-28"
BATCH_SIZE = 100
TMP_DIR = "/tmp/hf_batch"


def get_remaining():
    src_files = set()
    for f in src_api.list_repo_tree(SRC_REPO, repo_type="dataset", recursive=True):
        if hasattr(f, "size") and f.size:
            src_files.add(f.path)

    dst_files = set()
    try:
        for f in dst_api.list_repo_tree(DST_REPO, repo_type="dataset", recursive=True):
            if hasattr(f, "size") and f.size:
                dst_files.add(f.path)
    except Exception:
        pass

    return sorted(src_files - dst_files)


def upload_batch(batch, batch_num, total_batches):
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR)

    for i, fpath in enumerate(batch):
        hf_hub_download(
            repo_id=SRC_REPO,
            filename=fpath,
            repo_type="dataset",
            token=SRC_TOKEN,
            local_dir=TMP_DIR,
            force_download=True,
        )
        if (i + 1) % 25 == 0:
            print(f"  Downloaded {i+1}/{len(batch)}")

    print(f"  Downloaded all {len(batch)} files, uploading...")

    max_retries = 5
    for attempt in range(max_retries):
        try:
            dst_api.upload_folder(
                folder_path=TMP_DIR,
                repo_id=DST_REPO,
                repo_type="dataset",
                commit_message=f"Batch {batch_num}/{total_batches}",
            )
            print(f"  Batch {batch_num} uploaded!")
            break
        except Exception as e:
            err_str = str(e)
            retry_match = re.search(r"Retry after (\d+) seconds", err_str)
            rate_match = re.search(r"retry this action in (\d+) minutes", err_str)

            if "429" in err_str:
                if rate_match:
                    wait = int(rate_match.group(1)) * 60 + 60
                elif retry_match:
                    wait = int(retry_match.group(1)) + 10
                else:
                    wait = 300

                print(f"  Rate limited. Waiting {wait}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait)
            else:
                raise

    shutil.rmtree(TMP_DIR, ignore_errors=True)


def main():
    dst_api.create_repo(repo_id=DST_REPO, repo_type="dataset", exist_ok=True)

    remaining = get_remaining()
    print(f"Remaining: {len(remaining)} files")

    batch_idx = 0
    while remaining:
        batch = remaining[:BATCH_SIZE]
        batch_num = batch_idx + 1
        total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n=== Batch {batch_num}/{total_batches} ({len(batch)} files, {len(remaining)} total remaining) ===")

        upload_batch(batch, batch_num, total_batches)

        remaining = get_remaining()
        batch_idx += 1

    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
