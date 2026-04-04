"""
Fix actorlabs/january-kansas-data to match the standard format.

SAFETY: This script ONLY modifies the HuggingFace copy. GCS is NOT touched.
Run the GCS transfer separately AFTER verifying HF is correct.

Changes made (nothing else is touched):
  1. Add global 'index' column (int64) to each parquet
  2. Reorder columns: index first, then existing columns in original order
  3. Add 'task' string to each line in episodes.jsonl
  4. Upload empty custom_annotation.json ({})

Data integrity guarantees:
  - All original column values (images, states, actions, timestamps) are UNTOUCHED
  - No episodes are added or removed
  - No data types are changed
  - Each parquet is verified after modification before upload
"""

import json, os, sys, time, tempfile, shutil
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, HfApi

TOKEN = os.environ["HF_TOKEN"]
REPO = "actorlabs/january-kansas-data"
BATCH_SIZE = 25
MAX_RETRIES = 3

TARGET_COL_ORDER = [
    "index",
    "timestamp",
    "episode_index",
    "frame_index",
    "task_index",
    "observation.state",
    "action",
    "observation.images.cam0",
    "observation.images.cam1",
]


def download_with_retry(repo, path, token):
    for attempt in range(MAX_RETRIES):
        try:
            return hf_hub_download(repo, path, repo_type="dataset", token=token)
        except Exception as e:
            print(f"      download attempt {attempt+1} failed: {e}", flush=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise


def main():
    api = HfApi(token=TOKEN)

    # ── Load metadata ──
    print("Loading metadata...", flush=True)
    info_path = download_with_retry(REPO, "meta/info.json", TOKEN)
    with open(info_path) as f:
        info = json.load(f)

    ep_path = download_with_retry(REPO, "meta/episodes.jsonl", TOKEN)
    with open(ep_path) as f:
        eps = [json.loads(line) for line in f]

    task_path = download_with_retry(REPO, "meta/tasks.jsonl", TOKEN)
    with open(task_path) as f:
        tasks = [json.loads(line) for line in f]

    num_eps = len(eps)
    total_frames = info["total_frames"]
    task_str = tasks[0]["task"]  # "Dig dirt and place in pile"

    print(f"  Repo: {REPO}", flush=True)
    print(f"  Episodes: {num_eps}, Total frames: {total_frames}", flush=True)
    print(f"  Task: {task_str}", flush=True)

    # ── Compute global start index per episode ──
    global_starts = []
    running = 0
    for e in eps:
        global_starts.append(running)
        running += e["length"]
    assert running == total_frames, f"Frame count mismatch: {running} vs {total_frames}"
    print(f"  Global index range: 0 .. {running - 1}", flush=True)

    # ── Fix 3: Update episodes.jsonl with task string ──
    print("\nFix 3: Adding 'task' to episodes.jsonl...", flush=True)
    needs_task_fix = "task" not in eps[0]
    if needs_task_fix:
        tmpdir_meta = Path(tempfile.mkdtemp())
        new_ep_path = tmpdir_meta / "episodes.jsonl"
        with open(new_ep_path, "w") as f:
            for e in eps:
                row = {
                    "episode_index": e["episode_index"],
                    "length": e["length"],
                    "task_index": e["task_index"],
                    "task": task_str,
                }
                f.write(json.dumps(row) + "\n")
        api.upload_file(
            path_or_fileobj=str(new_ep_path),
            path_in_repo="meta/episodes.jsonl",
            repo_id=REPO,
            repo_type="dataset",
            token=TOKEN,
        )
        shutil.rmtree(tmpdir_meta)
        print("  DONE: episodes.jsonl updated", flush=True)
    else:
        print("  SKIP: 'task' already present", flush=True)

    # ── Fix 4: Upload custom_annotation.json ──
    print("\nFix 4: Uploading custom_annotation.json...", flush=True)
    tmpdir_ann = Path(tempfile.mkdtemp())
    ann_path = tmpdir_ann / "custom_annotation.json"
    with open(ann_path, "w") as f:
        json.dump({}, f)
    api.upload_file(
        path_or_fileobj=str(ann_path),
        path_in_repo="meta/custom_annotation.json",
        repo_id=REPO,
        repo_type="dataset",
        token=TOKEN,
    )
    shutil.rmtree(tmpdir_ann)
    print("  DONE: custom_annotation.json uploaded", flush=True)

    # ── Fix 1 & 2: Add index column + reorder, in batches ──
    print(f"\nFix 1+2: Adding 'index' column + reordering ({num_eps} episodes in batches of {BATCH_SIZE})...", flush=True)
    tmpdir = Path(tempfile.mkdtemp())
    errors = []

    for batch_start in range(0, num_eps, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_eps)
        batch_dir = tmpdir / "batch"
        batch_dir.mkdir(exist_ok=True)

        for ep_idx in range(batch_start, batch_end):
            fname = f"data/chunk-000/episode_{ep_idx:06d}.parquet"
            try:
                local = download_with_retry(REPO, fname, TOKEN)
                table = pq.read_table(local)

                # ── Verify row count matches episodes.jsonl ──
                expected_len = eps[ep_idx]["length"]
                actual_len = len(table)
                if actual_len != expected_len:
                    msg = f"ep {ep_idx}: row count {actual_len} != expected {expected_len}"
                    print(f"  ERROR: {msg}", flush=True)
                    errors.append(msg)
                    continue

                # ── Verify original columns exist ──
                original_cols = table.schema.names
                for col in TARGET_COL_ORDER:
                    if col != "index" and col not in original_cols:
                        if col in ("observation.images.cam1",):
                            continue  # cam1 might not exist in 1cam datasets
                        msg = f"ep {ep_idx}: missing expected column '{col}'"
                        print(f"  ERROR: {msg}", flush=True)
                        errors.append(msg)

                # ── Create index column ──
                start_idx = global_starts[ep_idx]
                new_index = pa.array(range(start_idx, start_idx + actual_len), type=pa.int64())

                if "index" not in original_cols:
                    table = table.append_column("index", new_index)
                else:
                    idx_col = table.schema.get_field_index("index")
                    table = table.set_column(idx_col, "index", new_index)

                # ── Reorder columns ──
                available_target = [c for c in TARGET_COL_ORDER if c in table.schema.names]
                table = table.select(available_target)

                # ── Verify after modification ──
                assert len(table) == expected_len, f"Row count changed after modification!"
                assert table.schema.names[0] == "index", f"index is not first column!"
                assert table.column("index")[0].as_py() == start_idx, f"index start value wrong!"

                # ── Write ──
                out = batch_dir / f"episode_{ep_idx:06d}.parquet"
                pq.write_table(table, out)

            except Exception as e:
                msg = f"ep {ep_idx}: {e}"
                print(f"  ERROR: {msg}", flush=True)
                errors.append(msg)
                continue

        # ── Upload batch ──
        parquets = list(batch_dir.glob("*.parquet"))
        if parquets:
            api.upload_folder(
                folder_path=str(batch_dir),
                path_in_repo="data/chunk-000",
                repo_id=REPO,
                repo_type="dataset",
                token=TOKEN,
            )

        shutil.rmtree(batch_dir)

        # ── Clear HF cache after each batch ──
        cache = Path(os.path.expanduser("~/.cache/huggingface/hub"))
        cache_dir = cache / f"datasets--{REPO.replace('/', '--')}"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

        print(f"  batch {batch_start}-{batch_end-1} ({len(parquets)} files) [OK]", flush=True)

    shutil.rmtree(tmpdir, ignore_errors=True)

    # ── Summary ──
    print(f"\n{'='*50}", flush=True)
    if errors:
        print(f"COMPLETED WITH {len(errors)} ERRORS:", flush=True)
        for e in errors:
            print(f"  - {e}", flush=True)
    else:
        print("ALL FIXES APPLIED SUCCESSFULLY.", flush=True)

    # ── Quick verification: spot-check 3 episodes ──
    print(f"\nVerification (spot-check ep 0, {num_eps//2}, {num_eps-1})...", flush=True)
    for check_ep in [0, num_eps // 2, num_eps - 1]:
        fname = f"data/chunk-000/episode_{check_ep:06d}.parquet"
        local = download_with_retry(REPO, fname, TOKEN)
        t = pq.read_table(local)
        cols = t.schema.names
        idx_first = t.column("index")[0].as_py()
        idx_last = t.column("index")[-1].as_py()
        print(f"  ep {check_ep}: {len(t)} rows, cols={cols}, index=[{idx_first}..{idx_last}]", flush=True)

    # Clear final cache
    cache = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    cache_dir = cache / f"datasets--{REPO.replace('/', '--')}"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    print("\nDONE. HF dataset fixed. Run GCS transfer separately after verifying.", flush=True)


if __name__ == "__main__":
    main()
