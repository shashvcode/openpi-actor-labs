"""Fix task_index in actor_teleop_300 parquets to match episodes.jsonl.
Only scans eps 241-299 (the ones inherited from source eps 242+ with the bug)."""
import json, os, sys, time, tempfile, pathlib
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, HfApi

TOKEN = os.environ["HF_TOKEN"]
REPO = "verm11/actor_teleop_300"
START_EP = 241
MAX_RETRIES = 3

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

    print("[1/4] Downloading metadata...", flush=True)
    info_path = download_with_retry(REPO, "meta/info.json", TOKEN)
    eps_path = download_with_retry(REPO, "meta/episodes.jsonl", TOKEN)
    with open(info_path) as f:
        info = json.load(f)
    with open(eps_path) as f:
        eps = [json.loads(l) for l in f]
    print(f"    {len(eps)} episodes total, scanning {START_EP}-{len(eps)-1}", flush=True)

    print(f"[2/4] Scanning eps {START_EP}-299 for mismatches...", flush=True)
    mismatched = []
    for i in range(START_EP, len(eps)):
        pf = info["data_path"].format(file_index=i, chunk_index=0)
        local = download_with_retry(REPO, pf, TOKEN)
        t = pq.read_table(local)
        pq_ti = t.column("task_index")[0].as_py()
        jsonl_ti = eps[i]["task_index"]
        if pq_ti != jsonl_ti:
            mismatched.append((i, jsonl_ti, local))
            print(f"    ep {i}: parquet={pq_ti} -> should be {jsonl_ti}  MISMATCH", flush=True)
        else:
            print(f"    ep {i}: OK (ti={pq_ti})", flush=True)

    print(f"    TOTAL mismatches: {len(mismatched)}", flush=True)
    if not mismatched:
        print("Nothing to fix!")
        return

    print("[3/4] Rewriting parquets...", flush=True)
    tmpdir = pathlib.Path(tempfile.mkdtemp())
    data_dir = tmpdir / "data" / "chunk-000"
    data_dir.mkdir(parents=True)

    for count, (ep_idx, correct_ti, local_path) in enumerate(mismatched, 1):
        t = pq.read_table(local_path)
        new_col = pa.array([correct_ti] * len(t), type=pa.int64())
        col_idx = t.schema.get_field_index("task_index")
        t = t.set_column(col_idx, "task_index", new_col)
        pq.write_table(t, data_dir / f"episode_{ep_idx:06d}.parquet")
        print(f"    wrote ep {ep_idx} [{count}/{len(mismatched)}]", flush=True)

    print("[4/4] Uploading to HF...", flush=True)
    api.upload_folder(
        repo_id=REPO,
        repo_type="dataset",
        folder_path=str(data_dir),
        path_in_repo="data/chunk-000",
        commit_message="Fix task_index in parquets to match episodes.jsonl",
    )
    print("    Upload complete!", flush=True)

    import shutil
    shutil.rmtree(tmpdir)
    print(f"\nDONE. Fixed {len(mismatched)} episodes in {REPO}.", flush=True)

if __name__ == "__main__":
    main()
