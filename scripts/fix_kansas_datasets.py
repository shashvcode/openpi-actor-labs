"""Fix Kansas datasets: add index column, fix task_index, reorder columns.
Processes in batches to manage disk space. Uploads fixed parquets back to HF."""

import json, os, sys, time, tempfile, pathlib, shutil
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, HfApi

TOKEN = os.environ["HF_TOKEN"]
BATCH_SIZE = 30
MAX_RETRIES = 3

def download(repo, path, token):
    for attempt in range(MAX_RETRIES):
        try:
            return hf_hub_download(repo, path, repo_type="dataset", token=token)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise

def fix_dataset(repo):
    api = HfApi(token=TOKEN)

    print(f"\n{'='*60}", flush=True)
    print(f"FIXING: {repo}", flush=True)
    print(f"{'='*60}", flush=True)

    # Load metadata
    print("[1/5] Loading metadata...", flush=True)
    info_path = download(repo, "meta/info.json", TOKEN)
    eps_path = download(repo, "meta/episodes.jsonl", TOKEN)
    tasks_path = download(repo, "meta/tasks.jsonl", TOKEN)

    with open(info_path) as f:
        info = json.load(f)
    with open(eps_path) as f:
        eps = [json.loads(l) for l in f]
    with open(tasks_path) as f:
        tasks = [json.loads(l) for l in f]

    tasks_map = {t["task_index"]: t["task"] for t in tasks}
    num_eps = len(eps)
    print(f"    {num_eps} episodes, {info['total_frames']} frames", flush=True)

    # Pre-compute global index starts
    print("[2/5] Computing global index offsets...", flush=True)
    global_starts = []
    running = 0
    for e in eps:
        global_starts.append(running)
        running += e["length"]
    print(f"    Global index range: 0-{running - 1}", flush=True)

    # Determine target column order based on what image cols exist
    sample_path = download(repo, info["data_path"].format(file_index=0, chunk_index=0), TOKEN)
    sample_table = pq.read_table(sample_path)
    image_cols = [c for c in sample_table.schema.names if "images" in c]
    target_order = ["observation.state", "action"] + sorted(image_cols) + [
        "timestamp", "frame_index", "episode_index", "index", "task_index"
    ]
    print(f"    Target column order: {target_order}", flush=True)

    # Update episodes.jsonl to include task string
    print("[3/5] Updating episodes.jsonl (adding task strings)...", flush=True)
    tmpdir = pathlib.Path(tempfile.mkdtemp())
    meta_dir = tmpdir / "meta"
    meta_dir.mkdir()

    new_eps = []
    for e in eps:
        ne = dict(e)
        ne["task"] = tasks_map[e["task_index"]]
        new_eps.append(ne)

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ne in new_eps:
            f.write(json.dumps(ne) + "\n")

    api.upload_file(
        path_or_fileobj=str(meta_dir / "episodes.jsonl"),
        path_in_repo="meta/episodes.jsonl",
        repo_id=repo,
        repo_type="dataset",
        commit_message="Add task strings to episodes.jsonl",
        token=TOKEN,
    )
    print("    episodes.jsonl uploaded", flush=True)

    # Process parquets in batches
    print(f"[4/5] Processing {num_eps} parquets in batches of {BATCH_SIZE}...", flush=True)
    data_dir = tmpdir / "data" / "chunk-000"
    data_dir.mkdir(parents=True)

    for batch_start in range(0, num_eps, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_eps)
        batch_eps = list(range(batch_start, batch_end))

        for ep_idx in batch_eps:
            pf = info["data_path"].format(file_index=ep_idx, chunk_index=ep_idx // 1000)
            local = download(repo, pf, TOKEN)
            table = pq.read_table(local)
            n = len(table)

            correct_ti = eps[ep_idx]["task_index"]
            g_start = global_starts[ep_idx]

            # Fix task_index
            new_ti = pa.array([correct_ti] * n, type=pa.int64())
            ti_idx = table.schema.get_field_index("task_index")
            table = table.set_column(ti_idx, "task_index", new_ti)

            # Add index column
            new_index = pa.array(list(range(g_start, g_start + n)), type=pa.int64())
            table = table.append_column("index", new_index)

            # Reorder columns
            table = table.select(target_order)

            out = data_dir / f"episode_{ep_idx:06d}.parquet"
            pq.write_table(table, out)

        # Upload batch
        api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=str(data_dir),
            path_in_repo="data/chunk-000",
            commit_message=f"Fix parquets batch {batch_start}-{batch_end-1}: add index, fix task_index, reorder cols",
            token=TOKEN,
        )

        # Cleanup
        for f in data_dir.iterdir():
            f.unlink()

        print(f"    batch {batch_start}-{batch_end-1} ({len(batch_eps)} files) UPLOADED", flush=True)

    # Verify
    print("[5/5] Quick verification...", flush=True)
    for ep_idx in [0, num_eps // 2, num_eps - 1]:
        pf = info["data_path"].format(file_index=ep_idx, chunk_index=ep_idx // 1000)
        local = download(repo, pf, TOKEN)
        table = pq.read_table(local, use_pandas_metadata=False)
        n = len(table)
        cols = table.schema.names
        pq_ti = table.column("task_index")[0].as_py()
        idx = table.column("index").to_pylist()
        errs = []
        if cols != target_order:
            errs.append(f"col order: {cols}")
        if pq_ti != eps[ep_idx]["task_index"]:
            errs.append(f"task_index pq={pq_ti} jsonl={eps[ep_idx]['task_index']}")
        if idx != list(range(global_starts[ep_idx], global_starts[ep_idx] + n)):
            errs.append("index wrong")
        status = "PASS" if not errs else f"FAIL: {errs}"
        print(f"    ep {ep_idx}: {n} rows, idx {idx[0]}-{idx[-1]}, ti={pq_ti}, cols={len(cols)}  [{status}]", flush=True)

    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"\nDONE: {repo}", flush=True)


def main():
    for repo in ["verm11/pi_kansas_data_1cam", "verm11/pi_kansasdata"]:
        fix_dataset(repo)
    print("\nALL DONE.", flush=True)


if __name__ == "__main__":
    main()
