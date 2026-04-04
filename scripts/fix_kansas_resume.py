"""Resume fixing pi_kansasdata from episode 660."""
import json, os, time, tempfile, pathlib, shutil
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, HfApi

TOKEN = os.environ["HF_TOKEN"]
REPO = "verm11/pi_kansasdata"
BATCH_SIZE = 30
START_EP = 660

def download(repo, path, token):
    for attempt in range(3):
        try:
            return hf_hub_download(repo, path, repo_type="dataset", token=token)
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise

def main():
    api = HfApi(token=TOKEN)

    print(f"Resuming {REPO} from ep {START_EP}...", flush=True)
    info_path = download(REPO, "meta/info.json", TOKEN)
    eps_path = download(REPO, "meta/episodes.jsonl", TOKEN)
    with open(info_path) as f:
        info = json.load(f)
    with open(eps_path) as f:
        eps = [json.loads(l) for l in f]

    num_eps = len(eps)
    global_starts = []
    running = 0
    for e in eps:
        global_starts.append(running)
        running += e["length"]

    sample_path = download(REPO, info["data_path"].format(file_index=0, chunk_index=0), TOKEN)
    sample_table = pq.read_table(sample_path)
    image_cols = [c for c in sample_table.schema.names if "images" in c]

    # Check if index already exists from previous run
    if "index" in sample_table.schema.names:
        target_order = ["observation.state", "action"] + sorted(image_cols) + [
            "timestamp", "frame_index", "episode_index", "index", "task_index"
        ]
    else:
        target_order = ["observation.state", "action"] + sorted(image_cols) + [
            "timestamp", "frame_index", "episode_index", "index", "task_index"
        ]

    print(f"  {num_eps} episodes, resuming {START_EP}-{num_eps-1}", flush=True)

    tmpdir = pathlib.Path(tempfile.mkdtemp())
    data_dir = tmpdir / "data" / "chunk-000"
    data_dir.mkdir(parents=True)

    for batch_start in range(START_EP, num_eps, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_eps)

        for ep_idx in range(batch_start, batch_end):
            pf = info["data_path"].format(file_index=ep_idx, chunk_index=ep_idx // 1000)
            local = download(REPO, pf, TOKEN)
            table = pq.read_table(local)
            n = len(table)

            correct_ti = eps[ep_idx]["task_index"]
            g_start = global_starts[ep_idx]

            new_ti = pa.array([correct_ti] * n, type=pa.int64())
            ti_idx = table.schema.get_field_index("task_index")
            table = table.set_column(ti_idx, "task_index", new_ti)

            if "index" not in table.schema.names:
                new_index = pa.array(list(range(g_start, g_start + n)), type=pa.int64())
                table = table.append_column("index", new_index)
            else:
                new_index = pa.array(list(range(g_start, g_start + n)), type=pa.int64())
                idx_col = table.schema.get_field_index("index")
                table = table.set_column(idx_col, "index", new_index)

            table = table.select(target_order)
            pq.write_table(table, data_dir / f"episode_{ep_idx:06d}.parquet")

        api.upload_folder(
            repo_id=REPO, repo_type="dataset",
            folder_path=str(data_dir), path_in_repo="data/chunk-000",
            commit_message=f"Fix parquets batch {batch_start}-{batch_end-1}",
            token=TOKEN,
        )
        for f in data_dir.iterdir():
            f.unlink()
        print(f"  batch {batch_start}-{batch_end-1} UPLOADED", flush=True)

    # Verify
    print("\nVerifying...", flush=True)
    for ep_idx in [0, 660, num_eps - 1]:
        pf = info["data_path"].format(file_index=ep_idx, chunk_index=ep_idx // 1000)
        local = download(REPO, pf, TOKEN)
        table = pq.read_table(local)
        n = len(table)
        pq_ti = table.column("task_index")[0].as_py()
        idx = table.column("index").to_pylist()
        cols = table.schema.names
        errs = []
        if cols != target_order:
            errs.append("col order")
        if pq_ti != eps[ep_idx]["task_index"]:
            errs.append(f"ti pq={pq_ti} jsonl={eps[ep_idx]['task_index']}")
        if idx != list(range(global_starts[ep_idx], global_starts[ep_idx] + n)):
            errs.append("index")
        status = "PASS" if not errs else f"FAIL: {errs}"
        print(f"  ep {ep_idx}: {n} rows, idx {idx[0]}-{idx[-1]}, ti={pq_ti} [{status}]", flush=True)

    shutil.rmtree(tmpdir, ignore_errors=True)
    print("\nDONE.", flush=True)

if __name__ == "__main__":
    main()
