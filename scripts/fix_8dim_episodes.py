"""Truncate 8-dim state/action to 6-dim for episodes 0-45 and regenerate episodes_stats.jsonl."""

import json
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = pathlib.Path("./so100_merged/data/chunk-000")
META_DIR = pathlib.Path("./so100_merged/meta")


def truncate_list_column(table: pa.Table, col_name: str, target_dim: int) -> pa.Table:
    col = table.column(col_name)
    values = col.to_pylist()
    truncated = [v[:target_dim] for v in values]
    new_col = pa.array(truncated, type=pa.list_(pa.float32()))
    idx = table.schema.get_field_index(col_name)
    return table.set_column(idx, col_name, new_col)


def compute_stats_for_array(arr: np.ndarray) -> dict:
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [len(arr)] * arr.shape[1],
    }


def main():
    fixed = 0
    for i in range(105):
        fpath = DATA_DIR / f"file-{i:03d}.parquet"
        pf = pq.ParquetFile(fpath)
        batches = list(pf.iter_batches(batch_size=50000))
        table = pa.Table.from_batches(batches)

        state_dim = len(table.column("observation.state").to_pylist()[0])
        if state_dim > 6:
            table = truncate_list_column(table, "observation.state", 6)
            table = truncate_list_column(table, "action", 6)
            pq.write_table(table, fpath)
            fixed += 1
            ep = table.column("episode_index").to_pylist()[0]
            print(f"  Fixed file-{i:03d}.parquet (episode {ep}): {state_dim} -> 6 dims")

    print(f"\nFixed {fixed} files.")

    # Regenerate episodes_stats.jsonl
    print("\nRegenerating episodes_stats.jsonl...")
    ep_data: dict[int, dict[str, list]] = {}

    for i in range(105):
        fpath = DATA_DIR / f"file-{i:03d}.parquet"
        for batch in pq.ParquetFile(fpath).iter_batches(batch_size=5000, columns=["episode_index", "observation.state", "action"]):
            for ep_idx, state, action in zip(
                batch.column("episode_index").to_pylist(),
                batch.column("observation.state").to_pylist(),
                batch.column("action").to_pylist(),
            ):
                if ep_idx not in ep_data:
                    ep_data[ep_idx] = {"states": [], "actions": []}
                ep_data[ep_idx]["states"].append(state)
                ep_data[ep_idx]["actions"].append(action)

    stats_lines = []
    for ep_idx in sorted(ep_data.keys()):
        states = np.array(ep_data[ep_idx]["states"], dtype=np.float32)
        actions = np.array(ep_data[ep_idx]["actions"], dtype=np.float32)
        ep_stats = {
            "observation.state": compute_stats_for_array(states),
            "action": compute_stats_for_array(actions),
        }
        stats_lines.append({"episode_index": ep_idx, "stats": ep_stats})

    out_path = META_DIR / "episodes_stats.jsonl"
    with open(out_path, "w") as f:
        for line in stats_lines:
            f.write(json.dumps(line) + "\n")

    print(f"Wrote {len(stats_lines)} episode stats (all 6-dim).")


if __name__ == "__main__":
    main()
