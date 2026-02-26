"""Generate episodes_stats.jsonl for the merged SO-100 dataset.

Each line: {"episode_index": N, "stats": {"observation.state": {"min": [...], "max": [...], "mean": [...], "std": [...]}, "action": {...}}}
"""

import json
import pathlib

import numpy as np
import pyarrow.parquet as pq

DATA_DIR = pathlib.Path("./so100_merged/data/chunk-000")
META_DIR = pathlib.Path("./so100_merged/meta")


def compute_stats_for_array(arr: np.ndarray) -> dict:
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
    }


def main():
    episodes = []
    with open(META_DIR / "episodes.jsonl") as f:
        for line in f:
            episodes.append(json.loads(line))

    ep_data: dict[int, dict[str, list]] = {}

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    print(f"Reading {len(parquet_files)} parquet files...")

    for i, pf in enumerate(parquet_files):
        for batch in pq.ParquetFile(pf).iter_batches(batch_size=5000, columns=["episode_index", "observation.state", "action"]):
            ep_indices = batch.column("episode_index").to_pylist()
            states = batch.column("observation.state").to_pylist()
            actions = batch.column("action").to_pylist()

            for ep_idx, state, action in zip(ep_indices, states, actions):
                if ep_idx not in ep_data:
                    ep_data[ep_idx] = {"states": [], "actions": []}
                ep_data[ep_idx]["states"].append(state)
                ep_data[ep_idx]["actions"].append(action)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(parquet_files)} files...")

    print(f"Computing stats for {len(ep_data)} episodes...")
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

    print(f"Wrote {len(stats_lines)} episode stats to {out_path}")


if __name__ == "__main__":
    main()
