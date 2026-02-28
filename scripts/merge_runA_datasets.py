"""Merge verm11/runA_1 episodes into verm11/runA.

Downloads each parquet from runA_1 one at a time, renumbers episode_index
and global index to continue from where runA left off, then uploads to runA.
Finally regenerates all metadata files.
"""

import io
import json
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

SRC_REPO = "verm11/runA_1"
SRC_PREFIX = "so100_v2/"
DST_REPO = "verm11/runA"
DST_EPISODES_START = 215
SRC_EPISODE_COUNT = 288

api = HfApi()


def get_episode_lengths_from_dst():
    """Read existing episodes.jsonl from DST_REPO to get lengths for episodes 0-214."""
    path = hf_hub_download(DST_REPO, "meta/episodes.jsonl", repo_type="dataset")
    lengths = {}
    with open(path) as f:
        for line in f:
            ep = json.loads(line)
            lengths[ep["episode_index"]] = ep["length"]
    return lengths


def compute_global_index_start(existing_lengths: dict) -> int:
    return sum(existing_lengths.values())


def renumber_and_upload(src_ep: int, dst_ep: int, global_idx_start: int) -> int:
    """Download one episode from SRC, renumber, upload to DST. Returns frame count."""
    src_path = f"{SRC_PREFIX}data/chunk-000/episode_{src_ep:06d}.parquet"
    local = hf_hub_download(SRC_REPO, src_path, repo_type="dataset")

    table = pq.read_table(local)
    n = len(table)

    new_ep_col = pa.array([dst_ep] * n, type=pa.int64())
    new_idx_col = pa.array(list(range(global_idx_start, global_idx_start + n)), type=pa.int64())

    col_names = table.column_names
    ep_i = col_names.index("episode_index")
    idx_i = col_names.index("index")

    table = table.set_column(ep_i, "episode_index", new_ep_col)
    table = table.set_column(idx_i, "index", new_idx_col)

    dst_path = f"data/chunk-000/episode_{dst_ep:06d}.parquet"
    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        pq.write_table(table, tmp.name)
        api.upload_file(
            path_or_fileobj=tmp.name,
            path_in_repo=dst_path,
            repo_id=DST_REPO,
            repo_type="dataset",
        )
    print(f"  ep {src_ep} -> {dst_ep}  ({n} frames, global {global_idx_start}-{global_idx_start + n - 1})")
    return n


def build_episode_stats(ep_idx: int, table: pa.Table) -> dict:
    """Compute min/max/mean/std/count for state and action columns."""
    stats = {}
    for col_name in ["observation.state", "action"]:
        col = table.column(col_name)
        arr = np.array([row.as_py() for row in col], dtype=np.float32)
        stats[col_name] = {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [int(len(arr))],
        }
    return {"episode_index": ep_idx, "stats": stats}


def regenerate_metadata(all_lengths: dict):
    """Rebuild info.json, episodes.jsonl, episodes_stats.jsonl and upload."""
    total_eps = len(all_lengths)
    total_frames = sum(all_lengths.values())

    info_path = hf_hub_download(DST_REPO, "meta/info.json", repo_type="dataset")
    with open(info_path) as f:
        info = json.load(f)
    info["total_episodes"] = total_eps
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{total_eps}"}

    info_bytes = json.dumps(info, indent=2).encode()
    api.upload_file(
        path_or_fileobj=io.BytesIO(info_bytes),
        path_in_repo="meta/info.json",
        repo_id=DST_REPO,
        repo_type="dataset",
    )
    print(f"Updated info.json: {total_eps} episodes, {total_frames} frames")

    task_str = "Pick up the bottle and place it on the yellow outlined square."
    episodes_lines = []
    for ep_idx in sorted(all_lengths.keys()):
        episodes_lines.append(json.dumps({
            "episode_index": ep_idx,
            "length": all_lengths[ep_idx],
            "task_index": 0,
            "task": task_str,
        }))
    episodes_bytes = ("\n".join(episodes_lines) + "\n").encode()
    api.upload_file(
        path_or_fileobj=io.BytesIO(episodes_bytes),
        path_in_repo="meta/episodes.jsonl",
        repo_id=DST_REPO,
        repo_type="dataset",
    )
    print(f"Updated episodes.jsonl: {len(episodes_lines)} entries")

    # Rebuild episodes_stats from existing + new
    existing_stats_path = hf_hub_download(DST_REPO, "meta/episodes_stats.jsonl", repo_type="dataset")
    existing_stats = {}
    with open(existing_stats_path) as f:
        for line in f:
            entry = json.loads(line)
            existing_stats[entry["episode_index"]] = entry

    stats_lines = []
    for ep_idx in sorted(all_lengths.keys()):
        if ep_idx in existing_stats:
            stats_lines.append(json.dumps(existing_stats[ep_idx]))
        else:
            parquet_path = f"data/chunk-000/episode_{ep_idx:06d}.parquet"
            local = hf_hub_download(DST_REPO, parquet_path, repo_type="dataset")
            table = pq.read_table(local)
            stat = build_episode_stats(ep_idx, table)
            stats_lines.append(json.dumps(stat))
            if ep_idx % 50 == 0:
                print(f"  computed stats for ep {ep_idx}")

    stats_bytes = ("\n".join(stats_lines) + "\n").encode()
    api.upload_file(
        path_or_fileobj=io.BytesIO(stats_bytes),
        path_in_repo="meta/episodes_stats.jsonl",
        repo_id=DST_REPO,
        repo_type="dataset",
    )
    print(f"Updated episodes_stats.jsonl: {len(stats_lines)} entries")


def main():
    print(f"Merging {SRC_REPO} into {DST_REPO}")
    print(f"  Source episodes: 0-{SRC_EPISODE_COUNT - 1} (from {SRC_PREFIX})")
    print(f"  Destination: episodes {DST_EPISODES_START}-{DST_EPISODES_START + SRC_EPISODE_COUNT - 1}")

    existing_lengths = get_episode_lengths_from_dst()
    print(f"  Existing episodes in DST: {len(existing_lengths)}")

    global_idx = compute_global_index_start(existing_lengths)
    print(f"  Global index starts at: {global_idx}")

    all_lengths = dict(existing_lengths)

    for src_ep in range(SRC_EPISODE_COUNT):
        dst_ep = DST_EPISODES_START + src_ep
        if dst_ep in all_lengths:
            print(f"  ep {dst_ep} already exists, skipping")
            global_idx += all_lengths[dst_ep]
            continue
        n = renumber_and_upload(src_ep, dst_ep, global_idx)
        all_lengths[dst_ep] = n
        global_idx += n

    print(f"\nAll {SRC_EPISODE_COUNT} episodes uploaded. Regenerating metadata...")
    regenerate_metadata(all_lengths)
    print("Done!")


if __name__ == "__main__":
    main()
