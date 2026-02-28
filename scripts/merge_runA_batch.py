"""Batch merge remaining verm11/runA_1 episodes into verm11/runA.

Episodes 215-338 are already merged. This script handles 339-502 (src episodes 124-287)
by downloading, renumbering, and batch-uploading via upload_folder.
"""

import json
import os
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

SRC_REPO = "verm11/runA_1"
SRC_PREFIX = "so100_v2/"
DST_REPO = "verm11/runA"

ALREADY_MERGED_UP_TO_SRC = 124  # src episodes 0-123 already uploaded as dst 215-338
SRC_EPISODE_COUNT = 288
DST_EPISODES_START = 215

api = HfApi()


def main():
    # Calculate where to resume
    resume_src = ALREADY_MERGED_UP_TO_SRC
    resume_dst = DST_EPISODES_START + resume_src
    remaining = SRC_EPISODE_COUNT - resume_src
    print(f"Resuming merge: src episodes {resume_src}-{SRC_EPISODE_COUNT-1} -> dst {resume_dst}-{DST_EPISODES_START + SRC_EPISODE_COUNT - 1}")
    print(f"  {remaining} episodes to go")

    # Get existing episode lengths to compute global index
    ep_path = hf_hub_download(DST_REPO, "meta/episodes.jsonl", repo_type="dataset")
    existing_lengths = {}
    with open(ep_path) as f:
        for line in f:
            ep = json.loads(line)
            existing_lengths[ep["episode_index"]] = ep["length"]
    global_idx = sum(existing_lengths.values())
    print(f"  Global index starts at: {global_idx}")

    staging = "/tmp/runA_merge_staging"
    if True:
        data_dir = os.path.join(staging, "data", "chunk-000")
        os.makedirs(data_dir, exist_ok=True)

        all_lengths = dict(existing_lengths)
        new_stats = []

        for src_ep in range(resume_src, SRC_EPISODE_COUNT):
            dst_ep = DST_EPISODES_START + src_ep
            src_path = f"{SRC_PREFIX}data/chunk-000/episode_{src_ep:06d}.parquet"
            local = hf_hub_download(SRC_REPO, src_path, repo_type="dataset")

            table = pq.read_table(local)
            n = len(table)

            col_names = table.column_names
            ep_i = col_names.index("episode_index")
            idx_i = col_names.index("index")
            table = table.set_column(ep_i, "episode_index", pa.array([dst_ep] * n, type=pa.int64()))
            table = table.set_column(idx_i, "index", pa.array(list(range(global_idx, global_idx + n)), type=pa.int64()))

            dst_file = os.path.join(data_dir, f"episode_{dst_ep:06d}.parquet")
            pq.write_table(table, dst_file)

            all_lengths[dst_ep] = n

            # Compute stats
            stats_entry = {"episode_index": dst_ep, "stats": {}}
            for col_name in ["observation.state", "action"]:
                arr = np.array([row.as_py() for row in table.column(col_name)], dtype=np.float32)
                stats_entry["stats"][col_name] = {
                    "min": arr.min(axis=0).tolist(),
                    "max": arr.max(axis=0).tolist(),
                    "mean": arr.mean(axis=0).tolist(),
                    "std": arr.std(axis=0).tolist(),
                    "count": [int(len(arr))],
                }
            new_stats.append(stats_entry)

            global_idx += n
            if (src_ep - resume_src) % 20 == 0:
                print(f"  Processed {src_ep - resume_src + 1}/{remaining}: ep {src_ep} -> {dst_ep} ({n} frames)")

        print(f"\nAll {remaining} parquets staged. Writing metadata...")

        # Build metadata in staging
        meta_dir = os.path.join(staging, "meta")
        os.makedirs(meta_dir)

        total_eps = len(all_lengths)
        total_frames = sum(all_lengths.values())

        # info.json
        info_local = hf_hub_download(DST_REPO, "meta/info.json", repo_type="dataset")
        with open(info_local) as f:
            info = json.load(f)
        info["total_episodes"] = total_eps
        info["total_frames"] = total_frames
        info["splits"] = {"train": f"0:{total_eps}"}
        with open(os.path.join(meta_dir, "info.json"), "w") as f:
            json.dump(info, f, indent=2)

        # episodes.jsonl
        task_str = "Pick up the bottle and place it on the yellow outlined square."
        with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
            for ep_idx in sorted(all_lengths.keys()):
                f.write(json.dumps({
                    "episode_index": ep_idx,
                    "length": all_lengths[ep_idx],
                    "task_index": 0,
                    "task": task_str,
                }) + "\n")

        # episodes_stats.jsonl
        existing_stats_path = hf_hub_download(DST_REPO, "meta/episodes_stats.jsonl", repo_type="dataset")
        existing_stats_lines = {}
        with open(existing_stats_path) as f:
            for line in f:
                entry = json.loads(line)
                existing_stats_lines[entry["episode_index"]] = line.strip()
        with open(os.path.join(meta_dir, "episodes_stats.jsonl"), "w") as f:
            for ep_idx in sorted(all_lengths.keys()):
                if ep_idx in existing_stats_lines:
                    f.write(existing_stats_lines[ep_idx] + "\n")
                else:
                    match = [s for s in new_stats if s["episode_index"] == ep_idx]
                    if match:
                        f.write(json.dumps(match[0]) + "\n")

        # tasks.jsonl
        with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
            f.write(json.dumps({"task_index": 0, "task": task_str}) + "\n")

        print(f"Metadata: {total_eps} episodes, {total_frames} frames")
        print(f"Staging dir: {staging}")
        print(f"Uploading to {DST_REPO}...")

        api.upload_folder(
            folder_path=staging,
            repo_id=DST_REPO,
            repo_type="dataset",
        )
        print("Upload complete!")


if __name__ == "__main__":
    main()
