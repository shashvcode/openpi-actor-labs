"""Fix verm11/excavator-teleop metadata to match LeRobot v2.1 format.

The dataset has 7 cameras but training only uses csi_0_imx219.
This script:
  1. Rebuilds info.json with ONLY csi_0 declared (so LeRobot doesn't download unused videos)
  2. Rebuilds episodes.jsonl with only episodes that have csi_0 video, re-indexed contiguously
  3. Rebuilds episodes_stats.jsonl to match
  4. Recreates the v2.1 tag
"""

import json
import os
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, upload_file

REPO_ID = "verm11/excavator-teleop"
REPO_TYPE = "dataset"


def find_csi0_episodes(api: HfApi) -> list[int]:
    """Find which episode indices have csi_0_imx219 video files."""
    print("\n=== Scanning repo for csi_0 episodes ===")
    all_files = list(api.list_repo_tree(REPO_ID, repo_type=REPO_TYPE, recursive=True))
    file_paths = [f.rfilename for f in all_files if hasattr(f, "rfilename")]

    csi0_videos = sorted([f for f in file_paths if "csi_0_imx219" in f and f.endswith(".mp4")])
    csi0_episodes = set()
    for vf in csi0_videos:
        fname = Path(vf).stem
        ep_num = int(fname.replace("episode_", ""))
        csi0_episodes.add(ep_num)

    all_parquets = sorted([f for f in file_paths if f.endswith(".parquet")])
    all_episodes = set()
    for pf in all_parquets:
        fname = Path(pf).stem
        ep_num = int(fname.replace("episode_", ""))
        all_episodes.add(ep_num)

    missing = sorted(all_episodes - csi0_episodes)
    print(f"  Episodes with parquet data: {len(all_episodes)}")
    print(f"  Episodes with csi_0 video:  {len(csi0_episodes)}")
    if missing:
        print(f"  Episodes WITHOUT csi_0 ({len(missing)}): {missing}")
    return sorted(csi0_episodes)


def fix_info_json(api: HfApi, csi0_episodes: list[int]):
    print("\n=== Rebuilding info.json ===")

    dl = hf_hub_download(REPO_ID, "meta/info.json", repo_type=REPO_TYPE, force_download=True)
    with open(dl) as f:
        old_info = json.load(f)

    # Load episodes.jsonl to compute total_frames for csi_0-only episodes
    ep_dl = hf_hub_download(REPO_ID, "meta/episodes.jsonl", repo_type=REPO_TYPE, force_download=True)
    with open(ep_dl) as f:
        all_episodes = [json.loads(line) for line in f if line.strip()]

    ep_by_idx = {e["episode_index"]: e for e in all_episodes}
    total_frames = sum(ep_by_idx[i]["length"] for i in csi0_episodes if i in ep_by_idx)

    info = {
        "codebase_version": "v2.1",
        "robot_type": old_info.get("robot_type", "excavator"),
        "total_episodes": len(csi0_episodes),
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": old_info.get("fps", 10),
        "splits": {"train": f"0:{len(csi0_episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [4],
                "names": [["left_x", "left_y", "right_x", "right_y"]]
            },
            "action": {
                "dtype": "float32",
                "shape": [4],
                "names": [["left_x", "left_y", "right_x", "right_y"]]
            },
            "observation.images.csi_0_imx219": {
                "dtype": "video",
                "shape": [3, 480, 640],
                "names": ["channels", "height", "width"],
                "video_info": {
                    "video.fps": 10.0,
                    "video.height": 1232,
                    "video.width": 1640,
                    "video.codec": "av1",
                    "has_audio": False
                }
            },
            "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
            "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
            "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
            "index":         {"dtype": "int64",   "shape": [1], "names": None},
            "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(info, f, indent=2)
        tmp_path = f.name

    upload_file(path_or_fileobj=tmp_path, path_in_repo="meta/info.json",
                repo_id=REPO_ID, repo_type=REPO_TYPE)
    os.unlink(tmp_path)

    print(f"  total_episodes: {info['total_episodes']}")
    print(f"  total_frames:   {info['total_frames']}")
    print(f"  features:       {list(info['features'].keys())}")
    print(f"  Removed USB cameras — only csi_0_imx219 declared")
    return info, ep_by_idx


def fix_episodes_jsonl(api: HfApi, csi0_episodes: list[int], ep_by_idx: dict):
    """Rebuild episodes.jsonl with only csi_0 episodes, keeping original indices."""
    print("\n=== Rebuilding episodes.jsonl ===")

    lines = []
    for ep_idx in csi0_episodes:
        if ep_idx not in ep_by_idx:
            continue
        ep = ep_by_idx[ep_idx]
        lines.append({
            "episode_index": ep["episode_index"],
            "length": ep["length"],
            "task_index": ep.get("task_index", 0),
            "task": ep.get("task", "Scoop up packing peanuts from large pool and put in to small pool"),
        })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for entry in lines:
            f.write(json.dumps(entry) + "\n")
        tmp_path = f.name

    upload_file(path_or_fileobj=tmp_path, path_in_repo="meta/episodes.jsonl",
                repo_id=REPO_ID, repo_type=REPO_TYPE)
    os.unlink(tmp_path)
    print(f"  Wrote {len(lines)} episodes (only those with csi_0 video)")
    if lines:
        print(f"  Index range: {lines[0]['episode_index']} – {lines[-1]['episode_index']}")


def fix_episodes_stats(api: HfApi, csi0_episodes: list[int]):
    print("\n=== Rebuilding episodes_stats.jsonl ===")

    dl = hf_hub_download(REPO_ID, "meta/episodes_stats.jsonl", repo_type=REPO_TYPE, force_download=True)
    with open(dl) as f:
        all_stats = [json.loads(line) for line in f if line.strip()]

    stats_by_idx = {}
    for entry in all_stats:
        idx = entry.get("episode_index", entry.get("index", -1))
        stats_by_idx[idx] = entry

    csi0_set = set(csi0_episodes)
    fixed_lines = []
    for entry in all_stats:
        ep_idx = entry.get("episode_index", entry.get("index", -1))
        if ep_idx not in csi0_set:
            continue

        if "stats" in entry:
            stats = entry["stats"]
        else:
            stats = {k: v for k, v in entry.items()
                     if k not in ("episode_index", "index", "length", "task_index", "pi_episode_id")}

        for key in stats:
            s = stats[key]
            if "count" not in s:
                s["count"] = [300]
            elif isinstance(s["count"], list) and len(s["count"]) > 1:
                s["count"] = [s["count"][0]]
            elif isinstance(s["count"], (int, float)):
                s["count"] = [int(s["count"])]

        fixed_lines.append({"episode_index": ep_idx, "stats": stats})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for entry in fixed_lines:
            f.write(json.dumps(entry) + "\n")
        tmp_path = f.name

    upload_file(path_or_fileobj=tmp_path, path_in_repo="meta/episodes_stats.jsonl",
                repo_id=REPO_ID, repo_type=REPO_TYPE)
    os.unlink(tmp_path)
    print(f"  Wrote {len(fixed_lines)} episode stats entries")


def recreate_tag(api: HfApi):
    print("\n=== Recreating v2.1 tag ===")
    try:
        api.delete_tag(REPO_ID, tag="v2.1", repo_type=REPO_TYPE)
        print("  Deleted old v2.1 tag")
    except Exception:
        print("  No existing v2.1 tag to delete")
    time.sleep(2)
    api.create_tag(REPO_ID, tag="v2.1", repo_type=REPO_TYPE)
    print("  Created new v2.1 tag")


def main():
    api = HfApi()

    csi0_episodes = find_csi0_episodes(api)
    info, ep_by_idx = fix_info_json(api, csi0_episodes)
    fix_episodes_jsonl(api, csi0_episodes, ep_by_idx)
    fix_episodes_stats(api, csi0_episodes)
    recreate_tag(api)

    print("\n" + "=" * 60)
    print("DONE. Next steps on the pod:")
    print("=" * 60)
    print()
    print("1. Clear local cache:")
    print("   rm -rf /workspace/.hf_home/lerobot/verm11/excavator-teleop")
    print("   rm -rf /root/.cache/huggingface/hub/datasets--verm11--excavator-teleop")
    print()
    print("2. Pre-download ONLY needed files (parquet + csi_0 videos):")
    print("   HF_HOME=/workspace/.hf_home uv run huggingface-cli download \\")
    print("     verm11/excavator-teleop \\")
    print("     --repo-type dataset \\")
    print('     --include "meta/*" "data/*" "videos/*/observation.images.csi_0_imx219/*" \\')
    print("     --local-dir /workspace/.hf_home/lerobot/verm11/excavator-teleop")
    print()
    print("3. Then run norm stats + training:")
    print("   nohup bash -c 'source $HOME/.local/bin/env && cd /workspace/openpi && \\")
    print("   HF_HOME=/workspace/.hf_home uv run python scripts/compute_norm_stats.py --config-name pi05_excavator_lora && \\")
    print("   HF_HOME=/workspace/.hf_home uv run python scripts/train.py --config-name pi05_excavator_lora' \\")
    print("   > /workspace/train_excavator.log 2>&1 &")


if __name__ == "__main__":
    main()
