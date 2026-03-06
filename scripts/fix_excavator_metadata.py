"""Fix verm11/excavator-teleop metadata to match LeRobot v2.1 format.

Fixes:
  1. info.json: add chunks_size, data_path, video_path, splits, csi_0 feature, metadata columns
  2. episodes_stats.jsonl: ensure correct format with "stats" wrapper and scalar "count"
  3. Recreate v2.1 tag
"""

import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, upload_file

REPO_ID = "verm11/excavator-teleop"
REPO_TYPE = "dataset"


def fix_info_json(api: HfApi):
    print("\n=== Fixing info.json ===")

    dl = hf_hub_download(REPO_ID, "meta/info.json", repo_type=REPO_TYPE, force_download=True)
    with open(dl) as f:
        info = json.load(f)

    print(f"  Original features: {list(info.get('features', {}).keys())}")
    print(f"  Original total_episodes: {info.get('total_episodes')}")

    info["chunks_size"] = 1000
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

    total_ep = info.get("total_episodes", 624)
    info["splits"] = {"train": f"0:{total_ep}"}

    features = info.get("features", {})

    if "observation.images.csi_0_imx219" not in features:
        features["observation.images.csi_0_imx219"] = {
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
        }
        print("  Added csi_0_imx219 to features")

    for col_name, col_def in [
        ("timestamp", {"dtype": "float32", "shape": [1], "names": None}),
        ("frame_index", {"dtype": "int64", "shape": [1], "names": None}),
        ("episode_index", {"dtype": "int64", "shape": [1], "names": None}),
        ("index", {"dtype": "int64", "shape": [1], "names": None}),
        ("task_index", {"dtype": "int64", "shape": [1], "names": None}),
    ]:
        if col_name not in features:
            features[col_name] = col_def

    info["features"] = features

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(info, f, indent=2)
        tmp_path = f.name

    upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="meta/info.json",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )
    os.unlink(tmp_path)
    print(f"  Uploaded fixed info.json with {len(features)} features")
    print(f"  Features now: {list(features.keys())}")
    return info


def fix_episodes_stats(api: HfApi, info: dict):
    print("\n=== Fixing episodes_stats.jsonl ===")

    dl = hf_hub_download(REPO_ID, "meta/episodes_stats.jsonl", repo_type=REPO_TYPE, force_download=True)
    with open(dl) as f:
        lines = [json.loads(line) for line in f if line.strip()]

    print(f"  Total entries: {len(lines)}")
    if lines:
        print(f"  Sample keys in first entry: {list(lines[0].keys())}")

    fixed_lines = []
    for entry in lines:
        ep_idx = entry.get("episode_index", entry.get("index", 0))
        length = entry.get("length", 300)

        if "stats" in entry:
            stats = entry["stats"]
        else:
            stats = {k: v for k, v in entry.items() if k not in ("episode_index", "index", "length", "task_index")}

        for key in stats:
            s = stats[key]
            if "count" not in s:
                s["count"] = [length]
            elif isinstance(s["count"], list) and len(s["count"]) > 1:
                s["count"] = [s["count"][0]]
            elif isinstance(s["count"], (int, float)):
                s["count"] = [int(s["count"])]

        fixed_lines.append({"episode_index": ep_idx, "stats": stats})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for entry in fixed_lines:
            f.write(json.dumps(entry) + "\n")
        tmp_path = f.name

    upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="meta/episodes_stats.jsonl",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )
    os.unlink(tmp_path)
    print(f"  Uploaded fixed episodes_stats.jsonl ({len(fixed_lines)} entries)")


def find_csi0_episodes(api: HfApi):
    """Find which episodes have csi_0_imx219 video files."""
    print("\n=== Finding episodes with csi_0 camera ===")
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
    print(f"  Total episodes with parquet data: {len(all_episodes)}")
    print(f"  Episodes with csi_0 video: {len(csi0_episodes)}")
    print(f"  Episodes WITHOUT csi_0 ({len(missing)}): {missing}")

    return sorted(csi0_episodes)


def recreate_tag(api: HfApi):
    print("\n=== Recreating v2.1 tag ===")
    try:
        api.delete_tag(REPO_ID, tag="v2.1", repo_type=REPO_TYPE)
        print("  Deleted old v2.1 tag")
    except Exception:
        print("  No existing v2.1 tag to delete")

    api.create_tag(REPO_ID, tag="v2.1", repo_type=REPO_TYPE)
    print("  Created new v2.1 tag")


def main():
    api = HfApi()

    info = fix_info_json(api)
    fix_episodes_stats(api, info)
    csi0_episodes = find_csi0_episodes(api)
    recreate_tag(api)

    print("\n=== DONE ===")
    print(f"  {len(csi0_episodes)} episodes have csi_0 camera data")
    print("Now clear the local cache and retry:")
    print("  rm -rf /workspace/.hf_home/lerobot/verm11/excavator-teleop")
    print("  rm -rf /root/.cache/huggingface/hub/datasets--verm11--excavator-teleop")


if __name__ == "__main__":
    main()
