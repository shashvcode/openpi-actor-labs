"""Audit the verm11/excavator-teleop dataset for training readiness.

Checks:
  1. info.json structure and required fields
  2. Camera keys declared in features
  3. Whether csi_0_imx219 video files actually exist
  4. episodes.jsonl integrity
  5. Parquet data files existence
  6. Sample a video frame from csi_0_imx219 and save it
  7. State and action dimensions
"""

import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, list_repo_tree

REPO_ID = "verm11/excavator-teleop"
REPO_TYPE = "dataset"
OUT_DIR = Path("/workspace/excavator_audit")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    # ── 1. Download and inspect info.json ──
    print("\n" + "=" * 60)
    print("1. CHECKING info.json")
    print("=" * 60)
    try:
        info_path = hf_hub_download(
            repo_id=REPO_ID, repo_type=REPO_TYPE,
            filename="meta/info.json",
            local_dir=str(OUT_DIR / "meta_download"),
            force_download=True,
        )
        with open(info_path) as f:
            info = json.load(f)

        print(f"   codebase_version : {info.get('codebase_version', 'MISSING')}")
        print(f"   fps              : {info.get('fps', 'MISSING')}")
        print(f"   robot_type       : {info.get('robot_type', 'MISSING')}")
        print(f"   total_episodes   : {info.get('total_episodes', 'MISSING')}")
        print(f"   total_frames     : {info.get('total_frames', 'MISSING')}")
        print(f"   chunks_size      : {info.get('chunks_size', 'MISSING')}")
        print(f"   data_path        : {info.get('data_path', 'MISSING')}")
        print(f"   video_path       : {info.get('video_path', 'MISSING')}")

        features = info.get("features", {})
        print(f"\n   Features ({len(features)} total):")
        video_keys = []
        image_keys = []
        state_key = None
        action_key = None
        for k, v in features.items():
            dtype = v.get("dtype", "?")
            shape = v.get("shape", "?")
            names = v.get("names", "")
            is_video = "video" in str(dtype).lower() or "video" in str(v).lower()
            is_image = "image" in str(dtype).lower() or "image" in str(v).lower()
            tag = ""
            if is_video:
                video_keys.append(k)
                tag = " [VIDEO]"
            elif is_image:
                image_keys.append(k)
                tag = " [IMAGE]"
            if k == "observation.state":
                state_key = k
                tag += " [STATE]"
            if k == "action":
                action_key = k
                tag += " [ACTION]"
            print(f"      {k}: dtype={dtype}, shape={shape}{tag}")

        print(f"\n   Video keys: {video_keys}")
        print(f"   Image keys: {image_keys}")

        has_csi0 = any("csi_0" in k for k in video_keys + image_keys + list(features.keys()))
        print(f"\n   >>> csi_0 camera present in features: {has_csi0}")
        if not has_csi0:
            print("   !!! WARNING: csi_0_imx219 is NOT declared in features!")
            print("   Available observation keys:")
            for k in features:
                if k.startswith("observation"):
                    print(f"       {k}")
    except Exception as e:
        print(f"   ERROR fetching info.json: {e}")
        info = {}

    # ── 2. List repo files to find video/data structure ──
    print("\n" + "=" * 60)
    print("2. REPO FILE STRUCTURE")
    print("=" * 60)
    try:
        all_files = list(api.list_repo_tree(REPO_ID, repo_type=REPO_TYPE, recursive=True))
        file_paths = [f.rfilename for f in all_files if hasattr(f, "rfilename")]
        print(f"   Total files in repo: {len(file_paths)}")

        meta_files = [f for f in file_paths if f.startswith("meta/")]
        print(f"\n   Meta files ({len(meta_files)}):")
        for f in sorted(meta_files):
            print(f"      {f}")

        video_files = [f for f in file_paths if f.endswith(".mp4")]
        print(f"\n   Video files (.mp4): {len(video_files)}")
        if video_files:
            cam_names = set()
            for vf in video_files:
                parts = vf.split("/")
                if len(parts) >= 3:
                    cam_names.add(parts[-2] if "observation" not in parts[-1] else parts[-1].replace(".mp4", ""))
            # Better: extract camera name from path like videos/chunk-000/observation.images.csi_0_imx219/episode_000000.mp4
            cam_dirs = set()
            for vf in video_files:
                parent = str(Path(vf).parent.name)
                cam_dirs.add(parent)
            print(f"   Camera directories found: {cam_dirs}")

            csi0_videos = [f for f in video_files if "csi_0" in f]
            print(f"   csi_0 video files: {len(csi0_videos)}")
            if csi0_videos:
                print(f"   First 5 csi_0 videos:")
                for f in sorted(csi0_videos)[:5]:
                    print(f"      {f}")
            else:
                print("   !!! NO csi_0 video files found!")
                print("   Sample video paths:")
                for f in sorted(video_files)[:10]:
                    print(f"      {f}")

        parquet_files = [f for f in file_paths if f.endswith(".parquet")]
        print(f"\n   Parquet data files: {len(parquet_files)}")
        if parquet_files:
            print(f"   First 5:")
            for f in sorted(parquet_files)[:5]:
                print(f"      {f}")

    except Exception as e:
        print(f"   ERROR listing repo: {e}")
        file_paths = []
        csi0_videos = []

    # ── 3. Check episodes.jsonl ──
    print("\n" + "=" * 60)
    print("3. EPISODES.JSONL")
    print("=" * 60)
    try:
        ep_path = hf_hub_download(
            repo_id=REPO_ID, repo_type=REPO_TYPE,
            filename="meta/episodes.jsonl",
            local_dir=str(OUT_DIR / "meta_download"),
            force_download=True,
        )
        with open(ep_path) as f:
            episodes = [json.loads(line) for line in f if line.strip()]
        print(f"   Total episodes in episodes.jsonl: {len(episodes)}")
        if episodes:
            print(f"   First episode: {episodes[0]}")
            print(f"   Last episode:  {episodes[-1]}")
            ep_indices = [e.get("episode_index", e.get("index", "?")) for e in episodes]
            print(f"   Episode index range: {min(ep_indices)} - {max(ep_indices)}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # ── 4. Download and read one parquet to check columns ──
    print("\n" + "=" * 60)
    print("4. PARQUET DATA SAMPLE")
    print("=" * 60)
    if parquet_files:
        try:
            import pyarrow.parquet as pq
            pq_path = hf_hub_download(
                repo_id=REPO_ID, repo_type=REPO_TYPE,
                filename=parquet_files[0],
                local_dir=str(OUT_DIR / "data_download"),
                force_download=True,
            )
            table = pq.read_table(pq_path)
            print(f"   Parquet columns: {table.column_names}")
            print(f"   Rows in first file: {len(table)}")
            if "observation.state" in table.column_names:
                states = table["observation.state"].to_pylist()
                print(f"   State dim (first row): {len(states[0]) if states else '?'}")
                print(f"   State sample (first row): {states[0] if states else '?'}")
            if "action" in table.column_names:
                actions = table["action"].to_pylist()
                print(f"   Action dim (first row): {len(actions[0]) if actions else '?'}")
                print(f"   Action sample (first row): {actions[0] if actions else '?'}")
        except Exception as e:
            print(f"   ERROR reading parquet: {e}")

    # ── 5. Try to decode one csi_0 video frame ──
    print("\n" + "=" * 60)
    print("5. VIDEO FRAME SAMPLE")
    print("=" * 60)
    if csi0_videos:
        try:
            vid_file = csi0_videos[0]
            vid_path = hf_hub_download(
                repo_id=REPO_ID, repo_type=REPO_TYPE,
                filename=vid_file,
                local_dir=str(OUT_DIR / "video_download"),
            )
            import cv2
            cap = cv2.VideoCapture(vid_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                out_img = str(OUT_DIR / "csi_0_sample_frame.jpg")
                cv2.imwrite(out_img, frame)
                print(f"   Frame shape: {frame.shape}")
                print(f"   Saved sample frame to: {out_img}")
            else:
                print("   !!! Could not read frame from video")
        except Exception as e:
            print(f"   ERROR decoding video: {e}")
    elif video_files:
        try:
            vid_file = video_files[0]
            print(f"   No csi_0 videos, trying first available: {vid_file}")
            vid_path = hf_hub_download(
                repo_id=REPO_ID, repo_type=REPO_TYPE,
                filename=vid_file,
                local_dir=str(OUT_DIR / "video_download"),
            )
            import cv2
            cap = cv2.VideoCapture(vid_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                out_img = str(OUT_DIR / "first_cam_sample_frame.jpg")
                cv2.imwrite(out_img, frame)
                print(f"   Frame shape: {frame.shape}")
                print(f"   Saved sample frame to: {out_img}")
            else:
                print("   !!! Could not read frame from video")
        except Exception as e:
            print(f"   ERROR decoding video: {e}")
    else:
        print("   No video files found to sample!")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"   Dataset: {REPO_ID}")
    print(f"   Total episodes: {info.get('total_episodes', '?')}")
    print(f"   Total frames:   {info.get('total_frames', '?')}")
    print(f"   FPS:            {info.get('fps', '?')}")
    print(f"   Video files:    {len(video_files) if 'video_files' in dir() else '?'}")
    csi0_count = len(csi0_videos) if 'csi0_videos' in dir() else 0
    print(f"   csi_0 videos:   {csi0_count}")
    if csi0_count == 0:
        print("\n   >>> PROBLEM: No csi_0 camera data found!")
        print("   >>> The training config expects 'observation.images.csi_0_imx219'")
        print("   >>> You may need to use a different camera key.")
    else:
        print(f"\n   >>> csi_0 camera looks available with {csi0_count} video files")


if __name__ == "__main__":
    main()
