"""Convert a video-format LeRobot dataset to image-in-parquet format.

Reads each episode's video file(s) and parquet data, decodes every frame,
encodes as JPEG, and writes new parquet files with embedded images.
Also rebuilds all metadata files.

Usage:
    HF_HOME=/workspace/.hf_home uv run python scripts/convert_video_to_parquet.py \
        --repo-id verm11/excavator-teleop \
        --output-dir /workspace/excavator_v2_converted \
        --camera-key observation.images.csi_0_imx219
"""

import argparse
import io
import json
import logging
import math
import pathlib
import sys

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def decode_video_frames(video_path: pathlib.Path) -> list[bytes]:
    """Decode all frames from a video file and return as JPEG bytes."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frames.append(jpeg.tobytes())

    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="HF dataset repo id (e.g. verm11/excavator-teleop)")
    parser.add_argument("--output-dir", required=True, help="Local output directory for converted dataset")
    parser.add_argument("--camera-keys", nargs="+", required=True,
                        help="Camera feature keys to convert (e.g. observation.images.csi_0_imx219)")
    parser.add_argument("--hf-home", default=None, help="HF_HOME override")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--task", default="Scoop packing peanuts from large pool and dump into small pool")
    parser.add_argument("--chunks-size", type=int, default=1000)
    args = parser.parse_args()

    if args.hf_home:
        import os
        os.environ["HF_HOME"] = args.hf_home

    hf_cache_root = pathlib.Path(args.hf_home or "~/.cache/huggingface").expanduser()
    lerobot_root = hf_cache_root / "lerobot" / args.repo_id
    if not lerobot_root.exists():
        log.error("Dataset not found at %s. Download it first with huggingface-cli.", lerobot_root)
        sys.exit(1)

    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    data_dir = lerobot_root / "data"
    videos_dir = lerobot_root / "videos"

    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    log.info("Found %d parquet files in source dataset", len(parquet_files))

    video_episodes = {}
    for cam_key in args.camera_keys:
        video_episodes[cam_key] = set()
        for chunk_dir in sorted(videos_dir.glob("chunk-*")):
            cam_dir = chunk_dir / cam_key
            if cam_dir.exists():
                for vf in cam_dir.glob("episode_*.mp4"):
                    video_episodes[cam_key].add(int(vf.stem.replace("episode_", "")))
        log.info("Camera %s: found %d episodes with video", cam_key, len(video_episodes[cam_key]))

    valid_episodes = None
    for cam_key in args.camera_keys:
        if valid_episodes is None:
            valid_episodes = video_episodes[cam_key].copy()
        else:
            valid_episodes &= video_episodes[cam_key]

    parquet_ep_map = {}
    for pf in parquet_files:
        ep_idx = int(pf.stem.replace("episode_", ""))
        if valid_episodes is not None and ep_idx in valid_episodes:
            parquet_ep_map[ep_idx] = pf

    available = sorted(parquet_ep_map.keys())
    log.info("Episodes with both parquet + all camera videos: %d", len(available))

    if not available:
        log.error("No valid episodes found!")
        sys.exit(1)

    episodes_meta = []
    episodes_stats = []
    global_index = 0
    new_ep_idx = 0

    for orig_ep_idx in tqdm.tqdm(available, desc="Converting episodes"):
        pf = parquet_ep_map[orig_ep_idx]
        table = pq.read_table(pf)
        n_rows = len(table)

        cam_frames = {}
        for cam_key in args.camera_keys:
            chunk_idx = orig_ep_idx // args.chunks_size
            video_path = videos_dir / f"chunk-{chunk_idx:03d}" / cam_key / f"episode_{orig_ep_idx:06d}.mp4"
            if not video_path.exists():
                log.warning("Skipping episode %d: missing video %s", orig_ep_idx, video_path)
                cam_frames = None
                break
            frames = decode_video_frames(video_path)
            if len(frames) < n_rows:
                log.warning("Episode %d: video has %d frames but parquet has %d rows, truncating parquet",
                            orig_ep_idx, len(frames), n_rows)
                n_rows = len(frames)
            elif len(frames) > n_rows:
                frames = frames[:n_rows]
            cam_frames[cam_key] = frames

        if cam_frames is None:
            continue

        cols = table.to_pydict()

        new_rows = []
        for i in range(n_rows):
            row = {
                "observation.state": cols["observation.state"][i],
                "action": cols["action"][i],
                "timestamp": i / args.fps,
                "frame_index": i,
                "episode_index": new_ep_idx,
                "index": global_index,
                "task_index": 0,
            }
            for cam_key in args.camera_keys:
                row[cam_key] = {"bytes": cam_frames[cam_key][i], "path": None}
            new_rows.append(row)
            global_index += 1

        new_table = pa.Table.from_pylist(new_rows)
        chunk_idx = new_ep_idx // args.chunks_size
        out_chunk = output / "data" / f"chunk-{chunk_idx:03d}"
        out_chunk.mkdir(parents=True, exist_ok=True)
        pq.write_table(new_table, out_chunk / f"episode_{new_ep_idx:06d}.parquet")

        state_arr = np.array([r["observation.state"] for r in new_rows], dtype=np.float32)
        action_arr = np.array([r["action"] for r in new_rows], dtype=np.float32)

        def compute_stats(arr, length):
            return {
                "min": arr.min(axis=0).tolist(),
                "max": arr.max(axis=0).tolist(),
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0).tolist(),
                "count": [length],
            }

        episodes_meta.append({
            "episode_index": new_ep_idx,
            "length": n_rows,
            "task_index": 0,
            "task": args.task,
        })
        episodes_stats.append({
            "episode_index": new_ep_idx,
            "stats": {
                "observation.state": compute_stats(state_arr, n_rows),
                "action": compute_stats(action_arr, n_rows),
            }
        })

        new_ep_idx += 1

    total_episodes = new_ep_idx
    total_frames = global_index
    log.info("Converted %d episodes, %d total frames", total_episodes, total_frames)

    meta_dir = output / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [len(episodes_meta[0]["length"] if isinstance(episodes_meta[0]["length"], list) else
                         cols["observation.state"][0])],
            "names": [["left_x", "left_y", "right_x", "right_y"]],
        },
        "action": {
            "dtype": "float32",
            "shape": [len(cols["action"][0])],
            "names": [["left_x", "left_y", "right_x", "right_y"]],
        },
    }
    for cam_key in args.camera_keys:
        features[cam_key] = {
            "dtype": "image",
            "shape": [3, 480, 640],
            "names": ["channels", "height", "width"],
        }
    features.update({
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    })

    info = {
        "codebase_version": "v2.1",
        "robot_type": "excavator",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": args.chunks_size,
        "fps": args.fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": features,
    }

    state_dim = len(cols["observation.state"][0])
    info["features"]["observation.state"]["shape"] = [state_dim]
    info["features"]["action"]["shape"] = [state_dim]

    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    with open(meta_dir / "episodes_stats.jsonl", "w") as f:
        for es in episodes_stats:
            f.write(json.dumps(es) + "\n")

    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": args.task}) + "\n")

    log.info("Done! Output at: %s", output)
    log.info("To upload: huggingface-cli upload verm11/excavator_v2 %s . --repo-type dataset", output)


if __name__ == "__main__":
    main()
