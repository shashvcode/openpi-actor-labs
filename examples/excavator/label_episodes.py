"""Label excavator episodes with varied prompts.

Downloads all episodes first, then plays them in a single persistent OpenCV
window. Label with keyboard — no terminal input needed.

Keyboard controls:
    PLAYBACK:
        SPACE      — pause / resume
        LEFT/RIGHT — seek ±30 frames
        UP/DOWN    — seek ±5 frames (fine)
        R          — replay from start

    LABELING (press anytime during or after playback):
        1  — dump into pool on the left
        2  — dump into pool on the right
        3  — dump into the smallest pool
        4  — dump into the medium sized pool
        5  — dump into small pool (generic)
        S  — skip this episode
        Q  — quit and save

Usage:
    python examples/excavator/label_episodes.py --start 50
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO_ID = "verm11/excavator_v3"
CAMERA_COL = "observation.images.csi_0_imx219"
SAVE_FILE = os.path.join(os.path.dirname(__file__), "episode_labels.json")

PROMPTS = {
    "1": "Scoop packing peanuts from large pool and dump into pool on the left",
    "2": "Scoop packing peanuts from large pool and dump into pool on the right",
    "3": "Scoop packing peanuts from large pool and dump into the smallest pool",
    "4": "Scoop packing peanuts from large pool and dump into the medium sized pool",
    "5": "Scoop packing peanuts from large pool and dump into small pool",
}

KEY_MAP = {
    ord("1"): "1", ord("2"): "2", ord("3"): "3", ord("4"): "4", ord("5"): "5",
    ord("s"): "s", ord("S"): "s",
}


def load_labels() -> dict:
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE) as f:
            return json.load(f)
    return {}


def save_labels(labels: dict):
    with open(SAVE_FILE, "w") as f:
        json.dump(labels, f, indent=2)


def download_episodes(start: int, end: int, token: str) -> dict:
    """Download all parquet files and return {ep_index: local_path}."""
    paths = {}
    total = end - start
    for i, ep in enumerate(range(start, end)):
        chunk = ep // 1000
        repo_path = f"data/chunk-{chunk:03d}/episode_{ep:06d}.parquet"
        sys.stdout.write(f"\r  Downloading episode {ep} ({i+1}/{total})...")
        sys.stdout.flush()
        try:
            local = hf_hub_download(REPO_ID, repo_path, repo_type="dataset", token=token)
            paths[ep] = local
        except Exception as e:
            print(f"\n  FAILED ep {ep}: {e}")
    print(f"\r  Downloaded {len(paths)} episodes.                    ")
    return paths


def load_frames(parquet_path: str) -> list[np.ndarray]:
    """Load all frames from a parquet file into memory as BGR numpy arrays."""
    table = pq.read_table(parquet_path)
    col = table.column(CAMERA_COL)
    frames = []
    for i in range(len(col)):
        jpeg_bytes = col[i]["bytes"].as_py()
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        frames.append(bgr)
    return frames


def draw_overlay(frame: np.ndarray, ep_idx: int, frame_idx: int,
                 n_frames: int, paused: bool, label_mode: bool,
                 current_label: str | None, total_eps: int,
                 labeled_count: int) -> np.ndarray:
    """Draw HUD overlay on the frame."""
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    bar_h = 45
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, overlay, -1)
    # Re-draw black bar with alpha
    roi = overlay[0:bar_h, 0:w]
    black = np.zeros_like(roi)
    cv2.addWeighted(roi, 0.5, black, 0.5, 0, roi)

    info = f"Episode {ep_idx} | Frame {frame_idx}/{n_frames-1} | {labeled_count} labeled"
    cv2.putText(overlay, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    status = ""
    if paused:
        status = "PAUSED"
    if current_label:
        status = f"LABELED: [{current_label}]"
    if status:
        cv2.putText(overlay, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

    # Progress bar
    progress = frame_idx / max(n_frames - 1, 1)
    bar_y = bar_h - 3
    cv2.rectangle(overlay, (0, bar_y), (int(w * progress), bar_y + 3), (0, 255, 0), -1)

    if label_mode or paused:
        # Show label options at bottom
        labels_bg_y = h - 130
        cv2.rectangle(overlay, (0, labels_bg_y), (w, h), (0, 0, 0), -1)
        roi2 = overlay[labels_bg_y:h, 0:w]
        black2 = np.zeros_like(roi2)
        cv2.addWeighted(roi2, 0.4, black2, 0.6, 0, roi2)

        y = labels_bg_y + 22
        cv2.putText(overlay, "Press 1-5 to label, S=skip, R=replay, Q=quit:",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 22
        for k, v in PROMPTS.items():
            short = v.split("dump into ")[-1] if "dump into " in v else v
            cv2.putText(overlay, f"[{k}] {short}",
                        (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
            y += 20

    return overlay


def main():
    parser = argparse.ArgumentParser(description="Label excavator episodes")
    parser.add_argument("--start", type=int, default=0,
                        help="Episode index to start from")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default 1.0 = realtime at 10fps)")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        from dotenv import load_dotenv
        load_dotenv()
        token = os.environ.get("HF_TOKEN")

    print("Fetching dataset info...")
    info_path = hf_hub_download(REPO_ID, "meta/info.json", repo_type="dataset", token=token)
    with open(info_path) as f:
        info = json.load(f)
    total = info["total_episodes"]

    labels = load_labels()
    labeled_count = len([k for k, v in labels.items() if v != "s"])

    # Figure out which episodes need labeling
    to_label = []
    for ep in range(args.start, total):
        key = str(ep)
        if key not in labels or labels[key] == "s":
            to_label.append(ep)

    print(f"Dataset: {total} episodes | Already labeled: {labeled_count} | To label: {len(to_label)}")

    if not to_label:
        print("Nothing to label!")
        return

    # Download all needed episodes
    print(f"\nDownloading {len(to_label)} episodes...")
    ep_paths = download_episodes(to_label[0], to_label[-1] + 1, token)

    # Create persistent window
    win = "Excavator Episode Labeler"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 720)

    base_delay = int(100 / args.speed)  # ms per frame at 10fps
    quit_all = False

    for ep_idx in to_label:
        if quit_all:
            break

        key = str(ep_idx)
        if key in labels and labels[key] != "s":
            continue
        if ep_idx not in ep_paths:
            continue

        # Load all frames into memory
        frames = load_frames(ep_paths[ep_idx])
        n_frames = len(frames)

        frame_idx = 0
        paused = False
        chosen = None
        playback_done = False

        while True:
            label_mode = playback_done or paused
            overlay = draw_overlay(
                frames[frame_idx], ep_idx, frame_idx, n_frames,
                paused, label_mode, chosen, total, labeled_count,
            )
            cv2.imshow(win, overlay)

            wait_ms = 0 if paused or playback_done else base_delay
            raw_key = cv2.waitKey(max(wait_ms, 1)) & 0xFF

            # Label keys — work anytime
            if raw_key in KEY_MAP:
                choice = KEY_MAP[raw_key]
                if choice == "s":
                    labels[key] = "s"
                    save_labels(labels)
                    print(f"  Episode {ep_idx}: SKIPPED")
                    break
                else:
                    labels[key] = choice
                    labeled_count += 1
                    save_labels(labels)
                    short = PROMPTS[choice].split("dump into ")[-1]
                    print(f"  Episode {ep_idx}: [{choice}] {short}")
                    break

            elif raw_key == ord("q") or raw_key == ord("Q"):
                quit_all = True
                break

            elif raw_key == ord("r") or raw_key == ord("R"):
                frame_idx = 0
                paused = False
                playback_done = False

            elif raw_key == ord(" "):
                paused = not paused

            elif raw_key in (81, 2):  # LEFT
                frame_idx = max(0, frame_idx - 30)
            elif raw_key in (83, 3):  # RIGHT
                frame_idx = min(n_frames - 1, frame_idx + 30)
            elif raw_key in (82, 0):  # UP (fine seek back)
                frame_idx = max(0, frame_idx - 5)
            elif raw_key in (84, 1):  # DOWN (fine seek forward)
                frame_idx = min(n_frames - 1, frame_idx + 5)

            # Auto-advance playback
            elif not paused and not playback_done:
                frame_idx += 1
                if frame_idx >= n_frames:
                    frame_idx = n_frames - 1
                    playback_done = True

    cv2.destroyAllWindows()
    save_labels(labels)

    labeled_count = len([k for k, v in labels.items() if v != "s"])
    print(f"\nDone. {labeled_count}/{total} episodes labeled.")
    for pk, prompt in PROMPTS.items():
        count = sum(1 for v in labels.values() if v == pk)
        if count:
            short = prompt.split("dump into ")[-1]
            print(f"  [{pk}] {count:3d} eps — {short}")


if __name__ == "__main__":
    main()
