"""Review pi_teleop episodes with synchronized joystick visualization.

This tool downloads episodes from a Hugging Face dataset one-by-one, plays the
cab and wrist camera streams side-by-side, and renders the stored joystick
action/state underneath so video and controls can be inspected together.

Default dataset:
    verm11/pi_teleopv2v3

Keyboard controls:
    SPACE      pause / resume
    LEFT/RIGHT seek -/+ 15 frames
    UP/DOWN    seek -/+ 1 frame
    N          next batch (jump by --step episodes)
    P          previous batch (jump by --step episodes)
    R          restart current episode
    S          toggle joystick source (action/state)
    +/=        increase USB cam offset by 1 frame (USB starts later)
    -          decrease USB cam offset by 1 frame
    0          reset USB cam offset to 0
    Q / ESC    quit

Usage:
    python examples/excavator/view_pi_teleop_dataset.py
    python examples/excavator/view_pi_teleop_dataset.py --start 100
    python examples/excavator/view_pi_teleop_dataset.py --step 5
    python examples/excavator/view_pi_teleop_dataset.py --repo-id verm11/excavator_v2 --step 5
    python examples/excavator/view_pi_teleop_dataset.py --save-preview src/pi_teleop_preview.jpg
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DEFAULT_REPO_ID = "verm11/pi_teleopv2v3"
CAB_COL = "observation.images.csi_0_imx219"
WRIST_COL = "observation.images.usb_0"

GREEN = (80, 255, 80)
WHITE = (245, 245, 245)
YELLOW = (80, 220, 255)
CYAN = (255, 220, 80)
GRAY = (160, 160, 160)
RED = (80, 80, 255)
BG = (18, 18, 18)
PANEL = (32, 32, 32)


@dataclass
class EpisodeData:
    frames_cab: list[np.ndarray]
    frames_wrist: list[np.ndarray]
    action: np.ndarray
    state: np.ndarray
    task: str | None
    length: int


def load_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    root = Path(__file__).resolve()
    for parent in [root.parent, *root.parents]:
        env_path = parent / ".env"
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return os.environ.get("HF_TOKEN")
    return os.environ.get("HF_TOKEN")


def build_episode_path(info: dict, episode_idx: int) -> str:
    template = info["data_path"]
    chunk = episode_idx // 1000
    values = {
        "episode_index": episode_idx,
        "file_index": episode_idx,
        "episode_chunk": chunk,
        "chunk_index": chunk,
    }
    return template.format(**values)


def read_jsonl_map(path: str, key: str) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            rows[int(row[key])] = row
    return rows


def decode_image_column(table: pq.Table, col_name: str) -> list[np.ndarray]:
    col = table.column(col_name)
    frames: list[np.ndarray] = []
    for i in range(len(col)):
        item = col[i]
        jpeg_bytes = item["bytes"].as_py()
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        frames.append(bgr)
    return frames


def load_episode(repo_id: str, info: dict, episode_meta: dict | None, episode_idx: int, token: str | None) -> EpisodeData:
    parquet_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=build_episode_path(info, episode_idx),
        token=token,
    )
    table = pq.read_table(parquet_path)
    frames_cab = decode_image_column(table, CAB_COL)
    frames_wrist = decode_image_column(table, WRIST_COL)
    action = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    task = episode_meta.get("task") if episode_meta else None
    return EpisodeData(
        frames_cab=frames_cab,
        frames_wrist=frames_wrist,
        action=action,
        state=state,
        task=task,
        length=len(frames_cab),
    )


def fit_image(img: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = img.shape[:2]
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = BG
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def draw_panel_label(img: np.ndarray, label: str) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(img, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 1, cv2.LINE_AA)


def draw_stick(canvas: np.ndarray, origin: tuple[int, int], size: int, x_val: float, y_val: float, title: str) -> None:
    x0, y0 = origin
    cx = x0 + size // 2
    cy = y0 + size // 2
    radius = size // 2 - 16

    cv2.rectangle(canvas, (x0, y0), (x0 + size, y0 + size), PANEL, -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + size, y0 + size), (70, 70, 70), 1)
    cv2.putText(canvas, title, (x0 + 12, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)

    cv2.circle(canvas, (cx, cy), radius, (95, 95, 95), 1, cv2.LINE_AA)
    cv2.line(canvas, (cx - radius, cy), (cx + radius, cy), (85, 85, 85), 1, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - radius), (cx, cy + radius), (85, 85, 85), 1, cv2.LINE_AA)

    knob_x = int(cx + np.clip(float(x_val), -1.0, 1.0) * radius)
    knob_y = int(cy - np.clip(float(y_val), -1.0, 1.0) * radius)
    cv2.circle(canvas, (knob_x, knob_y), 12, YELLOW, -1, cv2.LINE_AA)
    cv2.circle(canvas, (knob_x, knob_y), 12, (20, 20, 20), 1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"x={x_val:+.2f}  y={y_val:+.2f}",
        (x0 + 12, y0 + size - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        GREEN,
        1,
        cv2.LINE_AA,
    )


def render_frame(
    episode: EpisodeData,
    episode_idx: int,
    frame_idx: int,
    total_episodes: int,
    source_name: str,
    joystick_vec: np.ndarray,
    paused: bool,
    step: int = 1,
    usb_offset: int = 0,
) -> np.ndarray:
    width = 1500
    height = 1020
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = BG

    step_label = f"  (step {step})" if step > 1 else ""
    cv2.putText(canvas, f"Dataset Viewer | Episode {episode_idx}/{total_episodes - 1}{step_label}", (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Frame {frame_idx + 1}/{episode.length} | Source: {source_name} | {'PAUSED' if paused else 'PLAYING'}",
                (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, GREEN if not paused else YELLOW, 1, cv2.LINE_AA)

    if episode.task:
        task = episode.task
        if len(task) > 90:
            task = task[:87] + "..."
        cv2.putText(canvas, f"Task: {task}", (18, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 1, cv2.LINE_AA)

    usb_frame_idx = min(max(frame_idx + usb_offset, 0), episode.length - 1)

    cam_h = 460
    cam_w = 720
    cab = fit_image(episode.frames_cab[frame_idx], cam_w, cam_h)
    wrist = fit_image(episode.frames_wrist[usb_frame_idx], cam_w, cam_h)
    draw_panel_label(cab, f"Cab (csi_0)  frame {frame_idx}")
    offset_color = RED if usb_offset != 0 else WHITE
    offset_str = f"  [offset {usb_offset:+d}]" if usb_offset != 0 else ""
    draw_panel_label(wrist, f"USB (usb_0)  frame {usb_frame_idx}{offset_str}")
    if usb_offset != 0:
        cv2.rectangle(wrist, (0, 0), (wrist.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(wrist, f"USB (usb_0)  frame {usb_frame_idx}  [offset {usb_offset:+d}]",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, offset_color, 1, cv2.LINE_AA)
    canvas[110:110 + cam_h, 20:20 + cam_w] = cab
    canvas[110:110 + cam_h, 760:760 + cam_w] = wrist

    bottom_y = 600
    cv2.rectangle(canvas, (20, bottom_y), (1480, 990), PANEL, -1)
    cv2.rectangle(canvas, (20, bottom_y), (1480, 990), (70, 70, 70), 1)
    cv2.putText(canvas, "Joystick order: [LX, LY, RX, RY]", (40, bottom_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 1, cv2.LINE_AA)

    left_x = float(joystick_vec[0])
    left_y = float(joystick_vec[1])
    right_x = float(joystick_vec[2])
    right_y = float(joystick_vec[3])

    draw_stick(canvas, (120, bottom_y + 55), 250, left_x, left_y, "Left Stick (channels 0,1)")
    draw_stick(canvas, (430, bottom_y + 55), 250, right_x, right_y, "Right Stick (channels 2,3)")

    text_x = 760
    cv2.putText(canvas, f"{source_name}: [{left_x:+.3f}, {left_y:+.3f}, {right_x:+.3f}, {right_y:+.3f}]",
                (text_x, bottom_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 1, cv2.LINE_AA)
    state = episode.state[frame_idx]
    action = episode.action[frame_idx]
    cv2.putText(canvas, f"Action: [{action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f}, {action[3]:+.3f}]",
                (text_x, bottom_y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.62, WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"State:  [{state[0]:+.3f}, {state[1]:+.3f}, {state[2]:+.3f}, {state[3]:+.3f}]",
                (text_x, bottom_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.62, WHITE, 1, cv2.LINE_AA)

    help_lines = [
        "Controls:",
        "SPACE pause/resume",
        "LEFT/RIGHT seek +/- 15",
        "UP/DOWN seek +/- 1",
        f"N next (+{step}), P prev (-{step})",
        "R restart, S toggle src",
        "+/- USB offset, 0 reset",
        "Q or ESC quit",
    ]
    if usb_offset != 0:
        help_lines.insert(1, f"USB OFFSET: {usb_offset:+d} frames")
    y = bottom_y + 205
    for line in help_lines:
        cv2.putText(canvas, line, (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, CYAN if line == "Controls:" else GRAY, 1, cv2.LINE_AA)
        y += 28

    progress_x0 = 40
    progress_x1 = 1460
    progress_y = 970
    cv2.rectangle(canvas, (progress_x0, progress_y - 8), (progress_x1, progress_y + 8), (55, 55, 55), -1)
    filled = int((frame_idx / max(episode.length - 1, 1)) * (progress_x1 - progress_x0))
    cv2.rectangle(canvas, (progress_x0, progress_y - 8), (progress_x0 + filled, progress_y + 8), GREEN, -1)

    return canvas


def save_preview(args: argparse.Namespace, info: dict, episodes: dict[int, dict], token: str | None) -> None:
    episode = load_episode(args.repo_id, info, episodes.get(args.start), args.start, token)
    source_name = "action"
    frame = render_frame(
        episode=episode,
        episode_idx=args.start,
        frame_idx=0,
        total_episodes=int(info["total_episodes"]),
        source_name=source_name,
        joystick_vec=episode.action[0],
        paused=True,
        step=1,
    )
    out_path = Path(args.save_preview)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    print(f"Saved preview to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="View pi teleop dataset with joystick overlay")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo")
    parser.add_argument("--start", type=int, default=0, help="Episode index to start from")
    parser.add_argument("--step", type=int, default=1, help="Episode step size for N/P navigation (e.g. 5 to jump 0,5,10,...)")
    parser.add_argument("--usb-offset", type=int, default=0, help="Initial USB camera frame offset (positive = USB starts later)")
    parser.add_argument("--trim-ms", type=int, default=0, help="Trim N ms from USB start and CSI/joystick end to test sync alignment (visualization only)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--save-preview", default=None, help="Render first frame of the start episode to an image and exit")
    args = parser.parse_args()

    token = load_hf_token()

    info_path = hf_hub_download(args.repo_id, "meta/info.json", repo_type="dataset", token=token)
    episodes_path = hf_hub_download(args.repo_id, "meta/episodes.jsonl", repo_type="dataset", token=token)
    with open(info_path) as f:
        info = json.load(f)
    episodes = read_jsonl_map(episodes_path, "episode_index")
    total_episodes = int(info["total_episodes"])

    if args.save_preview:
        save_preview(args, info, episodes, token)
        return

    step = max(1, args.step)

    win = "pi_teleop Dataset Viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1500, 1020)

    episode_idx = max(0, min(args.start, total_episodes - 1))
    source_name = "action"
    usb_offset = args.usb_offset
    quit_all = False

    trim_frames = int(args.trim_ms / (1000.0 / info.get("fps", 10))) if args.trim_ms > 0 else 0

    while not quit_all:
        episode = load_episode(args.repo_id, info, episodes.get(episode_idx), episode_idx, token)
        if trim_frames > 0 and trim_frames < episode.length:
            episode = EpisodeData(
                frames_cab=episode.frames_cab[:episode.length - trim_frames],
                frames_wrist=episode.frames_wrist[trim_frames:],
                action=episode.action[:episode.length - trim_frames],
                state=episode.state[:episode.length - trim_frames],
                task=episode.task,
                length=episode.length - trim_frames,
            )
        frame_idx = 0
        paused = False
        next_episode_delta = 0
        base_delay = max(1, int(100 / max(args.speed, 1e-6)))

        while True:
            joystick_vec = episode.action[frame_idx] if source_name == "action" else episode.state[frame_idx]
            canvas = render_frame(
                episode=episode,
                episode_idx=episode_idx,
                frame_idx=frame_idx,
                total_episodes=total_episodes,
                source_name=source_name,
                joystick_vec=joystick_vec,
                paused=paused,
                step=step,
                usb_offset=usb_offset,
            )
            cv2.imshow(win, canvas)

            wait_ms = 0 if paused else base_delay
            key = cv2.waitKey(max(wait_ms, 1)) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                quit_all = True
                break
            if key == ord(" "):
                paused = not paused
                continue
            if key in (81, 2):  # left
                frame_idx = max(0, frame_idx - 15)
                paused = True
                continue
            if key in (83, 3):  # right
                frame_idx = min(episode.length - 1, frame_idx + 15)
                paused = True
                continue
            if key in (82, 0):  # up
                frame_idx = max(0, frame_idx - 1)
                paused = True
                continue
            if key in (84, 1):  # down
                frame_idx = min(episode.length - 1, frame_idx + 1)
                paused = True
                continue
            if key in (ord("r"), ord("R")):
                frame_idx = 0
                paused = False
                continue
            if key in (ord("s"), ord("S")):
                source_name = "state" if source_name == "action" else "action"
                continue
            if key in (ord("+"), ord("=")):
                usb_offset += 1
                continue
            if key == ord("-"):
                usb_offset -= 1
                continue
            if key == ord("0"):
                usb_offset = 0
                continue
            if key in (ord("n"), ord("N")):
                next_episode_delta = step
                break
            if key in (ord("p"), ord("P")):
                next_episode_delta = -step
                break

            if not paused:
                frame_idx += 1
                if frame_idx >= episode.length:
                    frame_idx = episode.length - 1
                    paused = True

        if quit_all:
            break
        episode_idx = int(np.clip(episode_idx + next_episode_delta, 0, total_episodes - 1))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
