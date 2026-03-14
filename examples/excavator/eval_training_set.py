"""Evaluate the trained excavator model on its own training set.

Loads episodes from the HuggingFace dataset, feeds observations to the
policy server (localhost), and compares predicted actions to ground truth.

Run on the same RunPod where the model is being served.

Usage:
    # Serve the model first (in another terminal):
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config pi05_excavator_v3 \
        --policy.dir checkpoints/pi05_excavator_v3/excavator_v3_run1/14999

    # Then run eval:
    uv run python examples/excavator/eval_training_set.py --num-episodes 30
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from openpi_client import websocket_client_policy as wcp

REPO_ID = "verm11/excavator_v3"
CAB_COL = "observation.images.csi_0_imx219"
SIDE_COL = "observation.images.usb_0"
ACTION_DIM = 4


def decode_image(table, col: str, idx: int) -> np.ndarray:
    import cv2
    jpeg_bytes = table.column(col)[idx]["bytes"].as_py()
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.uint8)


def eval_episode(policy, table, episode_index: int, prompt: str) -> dict:
    n_frames = len(table)
    action_col = table.column("action")

    all_pred = []
    all_gt = []
    infer_times = []

    state = np.zeros(ACTION_DIM, dtype=np.float32)

    step = 0
    while step < n_frames:
        cab_img = decode_image(table, CAB_COL, step)
        side_img = decode_image(table, SIDE_COL, step)

        gt_action = np.array(action_col[step].as_py()[:ACTION_DIM], dtype=np.float32)

        obs = {
            "observation/state": state.copy(),
            "observation/image_cab": cab_img,
            "observation/image_side": side_img,
            "prompt": prompt,
        }

        t0 = time.perf_counter()
        result = policy.infer(obs)
        infer_ms = (time.perf_counter() - t0) * 1000
        infer_times.append(infer_ms)

        actions = result["actions"]
        pred_action = actions[0][:ACTION_DIM]

        all_pred.append(pred_action)
        all_gt.append(gt_action)

        state = gt_action.copy()

        chunk_size = len(actions)
        step += chunk_size

    all_pred = np.array(all_pred)
    all_gt = np.array(all_gt)

    mse_per_axis = np.mean((all_pred - all_gt) ** 2, axis=0)
    mae_per_axis = np.mean(np.abs(all_pred - all_gt), axis=0)
    mse_total = np.mean(mse_per_axis)
    mae_total = np.mean(mae_per_axis)

    return {
        "episode": episode_index,
        "frames": n_frames,
        "infer_steps": len(all_pred),
        "mse_total": float(mse_total),
        "mae_total": float(mae_total),
        "mse_lx": float(mse_per_axis[0]),
        "mse_ly": float(mse_per_axis[1]),
        "mse_rx": float(mse_per_axis[2]),
        "mse_ry": float(mse_per_axis[3]),
        "mae_lx": float(mae_per_axis[0]),
        "mae_ly": float(mae_per_axis[1]),
        "mae_rx": float(mae_per_axis[2]),
        "mae_ry": float(mae_per_axis[3]),
        "avg_infer_ms": float(np.mean(infer_times)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate excavator model on training set")
    parser.add_argument("--host", default="localhost", help="Policy server host")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--start", type=int, default=0, help="First episode index")
    parser.add_argument("--spread", action="store_true",
                        help="Spread episodes evenly across dataset instead of sequential")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")

    # Download metadata
    print("Fetching dataset metadata...")
    info_path = hf_hub_download(REPO_ID, "meta/info.json", repo_type="dataset", token=token)
    episodes_path = hf_hub_download(REPO_ID, "meta/episodes.jsonl", repo_type="dataset", token=token)
    tasks_path = hf_hub_download(REPO_ID, "meta/tasks.jsonl", repo_type="dataset", token=token)

    with open(info_path) as f:
        info = json.load(f)
    with open(episodes_path) as f:
        episodes_meta = [json.loads(l) for l in f]
    with open(tasks_path) as f:
        tasks = {t["task_index"]: t["task"] for t in (json.loads(l) for l in f)}

    total = info["total_episodes"]
    print(f"Dataset: {total} episodes, {info['total_tasks']} tasks\n")

    # Pick episodes
    if args.spread:
        indices = np.linspace(0, total - 1, args.num_episodes, dtype=int).tolist()
    else:
        indices = list(range(args.start, min(args.start + args.num_episodes, total)))

    print(f"Evaluating {len(indices)} episodes: {indices[0]}..{indices[-1]}")

    # Connect to policy server
    print(f"Connecting to policy server at {args.host}:{args.port}...")
    policy = wcp.WebsocketClientPolicy(host=args.host, port=args.port)
    print("Connected.\n")

    # Download episodes
    print("Downloading episodes...")
    ep_tables = {}
    for i, ep_idx in enumerate(indices):
        chunk = ep_idx // 1000
        path = f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
        sys.stdout.write(f"\r  {i+1}/{len(indices)}...")
        sys.stdout.flush()
        local = hf_hub_download(REPO_ID, path, repo_type="dataset", token=token)
        ep_tables[ep_idx] = pq.read_table(local)
    print(f"\r  Downloaded {len(ep_tables)} episodes.      \n")

    # Run eval
    results = []
    for i, ep_idx in enumerate(indices):
        table = ep_tables[ep_idx]
        meta = episodes_meta[ep_idx]
        task_str = tasks.get(meta["task_index"], "unknown")

        # Inject prompt into eval
        sys.stdout.write(f"  Episode {ep_idx:3d} ({meta['length']:3d} frames) [{task_str[:50]}]...")
        sys.stdout.flush()

        r = eval_episode(policy, table, ep_idx, task_str)
        r["task"] = task_str
        results.append(r)

        print(f" MSE={r['mse_total']:.4f}  MAE={r['mae_total']:.4f}  ({r['avg_infer_ms']:.0f}ms/step)")

    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    avg_mse = np.mean([r["mse_total"] for r in results])
    avg_mae = np.mean([r["mae_total"] for r in results])
    avg_infer = np.mean([r["avg_infer_ms"] for r in results])

    print(f"Episodes evaluated: {len(results)}")
    print(f"Avg MSE (total):    {avg_mse:.4f}")
    print(f"Avg MAE (total):    {avg_mae:.4f}")
    print(f"Avg inference:      {avg_infer:.0f} ms/step")

    print(f"\nPer-axis MSE:")
    for axis, name in enumerate(["lx", "ly", "rx", "ry"]):
        avg = np.mean([r[f"mse_{name}"] for r in results])
        print(f"  {name}: {avg:.4f}")

    print(f"\nPer-axis MAE:")
    for axis, name in enumerate(["lx", "ly", "rx", "ry"]):
        avg = np.mean([r[f"mae_{name}"] for r in results])
        print(f"  {name}: {avg:.4f}")

    # Per-task breakdown
    task_results = {}
    for r in results:
        t = r["task"]
        if t not in task_results:
            task_results[t] = []
        task_results[t].append(r)

    print(f"\nPer-task MSE:")
    for t, rs in sorted(task_results.items()):
        avg = np.mean([r["mse_total"] for r in rs])
        print(f"  ({len(rs):2d} eps) MSE={avg:.4f} — {t[:60]}")

    # Best and worst
    results.sort(key=lambda r: r["mse_total"])
    print(f"\nBest 5 episodes (lowest MSE):")
    for r in results[:5]:
        print(f"  ep {r['episode']:3d}: MSE={r['mse_total']:.4f}  [{r['task'][:40]}]")

    print(f"\nWorst 5 episodes (highest MSE):")
    for r in results[-5:]:
        print(f"  ep {r['episode']:3d}: MSE={r['mse_total']:.4f}  [{r['task'][:40]}]")


if __name__ == "__main__":
    main()
