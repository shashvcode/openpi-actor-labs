"""Offline evaluation of a trained SO-100 joystick policy.

Loads the trained checkpoint and runs inference on episodes, comparing
predicted actions against ground truth. Reports MSE, MAE, cosine similarity,
and per-dimension errors.

Supports two modes:
  --source hf     : Evaluate on episodes from the HuggingFace training dataset (sanity check)
  --source s3     : Evaluate on held-out episodes from S3 that were NOT used for training (generalization)

Usage (on a GPU machine):
    # Sanity check on training data:
    uv run scripts/eval_offline.py --checkpoint-dir checkpoints/pi05_so100_lora/run1/4999 \
        --source hf --num-episodes 10 --stride 5

    # Generalization test on held-out S3 data:
    uv run scripts/eval_offline.py --checkpoint-dir checkpoints/pi05_so100_lora/run1/4999 \
        --source s3 --num-episodes 10 --stride 5
"""

import argparse
import io
import logging
import os
import pathlib

import numpy as np
from PIL import Image

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ACTION_DIM_NAMES = ["left_x", "left_y", "right_x", "right_y", "l2_trigger", "r2_trigger"]

# The nojoint_parquet dataset has 134 episodes; we used 35 for training (indices 115-149).
# The remaining 99 are held out for testing.
TRAIN_EPISODE_INDICES_FROM_NOJOINT = set(range(35))  # first 35 sampled during data prep


def decode_image(item) -> np.ndarray:
    if isinstance(item, dict) and "bytes" in item and item["bytes"] is not None:
        return np.array(Image.open(io.BytesIO(item["bytes"])).convert("RGB"))
    if isinstance(item, Image.Image):
        return np.array(item.convert("RGB"))
    return np.asarray(item)


def load_hf_episodes(repo_id: str, num_episodes: int, seed: int):
    """Load random episodes from the HuggingFace dataset."""
    from datasets import load_dataset

    logger.info("Loading HuggingFace dataset %s...", repo_id)
    ds = load_dataset(repo_id, split="train")

    episode_indices = sorted(set(row["episode_index"] for row in ds))
    rng = np.random.RandomState(seed)
    chosen = sorted(rng.choice(episode_indices, size=min(num_episodes, len(episode_indices)), replace=False))
    logger.info("Chose %d episodes from HF dataset: %s", len(chosen), list(chosen))

    episodes = {}
    for ep_idx in chosen:
        frames = sorted(
            [row for row in ds if row["episode_index"] == ep_idx],
            key=lambda x: x["frame_index"],
        )
        episodes[ep_idx] = {
            "frames": frames,
            "task": "Pick up the bottle and place it on the yellow outlined square.",
            "image_scene_key": "observation.images.scene",
            "image_wrist_key": "observation.images.wrist",
            "state_key": "observation.state",
            "action_key": "action",
        }
    return episodes


def load_s3_episodes(num_episodes: int, seed: int):
    """Load held-out episodes from S3 nojoint_parquet (not used in training)."""
    import boto3
    import pyarrow.parquet as pq

    env_path = pathlib.Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("AWS_REGION", "us-west-1"),
    )

    bucket = "lerobot-trajectories"
    key = "non-joint/nojoint_parquet/data/chunk-000/file-000.parquet"

    logger.info("Downloading held-out parquet from S3 (%s/%s)...", bucket, key)
    tmp_path = pathlib.Path("/tmp/nojoint_test.parquet")
    if not tmp_path.exists():
        s3.download_file(bucket, key, str(tmp_path))
        logger.info("Downloaded to %s", tmp_path)
    else:
        logger.info("Using cached %s", tmp_path)

    logger.info("Reading parquet file...")
    pf = pq.ParquetFile(str(tmp_path))
    all_rows = []
    for batch in pf.iter_batches(batch_size=500):
        table = pa.Table.from_batches([batch])
        for i in range(len(table)):
            row = {col: table.column(col)[i].as_py() for col in table.column_names}
            all_rows.append(row)
    logger.info("Read %d total rows from nojoint_parquet", len(all_rows))

    # Group by episode
    from collections import defaultdict
    ep_map = defaultdict(list)
    for row in all_rows:
        ep_map[row["episode_index"]].append(row)

    # Filter out episodes used for training
    held_out = {k: v for k, v in ep_map.items() if k not in TRAIN_EPISODE_INDICES_FROM_NOJOINT}
    logger.info("Found %d held-out episodes (excluded %d used in training)",
                len(held_out), len(ep_map) - len(held_out))

    # Sample
    rng = np.random.RandomState(seed)
    ep_keys = sorted(held_out.keys())
    chosen = sorted(rng.choice(ep_keys, size=min(num_episodes, len(ep_keys)), replace=False))
    logger.info("Chose %d episodes for evaluation: %s", len(chosen), chosen)

    episodes = {}
    for ep_idx in chosen:
        frames = sorted(held_out[ep_idx], key=lambda x: x["frame_index"])
        episodes[ep_idx] = {
            "frames": frames,
            "task": "Pick up the bottle and place it on the yellow outlined square.",
            "image_scene_key": "observation.images.scene",
            "image_wrist_key": "observation.images.wrist",
            "state_key": "observation.state",
            "action_key": "action",
        }
    return episodes


def build_observation(frame: dict, meta: dict) -> dict:
    scene_img = decode_image(frame[meta["image_scene_key"]])
    wrist_img = decode_image(frame[meta["image_wrist_key"]])
    state = np.array(frame[meta["state_key"]], dtype=np.float32)[:6]

    return {
        "observation/state": state,
        "observation/image_scene": scene_img,
        "observation/image_wrist": wrist_img,
        "prompt": meta["task"],
    }


def evaluate_episode(policy, frames: list[dict], meta: dict, stride: int = 1) -> dict:
    all_pred = []
    all_gt = []

    eval_indices = range(0, max(1, len(frames) - 1), stride)

    for i in eval_indices:
        obs = build_observation(frames[i], meta)
        result = policy.infer(obs)

        pred_actions = result["actions"]
        gt_action = np.array(frames[i][meta["action_key"]], dtype=np.float32)[:6]

        all_pred.append(pred_actions[0, :6])
        all_gt.append(gt_action)

    all_pred = np.array(all_pred)
    all_gt = np.array(all_gt)

    per_dim_mse = np.mean((all_pred - all_gt) ** 2, axis=0)
    overall_mse = np.mean(per_dim_mse)
    overall_mae = np.mean(np.abs(all_pred - all_gt))

    dots = np.sum(all_pred * all_gt, axis=1)
    pred_norms = np.linalg.norm(all_pred, axis=1) + 1e-8
    gt_norms = np.linalg.norm(all_gt, axis=1) + 1e-8
    cosine_sim = np.mean(dots / (pred_norms * gt_norms))

    return {
        "mse": float(overall_mse),
        "mae": float(overall_mae),
        "cosine_sim": float(cosine_sim),
        "per_dim_mse": per_dim_mse.tolist(),
        "num_frames_evaluated": len(all_pred),
    }


def main():
    import pyarrow as pa  # noqa: F811 (needed at module level for s3 path)

    parser = argparse.ArgumentParser(description="Offline evaluation of SO-100 policy")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--source", choices=["hf", "s3"], default="hf",
                        help="'hf' = training data (sanity check), 's3' = held-out data (generalization)")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--repo-id", type=str, default="verm11/so100_joystick_pickup")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("Loading trained policy from %s...", args.checkpoint_dir)
    train_config = _config.get_config("pi05_so100_lora")
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir)

    if args.source == "hf":
        logger.info("=== SANITY CHECK: evaluating on TRAINING data ===")
        episodes = load_hf_episodes(args.repo_id, args.num_episodes, args.seed)
    else:
        logger.info("=== GENERALIZATION TEST: evaluating on HELD-OUT S3 data ===")
        episodes = load_s3_episodes(args.num_episodes, args.seed)

    all_results = []
    for ep_idx, ep_data in episodes.items():
        frames = ep_data["frames"]
        if len(frames) < 2:
            logger.warning("Episode %d has < 2 frames, skipping", ep_idx)
            continue

        logger.info("Evaluating episode %d (%d frames)...", ep_idx, len(frames))
        result = evaluate_episode(policy, frames, ep_data, stride=args.stride)
        result["episode_index"] = int(ep_idx)
        result["episode_length"] = len(frames)
        all_results.append(result)

        logger.info("  MSE=%.4f  MAE=%.4f  CosSim=%.4f  (%d frames)",
                     result["mse"], result["mae"], result["cosine_sim"], result["num_frames_evaluated"])

    if all_results:
        avg_mse = np.mean([r["mse"] for r in all_results])
        avg_mae = np.mean([r["mae"] for r in all_results])
        avg_cos = np.mean([r["cosine_sim"] for r in all_results])
        avg_per_dim = np.mean([r["per_dim_mse"] for r in all_results], axis=0)

        source_label = "TRAINING DATA (sanity check)" if args.source == "hf" else "HELD-OUT DATA (generalization)"
        print("\n" + "=" * 60)
        print(f"OFFLINE EVALUATION — {source_label}")
        print("=" * 60)
        print(f"Episodes evaluated: {len(all_results)}")
        print(f"Total frames:       {sum(r['num_frames_evaluated'] for r in all_results)}")
        print(f"\nOverall MSE:        {avg_mse:.4f}")
        print(f"Overall MAE:        {avg_mae:.4f}")
        print(f"Cosine Similarity:  {avg_cos:.4f}")
        print(f"\nPer-dimension MSE:")
        for name, mse in zip(ACTION_DIM_NAMES, avg_per_dim):
            print(f"  {name:>12s}: {mse:.4f}")
        print("=" * 60)
    else:
        print("No episodes were evaluated.")


if __name__ == "__main__":
    main()
