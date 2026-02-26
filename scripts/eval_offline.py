"""Offline evaluation of a trained SO-100 joystick policy.

Loads the trained checkpoint and runs inference on episodes from the dataset,
comparing predicted actions against ground truth. Reports MSE, per-dimension
errors, and cosine similarity.

Usage (on a GPU machine):
    uv run scripts/eval_offline.py --checkpoint-dir checkpoints/pi05_so100_lora/run1/4999 \
        --num-episodes 10 --stride 5
"""

import argparse
import io
import logging
import pathlib

import numpy as np
from PIL import Image

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ACTION_DIM_NAMES = ["left_x", "left_y", "right_x", "right_y", "l2_trigger", "r2_trigger"]


def decode_image(item) -> np.ndarray:
    """Decode an image from the LeRobot dataset format ({bytes, path} struct)."""
    if isinstance(item, dict) and "bytes" in item and item["bytes"] is not None:
        return np.array(Image.open(io.BytesIO(item["bytes"])).convert("RGB"))
    if isinstance(item, Image.Image):
        return np.array(item.convert("RGB"))
    return np.asarray(item)


def load_dataset(repo_id: str):
    """Load the HuggingFace dataset."""
    from datasets import load_dataset
    ds = load_dataset(repo_id, split="train")
    return ds


def get_episode_frames(dataset, episode_idx: int) -> list[dict]:
    """Extract all frames for a given episode, sorted by frame_index."""
    frames = [row for row in dataset if row["episode_index"] == episode_idx]
    frames.sort(key=lambda x: x["frame_index"])
    return frames


def build_observation(frame: dict, task_label: str) -> dict:
    """Build the observation dict expected by the SO-100 policy."""
    scene_img = decode_image(frame["observation.images.scene"])
    wrist_img = decode_image(frame["observation.images.wrist"])

    state = np.array(frame["observation.state"], dtype=np.float32)

    return {
        "observation/state": state,
        "observation/image_scene": scene_img,
        "observation/image_wrist": wrist_img,
        "prompt": task_label,
    }


def evaluate_episode(policy, frames: list[dict], task_label: str, stride: int = 1) -> dict:
    """Evaluate the policy on a single episode.

    At each sampled timestep, runs inference and compares the first predicted
    action to the ground truth action at that timestep.
    """
    all_pred = []
    all_gt = []

    eval_indices = range(0, max(1, len(frames) - 1), stride)

    for i in eval_indices:
        frame = frames[i]
        obs = build_observation(frame, task_label)
        result = policy.infer(obs)

        pred_actions = result["actions"]  # (action_horizon, action_dim)
        gt_action = np.array(frames[i]["action"], dtype=np.float32)

        # Compare first predicted action to ground truth at current timestep
        first_pred = pred_actions[0, :ACTION_DIM_NAMES.__len__()]
        all_pred.append(first_pred)
        all_gt.append(gt_action)

    all_pred = np.array(all_pred)
    all_gt = np.array(all_gt)

    per_dim_mse = np.mean((all_pred - all_gt) ** 2, axis=0)
    overall_mse = np.mean(per_dim_mse)
    overall_mae = np.mean(np.abs(all_pred - all_gt))

    # Cosine similarity
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
    parser = argparse.ArgumentParser(description="Offline evaluation of SO-100 policy")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Path to checkpoint dir (e.g. checkpoints/pi05_so100_lora/run1/4999)")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of episodes to evaluate on")
    parser.add_argument("--stride", type=int, default=5,
                        help="Evaluate every N-th frame (1=all frames, 5=every 5th)")
    parser.add_argument("--repo-id", type=str, default="verm11/so100_joystick_pickup",
                        help="HuggingFace dataset repo ID")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    logger.info("Loading training config...")
    train_config = _config.get_config("pi05_so100_lora")

    logger.info("Loading trained policy from %s...", args.checkpoint_dir)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir)

    logger.info("Loading dataset %s...", args.repo_id)
    dataset = load_dataset(args.repo_id)

    # Get unique episode indices
    episode_indices = sorted(set(row["episode_index"] for row in dataset))
    logger.info("Dataset has %d episodes", len(episode_indices))

    # Sample episodes for evaluation
    eval_episodes = rng.choice(episode_indices, size=min(args.num_episodes, len(episode_indices)), replace=False)
    eval_episodes = sorted(eval_episodes)
    logger.info("Evaluating on episodes: %s", eval_episodes.tolist())

    # Load task label
    task_label = "Pick up the bottle and place it on the yellow outlined square."

    all_results = []
    for ep_idx in eval_episodes:
        logger.info("Evaluating episode %d...", ep_idx)
        frames = get_episode_frames(dataset, ep_idx)
        if len(frames) < 2:
            logger.warning("Episode %d has < 2 frames, skipping", ep_idx)
            continue

        result = evaluate_episode(policy, frames, task_label, stride=args.stride)
        result["episode_index"] = int(ep_idx)
        result["episode_length"] = len(frames)
        all_results.append(result)

        logger.info(
            "  Episode %d: MSE=%.4f  MAE=%.4f  CosSim=%.4f  (%d frames)",
            ep_idx, result["mse"], result["mae"], result["cosine_sim"], result["num_frames_evaluated"],
        )

    # Aggregate results
    if all_results:
        avg_mse = np.mean([r["mse"] for r in all_results])
        avg_mae = np.mean([r["mae"] for r in all_results])
        avg_cos = np.mean([r["cosine_sim"] for r in all_results])
        avg_per_dim = np.mean([r["per_dim_mse"] for r in all_results], axis=0)

        print("\n" + "=" * 60)
        print("OFFLINE EVALUATION RESULTS")
        print("=" * 60)
        print(f"Episodes evaluated: {len(all_results)}")
        print(f"Total frames evaluated: {sum(r['num_frames_evaluated'] for r in all_results)}")
        print(f"\nOverall MSE:        {avg_mse:.4f}")
        print(f"Overall MAE:        {avg_mae:.4f}")
        print(f"Cosine Similarity:  {avg_cos:.4f}")
        print(f"\nPer-dimension MSE:")
        for name, mse in zip(ACTION_DIM_NAMES, avg_per_dim):
            print(f"  {name:>12s}: {mse:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
