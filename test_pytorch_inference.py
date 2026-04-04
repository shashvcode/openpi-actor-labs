#!/usr/bin/env python3
"""Test PyTorch inference of the trained PI0.5 SO-100 encoder model.

Loads the model from checkpoint, runs inference with synthetic and realistic
inputs, and validates the output is sane before proceeding to ONNX export.

Usage:
    PYTHONPATH=src python3 test_pytorch_inference.py \
        --checkpoint ~/encoder_checkpoint/5000
"""

import argparse
import dataclasses
import json
import logging
import pathlib
import sys
import time

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class InferenceConfig:
    pi05: bool = True
    action_dim: int = 6
    action_horizon: int = 10
    num_denoising_steps: int = 7
    max_token_len: int = 200
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "bfloat16"
    pytorch_compile_mode: str = None
    discrete_state_input: bool = True

    @classmethod
    def from_metadata(cls, metadata_path, **overrides):
        meta = torch.load(metadata_path, weights_only=False, map_location="cpu")
        model_cfg = meta.get("config", {}).get("model", {})
        kwargs = {}
        field_names = {f.name for f in dataclasses.fields(cls)}
        for k, v in model_cfg.items():
            if k in field_names:
                kwargs[k] = v
        for key in ("paligemma_variant", "action_expert_variant"):
            if key in kwargs and isinstance(kwargs[key], str):
                kwargs[key] = kwargs[key].replace("_lora", "")
        kwargs["pytorch_compile_mode"] = None
        kwargs.update(overrides)
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Observation stub
# ---------------------------------------------------------------------------

class MockObservation:
    """Minimal observation object matching what PI0Pytorch.sample_actions expects."""
    def __init__(self, images, image_masks, state, tokenized_prompt,
                 tokenized_prompt_mask, token_ar_mask, token_loss_mask):
        self.images = images
        self.image_masks = image_masks
        self.state = state
        self.tokenized_prompt = tokenized_prompt
        self.tokenized_prompt_mask = tokenized_prompt_mask
        self.token_ar_mask = token_ar_mask
        self.token_loss_mask = token_loss_mask


def make_test_observation(config, device, state_values=None):
    """Build a test observation with dummy images and optional real state."""
    B = 1
    if state_values is not None:
        state = torch.tensor(state_values, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        state = torch.zeros(B, config.action_dim, dtype=torch.float32, device=device)

    images = {
        "base_0_rgb": torch.randn(B, 3, 224, 224, device=device),
        "left_wrist_0_rgb": torch.randn(B, 3, 224, 224, device=device),
        "right_wrist_0_rgb": torch.randn(B, 3, 224, 224, device=device),
    }
    image_masks = {
        "base_0_rgb": torch.ones(B, dtype=torch.bool, device=device),
        "left_wrist_0_rgb": torch.ones(B, dtype=torch.bool, device=device),
        "right_wrist_0_rgb": torch.zeros(B, dtype=torch.bool, device=device),
    }
    T = config.max_token_len
    tokenized_prompt = torch.zeros(B, T, dtype=torch.long, device=device)
    tokenized_prompt_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    token_ar_mask = torch.zeros(B, T, dtype=torch.int32, device=device)
    token_loss_mask = torch.zeros(B, T, dtype=torch.bool, device=device)

    return MockObservation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        token_ar_mask=token_ar_mask,
        token_loss_mask=token_loss_mask,
    )


# ---------------------------------------------------------------------------
# transformers_replace setup
# ---------------------------------------------------------------------------

def ensure_transformers_replace(repo_dir):
    """Copy transformers_replace files if not already installed."""
    import transformers
    import shutil

    tpath = pathlib.Path(transformers.__file__).parent
    replace_src = repo_dir / "src" / "openpi" / "models_pytorch" / "transformers_replace"

    if not replace_src.exists():
        log.warning("transformers_replace source not found at %s", replace_src)
        return False

    try:
        from transformers.models.siglip import check
        if check.check_whether_transformers_replace_is_installed_correctly():
            log.info("transformers_replace already installed correctly")
            return True
    except (ImportError, AttributeError):
        pass

    dest = tpath / "models" / "gemma"
    log.info("Copying transformers_replace files to %s ...", dest)
    for src_file in replace_src.glob("*.py"):
        shutil.copy2(src_file, dest / src_file.name)

    dest_siglip = tpath / "models" / "siglip"
    for src_file in replace_src.glob("*.py"):
        if "check" in src_file.name:
            shutil.copy2(src_file, dest_siglip / src_file.name)

    try:
        from importlib import reload
        from transformers.models.siglip import check as check_mod
        reload(check_mod)
        if check_mod.check_whether_transformers_replace_is_installed_correctly():
            log.info("transformers_replace installed successfully")
            return True
    except Exception as e:
        log.warning("transformers_replace check after copy: %s", e)

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint dir")
    parser.add_argument("--repo-dir", default=None,
                        help="Path to openpi-actor-labs repo (for transformers_replace)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-runs", type=int, default=5, help="Number of inference runs for timing")
    args = parser.parse_args()

    ckpt_dir = pathlib.Path(args.checkpoint)
    device = args.device

    # Find repo dir for transformers_replace
    if args.repo_dir:
        repo_dir = pathlib.Path(args.repo_dir)
    else:
        repo_dir = pathlib.Path.home() / "openpi-actor-labs"
    ensure_transformers_replace(repo_dir)

    # Load config from metadata
    log.info("Loading config from metadata.pt ...")
    config = InferenceConfig.from_metadata(ckpt_dir / "metadata.pt")
    log.info("Config: action_dim=%d, action_horizon=%d, denoise_steps=%d, pi05=%s",
             config.action_dim, config.action_horizon, config.num_denoising_steps, config.pi05)

    # Load model
    log.info("Creating PI0Pytorch model ...")
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    t0 = time.time()
    model = PI0Pytorch(config)
    log.info("Model created in %.1fs", time.time() - t0)

    log.info("Loading weights from %s ...", ckpt_dir / "model.safetensors")
    import safetensors.torch
    t0 = time.time()
    safetensors.torch.load_model(model, str(ckpt_dir / "model.safetensors"), device=device)
    log.info("Weights loaded in %.1fs", time.time() - t0)

    model.eval()
    model.to(device)
    log.info("Model on %s, dtype of first param: %s",
             device, next(model.parameters()).dtype)

    # Load norm stats for reference
    norm_stats_path = ckpt_dir / "assets" / "assets" / "verm11" / "so-100-encoders" / "norm_stats.json"
    norm_stats = None
    if norm_stats_path.exists():
        with open(norm_stats_path) as f:
            norm_stats = json.load(f).get("norm_stats", {})
        log.info("Loaded norm stats from %s", norm_stats_path)
        state_mean = norm_stats["state"]["mean"]
        state_q01 = norm_stats["state"]["q01"]
        state_q99 = norm_stats["state"]["q99"]
        log.info("  State mean:  %s", [f"{x:.0f}" for x in state_mean])
        log.info("  State q01:   %s", [f"{x:.0f}" for x in state_q01])
        log.info("  State q99:   %s", [f"{x:.0f}" for x in state_q99])

    # Test 1: Zero state, dummy images (smoke test)
    log.info("\n=== Test 1: Smoke test (zero state, random images) ===")
    obs = make_test_observation(config, device)
    with torch.no_grad():
        t0 = time.time()
        actions = model.sample_actions(device, obs, num_steps=config.num_denoising_steps)
        elapsed = time.time() - t0
    actions_np = actions.cpu().float().numpy()
    log.info("  Output shape: %s", actions_np.shape)
    log.info("  Actions[0]:\n%s", actions_np[0])
    log.info("  Range: [%.4f, %.4f]", actions_np.min(), actions_np.max())
    log.info("  Mean: %.4f  Std: %.4f", actions_np.mean(), actions_np.std())
    log.info("  Time: %.2fs", elapsed)
    log.info("  All finite: %s", np.all(np.isfinite(actions_np)))

    # Test 2: Normalized midpoint state (if norm stats available)
    if norm_stats:
        log.info("\n=== Test 2: Normalized midpoint state ===")
        q01 = np.array(state_q01)
        q99 = np.array(state_q99)
        midpoint = (q01 + q99) / 2.0
        normalized = (midpoint - q01) / (q99 - q01 + 1e-8) * 2.0 - 1.0
        log.info("  Raw midpoint state: %s", [f"{x:.0f}" for x in midpoint])
        log.info("  Normalized state:   %s", [f"{x:.4f}" for x in normalized])

        obs2 = make_test_observation(config, device, state_values=normalized.tolist())
        with torch.no_grad():
            actions2 = model.sample_actions(device, obs2, num_steps=config.num_denoising_steps)
        actions2_np = actions2.cpu().float().numpy()
        log.info("  Actions[0]:\n%s", actions2_np[0])
        log.info("  Range: [%.4f, %.4f]", actions2_np.min(), actions2_np.max())
        log.info("  Mean: %.4f  Std: %.4f", actions2_np.mean(), actions2_np.std())
        log.info("  All finite: %s", np.all(np.isfinite(actions2_np)))

    # Test 3: Determinism (same seed -> same output)
    log.info("\n=== Test 3: Determinism check ===")
    torch.manual_seed(42)
    noise1 = torch.randn(1, config.action_horizon, config.action_dim, device=device)
    obs3a = make_test_observation(config, device)
    with torch.no_grad():
        actions3a = model.sample_actions(device, obs3a, noise=noise1.clone(),
                                          num_steps=config.num_denoising_steps)

    obs3b = make_test_observation(config, device)
    with torch.no_grad():
        actions3b = model.sample_actions(device, obs3b, noise=noise1.clone(),
                                          num_steps=config.num_denoising_steps)

    diff = (actions3a - actions3b).abs().max().item()
    log.info("  Max diff between two runs with same noise: %.2e", diff)
    log.info("  Deterministic: %s", "YES" if diff < 1e-5 else "NO")

    # Test 4: Latency benchmark
    log.info("\n=== Test 4: Latency benchmark (%d runs) ===", args.num_runs)
    obs4 = make_test_observation(config, device)

    # Warmup
    with torch.no_grad():
        for _ in range(2):
            model.sample_actions(device, obs4, num_steps=config.num_denoising_steps)

    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for i in range(args.num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.sample_actions(device, obs4, num_steps=config.num_denoising_steps)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    times_arr = np.array(times)
    log.info("  Mean: %.1f ms  Std: %.1f ms  Min: %.1f ms  Max: %.1f ms",
             times_arr.mean(), times_arr.std(), times_arr.min(), times_arr.max())
    log.info("  Throughput: %.1f Hz", 1000.0 / times_arr.mean())

    log.info("\n=== All tests complete ===")
    if np.all(np.isfinite(actions_np)) and diff < 1e-5:
        log.info("RESULT: Model loads and produces finite, deterministic outputs. Ready for ONNX export.")
    else:
        log.warning("RESULT: Issues detected - review above output.")


if __name__ == "__main__":
    main()
