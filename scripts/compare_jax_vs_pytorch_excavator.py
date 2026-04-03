#!/usr/bin/env python3
"""Compare JAX vs PyTorch excavator model outputs at every pipeline stage.

Loads pre-saved JAX reference outputs (from jax_reference_excavator.py) and
the converted PyTorch checkpoint, feeds identical inputs, and compares at
multiple levels:
  1. Single-step velocity (v_t)
  2. Full sample_actions trajectory with fixed noise (8 denoise steps)
  3. Per-dimension action comparison
  4. Per-timestep action comparison

Usage:
    uv run python scripts/compare_jax_vs_pytorch_excavator.py \
        --jax_reference jax_reference_gpu.npz \
        --pytorch_checkpoint checkpoints/excavator_v1_pytorch \
        --config_name pi05_excavator_v2
"""

import argparse
import gc
import logging
import os
import sys
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_jax_reference(npz_path):
    """Load JAX reference outputs and inputs from npz file."""
    logger.info("Loading JAX reference from %s ...", npz_path)
    data = np.load(npz_path)

    jax_results = {
        "v_t": data["v_t"],
        "loss": float(data["loss"]),
        "sampled_actions": data["sampled_actions"],
        "sample_compile_ms": float(data["sample_compile_ms"]),
        "sample_warm_ms": float(data["sample_warm_ms"]),
        "single_step_compile_ms": float(data["single_step_compile_ms"]),
        "single_step_warm_ms": float(data["single_step_warm_ms"]),
    }

    inputs = {
        "state": data["input_state"],
        "sample_noise": data["input_sample_noise"],
        "actions": data["input_actions"],
        "noise": data["input_noise"],
        "time": data["input_time"],
        "images": {
            "base_0_rgb": data["input_image_base_0_rgb"],
            "left_wrist_0_rgb": data["input_image_left_wrist_0_rgb"],
            "right_wrist_0_rgb": data["input_image_right_wrist_0_rgb"],
        },
    }

    logger.info("JAX reference: v_t shape=%s, sampled shape=%s, loss=%.6f",
                jax_results["v_t"].shape, jax_results["sampled_actions"].shape, jax_results["loss"])
    logger.info("JAX timings: single-step warm=%.1fms, sample_actions warm=%.1fms",
                jax_results["single_step_warm_ms"], jax_results["sample_warm_ms"])

    return jax_results, inputs


def run_pytorch_inference(model_cfg, pytorch_checkpoint_path, inputs):
    """Run PyTorch model: single-step v_t + full sample_actions."""
    import torch
    import safetensors.torch
    import openpi.models.model as _model
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

    logger.info("Loading PyTorch model from %s ...", pytorch_checkpoint_path)
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt_model = PI0Pytorch(model_cfg).to(device)
    safetensors.torch.load_model(pt_model, f"{pytorch_checkpoint_path}/model.safetensors")
    pt_model.eval()
    logger.info("PyTorch model loaded on %s in %.1fs", device, time.time() - t0)

    # Images: JAX reference saved as NHWC (B, H, W, C), PyTorch needs NCHW (B, C, H, W)
    pt_images = {}
    for k, v in inputs["images"].items():
        img_nchw = np.transpose(v, (0, 3, 1, 2))
        pt_images[k] = torch.from_numpy(img_nchw).to(device)

    B = inputs["state"].shape[0]
    max_token_len = model_cfg.max_token_len
    tokenized_prompt_np = np.zeros((B, max_token_len), dtype=np.int32)
    tokenized_prompt_np[:, 0] = 2
    tokenized_prompt_mask_np = np.zeros((B, max_token_len), dtype=bool)
    tokenized_prompt_mask_np[:, 0] = True

    pt_image_masks = {
        "base_0_rgb": torch.ones((B,), dtype=torch.bool, device=device),
        "left_wrist_0_rgb": torch.ones((B,), dtype=torch.bool, device=device),
        "right_wrist_0_rgb": torch.zeros((B,), dtype=torch.bool, device=device),
    }
    pt_state = torch.from_numpy(inputs["state"]).to(device)
    pt_actions = torch.from_numpy(inputs["actions"]).to(device)
    pt_noise = torch.from_numpy(inputs["noise"]).to(device)
    pt_time = torch.from_numpy(inputs["time"]).to(device)

    pt_time_expanded = pt_time[:, None, None]
    pt_x_t = pt_time_expanded * pt_noise + (1 - pt_time_expanded) * pt_actions
    pt_u_t = pt_noise - pt_actions

    pt_observation = _model.Observation(
        images=pt_images,
        image_masks=pt_image_masks,
        state=pt_state,
        tokenized_prompt=torch.from_numpy(tokenized_prompt_np).to(device),
        tokenized_prompt_mask=torch.from_numpy(tokenized_prompt_mask_np).to(device),
    )

    # --- Single-step velocity ---
    logger.info("Running PyTorch single-step velocity...")
    with torch.no_grad():
        images_list, img_masks_list, lang_tokens, lang_masks, state = pt_model._preprocess_observation(
            pt_observation, train=False
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = pt_model.embed_prefix(
            images_list, img_masks_list, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = pt_model.embed_suffix(
            state, pt_x_t, pt_time
        )

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = pt_model._prepare_attention_masks_4d(att_2d_masks)

        if prefix_embs.dtype != suffix_embs.dtype:
            suffix_embs = suffix_embs.to(dtype=prefix_embs.dtype)

        t0 = time.time()
        (_, suffix_out), _ = pt_model.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        single_step_ms = (time.time() - t0) * 1000

        pt_suffix_out = suffix_out[:, -model_cfg.action_horizon:]
        pt_suffix_out = pt_suffix_out.to(dtype=torch.float32)
        pt_v_t = pt_model.action_out_proj(pt_suffix_out)
        pt_loss = torch.nn.functional.mse_loss(pt_u_t, pt_v_t).item()

    pt_v_t_np = pt_v_t.cpu().numpy()
    logger.info("PyTorch single-step: loss=%.6f, v_t mean=%.6f std=%.6f (%.1fms)",
                pt_loss, pt_v_t_np.mean(), pt_v_t_np.std(), single_step_ms)

    # Warm run
    with torch.no_grad():
        t0 = time.time()
        (_, suffix_out2), _ = pt_model.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        single_step_warm_ms = (time.time() - t0) * 1000
    logger.info("PyTorch single-step (warm): %.1fms", single_step_warm_ms)

    # --- Full sample_actions trajectory ---
    logger.info("Running PyTorch sample_actions (8 denoise steps)...")
    sample_noise = torch.from_numpy(inputs["sample_noise"]).to(device)
    with torch.no_grad():
        t0 = time.time()
        pt_sampled_actions = pt_model.sample_actions(
            device, pt_observation, noise=sample_noise, num_steps=8
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sample_time_ms = (time.time() - t0) * 1000

    pt_sampled_np = pt_sampled_actions.cpu().numpy()
    logger.info("PyTorch sample_actions: shape=%s, time=%.1fms, mean=%.6f std=%.6f",
                pt_sampled_np.shape, sample_time_ms,
                pt_sampled_np.mean(), pt_sampled_np.std())

    # Warm run
    with torch.no_grad():
        t0 = time.time()
        pt_sampled_actions_2 = pt_model.sample_actions(
            device, pt_observation, noise=sample_noise, num_steps=8
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sample_warm_ms = (time.time() - t0) * 1000
    logger.info("PyTorch sample_actions (warm): %.1fms", sample_warm_ms)

    del pt_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "v_t": pt_v_t_np,
        "loss": pt_loss,
        "sampled_actions": pt_sampled_np,
        "single_step_ms": single_step_ms,
        "single_step_warm_ms": single_step_warm_ms,
        "sample_time_ms": sample_time_ms,
        "sample_warm_ms": sample_warm_ms,
    }


def compare_results(jax_results, pt_results, action_dim):
    logger.info("")
    logger.info("=" * 70)
    logger.info("  COMPARISON RESULTS: JAX (GPU) vs PyTorch")
    logger.info("=" * 70)

    # Single-step velocity
    v_t_mse = float(np.mean((jax_results["v_t"] - pt_results["v_t"]) ** 2))
    v_t_max_diff = float(np.max(np.abs(jax_results["v_t"] - pt_results["v_t"])))
    loss_diff = abs(jax_results["loss"] - pt_results["loss"])

    logger.info("")
    logger.info("  1. Single-step velocity (v_t):")
    logger.info("     JAX loss:         %.6f", jax_results["loss"])
    logger.info("     PyTorch loss:     %.6f", pt_results["loss"])
    logger.info("     MSE:              %.2e", v_t_mse)
    logger.info("     Max abs diff:     %.2e", v_t_max_diff)
    logger.info("     Loss diff:        %.2e", loss_diff)

    # Full trajectory
    traj_mse = float(np.mean((jax_results["sampled_actions"] - pt_results["sampled_actions"]) ** 2))
    traj_max_diff = float(np.max(np.abs(jax_results["sampled_actions"] - pt_results["sampled_actions"])))

    logger.info("")
    logger.info("  2. Full trajectory (8 denoise steps, sample_actions):")
    logger.info("     MSE:              %.2e", traj_mse)
    logger.info("     Max abs diff:     %.2e", traj_max_diff)

    # Per-dimension trajectory comparison
    dim_names = ["left_x", "left_y", "right_x", "right_y"][:action_dim]
    jax_traj = jax_results["sampled_actions"][0]
    pt_traj = pt_results["sampled_actions"][0]

    logger.info("")
    logger.info("  3. Per-dimension trajectory comparison (first sample):")
    logger.info("     %-10s  %-12s  %-12s  %-12s  %-12s", "Dim", "JAX mean", "PT mean", "Diff mean", "MSE")
    for d in range(action_dim):
        jax_d = jax_traj[:, d]
        pt_d = pt_traj[:, d]
        dim_mse = float(np.mean((jax_d - pt_d) ** 2))
        name = dim_names[d] if d < len(dim_names) else f"dim_{d}"
        logger.info("     %-10s  %12.6f  %12.6f  %12.6f  %.2e",
                     name, jax_d.mean(), pt_d.mean(), (jax_d - pt_d).mean(), dim_mse)

    # Per-timestep comparison
    logger.info("")
    logger.info("  4. Per-timestep trajectory comparison:")
    logger.info("     %-6s  %-12s  %-12s  %-12s", "Step", "JAX norm", "PT norm", "MSE")
    for t in range(jax_traj.shape[0]):
        jax_step = jax_traj[t]
        pt_step = pt_traj[t]
        step_mse = float(np.mean((jax_step - pt_step) ** 2))
        logger.info("     %-6d  %12.6f  %12.6f  %.2e",
                     t, np.linalg.norm(jax_step), np.linalg.norm(pt_step), step_mse)

    # Latency
    logger.info("")
    logger.info("  5. Latency (warm):")
    logger.info("     JAX single-step:       %.1f ms", jax_results.get("single_step_warm_ms", 0))
    logger.info("     PyTorch single-step:    %.1f ms", pt_results.get("single_step_warm_ms", 0))
    logger.info("     JAX sample_actions:     %.1f ms", jax_results.get("sample_warm_ms", 0))
    logger.info("     PyTorch sample_actions:  %.1f ms", pt_results.get("sample_warm_ms", 0))

    # Verdict
    logger.info("")
    logger.info("=" * 70)
    THRESHOLD_SINGLE = 1e-4
    THRESHOLD_TRAJ = 1e-2
    if v_t_mse < THRESHOLD_SINGLE and traj_mse < THRESHOLD_TRAJ:
        logger.info("  PASS: Conversion is numerically faithful")
        logger.info("    v_t MSE %.2e < %.2e threshold", v_t_mse, THRESHOLD_SINGLE)
        logger.info("    trajectory MSE %.2e < %.2e threshold", traj_mse, THRESHOLD_TRAJ)
        return True
    else:
        logger.error("  FAIL: Significant divergence detected")
        if v_t_mse >= THRESHOLD_SINGLE:
            logger.error("    v_t MSE %.2e >= %.2e threshold", v_t_mse, THRESHOLD_SINGLE)
        if traj_mse >= THRESHOLD_TRAJ:
            logger.error("    trajectory MSE %.2e >= %.2e threshold", traj_mse, THRESHOLD_TRAJ)
        return False


def main():
    parser = argparse.ArgumentParser(description="Compare JAX vs PyTorch excavator model")
    parser.add_argument("--jax_reference", required=True, help="Path to JAX reference npz file")
    parser.add_argument("--pytorch_checkpoint", required=True, help="Path to converted PyTorch checkpoint directory")
    parser.add_argument("--config_name", default="pi05_excavator_v2", help="Training config name")
    parser.add_argument("--save_outputs", default="/tmp/jax_vs_pytorch_comparison.npz",
                        help="Save comparison outputs to npz file")
    args = parser.parse_args()

    import openpi.training.config as _config
    config = _config.get_config(args.config_name)
    model_cfg = config.model
    logger.info("Config: %s (action_dim=%d, action_horizon=%d, pi05=%s)",
                args.config_name, model_cfg.action_dim, model_cfg.action_horizon, model_cfg.pi05)

    jax_results, inputs = load_jax_reference(args.jax_reference)
    pt_results = run_pytorch_inference(model_cfg, args.pytorch_checkpoint, inputs)

    passed = compare_results(jax_results, pt_results, model_cfg.action_dim)

    if args.save_outputs:
        np.savez(args.save_outputs,
                 jax_v_t=jax_results["v_t"],
                 pt_v_t=pt_results["v_t"],
                 jax_sampled=jax_results["sampled_actions"],
                 pt_sampled=pt_results["sampled_actions"],
                 inputs_state=inputs["state"],
                 inputs_noise=inputs["sample_noise"])
        logger.info("Saved comparison outputs to %s", args.save_outputs)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
