#!/usr/bin/env python3
"""Validate TRT engines (FP32/FP16/INT8) against PyTorch reference for the excavator model.

Runs identical inputs through PyTorch and each TRT engine, comparing:
  1. Prefix encoder outputs (KV cache, prefix_pad_masks)
  2. Single denoise step velocity
  3. Full 8-step denoising trajectory (final actions)

Also saves calibration data from the PyTorch model for proper INT8 PTQ rebuilds.

Usage:
  uv run python scripts/validate_trt_vs_pytorch.py
"""
import dataclasses
import json
import logging
import os
import pathlib
import sys
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

WORKSPACE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE / "src"))
sys.path.insert(0, str(WORKSPACE / "scripts"))

ACTION_DIM = 4
ACTION_HORIZON = 11
NUM_DENOISE_STEPS = 8
SEED = 42


@dataclasses.dataclass
class ModelConfig:
    pi05: bool = True
    action_dim: int = ACTION_DIM
    action_horizon: int = ACTION_HORIZON
    paligemma_variant: str = "gemma_2b_lora"
    action_expert_variant: str = "gemma_300m_lora"
    dtype: str = "bfloat16"


def make_deterministic_inputs(device="cpu"):
    """Create reproducible inputs for comparison."""
    import torch
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    B = 1
    images_np = [np.random.randn(B, 3, 224, 224).astype(np.float32) * 0.3 for _ in range(3)]
    masks_np = [np.array([True]), np.array([True]), np.array([False])]
    tokens_np = np.zeros((B, 200), dtype=np.int64)
    tokens_np[0, :10] = [2, 6834, 1010, 573, 14402, 235265, 108, 7643, 235292, 139]
    token_masks_np = np.zeros((B, 200), dtype=bool)
    token_masks_np[0, :50] = True
    state_np = np.random.randn(B, ACTION_DIM).astype(np.float32) * 0.5

    images_torch = [torch.from_numpy(x).to(device) for x in images_np]
    masks_torch = [torch.tensor(m, dtype=torch.bool, device=device) for m in masks_np]
    tokens_torch = torch.from_numpy(tokens_np).to(device)
    token_masks_torch = torch.from_numpy(token_masks_np).to(device)
    state_torch = torch.from_numpy(state_np).to(device)

    return {
        "np": {
            "img0": images_np[0], "img1": images_np[1], "img2": images_np[2],
            "mask0": masks_np[0], "mask1": masks_np[1], "mask2": masks_np[2],
            "tokens": tokens_np, "token_masks": token_masks_np,
            "state": state_np,
        },
        "torch": {
            "images": images_torch, "masks": masks_torch,
            "tokens": tokens_torch, "token_masks": token_masks_torch,
            "state": state_torch,
        },
    }


def run_pytorch_reference(inputs_torch, device="cpu"):
    """Run PyTorch model and return reference outputs."""
    import torch
    import safetensors.torch
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

    from export_pytorch_onnx import _patch_float64_to_float32
    _patch_float64_to_float32()

    config = ModelConfig()
    original_compile = torch.compile
    torch.compile = lambda fn, **kwargs: fn
    try:
        model = PI0Pytorch(config)
    finally:
        torch.compile = original_compile

    ckpt = WORKSPACE / "checkpoints" / "excavator_v1_pytorch" / "model.safetensors"
    log.info("Loading PyTorch weights from %s", ckpt)
    safetensors.torch.load_model(model, str(ckpt), device=device)
    model = model.float().eval().to(device)

    log.info("Running PyTorch prefix encoder...")
    t0 = time.time()
    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            inputs_torch["images"], inputs_torch["masks"],
            inputs_torch["tokens"], inputs_torch["token_masks"],
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks.to(torch.int32), dim=1) - 1
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

    prefix_ms = (time.time() - t0) * 1000
    kv_keys = torch.stack(past_key_values.key_cache, dim=0)
    kv_values = torch.stack(past_key_values.value_cache, dim=0)

    log.info("  PyTorch prefix: %.1f ms", prefix_ms)
    log.info("  prefix_pad_masks sum: %d, shape: %s", prefix_pad_masks.sum().item(), prefix_pad_masks.shape)
    log.info("  kv_keys: mean=%.6f std=%.6f shape=%s", kv_keys.mean().item(), kv_keys.std().item(), kv_keys.shape)

    np.random.seed(SEED + 1000)
    x_t_np = np.random.standard_normal((1, ACTION_HORIZON, ACTION_DIM)).astype(np.float32)
    x_t = torch.from_numpy(x_t_np.copy()).to(device)

    log.info("Running PyTorch denoise loop (8 steps)...")
    t0 = time.time()
    with torch.no_grad():
        dt = -1.0 / NUM_DENOISE_STEPS
        t_val = 1.0
        velocities = []
        for step in range(NUM_DENOISE_STEPS):
            timestep = torch.tensor([t_val], device=device, dtype=torch.float32)
            v_t = model.denoise_step(
                inputs_torch["state"], prefix_pad_masks, past_key_values, x_t, timestep
            )
            velocities.append(v_t.cpu().numpy().copy())
            x_t = x_t + dt * v_t
            t_val += dt
    denoise_ms = (time.time() - t0) * 1000
    log.info("  PyTorch denoise: %.1f ms", denoise_ms)

    return {
        "prefix_pad_masks": prefix_pad_masks.cpu().numpy(),
        "kv_keys": kv_keys.cpu().numpy(),
        "kv_values": kv_values.cpu().numpy(),
        "velocities": velocities,
        "final_actions": x_t.cpu().numpy(),
        "initial_noise": x_t_np,
    }


def run_trt_inference(inputs_np, prefix_engine_path, denoise_engine_path, label="TRT"):
    """Run TRT inference and return outputs for comparison."""
    from trt_policy_server import TRTEngine, SHARED_TENSORS

    log.info("Loading %s engines: %s, %s", label, prefix_engine_path, denoise_engine_path)
    prefix_engine = TRTEngine(prefix_engine_path)
    shared = {name: prefix_engine.get_device_ptr(name) for name in SHARED_TENSORS}
    denoise_engine = TRTEngine(denoise_engine_path, shared_device_buffers=shared)

    t0 = time.time()
    prefix_engine.infer(
        {k: inputs_np[k] for k in ["img0", "img1", "img2", "mask0", "mask1", "mask2", "tokens", "token_masks"]},
        skip_d2h=set(SHARED_TENSORS),
    )
    prefix_pad_masks = prefix_engine.d2h("prefix_pad_masks")
    kv_keys = prefix_engine.d2h("kv_keys")
    kv_values = prefix_engine.d2h("kv_values")
    prefix_engine.sync()
    prefix_ms = (time.time() - t0) * 1000

    prefix_pad_masks = prefix_pad_masks.copy()
    kv_keys = kv_keys.copy()
    kv_values = kv_values.copy()

    log.info("  %s prefix: %.1f ms", label, prefix_ms)
    log.info("  prefix_pad_masks sum: %d", prefix_pad_masks.sum())
    log.info("  kv_keys: mean=%.6f std=%.6f", kv_keys.mean(), kv_keys.std())

    np.random.seed(SEED + 1000)
    x_t = np.random.standard_normal((1, ACTION_HORIZON, ACTION_DIM)).astype(np.float32)

    de = denoise_engine
    if "state" in de._io_info:
        de.h2d("state", inputs_np["state"])

    t0 = time.time()
    dt = np.float32(-1.0 / NUM_DENOISE_STEPS)
    t_val = np.float32(1.0)
    t_step = np.float32(-1.0 / NUM_DENOISE_STEPS)
    velocities = []
    for step in range(NUM_DENOISE_STEPS):
        de.h2d("x_t", x_t)
        de.h2d("timestep", np.array([t_val], dtype=np.float32))
        de.execute()
        vel_buf = de.d2h("velocity")
        de.sync()
        vel = vel_buf.copy()
        velocities.append(vel)
        x_t = x_t + dt * vel
        t_val += t_step
    denoise_ms = (time.time() - t0) * 1000
    log.info("  %s denoise: %.1f ms", label, denoise_ms)

    del prefix_engine, denoise_engine

    return {
        "prefix_pad_masks": prefix_pad_masks,
        "kv_keys": kv_keys,
        "kv_values": kv_values,
        "velocities": velocities,
        "final_actions": x_t,
    }


def compare_outputs(ref, test, label):
    """Compare reference and test outputs, printing detailed metrics."""
    log.info("")
    log.info("=" * 70)
    log.info("  %s vs PyTorch Reference", label)
    log.info("=" * 70)

    mask_match = np.array_equal(ref["prefix_pad_masks"], test["prefix_pad_masks"].astype(ref["prefix_pad_masks"].dtype))
    log.info("  prefix_pad_masks exact match: %s", mask_match)

    for name in ["kv_keys", "kv_values"]:
        r, t = ref[name], test[name]
        if r.shape != t.shape:
            log.warning("  %s shape mismatch: ref=%s test=%s", name, r.shape, t.shape)
            continue
        abs_diff = np.abs(r - t)
        rel_diff = abs_diff / (np.abs(r) + 1e-8)
        log.info("  %s:", name)
        log.info("    abs_diff: mean=%.2e  max=%.2e  p99=%.2e", abs_diff.mean(), abs_diff.max(), np.percentile(abs_diff, 99))
        log.info("    rel_diff: mean=%.2e  max=%.2e  p99=%.2e", rel_diff.mean(), rel_diff.max(), np.percentile(rel_diff, 99))
        log.info("    cosine_sim: %.8f", np.dot(r.ravel(), t.ravel()) / (np.linalg.norm(r.ravel()) * np.linalg.norm(t.ravel()) + 1e-10))

    for step_i in [0, 3, 7]:
        if step_i < len(ref["velocities"]) and step_i < len(test["velocities"]):
            rv = ref["velocities"][step_i]
            tv = test["velocities"][step_i]
            abs_diff = np.abs(rv - tv)
            log.info("  velocity step %d: abs_diff mean=%.2e max=%.2e", step_i, abs_diff.mean(), abs_diff.max())

    ra = ref["final_actions"]
    ta = test["final_actions"]
    action_abs = np.abs(ra - ta)
    action_mse = np.mean((ra - ta) ** 2)
    action_cos = np.dot(ra.ravel(), ta.ravel()) / (np.linalg.norm(ra.ravel()) * np.linalg.norm(ta.ravel()) + 1e-10)
    log.info("  final_actions:")
    log.info("    abs_diff: mean=%.2e  max=%.2e", action_abs.mean(), action_abs.max())
    log.info("    MSE: %.2e", action_mse)
    log.info("    cosine_sim: %.8f", action_cos)
    log.info("    ref  [0]: %s", ra[0, 0, :4] if ra.ndim == 3 else ra[0, :4])
    log.info("    test [0]: %s", ta[0, 0, :4] if ta.ndim == 3 else ta[0, :4])
    log.info("    ref  [5]: %s", ra[0, 5, :4] if ra.ndim == 3 else ra[5, :4])
    log.info("    test [5]: %s", ta[0, 5, :4] if ta.ndim == 3 else ta[5, :4])

    return {
        "masks_match": mask_match,
        "kv_keys_cos": float(np.dot(ref["kv_keys"].ravel(), test["kv_keys"].ravel()) / (np.linalg.norm(ref["kv_keys"].ravel()) * np.linalg.norm(test["kv_keys"].ravel()) + 1e-10)),
        "kv_values_cos": float(np.dot(ref["kv_values"].ravel(), test["kv_values"].ravel()) / (np.linalg.norm(ref["kv_values"].ravel()) * np.linalg.norm(test["kv_values"].ravel()) + 1e-10)),
        "action_mse": float(action_mse),
        "action_cos": float(action_cos),
    }


def save_calibration_data(ref_outputs, inputs_np):
    """Save calibration data from PyTorch reference for proper INT8 builds."""
    cal_dir = WORKSPACE / "calibration_data_fresh"
    prefix_dir = cal_dir / "prefix"
    denoise_dir = cal_dir / "denoise"
    prefix_dir.mkdir(parents=True, exist_ok=True)
    denoise_dir.mkdir(parents=True, exist_ok=True)

    for k in ["img0", "img1", "img2", "tokens", "token_masks"]:
        np.save(prefix_dir / f"{k}.npy", inputs_np[k])
    for k in ["mask0", "mask1", "mask2"]:
        np.save(prefix_dir / f"{k}.npy", inputs_np[k])

    np.save(denoise_dir / "state.npy", inputs_np["state"])
    np.save(denoise_dir / "prefix_pad_masks.npy", ref_outputs["prefix_pad_masks"])
    np.save(denoise_dir / "kv_keys.npy", ref_outputs["kv_keys"])
    np.save(denoise_dir / "kv_values.npy", ref_outputs["kv_values"])

    np.random.seed(SEED + 1000)
    x_t = np.random.standard_normal((1, ACTION_HORIZON, ACTION_DIM)).astype(np.float32)
    np.save(denoise_dir / "x_t.npy", x_t)
    np.save(denoise_dir / "timestep.npy", np.array([1.0], dtype=np.float32))

    npz_path = WORKSPACE / "calibration_data_fresh" / "calibration_batch.npz"
    np.savez(npz_path, **inputs_np,
             prefix_pad_masks=ref_outputs["prefix_pad_masks"],
             kv_keys=ref_outputs["kv_keys"],
             kv_values=ref_outputs["kv_values"])
    log.info("Saved fresh calibration data to %s", cal_dir)


def phase1_pytorch_reference():
    """Phase 1: Run PyTorch model, save reference outputs & calibration data.
    Run with: uv run python scripts/validate_trt_vs_pytorch.py --phase pytorch
    """
    import torch

    device = "cpu"
    log.info("Action dim: %d, Horizon: %d, Denoise steps: %d", ACTION_DIM, ACTION_HORIZON, NUM_DENOISE_STEPS)

    inputs = make_deterministic_inputs(device)
    ref = run_pytorch_reference(inputs["torch"], device)
    save_calibration_data(ref, inputs["np"])

    ref_path = WORKSPACE / "onnx_export" / "pytorch_reference.npz"
    np.savez(ref_path,
             prefix_pad_masks=ref["prefix_pad_masks"],
             kv_keys=ref["kv_keys"],
             kv_values=ref["kv_values"],
             final_actions=ref["final_actions"],
             initial_noise=ref["initial_noise"],
             **{f"vel_{i}": v for i, v in enumerate(ref["velocities"])},
             **{f"inp_{k}": v for k, v in inputs["np"].items()})
    log.info("Saved PyTorch reference to %s (%.1f MB)", ref_path, ref_path.stat().st_size / 1e6)


def phase2_trt_comparison():
    """Phase 2: Run TRT engines, compare against saved PyTorch reference.
    Run with: python3 scripts/validate_trt_vs_pytorch.py --phase trt
    (uses system Python with tensorrt)
    """
    ref_path = WORKSPACE / "onnx_export" / "pytorch_reference.npz"
    if not ref_path.exists():
        log.error("PyTorch reference not found at %s. Run --phase pytorch first.", ref_path)
        sys.exit(1)

    data = np.load(ref_path)
    ref = {
        "prefix_pad_masks": data["prefix_pad_masks"],
        "kv_keys": data["kv_keys"],
        "kv_values": data["kv_values"],
        "final_actions": data["final_actions"],
        "velocities": [data[f"vel_{i}"] for i in range(NUM_DENOISE_STEPS)],
    }

    inputs_np = {}
    for k in ["img0", "img1", "img2", "mask0", "mask1", "mask2", "tokens", "token_masks", "state"]:
        inputs_np[k] = data[f"inp_{k}"]

    log.info("Loaded PyTorch reference: kv_keys shape=%s, final_actions shape=%s",
             ref["kv_keys"].shape, ref["final_actions"].shape)

    engine_dir = WORKSPACE / "onnx_export"
    configs = []
    engine_combos = [
        ("FP32", "fp32", "fp32"),
        ("FP16", "fp16", "fp16"),
        ("BF16", "bf16_fresh", "bf16_fresh"),
        ("INT8", "int8_fresh", "int8_fresh"),
        ("FP32+FP16", "fp32", "fp16"),
        ("FP32+BF16", "fp32", "bf16_fresh"),
        ("FP32+INT8", "fp32", "int8_fresh"),
        ("BF16+FP32", "bf16_fresh", "fp32"),
    ]
    for label, prefix_suffix, denoise_suffix in engine_combos:
        prefix_path = engine_dir / f"prefix_encoder_{prefix_suffix}.engine"
        denoise_path = engine_dir / f"denoise_step_{denoise_suffix}.engine"
        if prefix_path.exists() and denoise_path.exists():
            configs.append((label, str(prefix_path), str(denoise_path)))
        else:
            log.warning("Skipping %s: engine files not found", label)

    results = {}
    for label, prefix_path, denoise_path in configs:
        log.info("\n" + "=" * 70)
        log.info("  %s TRT Inference", label)
        log.info("=" * 70)
        trt_out = run_trt_inference(inputs_np, prefix_path, denoise_path, label)
        metrics = compare_outputs(ref, trt_out, label)
        results[label] = metrics

    log.info("\n" + "=" * 70)
    log.info("  SUMMARY")
    log.info("=" * 70)
    log.info("%-8s %-12s %-12s %-12s %-12s %-8s", "Engine", "KV_keys_cos", "KV_vals_cos", "Action_MSE", "Action_cos", "Masks")
    for label, m in results.items():
        log.info("%-8s %-12.8f %-12.8f %-12.2e %-12.8f %-8s",
                 label, m["kv_keys_cos"], m["kv_values_cos"], m["action_mse"], m["action_cos"], m["masks_match"])

    results_path = WORKSPACE / "onnx_export" / "validation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", results_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["pytorch", "trt", "both"], default="both")
    args = parser.parse_args()

    if args.phase in ("pytorch", "both"):
        log.info("=" * 70)
        log.info("  PHASE 1: PyTorch Reference")
        log.info("=" * 70)
        phase1_pytorch_reference()

    if args.phase in ("trt", "both"):
        log.info("=" * 70)
        log.info("  PHASE 2: TRT Comparison")
        log.info("=" * 70)
        phase2_trt_comparison()


if __name__ == "__main__":
    main()
