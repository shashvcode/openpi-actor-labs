#!/usr/bin/env python3
"""Verify ONNX models against PyTorch reference.

Compares prefix encoder output (KV cache) and every denoising step's velocity
between the PyTorch model and the ONNX models via onnxruntime on CPU.

Usage (inside the trt_pipeline container):
  python /workspace/openpi/scripts/verify_onnx.py
"""

import dataclasses
import logging
import os
import pathlib
import sys
import time

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

WORKSPACE = pathlib.Path("/workspace/openpi")
CHECKPOINT_DIR = WORKSPACE / "checkpoints" / "runC_pytorch"
ONNX_DIR = WORKSPACE / "onnx_export"

sys.path.insert(0, str(WORKSPACE / "src"))


@dataclasses.dataclass
class ModelConfig:
    pi05: bool = True
    action_dim: int = 6
    action_horizon: int = 11
    paligemma_variant: str = "gemma_2b_lora"
    action_expert_variant: str = "gemma_300m_lora"
    dtype: str = "bfloat16"


def _patch_float64_to_float32():
    """Match the export: replace float64 with float32 in sinusoidal embedding."""
    import openpi.models_pytorch.pi0_pytorch as _pi0

    _original = _pi0.get_safe_dtype

    def _patched(target_dtype, device_type):
        if target_dtype == torch.float64:
            return torch.float32
        return _original(target_dtype, device_type)

    _pi0.get_safe_dtype = _patched


def load_pytorch_model(device="cpu"):
    import safetensors.torch
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    _patch_float64_to_float32()

    config = ModelConfig()
    original_compile = torch.compile
    torch.compile = lambda fn, **kwargs: fn
    try:
        model = PI0Pytorch(config)
    finally:
        torch.compile = original_compile

    safetensors.torch.load_model(model, str(CHECKPOINT_DIR / "model.safetensors"), device=device)
    model.eval().float().to(device)
    return model


def load_onnx_sessions():
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.log_severity_level = 3

    log.info("Loading ONNX prefix encoder...")
    prefix_sess = ort.InferenceSession(
        str(ONNX_DIR / "prefix_encoder.onnx"), opts, providers=["CPUExecutionProvider"]
    )
    log.info("Loading ONNX denoise step...")
    denoise_sess = ort.InferenceSession(
        str(ONNX_DIR / "denoise_step.onnx"), opts, providers=["CPUExecutionProvider"]
    )
    return prefix_sess, denoise_sess


def run_pytorch_inference(model, images, img_masks, tokens, token_masks, state, noise, num_steps=8):
    """Run full inference with PyTorch, capturing intermediate values."""
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, tokens, token_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        kv_keys = torch.stack(past_key_values.key_cache, dim=0)
        kv_values = torch.stack(past_key_values.value_cache, dim=0)

        dt = -1.0 / num_steps
        x_t = noise.clone()
        t = 1.0
        velocities = []
        for step in range(num_steps):
            expanded_time = torch.tensor([t], dtype=torch.float32)
            v_t = model.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
            velocities.append(v_t.clone())
            x_t = x_t + dt * v_t
            t += dt

    return {
        "prefix_pad_masks": prefix_pad_masks.numpy(),
        "kv_keys": kv_keys.float().numpy(),
        "kv_values": kv_values.float().numpy(),
        "velocities": [v.numpy() for v in velocities],
        "final_actions": x_t.numpy(),
    }


def run_onnx_inference(prefix_sess, denoise_sess, images, img_masks, tokens, token_masks,
                       state, noise, num_steps=8):
    """Run full inference with ONNX, capturing intermediate values."""

    prefix_inputs = {
        "img0": images[0].numpy(),
        "img1": images[1].numpy(),
        "img2": images[2].numpy(),
        "mask0": img_masks[0].numpy(),
        "mask1": img_masks[1].numpy(),
        "mask2": img_masks[2].numpy(),
        "tokens": tokens.numpy(),
        "token_masks": token_masks.numpy(),
    }

    prefix_pad_masks, kv_keys, kv_values = prefix_sess.run(None, prefix_inputs)

    dt = -1.0 / num_steps
    x_t = noise.numpy().copy()
    t = 1.0
    velocities = []
    for step in range(num_steps):
        denoise_inputs = {
            "state": state.numpy(),
            "prefix_pad_masks": prefix_pad_masks,
            "x_t": x_t.astype(np.float32),
            "timestep": np.array([t], dtype=np.float32),
            "kv_keys": kv_keys,
            "kv_values": kv_values,
        }
        (velocity,) = denoise_sess.run(None, denoise_inputs)
        velocities.append(velocity.copy())
        x_t = x_t + dt * velocity
        t += dt

    return {
        "prefix_pad_masks": prefix_pad_masks,
        "kv_keys": kv_keys.astype(np.float32),
        "kv_values": kv_values.astype(np.float32),
        "velocities": velocities,
        "final_actions": x_t,
    }


def compare_results(pt_results, onnx_results, threshold=0.01):
    """Compare PyTorch vs ONNX results and report MSE."""
    all_pass = True

    # Compare KV cache
    kv_keys_mse = np.mean((pt_results["kv_keys"] - onnx_results["kv_keys"]) ** 2)
    kv_values_mse = np.mean((pt_results["kv_values"] - onnx_results["kv_values"]) ** 2)
    log.info("KV cache comparison:")
    log.info("  keys   MSE: %.8f  max_abs_diff: %.6f", kv_keys_mse,
             np.max(np.abs(pt_results["kv_keys"] - onnx_results["kv_keys"])))
    log.info("  values MSE: %.8f  max_abs_diff: %.6f", kv_values_mse,
             np.max(np.abs(pt_results["kv_values"] - onnx_results["kv_values"])))

    if kv_keys_mse > threshold or kv_values_mse > threshold:
        log.error("  FAIL: KV cache MSE exceeds threshold %.4f", threshold)
        all_pass = False
    else:
        log.info("  PASS")

    # Compare each denoising step velocity
    log.info("Denoising step velocities:")
    for i, (pt_v, onnx_v) in enumerate(zip(pt_results["velocities"], onnx_results["velocities"])):
        mse = np.mean((pt_v - onnx_v) ** 2)
        max_diff = np.max(np.abs(pt_v - onnx_v))
        status = "PASS" if mse <= threshold else "FAIL"
        log.info("  step %d: MSE=%.8f  max_abs_diff=%.6f  [%s]", i, mse, max_diff, status)
        if mse > threshold:
            all_pass = False

    # Compare final actions
    final_mse = np.mean((pt_results["final_actions"] - onnx_results["final_actions"]) ** 2)
    final_max = np.max(np.abs(pt_results["final_actions"] - onnx_results["final_actions"]))
    status = "PASS" if final_mse <= threshold else "FAIL"
    log.info("Final actions: MSE=%.8f  max_abs_diff=%.6f  [%s]", final_mse, final_max, status)
    if final_mse > threshold:
        all_pass = False

    return all_pass


def main():
    device = "cpu"
    num_steps = 8

    log.info("Loading PyTorch model...")
    t0 = time.time()
    model = load_pytorch_model(device)
    log.info("PyTorch model loaded in %.1fs", time.time() - t0)

    log.info("Loading ONNX sessions...")
    t0 = time.time()
    prefix_sess, denoise_sess = load_onnx_sessions()
    log.info("ONNX sessions loaded in %.1fs", time.time() - t0)

    # Create deterministic test inputs
    torch.manual_seed(42)
    B = 1
    images = [torch.randn(B, 3, 224, 224) for _ in range(3)]
    img_masks = [torch.ones(B, dtype=torch.bool) for _ in range(3)]
    tokens = torch.randint(0, 1000, (B, 200), dtype=torch.long)
    token_masks = torch.ones(B, 200, dtype=torch.bool)
    state = torch.randn(B, model.config.action_dim)
    noise = torch.randn(B, model.config.action_horizon, model.config.action_dim)

    log.info("Running PyTorch inference...")
    t0 = time.time()
    pt_results = run_pytorch_inference(model, images, img_masks, tokens, token_masks, state, noise, num_steps)
    log.info("PyTorch inference done in %.1fs", time.time() - t0)

    log.info("Running ONNX inference...")
    t0 = time.time()
    onnx_results = run_onnx_inference(prefix_sess, denoise_sess, images, img_masks, tokens, token_masks,
                                      state, noise, num_steps)
    log.info("ONNX inference done in %.1fs", time.time() - t0)

    log.info("=" * 60)
    log.info("Comparing PyTorch vs ONNX results (threshold: 0.01)")
    log.info("=" * 60)
    all_pass = compare_results(pt_results, onnx_results, threshold=0.01)

    if all_pass:
        log.info("ALL CHECKS PASSED — ONNX models match PyTorch reference")
    else:
        log.error("SOME CHECKS FAILED — see above for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
