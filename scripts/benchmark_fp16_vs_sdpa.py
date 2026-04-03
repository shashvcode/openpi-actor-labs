#!/usr/bin/env python3
"""Head-to-head benchmark: TRT FP16 vs PyTorch SDPA+compile on Jetson Thor.

Runs both inference backends with identical inputs and reports timing comparison.

Usage (inside Docker):
  PYTHONPATH=/workspace/openpi/src:/workspace/openpi/packages/openpi-client/src \
    python3 /workspace/openpi/scripts/benchmark_fp16_vs_sdpa.py
"""

import gc
import json
import logging
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/workspace/openpi/src")
sys.path.insert(0, "/workspace/openpi/packages/openpi-client/src")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Constants
IMAGE_SIZE = 224
MAX_TOKEN_LEN = 200
ACTION_HORIZON = 11
NUM_STEPS = 8
ACTION_DIM = 4

CHECKPOINT_DIR = "/workspace/openpi/checkpoints/excavator_v1_pytorch"
NORM_STATS = "/workspace/openpi/checkpoints/excavator_v1_pytorch/norm_stats.json"
PREFIX_ENGINE = "/workspace/openpi/onnx_export/prefix_encoder_fp16.engine"
DENOISE_ENGINE = "/workspace/openpi/onnx_export/denoise_step_fp16.engine"
TOKENIZER = "/root/.cache/openpi/big_vision/paligemma_tokenizer.model"

N_WARMUP = 8
N_RUNS = 30


class QuantileNormalizer:
    def __init__(self, path):
        with open(path) as f:
            raw = json.load(f)["norm_stats"]
        self.state_q01 = np.array(raw["state"]["q01"], dtype=np.float32)
        self.state_q99 = np.array(raw["state"]["q99"], dtype=np.float32)
        self.action_q01 = np.array(raw["actions"]["q01"], dtype=np.float32)
        self.action_q99 = np.array(raw["actions"]["q99"], dtype=np.float32)

    def normalize_state(self, state):
        q01, q99 = self.state_q01[:state.shape[-1]], self.state_q99[:state.shape[-1]]
        return (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    def unnormalize_actions(self, actions):
        q01, q99 = self.action_q01, self.action_q99
        return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def make_dummy_inputs(normalizer):
    """Create identical dummy inputs for both backends."""
    state = np.zeros(ACTION_DIM, dtype=np.float32)
    norm_state = normalizer.normalize_state(state)

    import sentencepiece
    sp = sentencepiece.SentencePieceProcessor(model_file=TOKENIZER)
    discretized = np.digitize(norm_state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
    state_str = " ".join(map(str, discretized))
    full_prompt = f"Task: pick up the cube, State: {state_str};\nAction: "
    tokens_raw = sp.encode(full_prompt, add_bos=True)
    pad_len = MAX_TOKEN_LEN - len(tokens_raw)
    tokens = np.array(tokens_raw + [0] * pad_len, dtype=np.int64)
    token_mask = np.array([True] * len(tokens_raw) + [False] * pad_len, dtype=bool)

    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    img_proc = (img.astype(np.float32) / 255.0 * 2.0 - 1.0).transpose(2, 0, 1)[np.newaxis]

    return {
        "norm_state": norm_state,
        "tokens": tokens,
        "token_mask": token_mask,
        "img_proc": img_proc,
    }


# ---------------------------------------------------------------------------
# TRT FP16 Benchmark
# ---------------------------------------------------------------------------

def benchmark_trt_fp16(inputs, normalizer):
    """Benchmark TRT FP16 inference."""
    log.info("=" * 65)
    log.info("TRT FP16 BENCHMARK")
    log.info("=" * 65)

    try:
        import tensorrt as trt
        from cuda.bindings import runtime as cudart
    except ImportError:
        log.warning("TensorRT or cuda.bindings not available, skipping TRT benchmark")
        return None

    sys.path.insert(0, "/workspace/openpi/scripts")
    from trt_policy_server import TRTEngine, Tokenizer, preprocess_observation, SHARED_TENSORS

    log.info("Loading TRT engines...")
    prefix_engine = TRTEngine(PREFIX_ENGINE)
    shared = {name: prefix_engine.get_device_ptr(name) for name in SHARED_TENSORS}
    denoise_engine = TRTEngine(DENOISE_ENGINE, shared_device_buffers=shared)

    def run_trt_inference():
        trt_inputs = {
            "img0": inputs["img_proc"].copy(),
            "img1": inputs["img_proc"].copy(),
            "img2": np.full_like(inputs["img_proc"], -1.0),
            "mask0": np.array([True]),
            "mask1": np.array([True]),
            "mask2": np.array([False]),
            "tokens": inputs["tokens"][np.newaxis],
            "token_masks": inputs["token_mask"][np.newaxis],
            "state": inputs["norm_state"][np.newaxis],
        }

        prefix_engine.infer(
            {k: trt_inputs[k] for k in ["img0", "img1", "img2", "mask0", "mask1", "mask2", "tokens", "token_masks"]},
            skip_d2h=set(SHARED_TENSORS),
        )

        de = denoise_engine
        if "state" in de._io_info:
            de.h2d("state", trt_inputs["state"])

        rng = np.random.default_rng(42)
        x_t = rng.standard_normal((1, ACTION_HORIZON, ACTION_DIM)).astype(np.float32)
        dt = np.float32(-1.0 / NUM_STEPS)
        t = np.float32(1.0)

        for _ in range(NUM_STEPS):
            de.h2d("x_t", x_t)
            de.h2d("timestep", np.array([t], dtype=np.float32))
            de.execute()
            vel_buf = de.d2h("velocity")
            de.sync()
            x_t = x_t + dt * vel_buf
            t += dt

        return x_t[0, :, :ACTION_DIM]

    log.info("Warming up TRT (%d runs)...", N_WARMUP)
    for i in range(N_WARMUP):
        t0 = time.monotonic()
        _ = run_trt_inference()
        log.info("  Warmup %d/%d: %.1f ms", i + 1, N_WARMUP, (time.monotonic() - t0) * 1000)

    log.info("Timed TRT runs (%d)...", N_RUNS)
    timings = []
    for _ in range(N_RUNS):
        t0 = time.monotonic()
        actions = run_trt_inference()
        timings.append((time.monotonic() - t0) * 1000)

    timings = np.array(timings)
    log.info("TRT FP16: Mean=%.1f ms  Median=%.1f ms  Min=%.1f ms  P95=%.1f ms  (%.1f Hz)",
             timings.mean(), np.median(timings), timings.min(),
             np.percentile(timings, 95), 1000 / timings.mean())

    del prefix_engine, denoise_engine
    gc.collect()
    return timings


# ---------------------------------------------------------------------------
# PyTorch SDPA+compile Benchmark
# ---------------------------------------------------------------------------

def patch_attention_with_sdpa():
    """Monkey-patch eager_attention_forward with SDPA."""
    from transformers.models.gemma import modeling_gemma

    def sdpa_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        key_states = modeling_gemma.repeat_kv(key, module.num_key_value_groups)
        value_states = modeling_gemma.repeat_kv(value, module.num_key_value_groups)

        if attention_mask is not None:
            mask = attention_mask[:, :, :, :key_states.shape[-2]]
            if mask.dtype != query.dtype:
                mask = mask.to(query.dtype)
        else:
            mask = None

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key_states, value_states,
            attn_mask=mask,
            dropout_p=dropout if module.training else 0.0,
            scale=scaling,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, None

    modeling_gemma.eager_attention_forward = sdpa_attention_forward
    log.info("Patched attention -> SDPA")


def benchmark_pytorch_sdpa(inputs, normalizer, compile_mode="max-autotune"):
    """Benchmark PyTorch with SDPA + torch.compile."""
    log.info("=" * 65)
    log.info("PyTorch SDPA + torch.compile(%s) BENCHMARK", compile_mode)
    log.info("=" * 65)

    patch_attention_with_sdpa()

    import dataclasses
    import safetensors.torch
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    import openpi.models_pytorch.pi0_pytorch as _pi0
    from openpi.models import model as _model

    _orig = _pi0.get_safe_dtype
    def _patched(target_dtype, device_type):
        if target_dtype == torch.float64:
            return torch.float32
        return _orig(target_dtype, device_type)
    _pi0.get_safe_dtype = _patched

    @dataclasses.dataclass
    class ModelConfig:
        pi05: bool = True
        action_dim: int = ACTION_DIM
        action_horizon: int = ACTION_HORIZON
        paligemma_variant: str = "gemma_2b_lora"
        action_expert_variant: str = "gemma_300m_lora"
        dtype: str = "bfloat16"
        state_dim: int = ACTION_DIM
        max_token_len: int = MAX_TOKEN_LEN
        discrete_state_input: bool = True

    device = torch.device("cuda")

    log.info("Creating model...")
    orig_compile = torch.compile
    torch.compile = lambda fn, **kw: fn
    try:
        model = PI0Pytorch(ModelConfig())
    finally:
        torch.compile = orig_compile

    log.info("Loading weights...")
    t0 = time.time()
    safetensors.torch.load_model(model, f"{CHECKPOINT_DIR}/model.safetensors", device="cuda")
    log.info("Weights loaded in %.1fs", time.time() - t0)

    model.eval()
    model.to(device)

    if compile_mode != "none":
        log.info("Compiling denoise_step (mode=%s)...", compile_mode)
        model.denoise_step = torch.compile(model.denoise_step, mode=compile_mode)

    dummy_img = np.full((1, 3, IMAGE_SIZE, IMAGE_SIZE), -1.0, dtype=np.float32)
    obs_dict = {
        "image": {
            "base_0_rgb": torch.from_numpy(inputs["img_proc"]).to(device),
            "left_wrist_0_rgb": torch.from_numpy(inputs["img_proc"]).to(device),
            "right_wrist_0_rgb": torch.from_numpy(dummy_img).to(device),
        },
        "image_mask": {
            "base_0_rgb": torch.tensor([True], device=device),
            "left_wrist_0_rgb": torch.tensor([True], device=device),
            "right_wrist_0_rgb": torch.tensor([False], device=device),
        },
        "state": torch.from_numpy(inputs["norm_state"][np.newaxis]).to(device),
        "tokenized_prompt": torch.from_numpy(inputs["tokens"][np.newaxis]).to(device),
        "tokenized_prompt_mask": torch.from_numpy(inputs["token_mask"][np.newaxis]).to(device),
    }
    observation = _model.Observation.from_dict(obs_dict)

    def run_pytorch_inference():
        with torch.no_grad():
            return model.sample_actions(device, observation, num_steps=NUM_STEPS)

    log.info("Warming up PyTorch (%d runs, includes compile graph capture)...", N_WARMUP)
    for i in range(N_WARMUP):
        torch.cuda.synchronize()
        t0 = time.monotonic()
        _ = run_pytorch_inference()
        torch.cuda.synchronize()
        log.info("  Warmup %d/%d: %.1f ms", i + 1, N_WARMUP, (time.monotonic() - t0) * 1000)

    log.info("Timed PyTorch runs (%d)...", N_RUNS)
    timings = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        t0 = time.monotonic()
        actions = run_pytorch_inference()
        torch.cuda.synchronize()
        timings.append((time.monotonic() - t0) * 1000)

    timings = np.array(timings)
    log.info("PyTorch SDPA+compile: Mean=%.1f ms  Median=%.1f ms  Min=%.1f ms  P95=%.1f ms  (%.1f Hz)",
             timings.mean(), np.median(timings), timings.min(),
             np.percentile(timings, 95), 1000 / timings.mean())

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return timings


# ---------------------------------------------------------------------------
# Also benchmark PyTorch eager (baseline) for 3-way comparison
# ---------------------------------------------------------------------------

def benchmark_pytorch_eager(inputs, normalizer):
    """Benchmark PyTorch eager (no SDPA, no compile) as baseline."""
    log.info("=" * 65)
    log.info("PyTorch EAGER BASELINE BENCHMARK (no SDPA, no compile)")
    log.info("=" * 65)

    import importlib
    from transformers.models.gemma import modeling_gemma
    importlib.reload(modeling_gemma)

    import dataclasses
    import safetensors.torch
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    import openpi.models_pytorch.pi0_pytorch as _pi0
    from openpi.models import model as _model

    _orig = _pi0.get_safe_dtype
    def _patched(target_dtype, device_type):
        if target_dtype == torch.float64:
            return torch.float32
        return _orig(target_dtype, device_type)
    _pi0.get_safe_dtype = _patched

    @dataclasses.dataclass
    class ModelConfig:
        pi05: bool = True
        action_dim: int = ACTION_DIM
        action_horizon: int = ACTION_HORIZON
        paligemma_variant: str = "gemma_2b_lora"
        action_expert_variant: str = "gemma_300m_lora"
        dtype: str = "bfloat16"
        state_dim: int = ACTION_DIM
        max_token_len: int = MAX_TOKEN_LEN
        discrete_state_input: bool = True

    device = torch.device("cuda")

    log.info("Creating model (eager, no compile)...")
    orig_compile = torch.compile
    torch.compile = lambda fn, **kw: fn
    try:
        model = PI0Pytorch(ModelConfig())
    finally:
        torch.compile = orig_compile

    log.info("Loading weights...")
    safetensors.torch.load_model(model, f"{CHECKPOINT_DIR}/model.safetensors", device="cuda")
    model.eval()
    model.to(device)

    dummy_img = np.full((1, 3, IMAGE_SIZE, IMAGE_SIZE), -1.0, dtype=np.float32)
    obs_dict = {
        "image": {
            "base_0_rgb": torch.from_numpy(inputs["img_proc"]).to(device),
            "left_wrist_0_rgb": torch.from_numpy(inputs["img_proc"]).to(device),
            "right_wrist_0_rgb": torch.from_numpy(dummy_img).to(device),
        },
        "image_mask": {
            "base_0_rgb": torch.tensor([True], device=device),
            "left_wrist_0_rgb": torch.tensor([True], device=device),
            "right_wrist_0_rgb": torch.tensor([False], device=device),
        },
        "state": torch.from_numpy(inputs["norm_state"][np.newaxis]).to(device),
        "tokenized_prompt": torch.from_numpy(inputs["tokens"][np.newaxis]).to(device),
        "tokenized_prompt_mask": torch.from_numpy(inputs["token_mask"][np.newaxis]).to(device),
    }
    observation = _model.Observation.from_dict(obs_dict)

    def run_inference():
        with torch.no_grad():
            return model.sample_actions(device, observation, num_steps=NUM_STEPS)

    log.info("Warming up eager PyTorch (%d runs)...", N_WARMUP)
    for i in range(N_WARMUP):
        torch.cuda.synchronize()
        t0 = time.monotonic()
        _ = run_inference()
        torch.cuda.synchronize()
        log.info("  Warmup %d/%d: %.1f ms", i + 1, N_WARMUP, (time.monotonic() - t0) * 1000)

    log.info("Timed eager runs (%d)...", N_RUNS)
    timings = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        t0 = time.monotonic()
        _ = run_inference()
        torch.cuda.synchronize()
        timings.append((time.monotonic() - t0) * 1000)

    timings = np.array(timings)
    log.info("PyTorch Eager: Mean=%.1f ms  Median=%.1f ms  Min=%.1f ms  P95=%.1f ms  (%.1f Hz)",
             timings.mean(), np.median(timings), timings.min(),
             np.percentile(timings, 95), 1000 / timings.mean())

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return timings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_comparison(results):
    """Print comparison table."""
    log.info("")
    log.info("=" * 70)
    log.info("COMPARISON RESULTS")
    log.info("=" * 70)
    log.info("%-35s %8s %8s %8s %8s", "Backend", "Mean", "Median", "Min", "Hz")
    log.info("-" * 70)

    for name, timings in results.items():
        if timings is not None:
            log.info("%-35s %7.1fms %7.1fms %7.1fms %7.1f",
                     name, timings.mean(), np.median(timings),
                     timings.min(), 1000 / timings.mean())
        else:
            log.info("%-35s %s", name, "SKIPPED (deps not available)")

    log.info("-" * 70)

    valid = {k: v for k, v in results.items() if v is not None}
    if len(valid) >= 2:
        best_name = min(valid, key=lambda k: valid[k].mean())
        worst_name = max(valid, key=lambda k: valid[k].mean())
        speedup = valid[worst_name].mean() / valid[best_name].mean()
        log.info("Fastest: %s (%.1fx faster than %s)", best_name, speedup, worst_name)

    log.info("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-trt", action="store_true", help="Skip TRT benchmark")
    parser.add_argument("--skip-eager", action="store_true", help="Skip eager baseline")
    parser.add_argument("--skip-sdpa", action="store_true", help="Skip SDPA+compile benchmark")
    parser.add_argument("--compile-mode", default="max-autotune",
                        choices=["max-autotune", "reduce-overhead", "default", "none"])
    args = parser.parse_args()

    log.info("Jetson Thor Benchmark: TRT FP16 vs PyTorch SDPA+compile")
    log.info("GPU: %s", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

    normalizer = QuantileNormalizer(NORM_STATS)
    inputs = make_dummy_inputs(normalizer)
    log.info("Inputs prepared: state=%s, tokens=%s, img=%s",
             inputs["norm_state"].shape, inputs["tokens"].shape, inputs["img_proc"].shape)

    results = {}

    if not args.skip_trt:
        results["TRT FP16"] = benchmark_trt_fp16(inputs, normalizer)
        gc.collect()
        time.sleep(2)

    if not args.skip_eager:
        results["PyTorch Eager (baseline)"] = benchmark_pytorch_eager(inputs, normalizer)
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)

    if not args.skip_sdpa:
        results[f"PyTorch SDPA+compile({args.compile_mode})"] = benchmark_pytorch_sdpa(
            inputs, normalizer, compile_mode=args.compile_mode)

    print_comparison(results)


if __name__ == "__main__":
    main()
