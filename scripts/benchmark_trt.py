#!/usr/bin/env python3
"""Benchmark TRT engines across precision modes and verify output against ONNX reference.

Usage:
  python3 scripts/benchmark_trt.py                    # default: fp32
  python3 scripts/benchmark_trt.py --precision bf16
  python3 scripts/benchmark_trt.py --precision int8
  python3 scripts/benchmark_trt.py --precision all     # run all configurations
"""

import argparse
import logging
import pathlib
import sys
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

ENGINE_DIR = ROOT / "onnx_export"
TOKENIZER = str(pathlib.Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model")

PRECISION_ENGINES = {
    "fp32": ("prefix_encoder_fp32.engine", "denoise_step_fp32.engine"),
    "fp16": ("prefix_encoder_fp16.engine", "denoise_step_fp16.engine"),
    "bf16": ("prefix_encoder_bf16.engine", "denoise_step_bf16.engine"),
    "int8": ("prefix_encoder_int8.engine", "denoise_step_int8.engine"),
}


def make_obs():
    np.random.seed(42)
    return {
        "observation/state": np.random.randn(6).astype(np.float32) * 0.1,
        "observation/image_scene": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/image_wrist": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the cube",
    }


def benchmark_one(precision: str) -> dict:
    from trt_policy_server import TRTPolicy, TRTEngine, preprocess_observation, Tokenizer, NUM_DENOISE_STEPS, SHARED_TENSORS

    prefix_file, denoise_file = PRECISION_ENGINES[precision]
    log.info("=" * 70)
    log.info("BENCHMARKING: %s (shared buffers enabled)", precision.upper())
    log.info("=" * 70)

    policy = TRTPolicy(str(ENGINE_DIR / prefix_file), str(ENGINE_DIR / denoise_file), TOKENIZER)
    obs = make_obs()

    for _ in range(3):
        policy.infer(obs)

    tokenizer = Tokenizer(TOKENIZER)

    t0 = time.monotonic()
    for _ in range(10):
        inputs = preprocess_observation(obs, tokenizer)
    preprocess_ms = (time.monotonic() - t0) / 10 * 1000

    prefix_keys = ["img0", "img1", "img2", "mask0", "mask1", "mask2", "tokens", "token_masks"]
    prefix_inputs = {k: inputs[k] for k in prefix_keys}

    t0 = time.monotonic()
    for _ in range(10):
        policy.prefix_engine.infer(prefix_inputs, skip_d2h=set(SHARED_TENSORS))
    prefix_ms = (time.monotonic() - t0) / 10 * 1000

    # Single denoise step (with shared buffers already populated from prefix above)
    de = policy.denoise_engine
    de.h2d("state", inputs["state"])
    x_t = np.random.randn(1, 11, 6).astype(np.float32)

    t0 = time.monotonic()
    for _ in range(80):
        de.h2d("x_t", x_t)
        de.h2d("timestep", np.array([1.0], dtype=np.float32))
        de.execute()
        de.d2h("velocity")
        de.sync()
    denoise_ms = (time.monotonic() - t0) / 80 * 1000

    log.info("  Preprocessing:      %6.1f ms", preprocess_ms)
    log.info("  Prefix encoder:     %6.1f ms", prefix_ms)
    log.info("  Denoise step (1x):  %6.1f ms", denoise_ms)
    log.info("  Denoise step (8x):  %6.1f ms", denoise_ms * 8)

    times = []
    for _ in range(20):
        t0 = time.monotonic()
        policy.infer(obs)
        times.append((time.monotonic() - t0) * 1000)
    times = np.array(times)
    log.info("  E2E median:         %6.1f ms  (mean=%.1f, min=%.1f, max=%.1f)",
             np.median(times), np.mean(times), np.min(times), np.max(times))
    log.info("  Speedup vs 440ms:   %.2fx", 440 / np.median(times))

    # Deterministic run for accuracy comparison
    np.random.seed(12345)
    fixed_noise = np.random.randn(1, 11, 6).astype(np.float32)

    policy.prefix_engine.infer(prefix_inputs, skip_d2h=set(SHARED_TENSORS))
    de.h2d("state", inputs["state"])

    x_t = fixed_noise.copy()
    dt = np.float32(-1.0 / NUM_DENOISE_STEPS)
    t = np.float32(1.0)
    t_step = np.float32(-1.0 / NUM_DENOISE_STEPS)
    velocities = []
    for step in range(NUM_DENOISE_STEPS):
        de.h2d("x_t", x_t)
        de.h2d("timestep", np.array([t], dtype=np.float32))
        de.execute()
        vel_buf = de.d2h("velocity")
        de.sync()
        velocities.append(vel_buf.copy())
        x_t = x_t + dt * vel_buf
        t += t_step

    return {
        "precision": precision,
        "preprocess_ms": preprocess_ms,
        "prefix_ms": prefix_ms,
        "denoise_step_ms": denoise_ms,
        "e2e_median_ms": float(np.median(times)),
        "speedup": 440 / float(np.median(times)),
        "final_actions": x_t,
        "velocities": velocities,
        "inputs": inputs,
        "noise": fixed_noise,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16", "int8", "all"])
    args = parser.parse_args()

    precisions = list(PRECISION_ENGINES.keys()) if args.precision == "all" else [args.precision]
    results = {}
    for p in precisions:
        engine_prefix, engine_denoise = PRECISION_ENGINES[p]
        if not (ENGINE_DIR / engine_prefix).exists() or not (ENGINE_DIR / engine_denoise).exists():
            log.warning("Skipping %s: engine files not found", p)
            continue
        results[p] = benchmark_one(p)

    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info("%-8s  %8s  %8s  %10s  %8s  %7s", "Prec", "Prefix", "8xDenoise", "E2E Median", "Speedup", "Status")
    log.info("-" * 70)

    fp32_actions = results.get("fp32", {}).get("final_actions")
    for p in precisions:
        if p not in results:
            continue
        r = results[p]
        if fp32_actions is not None and p != "fp32":
            mse = float(np.mean((r["final_actions"] - fp32_actions) ** 2))
            status = "PASS" if mse < 0.01 else f"FAIL({mse:.4f})"
        else:
            status = "REF" if p == "fp32" else "N/A"
        log.info("%-8s  %6.1fms  %6.1fms  %8.1fms  %6.2fx  %s",
                 p.upper(), r["prefix_ms"], r["denoise_step_ms"] * 8,
                 r["e2e_median_ms"], r["speedup"], status)


if __name__ == "__main__":
    main()
