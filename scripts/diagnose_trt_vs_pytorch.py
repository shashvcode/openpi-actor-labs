#!/usr/bin/env python3
"""Compare TRT prefix encoder output vs PyTorch prefix encoder output.

Runs the same preprocessed input through both paths and compares KV caches
to pinpoint where the ONNX/TRT export diverges from PyTorch.

Run on host (uses TRT engines + sends to Docker for PyTorch comparison).
"""
import json
import logging
import sys
import time
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, "src")

ACTION_DIM = 6
ACTION_HORIZON = 11
IMAGE_SIZE = 224


def preprocess_like_trt_server(scene_rgb, wrist_rgb, state, prompt, norm_stats_path):
    """Exactly replicate the TRT server's preprocessing."""
    from PIL import Image as PILImage
    import sentencepiece

    with open(norm_stats_path) as f:
        raw = json.load(f)["norm_stats"]
    state_q01 = np.array(raw["state"]["q01"], dtype=np.float32)
    state_q99 = np.array(raw["state"]["q99"], dtype=np.float32)

    norm_state = (state - state_q01[:6]) / (state_q99[:6] - state_q01[:6] + 1e-6) * 2.0 - 1.0

    def preprocess_image(image):
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 127.5 - 1.0
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        if image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            cur_h, cur_w = image.shape[:2]
            ratio = max(cur_w / IMAGE_SIZE, cur_h / IMAGE_SIZE)
            new_h, new_w = int(cur_h / ratio), int(cur_w / ratio)
            img_u8 = np.clip((image + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)
            pil = PILImage.fromarray(img_u8).resize((new_w, new_h), PILImage.BILINEAR)
            resized = np.array(pil).astype(np.float32) / 255.0 * 2.0 - 1.0
            pad_h0 = (IMAGE_SIZE - new_h) // 2
            pad_h1 = IMAGE_SIZE - new_h - pad_h0
            pad_w0 = (IMAGE_SIZE - new_w) // 2
            pad_w1 = IMAGE_SIZE - new_w - pad_w0
            image = np.pad(resized, ((pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
                           mode="constant", constant_values=-1.0)
        image = np.transpose(image, (2, 0, 1))
        return image[np.newaxis].astype(np.float32)

    sp = sentencepiece.SentencePieceProcessor(
        model_file=str(__import__("pathlib").Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model"))
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    discretized = np.digitize(norm_state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
    state_str = " ".join(map(str, discretized))
    full_prompt = f"Task: {cleaned}, State: {state_str};\nAction: "
    tokens_raw = sp.encode(full_prompt, add_bos=True)
    pad_len = 200 - len(tokens_raw)
    tokens = np.array(tokens_raw + [0] * pad_len, dtype=np.int64)
    mask = np.array([True] * len(tokens_raw) + [False] * pad_len, dtype=bool)

    return {
        "img0": preprocess_image(scene_rgb),
        "img1": preprocess_image(wrist_rgb),
        "img2": np.full((1, 3, IMAGE_SIZE, IMAGE_SIZE), -1.0, dtype=np.float32),
        "mask0": np.array([True]),
        "mask1": np.array([True]),
        "mask2": np.array([False]),
        "tokens": tokens[np.newaxis],
        "token_masks": mask[np.newaxis],
        "state": norm_state[np.newaxis],
    }


def run_trt_inference(inputs, prefix_engine_path, denoise_engine_path):
    """Run full TRT inference and return actions + intermediate KV cache stats."""
    import tensorrt as trt
    from cuda.bindings import runtime as cudart

    sys.path.insert(0, "scripts")
    from trt_policy_server import TRTEngine, SHARED_TENSORS

    prefix_engine = TRTEngine(prefix_engine_path)
    shared = {name: prefix_engine.get_device_ptr(name) for name in SHARED_TENSORS}
    denoise_engine = TRTEngine(denoise_engine_path, shared_device_buffers=shared)

    prefix_out = prefix_engine.infer(
        {k: inputs[k] for k in ["img0", "img1", "img2", "mask0", "mask1", "mask2", "tokens", "token_masks"]},
        skip_d2h=set(SHARED_TENSORS),
    )

    prefix_pad_masks = prefix_engine.d2h("prefix_pad_masks")
    kv_keys = prefix_engine.d2h("kv_keys")
    kv_values = prefix_engine.d2h("kv_values")
    prefix_engine.sync()

    prefix_pad_masks = prefix_pad_masks.copy()
    kv_keys = kv_keys.copy()
    kv_values = kv_values.copy()

    log.info("TRT prefix_pad_masks: sum=%d, shape=%s", prefix_pad_masks.sum(), prefix_pad_masks.shape)
    log.info("TRT kv_keys: mean=%.6f, std=%.6f, shape=%s", kv_keys.mean(), kv_keys.std(), kv_keys.shape)
    log.info("TRT kv_values: mean=%.6f, std=%.6f, shape=%s", kv_values.mean(), kv_values.std(), kv_values.shape)

    np.random.seed(42)
    x_t = np.random.standard_normal((1, ACTION_HORIZON, ACTION_DIM)).astype(np.float32)
    log.info("Noise x_t[0,0]: %s", x_t[0, 0])

    de = denoise_engine
    de.h2d("state", inputs["state"])

    dt = np.float32(-1.0 / 8)
    t = np.float32(1.0)
    t_step = np.float32(-1.0 / 8)
    for step in range(8):
        de.h2d("x_t", x_t)
        de.h2d("timestep", np.array([t], dtype=np.float32))
        de.execute()
        vel_buf = de.d2h("velocity")
        de.sync()
        vel = vel_buf.copy()
        if step == 0:
            log.info("TRT step0 velocity[0,0]: %s", vel[0, 0])
        x_t = x_t + dt * vel
        t += t_step

    return x_t[0], kv_keys, kv_values, prefix_pad_masks


def main():
    data = np.load("/tmp/test_obs_real.npz")
    prompt = "Pick up the bottle and place it on the yellow outlined square."
    norm_stats_path = "checkpoints/runD_qat_pytorch/assets/verm11/runA/norm_stats.json"

    inputs = preprocess_like_trt_server(data["scene"], data["wrist"], data["state"], prompt, norm_stats_path)

    log.info("=== Input stats ===")
    log.info("img0: mean=%.4f std=%.4f", inputs["img0"].mean(), inputs["img0"].std())
    log.info("img1: mean=%.4f std=%.4f", inputs["img1"].mean(), inputs["img1"].std())
    log.info("tokens[:10]: %s", inputs["tokens"][0, :10])
    log.info("state: %s", inputs["state"])

    np.savez("/tmp/trt_preprocessed_inputs.npz", **{k: v for k, v in inputs.items()})
    log.info("Saved preprocessed inputs to /tmp/trt_preprocessed_inputs.npz")

    log.info("\n=== TRT FP32 inference ===")
    trt_actions, trt_kv_keys, trt_kv_values, trt_masks = run_trt_inference(
        inputs,
        "onnx_export/prefix_encoder_fp32_qat.engine",
        "onnx_export/denoise_step_fp32_qat.engine",
    )
    log.info("TRT FP32 final actions[0]: %s", trt_actions[0])
    log.info("TRT FP32 final actions[5]: %s", trt_actions[5])

    np.savez("/tmp/trt_intermediates.npz",
             actions=trt_actions, kv_keys=trt_kv_keys, kv_values=trt_kv_values, masks=trt_masks)
    log.info("Saved TRT intermediates to /tmp/trt_intermediates.npz")
    log.info("Now run the PyTorch comparison inside Docker with the same preprocessed inputs.")


if __name__ == "__main__":
    main()
