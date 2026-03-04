#!/usr/bin/env python3
"""Generate calibration data for INT8 TensorRT engine compilation.

Produces diverse observations and runs them through the ONNX prefix encoder
to generate KV-cache values needed for denoise-step calibration.

Must run inside the Docker container where onnxruntime is available:
  docker exec trt_pipeline bash -c "cd /workspace/openpi && python scripts/generate_calibration_data.py"
"""

import logging
import pathlib
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

NUM_SAMPLES = 200
IMAGE_SIZE = 224
ACTION_DIM = 6
ACTION_HORIZON = 11
MAX_TOKEN_LEN = 200
NUM_DENOISE_STEPS = 8

PROMPTS = [
    "pick up the cube",
    "place the block on the plate",
    "move to the left",
    "push the object forward",
    "grasp the red ball",
    "lift the cup",
    "stack the blocks",
    "put the toy in the bin",
    "reach for the handle",
    "turn the knob clockwise",
    "slide the drawer open",
    "press the button",
    "wave the arm",
    "return to home position",
    "pick up the screw",
    "tighten the bolt",
    "sweep the surface",
    "pour from the container",
    "flip the switch",
    "hold the object steady",
]


def generate_diverse_image(rng: np.random.Generator, idx: int) -> np.ndarray:
    """Generate a diverse 224x224x3 uint8 image using different patterns."""
    pattern = idx % 8
    h, w = IMAGE_SIZE, IMAGE_SIZE

    if pattern == 0:
        img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    elif pattern == 1:
        r = np.linspace(0, 255, h, dtype=np.uint8)[:, None, None]
        g = np.linspace(0, 255, w, dtype=np.uint8)[None, :, None]
        b = np.full((h, w, 1), 128, dtype=np.uint8)
        img = np.concatenate([r * np.ones((1, w, 1), dtype=np.uint8),
                              g * np.ones((h, 1, 1), dtype=np.uint8), b], axis=2)
    elif pattern == 2:
        color = rng.integers(0, 256, 3, dtype=np.uint8)
        img = np.full((h, w, 3), color, dtype=np.uint8)
    elif pattern == 3:
        base = rng.integers(50, 200, (h, w, 3), dtype=np.uint8)
        cx, cy = rng.integers(30, 194, 2)
        radius = rng.integers(20, 60)
        yy, xx = np.ogrid[:h, :w]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < radius ** 2
        base[mask] = rng.integers(0, 256, 3, dtype=np.uint8)
        img = base
    elif pattern == 4:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        stripe_w = rng.integers(5, 30)
        for s in range(0, w, stripe_w * 2):
            img[:, s:s + stripe_w] = rng.integers(100, 256, 3, dtype=np.uint8)
    elif pattern == 5:
        noise = rng.normal(128, 40, (h, w, 3))
        img = np.clip(noise, 0, 255).astype(np.uint8)
    elif pattern == 6:
        block_size = rng.integers(16, 56)
        small = rng.integers(0, 256, (h // block_size + 1, w // block_size + 1, 3), dtype=np.uint8)
        img = np.repeat(np.repeat(small, block_size, axis=0), block_size, axis=1)[:h, :w]
    else:
        img = np.full((h, w, 3), 128, dtype=np.uint8)
        n_rects = rng.integers(3, 10)
        for _ in range(n_rects):
            x1, y1 = rng.integers(0, 180, 2)
            x2, y2 = x1 + rng.integers(10, 44), y1 + rng.integers(10, 44)
            img[y1:y2, x1:x2] = rng.integers(0, 256, 3, dtype=np.uint8)

    return img


def preprocess_image_np(image: np.ndarray) -> np.ndarray:
    """uint8 HWC -> float32 [1,3,H,W] in [-1,1]."""
    img = image.astype(np.float32) / 127.5 - 1.0
    img = np.transpose(img, (2, 0, 1))
    return img[np.newaxis]


def main():
    import sentencepiece

    tokenizer_path = str(pathlib.Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model")
    sp = sentencepiece.SentencePieceProcessor(model_file=tokenizer_path)

    out_dir = ROOT / "calibration_data"
    prefix_dir = out_dir / "prefix"
    denoise_dir = out_dir / "denoise"
    prefix_dir.mkdir(parents=True, exist_ok=True)
    denoise_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=2026)

    log.info("Generating %d calibration samples...", NUM_SAMPLES)

    prefix_samples = []
    for i in range(NUM_SAMPLES):
        state = rng.uniform(-1, 1, ACTION_DIM).astype(np.float32)

        prompt = PROMPTS[i % len(PROMPTS)]
        discretized = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        state_str = " ".join(map(str, discretized))
        full_prompt = f"Task: {prompt}, State: {state_str};\nAction: "
        tokens = sp.encode(full_prompt, add_bos=True)

        if len(tokens) < MAX_TOKEN_LEN:
            pad_len = MAX_TOKEN_LEN - len(tokens)
            mask = [True] * len(tokens) + [False] * pad_len
            tokens = tokens + [0] * pad_len
        else:
            tokens = tokens[:MAX_TOKEN_LEN]
            mask = [True] * MAX_TOKEN_LEN

        tokens_arr = np.asarray(tokens, dtype=np.int64)[np.newaxis]
        mask_arr = np.asarray(mask, dtype=np.bool_)[np.newaxis]

        img_scene = generate_diverse_image(rng, i)
        img_wrist = generate_diverse_image(rng, i + 100)

        sample = {
            "img0": preprocess_image_np(img_scene),
            "img1": preprocess_image_np(img_wrist),
            "img2": np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32),
            "mask0": np.array([True]),
            "mask1": np.array([True]),
            "mask2": np.array([False]),
            "tokens": tokens_arr,
            "token_masks": mask_arr,
            "state": state[np.newaxis],
        }
        prefix_samples.append(sample)

        if (i + 1) % 50 == 0:
            log.info("  Generated %d/%d observations", i + 1, NUM_SAMPLES)

    log.info("Saving prefix calibration inputs...")
    for key in prefix_samples[0]:
        if key == "state":
            continue
        data = np.concatenate([s[key] for s in prefix_samples], axis=0)
        np.save(prefix_dir / f"{key}.npy", data)
        log.info("  %s: %s %s", key, data.shape, data.dtype)

    states = np.concatenate([s["state"] for s in prefix_samples], axis=0)
    np.save(denoise_dir / "state.npy", states)
    log.info("  state: %s %s", states.shape, states.dtype)

    log.info("Running prefix encoder on calibration data to get KV cache...")
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.log_severity_level = 3

    prefix_sess = ort.InferenceSession(
        str(ROOT / "onnx_export" / "prefix_encoder.onnx"), opts,
        providers=["CPUExecutionProvider"],
    )

    all_kv_keys = []
    all_kv_values = []
    all_pad_masks = []

    for i in range(NUM_SAMPLES):
        s = prefix_samples[i]
        prefix_in = {k: s[k] for k in ["img0", "img1", "img2", "mask0", "mask1", "mask2", "tokens", "token_masks"]}
        pad_masks, kv_keys, kv_values = prefix_sess.run(None, prefix_in)
        all_pad_masks.append(pad_masks)
        all_kv_keys.append(kv_keys)
        all_kv_values.append(kv_values)

        if (i + 1) % 20 == 0:
            log.info("  Prefix inference %d/%d", i + 1, NUM_SAMPLES)

    log.info("Generating denoise calibration inputs (varied timesteps and noisy actions)...")
    denoise_samples_per_prefix = 4
    all_denoise_x_t = []
    all_denoise_timesteps = []
    all_denoise_kv_keys = []
    all_denoise_kv_values = []
    all_denoise_pad_masks = []
    all_denoise_states = []

    timestep_values = np.linspace(1.0, 1.0 / NUM_DENOISE_STEPS, NUM_DENOISE_STEPS).astype(np.float32)

    for i in range(NUM_SAMPLES):
        for j in range(denoise_samples_per_prefix):
            x_t = rng.standard_normal((1, ACTION_HORIZON, ACTION_DIM)).astype(np.float32)
            t_idx = (i * denoise_samples_per_prefix + j) % NUM_DENOISE_STEPS
            timestep = np.array([timestep_values[t_idx]], dtype=np.float32)

            all_denoise_x_t.append(x_t)
            all_denoise_timesteps.append(timestep)
            all_denoise_kv_keys.append(all_kv_keys[i])
            all_denoise_kv_values.append(all_kv_values[i])
            all_denoise_pad_masks.append(all_pad_masks[i])
            all_denoise_states.append(prefix_samples[i]["state"])

    log.info("Saving denoise calibration inputs (%d samples)...", len(all_denoise_x_t))
    np.save(denoise_dir / "x_t.npy", np.concatenate(all_denoise_x_t, axis=0))
    np.save(denoise_dir / "timestep.npy", np.stack([t[0] for t in all_denoise_timesteps]))
    np.save(denoise_dir / "state.npy", np.concatenate(all_denoise_states, axis=0))

    kv_keys_stacked = np.stack(all_denoise_kv_keys, axis=0)
    kv_values_stacked = np.stack(all_denoise_kv_values, axis=0)
    pad_masks_stacked = np.concatenate(all_denoise_pad_masks, axis=0)

    log.info("  kv_keys: %s (%.1f MB)", kv_keys_stacked.shape,
             kv_keys_stacked.nbytes / 1e6)
    log.info("  kv_values: %s (%.1f MB)", kv_values_stacked.shape,
             kv_values_stacked.nbytes / 1e6)

    np.save(denoise_dir / "kv_keys.npy", kv_keys_stacked)
    np.save(denoise_dir / "kv_values.npy", kv_values_stacked)
    np.save(denoise_dir / "prefix_pad_masks.npy", pad_masks_stacked)

    log.info("Calibration data saved to %s", out_dir)
    log.info("  Prefix samples: %d", NUM_SAMPLES)
    log.info("  Denoise samples: %d", len(all_denoise_x_t))


if __name__ == "__main__":
    main()
