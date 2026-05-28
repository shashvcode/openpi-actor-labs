#!/usr/bin/env python3
"""Directly run the TRT engines with fixed noise for comparison with PyTorch.

Run on the Jetson host (not in Docker):
  python3 scripts/compare_trt_direct.py
"""

import json
import sys
import time

import numpy as np
import sentencepiece

sys.path.insert(0, "src")

CHECKPOINT_DIR = "checkpoints/runC_pytorch"
NORM_STATS_PATH = f"{CHECKPOINT_DIR}/assets/verm11/runA/norm_stats.json"
TOKENIZER_MODEL = "/home/Actor/.cache/openpi/big_vision/paligemma_tokenizer.model"
ENGINE_DIR = "onnx_export"
TEST_OBS_PATH = "/tmp/test_obs.npz"

ACTION_DIM = 6
ACTION_HORIZON = 11
NUM_STEPS = 8
IMAGE_SIZE = 224
MAX_TOKEN_LEN = 200


def load_norm_stats():
    with open(NORM_STATS_PATH) as f:
        raw = json.load(f)["norm_stats"]
    return {
        "state_q01": np.array(raw["state"]["q01"], dtype=np.float32),
        "state_q99": np.array(raw["state"]["q99"], dtype=np.float32),
        "action_q01": np.array(raw["actions"]["q01"], dtype=np.float32),
        "action_q99": np.array(raw["actions"]["q99"], dtype=np.float32),
    }


def normalize_state(state, ns):
    q01, q99 = ns["state_q01"][:state.shape[-1]], ns["state_q99"][:state.shape[-1]]
    return (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def unnormalize_actions(actions, ns):
    q01, q99 = ns["action_q01"], ns["action_q99"]
    return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def resize_with_pad(image, height, width):
    from PIL import Image as PILImage
    cur_h, cur_w = image.shape[:2]
    is_float = np.issubdtype(image.dtype, np.floating)
    if is_float:
        img_uint8 = np.clip((image + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)
    else:
        img_uint8 = image
    pil = PILImage.fromarray(img_uint8)
    ratio = max(cur_w / width, cur_h / height)
    new_w, new_h = int(cur_w / ratio), int(cur_h / ratio)
    pil = pil.resize((new_w, new_h), PILImage.BILINEAR)
    resized = np.array(pil)
    if is_float:
        resized = resized.astype(np.float32) / 255.0 * 2.0 - 1.0
        pad_val = -1.0
    else:
        pad_val = 0
    pad_h0 = (height - new_h) // 2
    pad_h1 = height - new_h - pad_h0
    pad_w0 = (width - new_w) // 2
    pad_w1 = width - new_w - pad_w0
    return np.pad(resized, ((pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
                  mode="constant", constant_values=pad_val)


def prep_img(img_uint8):
    img = img_uint8.astype(np.float32) / 127.5 - 1.0
    if img.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        img = resize_with_pad(img, IMAGE_SIZE, IMAGE_SIZE)
    return np.transpose(img, (2, 0, 1))[np.newaxis].astype(np.float32)


def main():
    sys.path.insert(0, "scripts")
    from trt_policy_server import TRTEngine

    ns = load_norm_stats()

    data = np.load(TEST_OBS_PATH)
    scene_img = data["scene"]
    wrist_img = data["wrist"]
    raw_state = np.zeros(ACTION_DIM, dtype=np.float32)

    norm_state = normalize_state(raw_state, ns)
    print(f"Normalized state: {norm_state}")

    # Tokenize
    sp = sentencepiece.SentencePieceProcessor(model_file=TOKENIZER_MODEL)
    prompt = "Pick up the bottle and place it on the yellow outlined square."
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    discretized = np.digitize(norm_state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
    state_str = " ".join(map(str, discretized))
    full_prompt = f"Task: {cleaned}, State: {state_str};\nAction: "
    tokens_raw = sp.encode(full_prompt, add_bos=True)

    if len(tokens_raw) < MAX_TOKEN_LEN:
        pad_len = MAX_TOKEN_LEN - len(tokens_raw)
        mask = [True] * len(tokens_raw) + [False] * pad_len
        tokens_list = tokens_raw + [0] * pad_len
    else:
        tokens_list = tokens_raw[:MAX_TOKEN_LEN]
        mask = [True] * MAX_TOKEN_LEN

    tokens = np.array(tokens_list, dtype=np.int64)[np.newaxis]
    token_masks = np.array(mask, dtype=bool)[np.newaxis]

    # Prep images
    scene_proc = prep_img(scene_img)
    wrist_proc = prep_img(wrist_img)
    dummy_proc = np.full((1, 3, IMAGE_SIZE, IMAGE_SIZE), -1.0, dtype=np.float32)

    print(f"Scene: shape={scene_proc.shape} range=[{scene_proc.min():.3f}, {scene_proc.max():.3f}]")
    print(f"Wrist: shape={wrist_proc.shape} range=[{wrist_proc.min():.3f}, {wrist_proc.max():.3f}]")

    # Load TRT engines
    SHARED_TENSORS = ("kv_keys", "kv_values", "prefix_pad_masks")

    print("\nLoading prefix encoder...")
    prefix_engine = TRTEngine(f"{ENGINE_DIR}/prefix_encoder_bf16.engine")

    shared = {name: prefix_engine.get_device_ptr(name) for name in SHARED_TENSORS}
    print("Loading denoise step...")
    denoise_engine = TRTEngine(f"{ENGINE_DIR}/denoise_step_bf16.engine", shared_device_buffers=shared)

    # Run prefix encoder
    print("\nRunning prefix encoder...")
    prefix_engine.infer(
        {
            "img0": scene_proc,
            "img1": wrist_proc,
            "img2": dummy_proc,
            "mask0": np.array([True]),
            "mask1": np.array([True]),
            "mask2": np.array([False]),
            "tokens": tokens,
            "token_masks": token_masks,
        },
        skip_d2h=set(SHARED_TENSORS),
    )

    # Run denoise with FIXED seed (matching PyTorch seed 42)
    print("\n=== Running TRT denoise with fixed seed 42 ===")
    rng = np.random.default_rng(42)
    x_t = rng.standard_normal((1, ACTION_HORIZON, ACTION_DIM)).astype(np.float32)
    print(f"Initial noise[0,0]: {x_t[0, 0]}")

    de = denoise_engine
    de.h2d("state", norm_state[np.newaxis])

    dt = np.float32(-1.0 / NUM_STEPS)
    t = np.float32(1.0)
    t_step = np.float32(-1.0 / NUM_STEPS)

    for step_i in range(NUM_STEPS):
        de.h2d("x_t", x_t)
        de.h2d("timestep", np.array([t], dtype=np.float32))
        de.execute()
        vel_buf = de.d2h("velocity")
        de.sync()
        if step_i == 0:
            print(f"Step 0 velocity[0,0]: {vel_buf[0, 0]}")
        x_t = x_t + dt * vel_buf
        t += t_step

    raw_trt = x_t[0, :, :ACTION_DIM]
    print(f"\nTRT raw output range: [{raw_trt.min():.4f}, {raw_trt.max():.4f}]")
    print(f"TRT raw action[0]: {raw_trt[0]}")

    unnorm_trt = unnormalize_actions(raw_trt, ns)
    print(f"TRT unnorm action[0]: {unnorm_trt[0]}")
    print(f"TRT unnorm range: [{unnorm_trt.min():.4f}, {unnorm_trt.max():.4f}]")

    # PyTorch reference (from seed 42 run)
    pytorch_raw = np.array([0.15211387, 1.0433533, -0.22778904, 0.7042536, -1.5304453, 0.5512792])
    pytorch_unnorm = np.array([0.15188396, 1.0429456, -0.2554338, 0.70391357, -0.2651699, 0.7754853])

    print(f"\n=== Comparison (seed 42) ===")
    print(f"PyTorch raw[0]:  {pytorch_raw}")
    print(f"TRT raw[0]:      {raw_trt[0]}")
    print(f"Difference:      {np.abs(raw_trt[0] - pytorch_raw)}")
    print(f"Max abs diff:    {np.max(np.abs(raw_trt[0] - pytorch_raw)):.6f}")

    print(f"\nPyTorch unnorm[0]: {pytorch_unnorm}")
    print(f"TRT unnorm[0]:     {unnorm_trt[0]}")

    # Run 3 more times with different seeds
    print("\n=== Multiple random seeds ===")
    for trial in range(3):
        rng2 = np.random.default_rng(trial + 100)
        x_t2 = rng2.standard_normal((1, ACTION_HORIZON, ACTION_DIM)).astype(np.float32)
        de.h2d("state", norm_state[np.newaxis])
        t2 = np.float32(1.0)
        for _ in range(NUM_STEPS):
            de.h2d("x_t", x_t2)
            de.h2d("timestep", np.array([t2], dtype=np.float32))
            de.execute()
            vel2 = de.d2h("velocity")
            de.sync()
            x_t2 = x_t2 + dt * vel2
            t2 += t_step
        ua = unnormalize_actions(x_t2[0, :, :ACTION_DIM], ns)
        print(f"Trial {trial+1} unnorm action[0]: {ua[0]}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
