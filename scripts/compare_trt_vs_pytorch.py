#!/usr/bin/env python3
"""Compare TRT server actions vs PyTorch model actions using the same inputs.

Run inside the Docker container where PyTorch is available:
  docker exec great_poitras python3 /workspace/openpi/scripts/compare_trt_vs_pytorch.py
"""

import dataclasses
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/workspace/openpi/src")

CHECKPOINT_DIR = "/workspace/openpi/checkpoints/runC_pytorch"
NORM_STATS_PATH = "/workspace/openpi/checkpoints/runC_pytorch/assets/verm11/runA/norm_stats.json"
TOKENIZER_PATH = "/root/.cache/openpi/big_vision/paligemma_tokenizer.model"
TEST_OBS_PATH = "/tmp/test_obs.npz"

ACTION_DIM = 6
ACTION_HORIZON = 11
NUM_STEPS = 8


@dataclasses.dataclass
class ModelConfig:
    pi05: bool = True
    action_dim: int = ACTION_DIM
    action_horizon: int = ACTION_HORIZON
    paligemma_variant: str = "gemma_2b_lora"
    action_expert_variant: str = "gemma_300m_lora"
    dtype: str = "bfloat16"
    state_dim: int = ACTION_DIM
    max_token_len: int = 200
    discrete_state_input: bool = True


def load_pytorch_model(device):
    import safetensors.torch
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, get_safe_dtype

    # Patch get_safe_dtype for float64 compat
    import openpi.models_pytorch.pi0_pytorch as _pi0
    _orig = _pi0.get_safe_dtype
    def _patched(target_dtype, device_type):
        if target_dtype == torch.float64:
            return torch.float32
        return _orig(target_dtype, device_type)
    _pi0.get_safe_dtype = _patched

    config = ModelConfig()
    print("Creating PI0Pytorch model...")
    orig_compile = torch.compile
    torch.compile = lambda fn, **kw: fn
    try:
        model = PI0Pytorch(config)
    finally:
        torch.compile = orig_compile

    safetensors_path = f"{CHECKPOINT_DIR}/model.safetensors"
    print(f"Loading weights from {safetensors_path}...")
    t0 = time.time()
    safetensors.torch.load_model(model, safetensors_path, device=str(device))
    print(f"Weights loaded in {time.time() - t0:.1f}s")

    model.eval()
    model.to(device)
    return model


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_pytorch_model(device)

    # Load norm stats
    ns = load_norm_stats()

    # Load test observation
    data = np.load(TEST_OBS_PATH)
    scene_img = data["scene"]   # uint8 HWC
    wrist_img = data["wrist"]   # uint8 HWC
    raw_state = np.zeros(ACTION_DIM, dtype=np.float32)

    print(f"Scene shape: {scene_img.shape}, Wrist shape: {wrist_img.shape}")

    # Normalize state
    norm_state = normalize_state(raw_state, ns)
    print(f"Raw state: {raw_state}")
    print(f"Normalized state: {norm_state}")

    # Tokenize (using sentencepiece directly to avoid JAX dependency)
    import sentencepiece
    sp = sentencepiece.SentencePieceProcessor(model_file=TOKENIZER_PATH)

    prompt = "Pick up the bottle and place it on the yellow outlined square."
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    discretized = np.digitize(norm_state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
    state_str = " ".join(map(str, discretized))
    full_prompt = f"Task: {cleaned}, State: {state_str};\nAction: "
    tokens_raw = sp.encode(full_prompt, add_bos=True)

    max_len = 200
    if len(tokens_raw) < max_len:
        pad_len = max_len - len(tokens_raw)
        mask = [True] * len(tokens_raw) + [False] * pad_len
        tokens_list = tokens_raw + [0] * pad_len
    else:
        tokens_list = tokens_raw[:max_len]
        mask = [True] * max_len

    tokens_np = np.array(tokens_list, dtype=np.int64)
    mask_np = np.array(mask, dtype=bool)

    # Prepare images - convert uint8 HWC to float32 CHW [-1,1]
    def prep_img(img_uint8):
        img_f = img_uint8.astype(np.float32) / 255.0 * 2.0 - 1.0
        h, w = img_f.shape[:2]
        if (h, w) != (224, 224):
            ratio = max(w / 224, h / 224)
            new_h, new_w = int(h / ratio), int(w / ratio)
            t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0)
            t = torch.nn.functional.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
            t = t.squeeze(0).permute(1, 2, 0).numpy()
            t = np.clip(t, -1.0, 1.0)
            pad_h0 = (224 - new_h) // 2
            pad_h1 = 224 - new_h - pad_h0
            pad_w0 = (224 - new_w) // 2
            pad_w1 = 224 - new_w - pad_w0
            img_f = np.pad(t, ((pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
                           mode="constant", constant_values=-1.0)
        img_f = np.transpose(img_f, (2, 0, 1))
        return img_f[np.newaxis].astype(np.float32)

    scene_proc = prep_img(scene_img)
    wrist_proc = prep_img(wrist_img)
    dummy_proc = np.full((1, 3, 224, 224), -1.0, dtype=np.float32)

    print(f"\nImage stats:")
    print(f"  scene: shape={scene_proc.shape} range=[{scene_proc.min():.3f}, {scene_proc.max():.3f}]")
    print(f"  wrist: shape={wrist_proc.shape} range=[{wrist_proc.min():.3f}, {wrist_proc.max():.3f}]")

    # Build observation in model format
    from openpi.models import model as _model

    obs_dict = {
        "image": {
            "base_0_rgb": torch.from_numpy(scene_proc).to(device),
            "left_wrist_0_rgb": torch.from_numpy(wrist_proc).to(device),
            "right_wrist_0_rgb": torch.from_numpy(dummy_proc).to(device),
        },
        "image_mask": {
            "base_0_rgb": torch.tensor([True], device=device),
            "left_wrist_0_rgb": torch.tensor([True], device=device),
            "right_wrist_0_rgb": torch.tensor([False], device=device),
        },
        "state": torch.from_numpy(norm_state[np.newaxis]).to(device),
        "tokenized_prompt": torch.from_numpy(tokens_np[np.newaxis]).to(device),
        "tokenized_prompt_mask": torch.from_numpy(mask_np[np.newaxis]).to(device),
    }

    observation = _model.Observation.from_dict(obs_dict)

    # Generate noise in numpy so it can be shared with TRT comparison
    print("\n=== Running PyTorch inference ===")
    rng = np.random.default_rng(42)
    noise_np = rng.standard_normal((1, ACTION_HORIZON, ACTION_DIM)).astype(np.float32)
    print(f"Fixed noise[0,0]: {noise_np[0, 0]}")
    np.save("/tmp/fixed_noise_42.npy", noise_np)

    noise_torch = torch.from_numpy(noise_np).to(device)

    with torch.no_grad():
        raw_actions = model.sample_actions(device, observation, noise=noise_torch, num_steps=NUM_STEPS)

    raw_actions_np = raw_actions.cpu().numpy()[0]  # [11, 6]
    print(f"Raw model output shape: {raw_actions_np.shape}")
    print(f"Raw model output range: [{raw_actions_np.min():.4f}, {raw_actions_np.max():.4f}]")
    print(f"Raw action[0]: {raw_actions_np[0]}")

    # Unnormalize
    unnorm_actions = unnormalize_actions(raw_actions_np, ns)
    print(f"\nUnnormalized action[0]: {unnorm_actions[0]}")
    print(f"Unnormalized actions range: [{unnorm_actions.min():.4f}, {unnorm_actions.max():.4f}]")

    # Run multiple times to see variance
    print("\n=== Running 3 more times with different noise ===")
    for trial in range(3):
        with torch.no_grad():
            actions_trial = model.sample_actions(device, observation, num_steps=NUM_STEPS)
        a = actions_trial.cpu().numpy()[0]
        ua = unnormalize_actions(a, ns)
        print(f"Trial {trial+1} action[0]: {ua[0]}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
