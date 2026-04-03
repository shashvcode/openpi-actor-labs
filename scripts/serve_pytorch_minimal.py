#!/usr/bin/env python3
"""Minimal PyTorch policy server for direct comparison with TRT server.

Loads the PI0.5 model in PyTorch, serves via same WebSocket protocol.
Run inside Docker: docker exec great_poitras python3 /workspace/openpi/scripts/serve_pytorch_minimal.py

Connects to port 8001 to avoid conflict with TRT server on 8000.
"""

import asyncio
import dataclasses
import http
import json
import logging
import sys
import time

import numpy as np
import torch
import websockets.asyncio.server as ws_server
import websockets.frames

sys.path.insert(0, "/workspace/openpi/src")

from openpi_client.msgpack_numpy import Packer as MsgPacker, unpackb as msg_unpackb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TOKENIZER_MODEL = "/root/.cache/openpi/big_vision/paligemma_tokenizer.model"

ACTION_HORIZON = 11
NUM_STEPS = 8
IMAGE_SIZE = 224
MAX_TOKEN_LEN = 200

ROBOT_CONFIGS = {
    "pi05_excavator_v2": {
        "action_dim": 4,
        "checkpoint_dir": "/workspace/openpi/checkpoints/excavator_v1_pytorch",
        "norm_stats": "/workspace/openpi/checkpoints/excavator_v1_pytorch/norm_stats.json",
        "obs_image_keys": [
            ("observation/image_cab", "base_0_rgb"),
            ("observation/image_side", "left_wrist_0_rgb"),
        ],
    },
    "pi05_so100": {
        "action_dim": 6,
        "checkpoint_dir": "/workspace/openpi/checkpoints/runD_qat_pytorch",
        "norm_stats": "/workspace/openpi/checkpoints/runD_qat_pytorch/assets/verm11/runA/norm_stats.json",
        "obs_image_keys": [
            ("observation/image_scene", "base_0_rgb"),
            ("observation/image_wrist", "left_wrist_0_rgb"),
        ],
    },
}


@dataclasses.dataclass
class ModelConfig:
    pi05: bool = True
    action_dim: int = 6
    action_horizon: int = ACTION_HORIZON
    paligemma_variant: str = "gemma_2b_lora"
    action_expert_variant: str = "gemma_300m_lora"
    dtype: str = "bfloat16"
    state_dim: int = 6
    max_token_len: int = MAX_TOKEN_LEN
    discrete_state_input: bool = True


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
        dim = q01.shape[-1]
        if dim < actions.shape[-1]:
            head = (actions[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
            return np.concatenate([head, actions[..., dim:]], axis=-1)
        return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def load_model(device, action_dim, checkpoint_dir):
    import safetensors.torch
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    import openpi.models_pytorch.pi0_pytorch as _pi0

    _orig = _pi0.get_safe_dtype
    def _patched(target_dtype, device_type):
        if target_dtype == torch.float64:
            return torch.float32
        return _orig(target_dtype, device_type)
    _pi0.get_safe_dtype = _patched

    config = ModelConfig(action_dim=action_dim, state_dim=action_dim)
    log.info("Creating PI0Pytorch model (action_dim=%d)...", action_dim)
    orig_compile = torch.compile
    torch.compile = lambda fn, **kw: fn
    try:
        model = PI0Pytorch(config)
    finally:
        torch.compile = orig_compile

    safetensors_path = f"{checkpoint_dir}/model.safetensors"
    log.info("Loading weights from %s...", safetensors_path)
    t0 = time.time()
    safetensors.torch.load_model(model, safetensors_path, device=str(device))
    log.info("Weights loaded in %.1fs", time.time() - t0)

    model.eval()
    model.to(device)
    return model


def _ensure_array(val, fallback_shape=None):
    if isinstance(val, np.ndarray):
        return val
    if isinstance(val, dict):
        d = {(k if isinstance(k, bytes) else k.encode()): v for k, v in val.items()}
        if b'__ndarray__' in d:
            dtype_str = d[b'dtype']
        elif b'type' in d:
            dtype_str = d[b'type']
        else:
            dtype_str = d.get(b'dtype', 'float32')
        if isinstance(dtype_str, bytes):
            dtype_str = dtype_str.decode()
        return np.ndarray(buffer=d[b'data'], dtype=np.dtype(dtype_str), shape=tuple(d[b'shape']))
    if fallback_shape is not None:
        return np.zeros(fallback_shape, dtype=np.float32)
    return np.asarray(val)


def preprocess_image(img_uint8):
    """uint8 HWC -> float32 [1,3,224,224] in [-1,1], matching Observation.from_dict"""
    img = img_uint8.astype(np.float32) / 255.0 * 2.0 - 1.0
    if img.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        from PIL import Image as PILImage
        h, w = img.shape[:2]
        ratio = max(w / IMAGE_SIZE, h / IMAGE_SIZE)
        new_w, new_h = int(w / ratio), int(h / ratio)
        img_u8 = np.clip((img + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)
        pil = PILImage.fromarray(img_u8).resize((new_w, new_h), PILImage.BILINEAR)
        resized = np.array(pil).astype(np.float32) / 255.0 * 2.0 - 1.0
        pad_h0 = (IMAGE_SIZE - new_h) // 2
        pad_h1 = IMAGE_SIZE - new_h - pad_h0
        pad_w0 = (IMAGE_SIZE - new_w) // 2
        pad_w1 = IMAGE_SIZE - new_w - pad_w0
        img = np.pad(resized, ((pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
                     mode="constant", constant_values=-1.0)
    img = np.transpose(img, (2, 0, 1))
    return img[np.newaxis].astype(np.float32)


class PyTorchPolicy:
    def __init__(self, model, device, normalizer, tokenizer_path, action_dim,
                 obs_image_keys=None):
        self.model = model
        self.device = device
        self.normalizer = normalizer
        self.action_dim = action_dim
        self.obs_image_keys = obs_image_keys or [
            ("observation/image_scene", "base_0_rgb"),
            ("observation/image_wrist", "left_wrist_0_rgb"),
        ]

        import sentencepiece
        self.sp = sentencepiece.SentencePieceProcessor(model_file=tokenizer_path)

    def tokenize(self, prompt, state):
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        discretized = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        state_str = " ".join(map(str, discretized))
        full_prompt = f"Task: {cleaned}, State: {state_str};\nAction: "
        tokens_raw = self.sp.encode(full_prompt, add_bos=True)

        if len(tokens_raw) < MAX_TOKEN_LEN:
            pad_len = MAX_TOKEN_LEN - len(tokens_raw)
            mask = [True] * len(tokens_raw) + [False] * pad_len
            tokens_list = tokens_raw + [0] * pad_len
        else:
            tokens_list = tokens_raw[:MAX_TOKEN_LEN]
            mask = [True] * MAX_TOKEN_LEN

        return np.array(tokens_list, dtype=np.int64), np.array(mask, dtype=bool)

    def infer(self, obs):
        start = time.monotonic()
        from openpi.models import model as _model

        raw_state = np.asarray(_ensure_array(obs.get("observation/state"), fallback_shape=(self.action_dim,)), dtype=np.float32)
        norm_state = self.normalizer.normalize_state(raw_state)

        tokens, token_mask = self.tokenize(obs.get("prompt", "pick up the cube"), norm_state)

        images = {}
        for obs_key, model_key in self.obs_image_keys:
            if obs_key in obs:
                img = np.asarray(_ensure_array(obs[obs_key]))
                if np.issubdtype(img.dtype, np.floating):
                    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                images[model_key] = img
            else:
                images[model_key] = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

        scene_proc = preprocess_image(images.get("base_0_rgb", np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)))
        wrist_proc = preprocess_image(images.get("left_wrist_0_rgb", np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)))
        dummy_proc = np.full((1, 3, IMAGE_SIZE, IMAGE_SIZE), -1.0, dtype=np.float32)

        obs_dict = {
            "image": {
                "base_0_rgb": torch.from_numpy(scene_proc).to(self.device),
                "left_wrist_0_rgb": torch.from_numpy(wrist_proc).to(self.device),
                "right_wrist_0_rgb": torch.from_numpy(dummy_proc).to(self.device),
            },
            "image_mask": {
                "base_0_rgb": torch.tensor([True], device=self.device),
                "left_wrist_0_rgb": torch.tensor([True], device=self.device),
                "right_wrist_0_rgb": torch.tensor([False], device=self.device),
            },
            "state": torch.from_numpy(norm_state[np.newaxis]).to(self.device),
            "tokenized_prompt": torch.from_numpy(tokens[np.newaxis]).to(self.device),
            "tokenized_prompt_mask": torch.from_numpy(token_mask[np.newaxis]).to(self.device),
        }

        observation = _model.Observation.from_dict(obs_dict)

        with torch.no_grad():
            raw_actions = self.model.sample_actions(self.device, observation, num_steps=NUM_STEPS)

        actions_np = raw_actions.cpu().numpy()[0, :, :self.action_dim]
        actions_unnorm = self.normalizer.unnormalize_actions(actions_np)

        infer_ms = (time.monotonic() - start) * 1000
        return {
            "state": norm_state,
            "actions": actions_unnorm,
            "policy_timing": {"infer_ms": infer_ms},
        }

    @property
    def metadata(self):
        return {"action_dim": self.action_dim, "action_horizon": ACTION_HORIZON, "model": "pi0.5-pytorch"}


async def handler(websocket, policy):
    log.info("Connection from %s", websocket.remote_address)
    packer = MsgPacker()
    await websocket.send(packer.pack(policy.metadata))

    while True:
        try:
            raw_bytes = await websocket.recv()
            obs = msg_unpackb(raw_bytes)
            start = time.monotonic()
            action = policy.infer(obs)
            infer_time = time.monotonic() - start
            action["server_timing"] = {"infer_ms": infer_time * 1000}
            await websocket.send(packer.pack(action))
        except websockets.ConnectionClosed:
            log.info("Connection closed")
            break
        except Exception:
            import traceback
            log.error(traceback.format_exc())
            await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason="Error")
            raise


def health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def main_async(policy, port):
    async with ws_server.serve(
        lambda ws: handler(ws, policy), "0.0.0.0", port,
        compression=None, max_size=None, process_request=health_check,
    ) as server:
        log.info("PyTorch policy server listening on port %d", port)
        await server.serve_forever()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pi05_excavator_v2",
                        choices=list(ROBOT_CONFIGS.keys()),
                        help="Robot config to use (sets ACTION_DIM and default paths)")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    robot_cfg = ROBOT_CONFIGS[args.config_name]
    action_dim = robot_cfg["action_dim"]
    checkpoint_dir = robot_cfg["checkpoint_dir"]
    norm_stats_path = robot_cfg["norm_stats"]
    obs_image_keys = robot_cfg["obs_image_keys"]
    log.info("Robot config: %s  ACTION_DIM=%d", args.config_name, action_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    model = load_model(device, action_dim=action_dim, checkpoint_dir=checkpoint_dir)
    normalizer = QuantileNormalizer(norm_stats_path)
    policy = PyTorchPolicy(model, device, normalizer, TOKENIZER_MODEL,
                           action_dim=action_dim, obs_image_keys=obs_image_keys)

    log.info("Warming up...")
    dummy_obs = {
        "observation/state": np.zeros(action_dim, dtype=np.float32),
        "prompt": "pick up the cube",
    }
    for obs_key, _ in obs_image_keys:
        dummy_obs[obs_key] = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    result = policy.infer(dummy_obs)
    log.info("Warmup done. Actions shape: %s, latency: %.1fms",
             result["actions"].shape, result["policy_timing"]["infer_ms"])

    asyncio.run(main_async(policy, args.port))


if __name__ == "__main__":
    main()
