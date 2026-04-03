#!/usr/bin/env python3
"""Optimized PyTorch policy server: SDPA + torch.compile for Jetson AGX Thor.

Replaces TRT with pure PyTorch optimization stack:
  1. SDPA attention (auto-dispatches to FlashAttention / memory-efficient kernels)
  2. torch.compile with max-autotune (CUDA graphs + autotuned kernels)
  3. Monkey-patched eager_attention_forward -> SDPA (zero model code changes)
  4. BF16 inference throughout

The SDPA patch replaces the eager matmul-softmax-matmul attention with
torch.nn.functional.scaled_dot_product_attention, which dispatches to:
  - FlashAttention kernel (causal / no-mask patterns)
  - Memory-efficient kernel (arbitrary masks, our prefix-LM pattern)
  - Math fallback (always correct)

Usage:
  python scripts/serve_pytorch_fa4.py --config-name pi05_so100 --port 8002
  python scripts/serve_pytorch_fa4.py --config-name pi05_excavator_v2 --benchmark
  python scripts/serve_pytorch_fa4.py --config-name pi05_excavator_v2 --verify
"""

import argparse
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


# ---------------------------------------------------------------------------
# SDPA Attention Patch
# ---------------------------------------------------------------------------

_original_eager_attention = None


def patch_attention_with_sdpa():
    """Monkey-patch eager_attention_forward with SDPA for automatic kernel dispatch.

    Works on all code paths (HF standard forward AND custom combined forward)
    because both call modeling_gemma.eager_attention_forward internally.
    """
    global _original_eager_attention
    from transformers.models.gemma import modeling_gemma

    _original_eager_attention = modeling_gemma.eager_attention_forward

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
    log.info("Patched eager_attention_forward -> SDPA (auto kernel dispatch)")


def unpatch_attention():
    """Restore original eager attention (for verification)."""
    if _original_eager_attention is not None:
        from transformers.models.gemma import modeling_gemma
        modeling_gemma.eager_attention_forward = _original_eager_attention
        log.info("Restored original eager_attention_forward")


# ---------------------------------------------------------------------------
# Model Config & Loading
# ---------------------------------------------------------------------------

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
    attn_impl: str = "eager"
    compile_mode: str = "reduce-overhead"


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


def load_model(device, action_dim, checkpoint_dir, compile_mode="max-autotune"):
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

    log.info("Creating PI0Pytorch (action_dim=%d)...", action_dim)
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

    if compile_mode != "none":
        log.info("Compiling denoise_step with mode=%s...", compile_mode)
        model.denoise_step = torch.compile(model.denoise_step, mode=compile_mode)

    return model


# ---------------------------------------------------------------------------
# Preprocessing (same as serve_pytorch_minimal.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class OptimizedPolicy:
    """Policy wrapper with SDPA attention and optional profiling."""

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
        self._profile = False

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

    def _build_observation(self, obs):
        from openpi.models import model as _model

        raw_state = np.asarray(
            _ensure_array(obs.get("observation/state"), fallback_shape=(self.action_dim,)),
            dtype=np.float32,
        )
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

        scene = preprocess_image(images.get("base_0_rgb", np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)))
        wrist = preprocess_image(images.get("left_wrist_0_rgb", np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)))
        dummy = np.full((1, 3, IMAGE_SIZE, IMAGE_SIZE), -1.0, dtype=np.float32)

        obs_dict = {
            "image": {
                "base_0_rgb": torch.from_numpy(scene).to(self.device),
                "left_wrist_0_rgb": torch.from_numpy(wrist).to(self.device),
                "right_wrist_0_rgb": torch.from_numpy(dummy).to(self.device),
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
        return _model.Observation.from_dict(obs_dict), norm_state

    def infer(self, obs):
        start = time.monotonic()

        observation, norm_state = self._build_observation(obs)

        with torch.no_grad():
            if self._profile and torch.cuda.is_available():
                torch.cuda.synchronize()
                t_prefix = time.monotonic()

            raw_actions = self.model.sample_actions(
                self.device, observation, num_steps=NUM_STEPS,
            )

            if self._profile and torch.cuda.is_available():
                torch.cuda.synchronize()
                t_done = time.monotonic()

        actions_np = raw_actions.cpu().numpy()[0, :, :self.action_dim]
        actions_unnorm = self.normalizer.unnormalize_actions(actions_np)

        infer_ms = (time.monotonic() - start) * 1000
        result = {
            "state": norm_state,
            "actions": actions_unnorm,
            "policy_timing": {"infer_ms": infer_ms},
        }
        if self._profile:
            result["policy_timing"]["model_ms"] = (t_done - t_prefix) * 1000
        return result

    @property
    def metadata(self):
        return {
            "action_dim": self.action_dim,
            "action_horizon": ACTION_HORIZON,
            "model": "pi0.5-pytorch-sdpa",
        }


# ---------------------------------------------------------------------------
# Benchmarking & Verification
# ---------------------------------------------------------------------------

def benchmark(policy, obs, n_warmup=8, n_runs=30):
    """Benchmark with detailed timing breakdown."""
    policy._profile = True

    log.info("Warming up (%d runs, includes torch.compile graph capture)...", n_warmup)
    for i in range(n_warmup):
        t0 = time.monotonic()
        result = policy.infer(obs)
        elapsed = (time.monotonic() - t0) * 1000
        log.info("  Warmup %d/%d: %.1f ms (model: %.1f ms)",
                 i + 1, n_warmup, elapsed,
                 result["policy_timing"].get("model_ms", 0))

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    log.info("Timed runs (%d)...", n_runs)
    timings = []
    model_timings = []
    for _ in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.monotonic()
        result = policy.infer(obs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.monotonic() - t0) * 1000)
        model_timings.append(result["policy_timing"].get("model_ms", 0))

    timings = np.array(timings)
    model_timings = np.array(model_timings)

    log.info("=" * 65)
    log.info("BENCHMARK RESULTS (%d runs)", n_runs)
    log.info("-" * 65)
    log.info("  E2E     Mean: %7.1f ms  Median: %7.1f ms  Std: %5.1f ms",
             timings.mean(), np.median(timings), timings.std())
    log.info("  Model   Mean: %7.1f ms  Median: %7.1f ms  Std: %5.1f ms",
             model_timings.mean(), np.median(model_timings), model_timings.std())
    log.info("  E2E     Min:  %7.1f ms  Max:    %7.1f ms  P95: %5.1f ms",
             timings.min(), timings.max(), np.percentile(timings, 95))
    log.info("  Control frequency: %.1f Hz", 1000.0 / timings.mean())
    log.info("=" * 65)

    policy._profile = False
    return timings


def verify_sdpa_vs_eager(policy, obs, atol=1e-2, rtol=1e-2):
    """Compare SDPA output against eager attention for correctness."""
    log.info("Verifying SDPA vs eager attention...")

    torch.manual_seed(42)
    result_sdpa = policy.infer(obs)
    actions_sdpa = result_sdpa["actions"]

    unpatch_attention()
    torch.manual_seed(42)
    result_eager = policy.infer(obs)
    actions_eager = result_eager["actions"]

    patch_attention_with_sdpa()

    diff = np.abs(actions_sdpa - actions_eager)
    max_diff = diff.max()
    mean_diff = diff.mean()
    rel_diff = diff / (np.abs(actions_eager) + 1e-8)

    log.info("  Max absolute diff:  %.6f", max_diff)
    log.info("  Mean absolute diff: %.6f", mean_diff)
    log.info("  Max relative diff:  %.6f", rel_diff.max())
    log.info("  SDPA actions[0]:  %s", actions_sdpa[0])
    log.info("  Eager actions[0]: %s", actions_eager[0])

    if max_diff < atol or rel_diff.max() < rtol:
        log.info("  PASS: SDPA matches eager within tolerance")
        return True
    else:
        log.warning("  WARN: SDPA differs from eager beyond tolerance (max_abs=%.4f, max_rel=%.4f)",
                    max_diff, rel_diff.max())
        log.warning("  This may be acceptable -- BF16 reordering causes small numerical differences")
        return False


# ---------------------------------------------------------------------------
# WebSocket Server
# ---------------------------------------------------------------------------

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
        log.info("Optimized PyTorch policy server listening on port %d", port)
        await server.serve_forever()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Optimized PyTorch policy server (SDPA + torch.compile)")
    parser.add_argument("--config-name", default="pi05_excavator_v2",
                        choices=list(ROBOT_CONFIGS.keys()))
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--compile-mode", default="max-autotune",
                        choices=["max-autotune", "reduce-overhead", "default", "none"],
                        help="torch.compile mode for denoise_step")
    parser.add_argument("--no-sdpa", action="store_true",
                        help="Disable SDPA patch (use eager attention)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark after warmup, then serve")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Run benchmark and exit (no server)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify SDPA output matches eager attention")
    parser.add_argument("--warmup-runs", type=int, default=8,
                        help="Number of warmup runs (needs >=3 for torch.compile)")
    parser.add_argument("--benchmark-runs", type=int, default=30)
    args = parser.parse_args()

    robot_cfg = ROBOT_CONFIGS[args.config_name]
    action_dim = robot_cfg["action_dim"]
    checkpoint_dir = robot_cfg["checkpoint_dir"]
    norm_stats_path = robot_cfg["norm_stats"]
    obs_image_keys = robot_cfg["obs_image_keys"]
    log.info("Robot config: %s  ACTION_DIM=%d", args.config_name, action_dim)

    if not args.no_sdpa:
        patch_attention_with_sdpa()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        log.info("GPU: %s (SM %d.%d, %.1f GB)",
                 props.name, props.major, props.minor,
                 props.total_mem / 1e9)

    model = load_model(device, action_dim=action_dim,
                       checkpoint_dir=checkpoint_dir,
                       compile_mode=args.compile_mode)

    # Enable SDPA for the combined forward path (training) too
    model.paligemma_with_expert._use_sdpa = not args.no_sdpa

    normalizer = QuantileNormalizer(norm_stats_path)
    policy = OptimizedPolicy(model, device, normalizer, TOKENIZER_MODEL,
                             action_dim=action_dim, obs_image_keys=obs_image_keys)

    dummy_obs = {
        "observation/state": np.zeros(action_dim, dtype=np.float32),
        "prompt": "pick up the cube",
    }
    for obs_key, _ in obs_image_keys:
        dummy_obs[obs_key] = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    log.info("Initial warmup...")
    result = policy.infer(dummy_obs)
    log.info("Warmup done. Actions shape: %s, latency: %.1fms",
             result["actions"].shape, result["policy_timing"]["infer_ms"])

    if args.verify:
        verify_sdpa_vs_eager(policy, dummy_obs)

    if args.benchmark or args.benchmark_only:
        benchmark(policy, dummy_obs,
                  n_warmup=args.warmup_runs,
                  n_runs=args.benchmark_runs)
        if args.benchmark_only:
            return

    asyncio.run(main_async(policy, args.port))


if __name__ == "__main__":
    main()
