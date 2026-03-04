#!/usr/bin/env python3
"""TensorRT-based policy server for pi-0.5 on Jetson Thor.

Loads two TRT engines (prefix encoder + denoise step), preprocesses
observations in numpy, and serves via the same WebSocket protocol as
the standard openpi server.

Usage:
  python scripts/trt_policy_server.py \
    --prefix-engine onnx_export/prefix_encoder_fp16.engine \
    --denoise-engine onnx_export/denoise_step_fp16.engine \
    --tokenizer-path <path_to_paligemma_tokenizer.model> \
    --port 8000
"""

import argparse
import asyncio
import http
import logging
import pathlib
import time

import msgpack_numpy
import numpy as np
import websockets.asyncio.server as ws_server
import websockets.frames

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Preprocessing (numpy only) ───────────────────────────────────────────────

IMAGE_SIZE = 224
MAX_TOKEN_LEN = 200
ACTION_DIM = 6
ACTION_HORIZON = 11
NUM_DENOISE_STEPS = 8


def resize_with_pad_np(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize image to (height, width) with letterbox padding. Input: [H, W, C] uint8 or float."""
    import cv2

    cur_h, cur_w = image.shape[:2]
    ratio = max(cur_w / width, cur_h / height)
    new_h = int(cur_h / ratio)
    new_w = int(cur_w / ratio)

    is_float = np.issubdtype(image.dtype, np.floating)

    if is_float:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        resized = np.clip(resized, -1.0, 1.0)
        pad_val = -1.0
    else:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        resized = np.clip(resized, 0, 255).astype(np.uint8)
        pad_val = 0

    pad_h0 = (height - new_h) // 2
    pad_h1 = height - new_h - pad_h0
    pad_w0 = (width - new_w) // 2
    pad_w1 = width - new_w - pad_w0

    padded = np.pad(resized, ((pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
                    mode="constant", constant_values=pad_val)
    return padded


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess a single image to [1, 3, 224, 224] float32 in [-1, 1]."""
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 127.5 - 1.0

    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    if image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        image = resize_with_pad_np(image, IMAGE_SIZE, IMAGE_SIZE)

    image = np.transpose(image, (2, 0, 1))
    return image[np.newaxis].astype(np.float32)


class Tokenizer:
    """Minimal PaliGemma tokenizer wrapper using sentencepiece."""

    def __init__(self, model_path: str, max_len: int = MAX_TOKEN_LEN):
        import sentencepiece
        self._sp = sentencepiece.SentencePieceProcessor(model_file=model_path)
        self._max_len = max_len

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            discretized = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
            state_str = " ".join(map(str, discretized))
            full_prompt = f"Task: {cleaned}, State: {state_str};\nAction: "
            tokens = self._sp.encode(full_prompt, add_bos=True)
        else:
            tokens = self._sp.encode(cleaned, add_bos=True) + self._sp.encode("\n")

        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            pad_len = self._max_len - tokens_len
            mask = [True] * tokens_len + [False] * pad_len
            tokens = tokens + [False] * pad_len
        else:
            tokens = tokens[:self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens, dtype=np.int64), np.asarray(mask, dtype=np.bool_)


def preprocess_observation(obs: dict, tokenizer: Tokenizer) -> dict:
    """Convert raw observation dict to TRT-ready numpy arrays."""
    state = np.asarray(obs.get("observation/state", np.zeros(ACTION_DIM)), dtype=np.float32)

    prompt = obs.get("prompt", "pick up the cube")
    tokens, token_mask = tokenizer.tokenize(prompt, state=state)

    img_keys = [
        ("observation/image_scene", "base_0_rgb"),
        ("observation/image_wrist", "left_wrist_0_rgb"),
    ]

    images = {}
    for obs_key, model_key in img_keys:
        if obs_key in obs:
            img = np.asarray(obs[obs_key])
            if np.issubdtype(img.dtype, np.floating):
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            images[model_key] = img
        else:
            images[model_key] = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    if "right_wrist_0_rgb" not in images:
        images["right_wrist_0_rgb"] = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    return {
        "img0": preprocess_image(images["base_0_rgb"]),
        "img1": preprocess_image(images["left_wrist_0_rgb"]),
        "img2": preprocess_image(images["right_wrist_0_rgb"]),
        "mask0": np.array([True]),
        "mask1": np.array([True]) if "left_wrist_0_rgb" in images else np.array([False]),
        "mask2": np.array([False]),
        "tokens": tokens[np.newaxis],
        "token_masks": token_mask[np.newaxis],
        "state": state[np.newaxis],
    }


# ─── TRT Engine Wrapper ───────────────────────────────────────────────────────

class TRTEngine:
    """Wraps a TensorRT engine for inference using cuda-python."""

    def __init__(self, engine_path: str, shared_device_buffers: dict[str, int] | None = None):
        """Load engine. shared_device_buffers maps tensor names to existing device pointers."""
        import tensorrt as trt
        from cuda.bindings import runtime as cudart

        self.trt = trt
        self.cudart = cudart
        self.logger = trt.Logger(trt.Logger.WARNING)

        log.info("Loading TRT engine: %s", engine_path)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self._io_info = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = list(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            mode = self.engine.get_tensor_mode(name)
            self._io_info[name] = {"shape": shape, "dtype": dtype, "is_input": mode == trt.TensorIOMode.INPUT}

        log.info("  Tensors: %s", {k: (v["shape"], v["dtype"].__name__, "in" if v["is_input"] else "out")
                                    for k, v in self._io_info.items()})

        self._device_buffers = {}
        self._host_buffers = {}
        self._owned_buffers = set()
        shared_device_buffers = shared_device_buffers or {}

        for name, info in self._io_info.items():
            nbytes = int(np.prod(info["shape"])) * np.dtype(info["dtype"]).itemsize
            if name in shared_device_buffers:
                ptr = shared_device_buffers[name]
                log.info("  Sharing device buffer for %s (not owned)", name)
            else:
                err, ptr = cudart.cudaMalloc(nbytes)
                assert err == 0 or (hasattr(err, "value") and err.value == 0), f"cudaMalloc failed for {name}: {err}"
                self._owned_buffers.add(name)
            self._device_buffers[name] = ptr
            self._host_buffers[name] = np.empty(info["shape"], dtype=info["dtype"])
            self.context.set_tensor_address(name, ptr)

        err, self._stream = cudart.cudaStreamCreate()
        assert err == 0 or (hasattr(err, "value") and err.value == 0), f"cudaStreamCreate failed: {err}"

    def _check(self, result, msg="CUDA error"):
        """Check CUDA return value (handles both tuple and scalar returns)."""
        if isinstance(result, tuple):
            err = result[0]
        else:
            err = result
        val = err.value if hasattr(err, "value") else int(err)
        assert val == 0, f"{msg}: {err}"

    def get_device_ptr(self, name: str) -> int:
        """Get device pointer for a tensor (for sharing with other engines)."""
        return self._device_buffers[name]

    def infer(self, inputs: dict[str, np.ndarray],
              skip_d2h: set[str] | None = None) -> dict[str, np.ndarray]:
        """Run inference. skip_d2h: output tensor names to leave on device (shared buffers)."""
        cudart = self.cudart
        H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        skip_d2h = skip_d2h or set()

        for name, arr in inputs.items():
            info = self._io_info[name]
            host_buf = np.ascontiguousarray(arr.astype(info["dtype"]))
            self._check(
                cudart.cudaMemcpyAsync(self._device_buffers[name], host_buf.ctypes.data,
                                       host_buf.nbytes, H2D, self._stream),
                f"H2D {name}"
            )

        self.context.execute_async_v3(self._stream)

        outputs = {}
        for name, info in self._io_info.items():
            if not info["is_input"] and name not in skip_d2h:
                out_buf = self._host_buffers[name]
                self._check(
                    cudart.cudaMemcpyAsync(out_buf.ctypes.data, self._device_buffers[name],
                                           out_buf.nbytes, D2H, self._stream),
                    f"D2H {name}"
                )
                outputs[name] = out_buf.copy()

        self._check(cudart.cudaStreamSynchronize(self._stream), "sync")
        return outputs

    def h2d(self, name: str, arr: np.ndarray):
        """Copy a single tensor H2D (async, no sync)."""
        info = self._io_info[name]
        host_buf = np.ascontiguousarray(arr.astype(info["dtype"]))
        self._check(
            self.cudart.cudaMemcpyAsync(
                self._device_buffers[name], host_buf.ctypes.data,
                host_buf.nbytes,
                self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self._stream),
            f"H2D {name}",
        )

    def d2h(self, name: str) -> np.ndarray:
        """Copy a single tensor D2H (async, no sync). Call sync() after."""
        out_buf = self._host_buffers[name]
        self._check(
            self.cudart.cudaMemcpyAsync(
                out_buf.ctypes.data, self._device_buffers[name],
                out_buf.nbytes,
                self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self._stream),
            f"D2H {name}",
        )
        return out_buf

    def execute(self):
        """Execute engine on stream (async, no sync)."""
        self.context.execute_async_v3(self._stream)

    def sync(self):
        """Synchronize the CUDA stream."""
        self._check(self.cudart.cudaStreamSynchronize(self._stream), "sync")

    def __del__(self):
        if hasattr(self, "_device_buffers") and hasattr(self, "_owned_buffers"):
            for name in self._owned_buffers:
                if name in self._device_buffers:
                    self.cudart.cudaFree(self._device_buffers[name])
        if hasattr(self, "_stream"):
            self.cudart.cudaStreamDestroy(self._stream)


# ─── TRT Policy ───────────────────────────────────────────────────────────────

SHARED_TENSORS = ("kv_keys", "kv_values", "prefix_pad_masks")


class TRTPolicy:
    """Pi-0.5 policy using TRT engines with shared device buffers."""

    def __init__(self, prefix_engine_path: str, denoise_engine_path: str, tokenizer_path: str):
        self.prefix_engine = TRTEngine(prefix_engine_path)

        shared = {name: self.prefix_engine.get_device_ptr(name) for name in SHARED_TENSORS}
        self.denoise_engine = TRTEngine(denoise_engine_path, shared_device_buffers=shared)

        self.tokenizer = Tokenizer(tokenizer_path)
        log.info("Shared device buffers for: %s", list(shared.keys()))

    def infer(self, obs: dict) -> dict:
        start = time.monotonic()
        inputs = preprocess_observation(obs, self.tokenizer)

        self.prefix_engine.infer(
            {
                "img0": inputs["img0"],
                "img1": inputs["img1"],
                "img2": inputs["img2"],
                "mask0": inputs["mask0"],
                "mask1": inputs["mask1"],
                "mask2": inputs["mask2"],
                "tokens": inputs["tokens"],
                "token_masks": inputs["token_masks"],
            },
            skip_d2h=set(SHARED_TENSORS),
        )

        de = self.denoise_engine
        de.h2d("state", inputs["state"])

        rng = np.random.default_rng()
        x_t = rng.standard_normal((1, ACTION_HORIZON, ACTION_DIM)).astype(np.float32)

        dt = np.float32(-1.0 / NUM_DENOISE_STEPS)
        t = np.float32(1.0)
        t_step = np.float32(-1.0 / NUM_DENOISE_STEPS)
        for _ in range(NUM_DENOISE_STEPS):
            de.h2d("x_t", x_t)
            de.h2d("timestep", np.array([t], dtype=np.float32))
            de.execute()
            vel_buf = de.d2h("velocity")
            de.sync()
            x_t = x_t + dt * vel_buf
            t += t_step

        infer_ms = (time.monotonic() - start) * 1000
        actions = x_t[0]

        return {
            "state": inputs["state"][0],
            "actions": actions[:, :ACTION_DIM],
            "policy_timing": {"infer_ms": infer_ms},
        }

    @property
    def metadata(self) -> dict:
        return {
            "action_dim": ACTION_DIM,
            "action_horizon": ACTION_HORIZON,
            "model": "pi0.5-trt",
        }


# ─── WebSocket Server ─────────────────────────────────────────────────────────

class TRTWebSocketServer:
    def __init__(self, policy: TRTPolicy, host: str = "0.0.0.0", port: int = 8000):
        self._policy = policy
        self._host = host
        self._port = port

    def serve_forever(self):
        asyncio.run(self._run())

    async def _run(self):
        async with ws_server.serve(
            self._handler, self._host, self._port,
            compression=None, max_size=None,
            process_request=self._health_check,
        ) as server:
            log.info("TRT policy server listening on %s:%d", self._host, self._port)
            await server.serve_forever()

    async def _handler(self, websocket: ws_server.ServerConnection):
        log.info("Connection from %s", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._policy.metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_start = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_start

                action["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                log.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                import traceback
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error",
                )
                raise

    @staticmethod
    def _health_check(connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TRT Policy Server for pi-0.5")
    parser.add_argument("--prefix-engine", help="Path to prefix encoder TRT engine")
    parser.add_argument("--denoise-engine", help="Path to denoise step TRT engine")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16", "int8"],
                        help="TRT precision mode (selects engine files automatically)")
    parser.add_argument("--engine-dir", default="onnx_export", help="Directory containing TRT engine files")
    parser.add_argument("--tokenizer-path",
                        default=str(pathlib.Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model"),
                        help="Path to PaliGemma sentencepiece model")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    if args.prefix_engine and args.denoise_engine:
        prefix_path = args.prefix_engine
        denoise_path = args.denoise_engine
    else:
        engine_dir = pathlib.Path(args.engine_dir)
        prefix_path = str(engine_dir / f"prefix_encoder_{args.precision}.engine")
        denoise_path = str(engine_dir / f"denoise_step_{args.precision}.engine")
        log.info("Using %s precision engines from %s", args.precision.upper(), engine_dir)

    policy = TRTPolicy(prefix_path, denoise_path, args.tokenizer_path)
    server = TRTWebSocketServer(policy, host=args.host, port=args.port)

    log.info("Warming up with dummy inference...")
    dummy_obs = {
        "observation/state": np.zeros(ACTION_DIM, dtype=np.float32),
        "observation/image_scene": np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        "observation/image_wrist": np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        "prompt": "pick up the cube",
    }
    result = policy.infer(dummy_obs)
    log.info("Warmup done. Action shape: %s, latency: %.1f ms",
             result["actions"].shape, result["policy_timing"]["infer_ms"])

    server.serve_forever()


if __name__ == "__main__":
    main()
