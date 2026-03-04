#!/usr/bin/env python3
"""Build INT8 TensorRT engines using calibration data.

Uses the host TRT Python API (v10.13.3) with IInt8EntropyCalibrator2.

Usage:
  python scripts/build_int8_engines.py
  python scripts/build_int8_engines.py --engine prefix   # prefix only
  python scripts/build_int8_engines.py --engine denoise  # denoise only
  python scripts/build_int8_engines.py --fallback-precision bf16  # mixed INT8+BF16
"""

import argparse
import logging
import pathlib
import sys

import numpy as np
import tensorrt as trt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parent.parent
ONNX_DIR = ROOT / "onnx_export"
CALIB_DIR = ROOT / "calibration_data"


class NpyCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibrator that reads pre-generated .npy calibration data."""

    def __init__(self, data_dir: pathlib.Path, input_names: list[str],
                 batch_size: int = 1, cache_file: str | None = None):
        super().__init__()
        self._batch_size = batch_size
        self._cache_file = cache_file
        self._current_idx = 0

        from cuda.bindings import runtime as cudart
        self._cudart = cudart

        self._data = {}
        self._num_samples = None
        for name in input_names:
            path = data_dir / f"{name}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Calibration data not found: {path}")
            arr = np.load(path, mmap_mode="r")
            self._data[name] = arr
            n = arr.shape[0]
            if self._num_samples is None:
                self._num_samples = n
            else:
                self._num_samples = min(self._num_samples, n)
            log.info("  Loaded %s: %s %s (%d samples)", name, arr.shape, arr.dtype, n)

        self._device_buffers = {}
        for name in input_names:
            sample = self._data[name][0]
            if sample.ndim == 0:
                sample = sample.reshape(1)
            nbytes = sample.nbytes
            err, ptr = cudart.cudaMalloc(nbytes)
            self._check(err, f"cudaMalloc for {name}")
            self._device_buffers[name] = (ptr, nbytes)

        self._input_names = input_names
        log.info("  Calibrator ready: %d samples, batch_size=%d", self._num_samples, batch_size)

    def _check(self, err, msg="CUDA error"):
        if isinstance(err, tuple):
            err = err[0]
        val = err.value if hasattr(err, "value") else int(err)
        assert val == 0, f"{msg}: {err}"

    def get_batch_size(self):
        return self._batch_size

    def get_batch(self, names):
        if self._current_idx >= self._num_samples:
            return None

        cudart = self._cudart
        H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        ptrs = []

        for name in names:
            raw = self._data[name][self._current_idx]
            sample = np.array(raw) if not isinstance(raw, np.ndarray) else raw.copy()
            if sample.ndim == 0:
                sample = sample.reshape(1)
            sample = np.ascontiguousarray(sample)
            ptr, nbytes = self._device_buffers[name]
            self._check(
                cudart.cudaMemcpy(ptr, sample.ctypes.data, sample.nbytes, H2D),
                f"H2D {name}",
            )
            ptrs.append(ptr)

        self._current_idx += 1
        if self._current_idx % 50 == 0:
            log.info("  Calibration batch %d/%d", self._current_idx, self._num_samples)

        return ptrs

    def read_calibration_cache(self):
        if self._cache_file and pathlib.Path(self._cache_file).exists():
            log.info("  Reading calibration cache: %s", self._cache_file)
            return pathlib.Path(self._cache_file).read_bytes()
        return None

    def write_calibration_cache(self, cache):
        if self._cache_file:
            pathlib.Path(self._cache_file).write_bytes(cache)
            log.info("  Wrote calibration cache: %s (%d bytes)", self._cache_file, len(cache))

    def __del__(self):
        if hasattr(self, "_device_buffers"):
            for ptr, _ in self._device_buffers.values():
                self._cudart.cudaFree(ptr)


def build_engine(onnx_path: str, output_path: str, calibrator: NpyCalibrator,
                 fallback_precision: str = "fp16", workspace_mb: int = 8192):
    """Build an INT8 TRT engine from ONNX with calibration."""
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    log.info("Parsing ONNX: %s", onnx_path)

    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            log.error("  ONNX parse error: %s", parser.get_error(i))
        raise RuntimeError("ONNX parse failed")

    log.info("  Network inputs: %d, outputs: %d", network.num_inputs, network.num_outputs)
    for i in range(network.num_inputs):
        t = network.get_input(i)
        log.info("    Input %d: %s %s %s", i, t.name, t.shape, t.dtype)
    for i in range(network.num_outputs):
        t = network.get_output(i)
        log.info("    Output %d: %s %s %s", i, t.name, t.shape, t.dtype)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)

    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator

    if fallback_precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif fallback_precision == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)

    log.info("Building INT8 engine (fallback=%s)...", fallback_precision)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    engine_bytes = bytes(serialized)
    log.info("Writing engine to %s (%.1f MB)", output_path, len(engine_bytes) / 1e6)
    with open(output_path, "wb") as f:
        f.write(engine_bytes)

    return output_path


def build_prefix_engine(fallback_precision: str = "fp16"):
    onnx_path = str(ONNX_DIR / "prefix_encoder.onnx")
    output_path = str(ONNX_DIR / "prefix_encoder_int8.engine")
    cache_path = str(ONNX_DIR / "prefix_encoder_int8.cache")

    prefix_inputs = ["img0", "img1", "img2", "mask0", "mask1", "mask2", "tokens", "token_masks"]

    log.info("Creating prefix encoder calibrator...")
    calibrator = NpyCalibrator(
        CALIB_DIR / "prefix", prefix_inputs,
        batch_size=1, cache_file=cache_path,
    )
    return build_engine(onnx_path, output_path, calibrator, fallback_precision)


def build_denoise_engine(fallback_precision: str = "fp16"):
    onnx_path = str(ONNX_DIR / "denoise_step.onnx")
    output_path = str(ONNX_DIR / "denoise_step_int8.engine")
    cache_path = str(ONNX_DIR / "denoise_step_int8.cache")

    denoise_inputs = ["state", "prefix_pad_masks", "x_t", "timestep", "kv_keys", "kv_values"]

    log.info("Creating denoise step calibrator...")
    calibrator = NpyCalibrator(
        CALIB_DIR / "denoise", denoise_inputs,
        batch_size=1, cache_file=cache_path,
    )
    return build_engine(onnx_path, output_path, calibrator, fallback_precision,
                        workspace_mb=4096)


def main():
    parser = argparse.ArgumentParser(description="Build INT8 TRT engines")
    parser.add_argument("--engine", choices=["prefix", "denoise", "both"], default="both")
    parser.add_argument("--fallback-precision", choices=["fp16", "bf16"], default="fp16",
                        help="Fallback precision for layers that don't quantize well")
    args = parser.parse_args()

    if args.engine in ("prefix", "both"):
        path = build_prefix_engine(args.fallback_precision)
        log.info("Prefix engine: %s", path)

    if args.engine in ("denoise", "both"):
        path = build_denoise_engine(args.fallback_precision)
        log.info("Denoise engine: %s", path)

    log.info("Done!")


if __name__ == "__main__":
    main()
