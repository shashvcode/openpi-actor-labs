#!/usr/bin/env python3
"""Build TRT engines with mixed precision: reduced-precision matmuls + FP32 sensitive ops.

Uses the TRT Python API to selectively set layer precision, keeping
softmax, normalization, and reduction layers in FP32 while allowing
GEMM/MatMul/Conv operations to run in FP16, BF16, or INT8.

For INT8, a calibrator is used with representative data from calibration_data_fresh/.

Usage:
  # FP16 matmuls + FP32 norm/softmax (recommended)
  python3 scripts/build_mixed_precision_engine.py --onnx onnx_export/prefix_encoder.onnx \
      --output onnx_export/prefix_encoder_mixed_fp16.engine --precision fp16

  # INT8 matmuls + FP32 norm/softmax
  python3 scripts/build_mixed_precision_engine.py --onnx onnx_export/prefix_encoder.onnx \
      --output onnx_export/prefix_encoder_mixed_int8.engine --precision int8 \
      --calib-data calibration_data_fresh/prefix
"""
import argparse
import logging
import os
import pathlib
import time

import numpy as np
import tensorrt as trt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FP32_LAYER_SUBSTRINGS = [
    "softmax", "Softmax", "SOFTMAX",
    "norm", "Norm", "NORM",
    "reduce", "Reduce", "REDUCE",
    "rsqrt", "Rsqrt",
    "pow", "Pow",
    "sqrt", "Sqrt",
    "exp", "Exp",
    "reciprocal",
]

SKIP_LAYER_TYPES = {
    trt.LayerType.CAST, trt.LayerType.SHAPE, trt.LayerType.IDENTITY,
    trt.LayerType.CONSTANT, trt.LayerType.CONCATENATION,
    trt.LayerType.GATHER, trt.LayerType.SHUFFLE, trt.LayerType.SLICE,
    trt.LayerType.FILL, trt.LayerType.CONDITION, trt.LayerType.ASSERTION,
}

FLOAT_DTYPES = (trt.float32, trt.float16, trt.bfloat16)


class NpzCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibrator that reads .npy files from a directory."""

    def __init__(self, data_dir, cache_path, network):
        super().__init__()
        self.cache_path = cache_path
        self.data_dir = pathlib.Path(data_dir)
        self.batch_idx = 0
        self.num_batches = 1

        self._device_buffers = {}
        self._input_names = []
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            self._input_names.append(inp.name)

        self._load_data()

    def _load_data(self):
        import cuda.bindings.driver as drv
        drv.cuInit(0)
        _, ctx = drv.cuCtxGetCurrent()

        self._host_data = {}
        self._device_ptrs = {}

        for name in self._input_names:
            npy_path = self.data_dir / f"{name}.npy"
            if npy_path.exists():
                arr = np.load(str(npy_path))
                log.info("  Calibration input %s: shape=%s dtype=%s", name, arr.shape, arr.dtype)
            else:
                log.warning("  Calibration input %s: not found at %s, using zeros", name, npy_path)
                arr = np.zeros(1, dtype=np.float32)

            self._host_data[name] = np.ascontiguousarray(arr)
            nbytes = self._host_data[name].nbytes
            err, ptr = drv.cuMemAlloc(nbytes)
            assert err == drv.CUresult.CUDA_SUCCESS, f"cuMemAlloc failed: {err}"
            self._device_ptrs[name] = ptr

            err, = drv.cuMemcpyHtoD(ptr, self._host_data[name].ctypes.data, nbytes)
            assert err == drv.CUresult.CUDA_SUCCESS, f"cuMemcpyHtoD failed: {err}"

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.batch_idx >= self.num_batches:
            return None
        self.batch_idx += 1
        return [int(self._device_ptrs[n]) for n in names]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_path):
            log.info("Reading calibration cache: %s", self.cache_path)
            with open(self.cache_path, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        log.info("Writing calibration cache: %s", self.cache_path)
        with open(self.cache_path, "wb") as f:
            f.write(cache)


def build_engine(onnx_path, engine_path, precision="fp16", workspace_gb=12,
                 calib_data=None, calib_cache=None):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    log.info("Parsing ONNX: %s", onnx_path)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read(), onnx_path):
            for i in range(parser.num_errors):
                log.error("  %s", parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        target_dtype = trt.float16
    elif precision == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)
        target_dtype = trt.bfloat16
    elif precision == "fp8":
        config.set_flag(trt.BuilderFlag.FP8)
        config.set_flag(trt.BuilderFlag.FP16)
        target_dtype = trt.fp8
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        target_dtype = trt.int8

        if calib_data is None:
            raise ValueError("INT8 requires --calib-data directory with .npy files")
        cache = calib_cache or engine_path.replace(".engine", ".cache")
        calibrator = NpzCalibrator(calib_data, cache, network)
        config.int8_calibrator = calibrator
        log.info("INT8 calibrator: data=%s cache=%s", calib_data, cache)
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    if precision in ("int8", "fp8"):
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    else:
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    fp32_count = 0
    reduced_count = 0
    total = network.num_layers

    for i in range(total):
        layer = network.get_layer(i)
        name = layer.name

        if layer.type in SKIP_LAYER_TYPES:
            continue

        has_float_output = any(
            layer.get_output(j).dtype in FLOAT_DTYPES
            for j in range(layer.num_outputs)
        )
        if not has_float_output:
            continue

        should_be_fp32 = False
        for substr in FP32_LAYER_SUBSTRINGS:
            if substr in name:
                should_be_fp32 = True
                break

        layer_type_name = str(layer.type).split(".")[-1]
        if layer_type_name in ("SOFTMAX", "REDUCE", "NORMALIZATION"):
            should_be_fp32 = True

        if should_be_fp32:
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                if layer.get_output(j).dtype in FLOAT_DTYPES:
                    layer.set_output_type(j, trt.float32)
            fp32_count += 1
        else:
            layer.precision = target_dtype
            if precision not in ("int8", "fp8"):
                for j in range(layer.num_outputs):
                    if layer.get_output(j).dtype in FLOAT_DTYPES:
                        layer.set_output_type(j, target_dtype)
            reduced_count += 1

    log.info("Layer precision: %d/%d in FP32, %d/%d in %s",
             fp32_count, total, reduced_count, total, precision.upper())

    log.info("Building engine (this may take several minutes)...")
    t0 = time.time()
    engine_bytes = builder.build_serialized_network(network, config)
    elapsed = time.time() - t0

    if engine_bytes is None:
        raise RuntimeError("Engine build failed")

    with open(engine_path, "wb") as f:
        f.write(bytes(engine_bytes))
    log.info("Saved: %s (%.1f MB) in %.1fs", engine_path, engine_bytes.nbytes / 1e6, elapsed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp8", "int8"])
    ap.add_argument("--workspace-gb", type=int, default=12)
    ap.add_argument("--calib-data", default=None,
                    help="Directory with .npy calibration files (required for INT8)")
    ap.add_argument("--calib-cache", default=None,
                    help="Path for calibration cache file")
    args = ap.parse_args()

    build_engine(args.onnx, args.output, args.precision, args.workspace_gb,
                 calib_data=args.calib_data, calib_cache=args.calib_cache)


if __name__ == "__main__":
    main()
