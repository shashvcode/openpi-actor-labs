#!/usr/bin/env python3
"""Build INT8 TRT engines from BF16 ONNX using real observation data for calibration.

Uses TRT's IInt8EntropyCalibrator2 with captured observations instead of random data.
"""
import logging
import pathlib
import sys

import numpy as np
import tensorrt as trt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ONNX_DIR = pathlib.Path("onnx_export")


class ObservationCalibrator(trt.IInt8EntropyCalibrator2):
    """Feeds real observation data for INT8 calibration."""

    def __init__(self, data_files, input_names, cache_path):
        super().__init__()
        self._cache_path = cache_path
        self._input_names = input_names
        self._batch_idx = 0

        self._data = []
        for f in data_files:
            d = np.load(f)
            self._data.append({k: np.ascontiguousarray(d[k]) for k in input_names if k in d})
        self._num_batches = len(self._data) * 20

        from cuda.bindings import runtime as cudart
        self._cudart = cudart
        self._device_buffers = {}
        for name in input_names:
            sample = self._data[0].get(name)
            if sample is not None:
                nbytes = sample.nbytes
                err, ptr = cudart.cudaMalloc(nbytes)
                assert err == 0 or (hasattr(err, "value") and err.value == 0)
                self._device_buffers[name] = (ptr, nbytes)

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self._batch_idx >= self._num_batches:
            return None
        data_entry = self._data[self._batch_idx % len(self._data)]
        ptrs = []
        for name in names:
            if name in data_entry and name in self._device_buffers:
                arr = data_entry[name]
                ptr, _ = self._device_buffers[name]
                self._cudart.cudaMemcpy(
                    ptr, arr.ctypes.data, arr.nbytes,
                    self._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
                )
                ptrs.append(ptr)
            else:
                ptrs.append(0)
        self._batch_idx += 1
        return ptrs

    def read_calibration_cache(self):
        if self._cache_path.exists():
            log.info("Reading calibration cache: %s", self._cache_path)
            return self._cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        log.info("Writing calibration cache: %s (%d bytes)", self._cache_path, len(cache))
        self._cache_path.write_bytes(cache)


def build_engine(onnx_path, engine_path, calibrator, cache_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    log.info("Parsing ONNX: %s", onnx_path)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read(), str(onnx_path)):
            for i in range(parser.num_errors):
                log.error("  %s", parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.BF16)
    config.int8_calibrator = calibrator

    log.info("Building INT8+BF16 engine (this may take a few minutes)...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Engine build failed")

    engine_path.write_bytes(engine_bytes)
    log.info("Saved engine: %s (%.1f MB)", engine_path, len(engine_bytes) / 1e6)


def main():
    data_files = list(pathlib.Path("/tmp").glob("trt_live_inputs_*.npz"))
    if not data_files:
        log.error("No calibration data found. Run TRT server with debug logging first.")
        sys.exit(1)
    log.info("Found %d calibration files: %s", len(data_files), [f.name for f in data_files])

    prefix_inputs = ["img0", "img1", "img2", "mask0", "mask1", "mask2", "tokens", "token_masks"]
    prefix_cache = ONNX_DIR / "prefix_int8_calib.cache"
    prefix_cal = ObservationCalibrator(data_files, prefix_inputs, prefix_cache)
    build_engine(
        ONNX_DIR / "prefix_encoder.onnx",
        ONNX_DIR / "prefix_encoder_int8_calibrated.engine",
        prefix_cal, prefix_cache,
    )

    denoise_inputs = ["state", "prefix_pad_masks", "x_t", "timestep", "kv_keys", "kv_values"]
    denoise_cache = ONNX_DIR / "denoise_int8_calib.cache"
    denoise_cal = ObservationCalibrator(data_files, denoise_inputs, denoise_cache)
    build_engine(
        ONNX_DIR / "denoise_step.onnx",
        ONNX_DIR / "denoise_step_int8_calibrated.engine",
        denoise_cal, denoise_cache,
    )

    log.info("Done! INT8 calibrated engines ready.")


if __name__ == "__main__":
    main()
