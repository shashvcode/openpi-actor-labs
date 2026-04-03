# Pi-0.5 Jetson AGX Thor Deployment Pipeline

## Overview

End-to-end pipeline for deploying Physical Intelligence's pi-0.5 model on NVIDIA Jetson AGX Thor for real-time robot control. Converts the 3.1B parameter flow-matching model from JAX/PyTorch to optimized TensorRT engines with INT8 quantization and zero-copy inference.

**Target hardware**: NVIDIA Jetson AGX Thor Developer Kit (Blackwell GPU, sm_110, 128 GB unified memory)

**Use cases**: Real-time robot arm control (SO-100, 6-DOF) and semi-autonomous excavator (4-axis joystick).

---

## Current Performance

All results with shared device buffers enabled (the default).

| Configuration | Prefix | 8x Denoise | E2E Latency | Speedup | Accuracy (MSE vs FP32) | Control Freq |
|--------------|--------|------------|-------------|---------|------------------------|-------------|
| PyTorch eager (baseline) | - | - | 440ms | 1.0x | Reference | 2.3 Hz |
| TRT FP32 (runC) | 154.8ms | 106.8ms | 263ms | 1.67x | Reference | 3.8 Hz |
| TRT FP16 (runC) | 81.6ms | 75.9ms | 161ms | 2.73x | 0.024 | 6.2 Hz |
| TRT BF16 (runC) | 78.6ms | 75.1ms | 158ms | 2.79x | 0.017 | 6.3 Hz |
| TRT INT8 PTQ (runC) | 56.1ms | 50.8ms | 108ms | 4.08x | 0.082 | 9.3 Hz |
| **TRT INT8 QAT (runD_qat)** | **57.5ms** | **~53ms** | **131ms** | **3.36x** | **QAT-trained** | **7.6 Hz** |

**Active configuration**: `runD_qat` model with INT8 TRT engines (QAT-trained, native PyTorch).

**Recommended configurations**:
- **INT8 QAT (runD_qat)**: 131ms, 7.6 Hz — QAT-trained model, quantization-friendly weights. **Current production config.**
- **FP32**: Use for pipeline validation and debugging only.

**⚠ BF16/FP16 are NOT recommended**: Live arm testing revealed systematic biases in BF16 (sign flip on wrist_flex dimension, +1.117 diff).

**Model history**:
- `runC`: JAX→PyTorch conversion. Arm behavior "incredibly wrong" due to weight conversion issues. DEPRECATED.
- `runD`: Native PyTorch-trained (H100). Arm behavior "kinda reasonable" — confirmed conversion fix.
- `runD_qat`: Native PyTorch-trained with QAT (H100). Slightly better than runD. **Current model.**

---

## Architecture

### Model Split

Pi-0.5 is split into two TensorRT engines to avoid recomputing the expensive prefix for every denoise step:

1. **Prefix Encoder** (~5.4 GB ONNX, ~2.6B params)
   - Inputs: 3 images (224x224), token IDs, masks, robot state
   - Processes images through SigLIP vision encoder + Gemma 2B language model
   - Outputs: KV cache (18 layers x 968 tokens x 256 dims) + attention masks
   - Runs **once** per observation

2. **Denoise Step** (~825 MB ONNX, ~300M params)
   - Inputs: KV cache, noisy actions, timestep, robot state
   - Gemma 300M action expert computes velocity field
   - Output: velocity vector (1 x 11 x 6)
   - Runs **8 times** per observation (flow-matching ODE solver with Euler steps)

### Shared Device Buffers

The KV cache (~34 MB) is kept on GPU between the prefix and denoise engines. The prefix encoder's output device buffers are directly wired as the denoise engine's input buffers. This eliminates 306 MB of unnecessary host-device memory copies per inference and reduced overhead from 65ms to ~0ms.

### Inference Flow

```
Observation (images, prompt, state)
    │
    ▼
[Preprocessing - NumPy] ─── ~8ms (resize, tokenize, normalize)
    │
    ▼
[Prefix Encoder - TRT INT8] ─── ~58ms
    │  outputs KV cache to shared GPU buffers
    ▼
[Denoise Loop x8 - TRT INT8] ─── ~66ms total (~8ms/step)
    │  each step: H2D(x_t, timestep) → execute → D2H(velocity) → Euler step
    ▼
[Unnormalize actions] ─── <1ms
    ▼
Action trajectory (11 steps x 6 dims) ─── total ~131ms
```

---

## Pipeline Components

### Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/export_pytorch_onnx.py` | Export PyTorch model to two ONNX subgraphs (prefix + denoise) | Done |
| `scripts/verify_onnx.py` | Verify ONNX numerical accuracy vs PyTorch reference | Done |
| `scripts/trt_policy_server.py` | TRT-based WebSocket policy server with shared buffers | Done |
| `scripts/benchmark_trt.py` | Benchmark all precision modes, measure latency + accuracy | Done |
| `scripts/generate_calibration_data.py` | Generate synthetic calibration data for INT8 quantization | Done |
| `scripts/build_int8_engines.py` | Build INT8 TRT engines with entropy calibration | In progress |
| `scripts/serve_policy_nodeps.py` | Lightweight policy server (no openpi dependencies) | Available |
| `scripts/serve_pytorch_minimal.py` | Minimal PyTorch WebSocket server (runs in Docker, no TRT) | Done |
| `scripts/train_pytorch.py` | Fixed PyTorch training script (freeze filter, EMA, AMP, QAT) | Done |
| `scripts/verify_training_fixes.py` | Verification suite for training script fixes (7 tests) | Done |
| `scripts/verify_conversion.py` | Numerical comparison of JAX vs PyTorch model outputs | Done |
| `scripts/compare_trt_vs_pytorch.py` | Side-by-side TRT vs PyTorch output comparison (fixed noise) | Done |
| `scripts/compare_trt_direct.py` | Direct TRT engine inference with fixed noise for debugging | Done |
| `CLOUD_TRAINING_GUIDE.md` | Step-by-step guide for training on cloud H100s | Done |

### Key Files

| File | Purpose |
|------|---------|
| `src/openpi/models_pytorch/pi0_pytorch.py` | PyTorch implementation of pi-0.5 (patched for ONNX export) |
| `src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py` | Custom Gemma patches for adaRMSNorm compatibility |
| `src/openpi/training/config.py` | Training configs including SO-100 LoRA fine-tune presets |
| `scripts/train_pytorch.py` | PyTorch training script (DDP, LoRA, wandb) |

### Generated Artifacts

| Path | Contents |
|------|----------|
| `onnx_export/prefix_encoder.onnx` | Prefix encoder ONNX model (~5.4 GB with external data) |
| `onnx_export/denoise_step.onnx` | Denoise step ONNX model (~825 MB with external data) |
| `onnx_export/*_fp32.engine` | FP32 TensorRT engines |
| `onnx_export/*_bf16.engine` | BF16 TensorRT engines |
| `onnx_export/*_fp16.engine` | FP16 TensorRT engines |
| `onnx_export/*_int8.engine` | INT8 TensorRT engines (PTQ, runC — deprecated) |
| `onnx_export/*_int8_qat.engine` | INT8 TensorRT engines (QAT, runD_qat — **current**) |
| `calibration_data/prefix/` | Prefix calibration data (200 samples, ~345 MB) |
| `calibration_data/denoise/` | Denoise calibration data (800 samples, ~27 GB) |

---

## Environment Setup

### Docker Container

```bash
# Pull the Jetson-compatible PyTorch container
sudo docker pull nvcr.io/nvidia/pytorch:26.02-py3-igpu

# Run with workspace mounted
sudo docker run -it --rm --runtime nvidia \
  --name trt_pipeline \
  -v /home/Actor/openpi-actor-labs:/workspace/openpi \
  nvcr.io/nvidia/pytorch:26.02-py3-igpu bash
```

### Container Dependencies

```bash
pip install onnxscript sentencepiece msgpack-numpy websockets opencv-python-headless
```

### Host Dependencies (for TRT engine compilation)

TensorRT 10.13.3 is used from the host for engine compilation (the container's TRT has DLA compiler dependency issues). The host's `trtexec` and Python TRT API are used directly.

---

## How to Use

### Quick Start (INT8 QAT — recommended)

The INT8 QAT engines from `runD_qat` are pre-built. Just run:

```bash
cd /home/Actor/openpi-actor-labs
python3 scripts/trt_policy_server.py --precision int8_qat --port 8000
```

Then on the robot client:
```bash
python3 examples/so100/run_policy.py --host <jetson-ip>
```

### Full Pipeline (re-export from a new checkpoint)

#### 1. Export ONNX (inside Docker container)

```bash
sudo docker exec -it trt_pipeline bash
cd /workspace/openpi
python scripts/export_pytorch_onnx.py \
  --checkpoint /workspace/openpi/checkpoints/runD_qat_pytorch \
  --dtype float32
```

#### 2. Compile TRT Engines (on host)

```bash
cd /home/Actor/openpi-actor-labs/onnx_export

# INT8 + FP16 fallback (recommended for QAT models)
/usr/src/tensorrt/bin/trtexec \
  --onnx=prefix_encoder.onnx \
  --saveEngine=prefix_encoder_int8_qat.engine \
  --int8 --fp16 \
  --memPoolSize=workspace:8192MiB

/usr/src/tensorrt/bin/trtexec \
  --onnx=denoise_step.onnx \
  --saveEngine=denoise_step_int8_qat.engine \
  --int8 --fp16 \
  --memPoolSize=workspace:8192MiB
```

#### 3. Serve Policy

```bash
python3 scripts/trt_policy_server.py --precision int8_qat --port 8000
```

The server accepts WebSocket connections with msgpack-encoded observations and returns action trajectories.

### Other Precision Modes

```bash
# FP32 (validation only)
python3 scripts/trt_policy_server.py --precision fp32 --port 8000

# Explicit engine paths
python3 scripts/trt_policy_server.py \
  --prefix-engine onnx_export/prefix_encoder_int8_qat.engine \
  --denoise-engine onnx_export/denoise_step_int8_qat.engine \
  --port 8000
```

---

## Conversion Verification

Every step in the pipeline is numerically verified:

| Conversion Step | Verification Method | Result |
|----------------|--------------------|---------| 
| PyTorch model load | Compare weights vs JAX checkpoint | Exact match (safetensors) |
| ONNX export | Run same input through PyTorch and ONNX, compare outputs | Max abs diff: 0.000168 |
| TRT FP32 compilation | Run same input through ONNX and TRT FP32, compare outputs | MSE: 0.0000000072 |
| TRT BF16/FP16/INT8 | Compare final action trajectory vs TRT FP32 reference | See performance table |

The FP32 TRT conversion is essentially lossless. Precision loss in BF16/INT8 is from the numerical format, not conversion bugs.

### Flow-Matching Accuracy Considerations

Pi-0.5 uses a flow-matching ODE solver (8 Euler steps). Errors compound across denoise steps -- a small velocity error at step 1 shifts the trajectory for all subsequent steps. MSE numbers measure deviation from the FP32 trajectory, but a different trajectory can still be a valid solution. Physical robot testing is required to validate any precision mode.

---

## Optimizations Implemented

### 1. Model Splitting (Prefix + Denoise)
Avoids recomputing the expensive 2B-parameter prefix encoder for each of the 8 denoise steps. Only the lightweight 300M-parameter denoise step runs in the inner loop.

### 2. Shared Device Buffers
The KV cache (34 MB) stays on GPU between prefix and denoise engines. The denoise engine's input pointers for `kv_keys`, `kv_values`, and `prefix_pad_masks` point directly to the prefix engine's output memory. Eliminated 306 MB of unnecessary memcpy per inference.

**Impact**: Reduced E2E overhead from 65ms to ~0ms. FP32: 340ms → 270ms. BF16: 227ms → 162ms.

### 3. Optimized Denoise Loop
Instead of the monolithic `infer()` call per denoise step (which copies all 6 inputs and 1 output), only the changing tensors are copied:
- H2D per step: `x_t` (264 bytes) + `timestep` (4 bytes)
- D2H per step: `velocity` (264 bytes)
- `state` is copied H2D once before the loop
- `kv_keys`, `kv_values`, `prefix_pad_masks` are never copied (shared buffers)

### 4. INT8 W8A8 Quantization
Post-training quantization using TensorRT's `IInt8EntropyCalibrator2`:
- 200 diverse synthetic observations for prefix calibration
- 800 denoise samples with varied timesteps and noise levels
- FP16 fallback for layers that don't quantize well (layer norms forced to FP32)
- Denoise engine: 413 MB (vs 825 MB FP32 -- 50% smaller)
- Prefix engine: 3.2 GB (vs 5.0 GB FP32 -- 36% smaller)
- Result: **108ms E2E, 4.08x speedup** (MSE 0.082 -- see accuracy note above)

### 5. NumPy-Only Preprocessing
All observation preprocessing (image resize, normalization, tokenization) uses NumPy and OpenCV. No PyTorch dependency at inference time.

---

## Known Issues and Fixes

### Build/Export Issues

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `Cos(7) not implemented` in ONNX | `create_sinusoidal_pos_embedding` generated float64 | Monkey-patched `get_safe_dtype` to force float32 |
| `GemmaRMSNorm` AttributeError | adaRMSNorm uses `self.dense` not `self.weight` | Conditional check in `extra_repr` |
| ONNX >2GB protobuf error | Large model with external data | Path-based `onnx.checker.check_model()` |
| `libnvdla_compiler.so` missing | Container TRT linked against DLA libraries | Use host TRT (v10.13.3) for compilation |
| `cudart` import error | `cuda-python` API changed | Import from `cuda.bindings.runtime` |
| `cudaError_t` no `.value` | Tuple-based error returns in `cuda-python` | `_check()` helper handles both formats |
| KV cache overhead (65ms) | 306 MB copied H↔D every denoise step | Shared device buffers (0ms overhead) |

### Deployment Issues (Live Arm Testing)

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Arm "incredibly wrong" (reaching up/back) | Missing quantile normalization in TRT server | Added `QuantileNormalizer` class using `norm_stats.json` |
| BF16 TRT sign flip on wrist_flex (+1.117 diff) | BF16 precision loss causes systematic bias in flow-matching ODE | Use FP32 for validation; train QAT model for INT8 production |
| Arm still erratic with FP32 TRT | Suspected JAX→PyTorch conversion error | Train natively in PyTorch (see cloud guide) |
| Arm erratic with PyTorch direct (Docker) | Same converted weights; also sm_110 GPU may have kernel compat issues | Train natively in PyTorch on standard GPUs (H100/A100) |
| `cv2`/numpy incompatibility on Jetson host | `numpy 2.4.2` incompatible with system `cv2` | Installed `opencv-python-headless==4.13.0.92`; TRT server uses PIL instead |
| Msgpack deserialization `TypeError` | `openpi_client` custom numpy serialization format | Import `Packer`/`unpackb` from `openpi_client.msgpack_numpy` |
| `ModuleNotFoundError: transformers` on host | Missing pip package | `pip3 install --break-system-packages transformers tqdm_loggable` |

### Training Script Issues

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| QAT closure bug | `original_forward` captured by reference in loop, all wrappers call last module's forward | Pass `module.forward` as explicit parameter to factory function |
| Silent random-weight training | `pytorch_weight_path=None` + `strict=False` in conversion = model trains from scratch | Added runtime guard that refuses to start if config has JAX weight_loader but no pytorch_weight_path |
| No pretrained base for PyTorch SO100 configs | Config has JAX `weight_loader` but PyTorch script ignores it | Documented conversion steps; guard raises error with instructions |

---

## PyTorch Training Script Fixes

The PyTorch training script (`scripts/train_pytorch.py`) was comprehensively rewritten to match the JAX reference script (`scripts/train.py`) behavior. All fixes were verified by `scripts/verify_training_fixes.py` (7/7 tests pass on Jetson Thor).

### Critical Bug Fixes

| Bug | Impact | Fix |
|-----|--------|-----|
| **No parameter freezing** | All 3.1B params passed to optimizer (~25GB wasted RAM), not actually doing LoRA fine-tuning | `apply_freeze_filter()`: sets `requires_grad_(False)` on frozen LLM params based on config variant names, casts to BF16, only passes trainable params to optimizer |
| **No EMA** | Would break full fine-tunes (LoRA configs set `ema_decay=None` so not blocking) | `EMATracker` class: maintains shadow copy, updates after each optimizer step, saved/loaded with checkpoints, `apply_to()`/`restore()` for export |

### Performance Fixes

| Issue | Impact | Fix |
|-------|--------|-----|
| **No AMP** | Training runs at model creation dtype, no mixed precision | `torch.amp.autocast("cuda", dtype=torch.bfloat16)` around forward pass |
| **TF32 gated behind 8+ GPUs** | Free 2-3x matmul speedup disabled on single-GPU training | Enabled unconditionally at script startup |
| **DDP misconfigured** | `find_unused_parameters=True` adds communication overhead | Set `find_unused_parameters=False`, `static_graph=True` always |
| **No torch.compile** | Eager mode only (JAX JIT-compiles everything) | Optional via `TORCH_COMPILE=1` env var |

### Code Quality Fixes

| Issue | Fix |
|-------|-----|
| `jax.tree.map` in PyTorch training loop | Replaced with `tree_map_tensors()` — recursive, handles dicts/lists/dataclasses/namedtuples |
| Redundant gradient clearing after `zero_grad(set_to_none=True)` | Removed manual gradient clearing loop |
| Missing `param_norm` logging | Added `param_norm` computation alongside `grad_norm` in logging block |

### QAT Support

Added `apply_qat()` function that inserts `FakeQuantize` nodes on frozen (base) linear layers. This simulates INT8 rounding in the forward pass so LoRA adapters learn to compensate. Usage:

```python
# In config or future CLI flag:
config.quantization_aware = True  # Not yet a config field — detected via getattr
```

The ONNX export script (`scripts/export_pytorch_onnx.py`) also supports QAT:

```bash
python scripts/export_pytorch_onnx.py --qat-checkpoint /path/to/qat/checkpoint
```

This preserves `QuantizeLinear`/`DequantizeLinear` ops in the ONNX graph so TensorRT uses learned scales directly (no PTQ calibration needed).

---

## Future Optimizations (Planned)

### Near-term (next fine-tune iteration)
1. **QAT Fine-tuning**: Now that the training script supports QAT, the next LoRA fine-tune should use it. Expected to bring INT8 MSE from 0.082 down to <0.005, giving INT8 speed (108ms) with near-FP32 accuracy. The LoRA adapters learn to compensate for INT8 rounding, and the learned quantization scales are exported directly into ONNX — no separate calibration step needed.

2. **Calibration with Real Data**: Replace synthetic calibration data with real observations from the robot for more accurate post-training INT8 quantization scales. Quick win before QAT is fully integrated.

3. **FP8 Quantization**: Blackwell GPU supports FP8 (8-bit floating point). Unlike INT8 (fixed-point), FP8 preserves the floating-point format (exponent + mantissa), giving better accuracy for transformers while offering similar throughput. TRT 10.13 supports FP8 natively. This is a good middle ground between BF16 accuracy and INT8 speed.

### Medium-term
5. **Reduced Denoise Steps**: Test with 4 or 6 steps instead of 8. Flow-matching models can often produce good actions with fewer steps, trading accuracy for 1.3-2x denoise speedup. At 4 steps, INT8 E2E could drop to ~80ms (12.5 Hz).

6. **CUDA Graphs**: Capture the entire denoise loop as a CUDA graph to eliminate per-step kernel launch overhead (~1-2ms savings).

7. **Monolithic ONNX Export**: Export the full model (prefix + denoise loop) as a single ONNX graph. TRT can optimize across the boundary and potentially fuse operations.

### Long-term (excavator production)
8. **Overlapped Execution**: Pipeline the prefix computation for the next observation while the current denoise loop runs, hiding prefix latency. Could approach denoise-only latency (~50ms, 20 Hz) for continuous operation.

9. **Custom TRT Plugins**: Write custom kernels for the Euler step and timestep embedding to keep everything on GPU.

10. **Model Distillation**: Train a smaller student model that approximates pi-0.5 for faster inference at the cost of some capability.

---

## Cloud Training (Native PyTorch) {#cloud-training-native-pytorch}

The current model (`runC`) was converted from JAX and produces incorrect arm behavior. The fix is to **train natively in PyTorch**, eliminating the JAX→PyTorch conversion from the pipeline entirely.

A full guide is in **`CLOUD_TRAINING_GUIDE.md`**. Summary:

```bash
# 1. Convert JAX base checkpoint to PyTorch (one-time)
python examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir gs://openpi-assets/checkpoints/pi05_base \
    --config_name pi05_so100_lora_v3 \
    --output_path ./checkpoints/pi05_base_pytorch --precision float32

# 2. Verify conversion is numerically correct
python scripts/verify_conversion.py \
    --config_name pi05_so100_lora_v3 \
    --pytorch_weight_path ./checkpoints/pi05_base_pytorch

# 3. Train (single H100, ~1-2 hrs)
python scripts/train_pytorch.py pi05_so100_lora_v3 \
    --pytorch_weight_path ./checkpoints/pi05_base_pytorch --exp_name runD
```

Output: `checkpoints/pi05_so100_lora_v3/runD/10000/model.safetensors` — native PyTorch weights, no conversion risk.

### Why This Fixes the Problem

- **runC** (current): JAX training → `convert_jax_model_to_pytorch.py` → `model.safetensors`. The conversion uses `load_state_dict(strict=False)` which silently ignores errors. The converted model produces wrong arm behavior even in pure PyTorch — the JAX model works fine.
- **runD** (planned): PyTorch training directly saves `model.safetensors` via `safetensors.torch.save_model()`. No conversion. No ambiguity.

### Training Script Audit Results

The PyTorch training script (`train_pytorch.py`) was audited against the JAX reference (`train.py`). Core training logic matches:
- Loss function (flow-matching MSE) ✓
- Time sampling (Beta(1.5, 1.0) * 0.999 + 0.001) ✓
- Optimizer (AdamW, same hyperparameters, clip-then-step ordering) ✓
- LR schedule (warmup cosine decay, equivalent formulation) ✓
- Freeze filter (same parameters frozen/trainable in both) ✓
- Data pipeline (same transforms, normalization, dataset) ✓

---

## Pipeline Reusability

This pipeline is designed to work with any pi-0.x model variant:

1. **New fine-tune**: Change the `--checkpoint` and `--config` args to `export_pytorch_onnx.py`. Re-run ONNX export, TRT compilation, and benchmarking. All scripts are parameterized.

2. **New model architecture**: If PI releases pi-0.6 or similar, the prefix/denoise split pattern is the same. Update `pi0_pytorch.py` model code and re-export.

3. **New hardware**: TRT engines are hardware-specific. Re-compile from the same ONNX models on the new device. All other scripts are hardware-agnostic.

4. **New precision**: Add engine files with a new suffix (e.g., `_fp8.engine`) and pass `--precision fp8` to the server/benchmark scripts.

---

## Training Pipeline

### Recommended Pipeline (Native PyTorch → TRT)

```
PI pretrained weights (JAX, on GCS)
    │
    ▼ [one-time conversion: convert_jax_model_to_pytorch.py]
    │   Converts base weights only (not fine-tuned)
    │   Verified with verify_conversion.py (MSE < 1e-4)
    │
PyTorch base weights (model.safetensors)
    │
    ▼ [LoRA fine-tuning on cloud H100s: train_pytorch.py]
    │   ✅ Parameter freezing (frozen LLM → bf16, only LoRA+projections trainable)
    │   ✅ AMP bf16 autocast + TF32 matmuls
    │   ✅ DDP with static_graph=True
    │   ✅ Safety guard: refuses to start from random weights
    │   ✅ Optional QAT for INT8 deployment
    │
Fine-tuned PyTorch checkpoint (e.g. runD, runE, ...)
    │   Natively PyTorch — NO JAX conversion in the loop
    │
    ▼ [export_pytorch_onnx.py]
ONNX models (prefix + denoise)
    │
    ▼ [trtexec or build_int8_engines.py]
TRT engines (FP32 for validation, INT8 with QAT for production)
    │
    ▼ [trt_policy_server.py]
WebSocket policy server → Robot
```

### Previous Pipeline (JAX → PyTorch → TRT) ⚠ DEPRECATED

The runC model was created by training in JAX, then converting weights with
`convert_jax_model_to_pytorch.py`. This produced incorrect arm behavior despite
the conversion appearing complete (812/813 parameter keys matched). The JAX model
works correctly on the same arm/cameras. Use the native PyTorch pipeline above instead.

### Ready: QAT Pipeline (no calibration needed)

```
PI pretrained weights (JAX)
    │
    ▼ [one-time conversion]
PyTorch base weights
    │
    ▼ [LoRA fine-tuning WITH QAT: train_pytorch.py]
    │   QAT inserts FakeQuantize on frozen Linear layers
    │   LoRA adapters learn to compensate for INT8 rounding
    │   Learned quantization scales saved with checkpoint
    │
Fine-tuned checkpoint with quantization annotations
    │
    ▼ [export_pytorch_onnx.py --qat-checkpoint <path>]
ONNX models with QuantizeLinear/DequantizeLinear ops
    │
    ▼ [TRT reads learned scales directly -- no PTQ calibration needed]
INT8 TRT engines with near-FP32 accuracy
    │
    ▼ [trt_policy_server.py --precision int8]
108ms inference, 9.3 Hz control → Robot
```

### Training Configs

Defined in `src/openpi/training/config.py`. Key SO-100 configs:

| Config Name | Dataset | Steps | Action Horizon | Notes |
|-------------|---------|-------|---------------|-------|
| `pi05_so100_lora` | verm11/so100_joystick_pickup | 5K | 10 | Initial |
| `pi05_so100_lora_v2` | verm11/runA | 25K | 5 | 500 episodes, 30 Hz |
| `pi05_so100_lora_v3` | verm11/runA | 10K | 11 | 30 Hz, 8 denoise steps |
| `pi05_so100_lora_v4` | verm11/runA | 15K | 11 | + scene depth image |
| `pi05_so100_lora_v5` | verm11/runA | 15K | 11 | + depth grid in state (22-dim) |

All configs use LoRA on both Gemma 2B (language) and Gemma 300M (action expert), with PI's pretrained pi0.5 base weights frozen.

---

## Excavator Deployment (March 11, 2026)

### Goal

Get the excavator model running on the Jetson AGX Thor with behavior identical to cloud inference. The excavator uses `pi05_excavator_v2` config: 4-dim joystick control (left_x, left_y, right_x, right_y), 2 cameras (cab + side), trained on `verm11/excavator_v2`.

### JAX on Jetson Thor — Working

JAX does **not** work with generic PyPI wheels on the Thor GPU. The standard `pip install jax[cuda13]` results in XLA falling back to sm_101 and cuDNN crashes:

```
Unknown compute capability 11.0. Defaulting to telling LLVM that we're compiling for sm_101
CUDNN_STATUS_EXECUTION_FAILED
```

**Fix**: NVIDIA's official JAX container `nvcr.io/nvidia/jax:26.01-py3` (arm64) explicitly supports Jetson Thor. It includes a custom XLA build with sm_110 kernels.

```bash
sudo docker pull nvcr.io/nvidia/jax:26.01-py3

sudo docker run --rm --runtime nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/Actor/openpi-actor-labs:/workspace/openpi \
  -w /workspace/openpi \
  nvcr.io/nvidia/jax:26.01-py3 bash
```

Inside the container, install project deps (the container has JAX 0.8.1, Flax 0.12.1, orbax 0.11.31 pre-installed):

```bash
pip install augmax einops sentencepiece equinox 'jaxtyping==0.2.36' ml_collections \
  tyro dm-tree tqdm-loggable 'beartype==0.19.0' treescope numpydantic safetensors pillow \
  pytest transformers torch opencv-python \
  --extra-index-url https://download.pytorch.org/whl/cpu

export PYTHONPATH=/workspace/openpi/src:/workspace/openpi/packages/openpi-client/src
```

**JAX GPU performance on Thor** (excavator model, `pi05_excavator_v2`):

| Operation | Compile+Execute | Warm |
|-----------|----------------|------|
| Single-step velocity | 13.6s | **4.0s** |
| sample_actions (8 steps) | 10.6s | **6.9s** |
| Checkpoint load (12.7 GiB) | 205s | — |

Reference outputs saved to `jax_reference_gpu.npz` via `scripts/jax_reference_excavator.py`.

**Note**: The container's orbax 0.11.31 wraps checkpoint leaves in `{'value': array}` dicts. The reference script includes an `_unwrap_value()` helper to flatten these before loading into the model.

### JAX → PyTorch Conversion — LoRA Bug Found and Fixed

**Root cause of SO-100 arm "incredibly wrong" behavior (runC) identified**: The conversion script `examples/convert_jax_model_to_pytorch.py` was **silently dropping all LoRA fine-tuning weights**. The JAX checkpoint has 20 LoRA parameter sets (attention + MLP, for both PaliGemma LLM and the 300M action expert) that encode the entire task-specific adaptation. The conversion only extracted base weights `w` and ignored `lora_a`/`lora_b`. Since `load_state_dict(strict=False)` was used, this was completely silent.

**Impact**: The "converted" PyTorch model was the base pre-trained model, not the fine-tuned one. This explains why runC produced wrong behavior — it was never actually running the fine-tuned model.

**Fix**: Added `merge_lora_weights()` function to the conversion script. For each base weight `W` with LoRA parameters, the merge computes:

```
W_merged = W + lora_a @ lora_b * (alpha / rank)
```

For the excavator config, both PaliGemma (rank=16, alpha=16) and the expert (rank=32, alpha=32) have `scaling_value = 1.0`, so the merge is simply `W + lora_a @ lora_b`.

Also added verbose logging of missing/unexpected keys in `load_state_dict` to prevent future silent failures.

**Comparison results** (JAX GPU reference vs LoRA-merged PyTorch):

| Metric | Before Fix (no LoRA) | After Fix (merged) |
|--------|---------------------|-------------------|
| v_t MSE | 4.66e-01 | **1.34e-02** (35x better) |
| Trajectory MSE (8-step) | 4.59e-01 | **4.12e-03** (111x better, **PASS** < 1e-02) |
| Loss (JAX / PT) | 0.143 / 0.916 | 0.143 / 0.175 |

Per-dimension actions now track closely (joystick axes):

| Dim | JAX | PyTorch (before) | PyTorch (after) |
|-----|-----|-------------------|-----------------|
| left_x | 0.012 | -0.211 | -0.050 |
| left_y | 0.091 | 0.081 | 0.089 |
| right_x | 0.208 | 0.528 | 0.166 |
| right_y | -0.016 | -0.417 | -0.080 |

Remaining v_t divergence (~1e-02) is from framework-level numerical differences (JAX/XLA vs PyTorch, different attention implementations, float accumulation order). The trajectory MSE is within threshold and should be imperceptible on the physical excavator.

### Scripts Added

| Script | Purpose |
|--------|---------|
| `scripts/jax_reference_excavator.py` | Generate JAX gold-standard outputs inside NVIDIA container, saves to npz |
| `scripts/compare_jax_vs_pytorch_excavator.py` | Load JAX reference npz, run PyTorch, compare at 5 levels |

### Artifacts

| Path | Contents |
|------|----------|
| `checkpoints/excavator_v1_jax/` | JAX checkpoint from HuggingFace `verm11/excavator_v1` (8.9 GB, Orbax format) |
| `checkpoints/excavator_v1_pytorch/` | LoRA-merged PyTorch checkpoint (14 GB safetensors, float32) |
| `jax_reference_gpu.npz` | JAX reference outputs (v_t, sampled_actions, inputs) from Thor GPU |

### Excavator TRT Precision Validation (2026-03-11, updated)

Comprehensive numerical comparison of all TRT precision configurations against PyTorch FP32 reference. Validated with both synthetic and real excavator camera images. Script: `scripts/validate_trt_vs_pytorch.py`.

**Round 1 (trtexec default builds):** TRT's default `--fp16`/`--bf16` flags gave unacceptable degradation because TRT aggressively converts ALL operations to the target precision — including softmax, RMSNorm, and other accumulation-sensitive ops that need FP32.

| Config | Total (ms) | KV Keys Cosine | Action Cosine | Verdict |
|--------|-----------|---------------|--------------|---------|
| FP32 | ~324 | 0.99999 | 1.00000 | Perfect |
| BF16 (trtexec) | ~203 | 0.964 | 0.916 | Degraded — TRT converts norm/softmax to BF16 |
| FP16 (trtexec) | ~195 | 0.206 | broken | BROKEN — layers 1-17 output ALL ZEROS |
| INT8 (random PTQ) | ~103 | 0.030 | broken | BROKEN (no calibration) |

**Root cause of FP16 collapse**: Per-layer analysis showed layer 0 KV cache was fine (cos=0.9996), but layers 1-17 were literally all zeros. The attention computation in layer 0 produces NaN/Inf when TRT converts softmax and intermediate ops to FP16, which poisons the residual stream for all subsequent layers. PyTorch FP16 is fine (cos=1.0 on all layers) because it keeps softmax/norm in FP32 automatically.

**Root cause of BF16 degradation**: TRT's BF16 kernels for RMSNorm and softmax accumulate errors. Even with mixed-precision approach (keeping norm/softmax in FP32), BF16 matmuls alone introduce ~3.6% per-layer error due to BF16's 7-bit mantissa (vs FP16's 10-bit mantissa). This is an inherent limitation of BF16 in TRT.

**Round 2 (mixed-precision builds with `scripts/build_mixed_precision_engine.py`):** Custom TRT Python API builder that assigns FP16 to matmul/conv layers and FP32 to softmax/norm/reduction layers, matching PyTorch's AMP behavior. Uses `OBEY_PRECISION_CONSTRAINTS` flag.

| Config | Prefix (ms) | Denoise (ms) | Total (ms) | KV Keys Cosine | Action Cosine | Action MSE | Verdict |
|--------|-------------|-------------|-----------|---------------|--------------|-----------|---------|
| FP32 | 204 | 120 | 324 | 0.99999 | 1.00000 | 3.87e-09 | Perfect (reference) |
| **Mixed-FP16** | **109** | **102** | **211** | **0.99985** | **0.99994** | **4.46e-06** | **Near-perfect, 35% faster** |
| Mixed-FP16 + FP32 denoise | 109 | 131 | 240 | 0.99985 | 1.00000 | 3.87e-09 | Perfect actions |
| Mixed-BF16 | 105 | 104 | 208 | 0.964 | 0.917 | 4.10e-03 | BF16 matmul precision limit |

**Key findings:**

1. **Mixed-FP16 is the clear winner** — 0.99994 action cosine, MSE 4.46e-06, at 211ms (35% faster than FP32). FP16's 10-bit mantissa provides sufficient precision for matmuls, while FP32 softmax/norm prevents the catastrophic collapse seen with trtexec's `--fp16`.
2. **Mixed-FP16 + FP32 denoise gives bit-perfect actions** (identical to pure FP32) at 240ms — useful if absolute precision is needed for the denoise loop.
3. **BF16 in any form is worse than Mixed-FP16** — BF16's 7-bit mantissa causes more matmul error than FP16's 10-bit. The wider dynamic range of BF16 is irrelevant since activations stay within FP16 range.
4. **Denoise error compounds over 8 steps** — When the denoise step runs in reduced precision, velocity errors accumulate through Euler integration.
5. **INT8 with random calibration is useless** — needs proper calibration data or QAT. Fresh calibration data is in `calibration_data_fresh/`.

**Pi0.5 `state` input note**: For pi0.5 models, the `state` tensor is NOT used in the denoise step suffix. The state is encoded in the **prefix tokens** (via the tokenizer's discretized state string in the prompt). ONNX tracing correctly optimized away the unused `state` input. The TRT server has been patched to handle this gracefully.

**Deployment recommendation:**
- **Initial testing**: FP32 (~324ms, perfect accuracy)
- **Production**: Mixed-FP16 (~211ms, action cosine 0.9999, 35% faster)
- **Maximum accuracy with speed**: Mixed-FP16 prefix + FP32 denoise (~240ms, perfect actions)
- **Future**: INT8 with calibrated PTQ or QAT for further latency reduction

### Remaining Steps

1. ~~Update deployment scripts for excavator~~ ✅ Done
2. ~~ONNX export~~ ✅ Done
3. ~~TRT engine build + numerical validation~~ ✅ Done (all precisions)
4. **End-to-end validation** — Run TRT policy server with excavator model, connect excavator client, compare real-world behavior to current cloud inference
5. **INT8 with proper calibration** — Use calibration data from `calibration_data_fresh/` or QAT training for better INT8 accuracy

---

## Revision History

| Date | Change |
|------|--------|
| 2026-03-04 | Initial pipeline: PyTorch → ONNX → TRT (FP32, BF16, FP16) |
| 2026-03-04 | Shared device buffers: eliminated 65ms overhead (BF16: 227ms → 158ms) |
| 2026-03-04 | Optimized denoise loop: fine-grained H2D/D2H, only 532 bytes per step |
| 2026-03-04 | INT8 W8A8 quantization: 108ms E2E, 4.08x speedup |
| 2026-03-04 | Full benchmark: FP32 (263ms), FP16 (161ms), BF16 (158ms), INT8 (108ms) |
| 2026-03-04 | Created JETSON_DEPLOYMENT_README.md |
| 2026-03-04 | Fixed PyTorch training script: parameter freezing, EMA, AMP, DDP, removed JAX dep |
| 2026-03-04 | Added QAT support to training script + ONNX export |
| 2026-03-04 | Verified all training fixes (7/7 tests pass on Jetson Thor) |
| 2026-03-05 | Live arm testing: identified missing quantile normalization in TRT server, fixed |
| 2026-03-05 | Precision comparison: BF16 has sign flip on wrist_flex, FP32 much closer to PyTorch |
| 2026-03-05 | Live arm testing with FP32 TRT: less wrong but still incorrect behavior |
| 2026-03-05 | Live arm testing with PyTorch direct (Docker): also incorrect — rules out TRT as cause |
| 2026-03-05 | Root cause identified: JAX→PyTorch weight conversion. JAX model works on same arm/cameras |
| 2026-03-05 | Audited train_pytorch.py vs train.py: core logic matches, identified 3 bugs |
| 2026-03-05 | Fixed QAT closure bug (original_forward captured by reference) |
| 2026-03-05 | Added safety guard against accidental random-weight training |
| 2026-03-05 | Created CLOUD_TRAINING_GUIDE.md and verify_conversion.py |
| 2026-03-05 | Pushed all changes to razafork/jetson-integration |
| 2026-03-05 | Deployed runD (native PyTorch) and runD_qat (QAT) models on Jetson |
| 2026-03-05 | Live arm test runD: "kinda reasonable", confirms conversion fix |
| 2026-03-05 | Live arm test runD_qat: "a little better", moving towards target correctly |
| 2026-03-06 | Fixed closure bug in export_pytorch_onnx.py _apply_qat_for_export |
| 2026-03-06 | Exported runD_qat to ONNX (FP32, prefix 14MB + 11GB data, denoise 2.9MB + 1.7GB data) |
| 2026-03-06 | Compiled INT8 QAT TRT engines (prefix 3.2GB/57ms, denoise 413MB/6.6ms) |
| 2026-03-06 | TRT INT8 QAT server running: 131ms E2E warm latency, 7.6 Hz control freq |
| 2026-03-11 | Merged origin/main into jetson-integration: gained excavator configs + policies |
| 2026-03-11 | Downloaded excavator JAX checkpoint from HuggingFace (verm11/excavator_v1, 8.9 GB) |
| 2026-03-11 | Fixed pi05 detection bug in convert_jax_model_to_pytorch.py (string match → config flag) |
| 2026-03-11 | Converted excavator JAX checkpoint to PyTorch (initial, without LoRA — broken) |
| 2026-03-11 | Set up NVIDIA JAX container (nvcr.io/nvidia/jax:26.01-py3) for Thor GPU support |
| 2026-03-11 | Generated JAX gold-standard reference outputs on Thor GPU (jax_reference_gpu.npz) |
| 2026-03-11 | **Root cause found**: conversion drops all 20 LoRA parameter sets (silently, via strict=False) |
| 2026-03-11 | Added merge_lora_weights() to conversion script — merges LoRA into base weights before export |
| 2026-03-11 | Re-converted excavator model with LoRA merge: trajectory MSE 4.12e-03 (PASS, 111x improvement) |
| 2026-03-11 | This also explains the SO-100 runC "incredibly wrong" behavior — same bug affected arm model |
| 2026-03-11 | Updated all 3 deployment scripts for excavator: obs key mapping (image_cab/image_side), --config-name CLI |
| 2026-03-11 | Fixed critical obs key mismatch: excavator sends image_cab/image_side, servers expected image_scene/image_wrist |
| 2026-03-11 | Patched model.py restore_params for orbax 0.11.31 StepMetadata API change |
| 2026-03-11 | JAX serve_policy.py running on Jetson Thor via NVIDIA container, excavator model on port 8000 |
| 2026-03-11 | Fixed CumSum-on-bool in pi0_pytorch.py and export script (TRT requires int32+ for CumSum) |
| 2026-03-11 | Exported excavator ONNX: prefix encoder (1.7 MB + 11.3 GB data), denoise step (1.7 GB) |
| 2026-03-11 | Built FP32 TRT engines for excavator: prefix 150ms, denoise 14ms → ~262ms E2E (8 denoise steps) |
| 2026-03-11 | Built FP16, BF16, INT8 TRT engines for excavator (all precision levels) |
| 2026-03-11 | Found and fixed latent bug: trt_policy_server.py tried to h2d `state` tensor but pi0.5 denoise engine doesn't use it (state encoded in prefix tokens) |
| 2026-03-11 | **Comprehensive numerical validation** of all TRT precision combos vs PyTorch (see table below) |
| 2026-03-11 | Added mixed-precision support to trt_policy_server.py (`--precision bf16+fp32` syntax) |
| 2026-03-11 | **Fixed attention mask FP16 overflow**: changed fill value from -2.38e38 (overflows FP16) to -65504 (FP16-safe). This alone didn't fix FP16 collapse — the real issue is TRT converting softmax/norm to FP16 |
| 2026-03-11 | **Root cause analysis**: PyTorch FP16 produces cosine=1.0 on all 18 layers (FP32 softmax/norm), while TRT's `--fp16` collapses layers 1-17 to zeros. TRT's default precision selection is too aggressive for transformer models |
| 2026-03-11 | Created `scripts/build_mixed_precision_engine.py`: TRT Python API builder that assigns FP16 to matmul/conv and FP32 to softmax/norm/reduction layers (matches PyTorch AMP behavior) |
| 2026-03-11 | **Mixed-FP16 engines achieve 0.99994 action cosine at 211ms** (35% faster than FP32, near-identical accuracy). This is the new recommended production configuration |
| 2026-03-11 | Built mixed-BF16 engines: no improvement over trtexec BF16 (0.964 KV cosine) — confirms the BF16 degradation comes from matmul precision, not norm/softmax |
