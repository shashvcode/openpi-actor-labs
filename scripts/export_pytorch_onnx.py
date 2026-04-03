#!/usr/bin/env python3
"""Export PI0 PyTorch model to ONNX (prefix encoder + denoise step).

Splits the model into two ONNX subgraphs:
  1. Prefix encoder: images + text -> KV cache + prefix_pad_masks
  2. Denoise step: KV cache + state + noisy actions + timestep -> velocity

Usage (inside the trt_pipeline container):
  python /workspace/openpi/scripts/export_pytorch_onnx.py
"""

import dataclasses
import json
import logging
import os
import pathlib
import sys
import time
import types

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

WORKSPACE = pathlib.Path(os.environ.get("OPENPI_WORKSPACE", "/workspace/openpi"))
if not WORKSPACE.exists():
    WORKSPACE = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIR = WORKSPACE / "onnx_export"

sys.path.insert(0, str(WORKSPACE / "src"))

ROBOT_CONFIGS = {
    "pi05_excavator_v2": {
        "action_dim": 4,
        "checkpoint_dir": WORKSPACE / "checkpoints" / "excavator_v1_pytorch",
    },
    "pi05_so100": {
        "action_dim": 6,
        "checkpoint_dir": WORKSPACE / "checkpoints" / "runC_pytorch",
    },
}


@dataclasses.dataclass
class ModelConfig:
    """Minimal config matching the fields PI0Pytorch.__init__ reads."""

    pi05: bool = True
    action_dim: int = 6
    action_horizon: int = 11
    paligemma_variant: str = "gemma_2b_lora"
    action_expert_variant: str = "gemma_300m_lora"
    dtype: str = "bfloat16"


class PrefixEncoderWrapper(nn.Module):
    """Wraps prefix encoding + VLM forward for ONNX export.

    Inputs:  3 images [B,3,224,224], 3 masks [B], tokens [B,200], token_masks [B,200]
    Outputs: prefix_pad_masks [B,P], kv_keys [L,B,H,P,D], kv_values [L,B,H,P,D]
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img0, img1, img2, mask0, mask1, mask2, tokens, token_masks):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            [img0, img1, img2], [mask0, mask1, mask2], tokens, token_masks
        )

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks.to(torch.int32), dim=1) - 1
        prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)

        self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        keys_stacked = torch.stack(past_key_values.key_cache, dim=0)
        values_stacked = torch.stack(past_key_values.value_cache, dim=0)

        return prefix_pad_masks, keys_stacked, values_stacked


class DenoiseStepWrapper(nn.Module):
    """Wraps the denoise step for ONNX export.

    The custom Gemma attention (use_cache=False) reads from the cache via indexing
    without mutation, so we can safely pre-fill a DynamicCache from stacked tensors.
    """

    def __init__(self, model, num_kv_layers):
        super().__init__()
        self.model = model
        self.num_kv_layers = num_kv_layers

    def forward(self, state, prefix_pad_masks, x_t, timestep, keys_stacked, values_stacked):
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        for i in range(self.num_kv_layers):
            cache.key_cache.append(keys_stacked[i])
            cache.value_cache.append(values_stacked[i])

        v_t = self.model.denoise_step(state, prefix_pad_masks, cache, x_t, timestep)
        return v_t


def _patch_float64_to_float32():
    """Replace float64 with float32 in sinusoidal embedding to ensure ONNX compatibility."""
    import openpi.models_pytorch.pi0_pytorch as _pi0

    _original_get_safe = _pi0.get_safe_dtype

    def _patched_get_safe(target_dtype, device_type):
        if target_dtype == torch.float64:
            return torch.float32
        return _original_get_safe(target_dtype, device_type)

    _pi0.get_safe_dtype = _patched_get_safe
    log.info("Patched get_safe_dtype: float64 → float32 for ONNX compatibility")


def load_model(device="cpu", export_dtype="float32", qat_checkpoint=None, checkpoint_override=None, action_dim=6):
    """Load PI0Pytorch without torch.compile, then load safetensors weights.

    If qat_checkpoint is provided, loads the QAT-trained checkpoint and applies
    fake quantization nodes so the ONNX export preserves QuantizeLinear/DequantizeLinear
    ops. TensorRT can then use the learned scales directly (no PTQ calibration needed).
    """
    import safetensors.torch
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    _patch_float64_to_float32()

    config = ModelConfig(action_dim=action_dim)

    log.info("Creating PI0Pytorch model (action_dim=%d, torch.compile disabled)...", action_dim)
    t0 = time.time()
    original_compile = torch.compile
    torch.compile = lambda fn, **kwargs: fn
    try:
        model = PI0Pytorch(config)
    finally:
        torch.compile = original_compile
    log.info("Model created in %.1fs", time.time() - t0)

    ckpt_dir = pathlib.Path(checkpoint_override) if checkpoint_override else None
    safetensors_path = ckpt_dir / "model.safetensors"
    log.info("Loading weights from %s...", safetensors_path)
    t0 = time.time()
    safetensors.torch.load_model(model, str(safetensors_path), device=device)
    log.info("Weights loaded in %.1fs", time.time() - t0)

    if qat_checkpoint:
        log.info("QAT mode: inserting fake quantization nodes for ONNX export...")
        _apply_qat_for_export(model, ckpt_dir)

    if export_dtype == "float32":
        log.info("Converting model to float32 for ONNX export...")
        model = model.float()
    elif export_dtype == "float16":
        log.info("Converting model to float16 for ONNX export...")
        model = model.half()
    elif export_dtype == "bfloat16":
        log.info("Keeping model in native bfloat16 for ONNX export...")

    model.eval()
    model.to(device)
    return model


def _apply_qat_for_export(model, ckpt_dir):
    """Restore trained FakeQuantize observers for ONNX export.

    Loads the learned scales/zero-points from quant_observers.pt (saved during
    QAT training) and attaches them to matching Linear layers. The ONNX graph
    will contain QuantizeLinear/DequantizeLinear ops with the correct trained
    scales so TensorRT can build INT8 engines without PTQ calibration.

    Falls back to fresh (untrained) observers with a warning if the saved
    state file is missing.
    """
    from torch.ao.quantization import default_weight_fake_quant

    quant_path = pathlib.Path(ckpt_dir) / "quant_observers.pt"
    if quant_path.exists():
        quant_state = torch.load(quant_path, weights_only=True)
        log.info("  Loaded trained QAT observer states from %s (%d layers)", quant_path, len(quant_state))
    else:
        quant_state = {}
        log.warning("  quant_observers.pt not found at %s — using fresh (untrained) observers", quant_path)

    count = 0
    restored = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            fq = default_weight_fake_quant()
            if name in quant_state:
                fq.load_state_dict(quant_state[name])
                restored += 1
            module.add_module("weight_fake_quant", fq)

            def make_fq_forward(mod, orig_forward):
                def fq_forward(x):
                    mod.weight.data = mod.weight_fake_quant(mod.weight.data)
                    return orig_forward(x)
                return fq_forward

            module.forward = make_fq_forward(module, module.forward)
            count += 1

    log.info("  Applied fake quantization to %d Linear layers (%d with trained scales)", count, restored)


def determine_shapes(model, device="cpu"):
    """Run a test forward pass to determine KV cache and prefix shapes."""
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    B = 1
    log.info("Running shape-determination forward pass...")
    with torch.no_grad():
        images = [torch.randn(B, 3, 224, 224, device=device) for _ in range(3)]
        img_masks = [torch.ones(B, dtype=torch.bool, device=device) for _ in range(3)]
        tokens = torch.zeros(B, 200, dtype=torch.long, device=device)
        token_masks = torch.ones(B, 200, dtype=torch.bool, device=device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, tokens, token_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks.to(torch.int32), dim=1) - 1
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

    num_layers = len(past_key_values.key_cache)
    kv_shape = past_key_values.key_cache[0].shape
    prefix_len = prefix_pad_masks.shape[1]

    log.info("  num_kv_layers: %d", num_layers)
    log.info("  kv_shape per layer: %s  dtype: %s", kv_shape, past_key_values.key_cache[0].dtype)
    log.info("  prefix_len: %d", prefix_len)

    # Also test one full denoise step
    state = torch.randn(B, model.config.action_dim, device=device)
    x_t = torch.randn(B, model.config.action_horizon, model.config.action_dim, device=device)
    timestep = torch.tensor([1.0], device=device)

    with torch.no_grad():
        v_t = model.denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep)
    log.info("  denoise output shape: %s  dtype: %s", v_t.shape, v_t.dtype)
    log.info("  denoise output sample: %s", v_t[0, 0, :3])

    return {
        "num_kv_layers": num_layers,
        "kv_shape": kv_shape,
        "kv_dtype": past_key_values.key_cache[0].dtype,
        "prefix_len": prefix_len,
    }


def export_prefix_encoder(model, shapes, output_dir, device="cpu"):
    """Export prefix encoder wrapper to ONNX."""
    log.info("=== Exporting prefix encoder to ONNX ===")

    wrapper = PrefixEncoderWrapper(model)
    wrapper.eval()

    B = 1
    dummy_inputs = (
        torch.randn(B, 3, 224, 224, device=device),
        torch.randn(B, 3, 224, 224, device=device),
        torch.randn(B, 3, 224, 224, device=device),
        torch.ones(B, dtype=torch.bool, device=device),
        torch.ones(B, dtype=torch.bool, device=device),
        torch.ones(B, dtype=torch.bool, device=device),
        torch.zeros(B, 200, dtype=torch.long, device=device),
        torch.ones(B, 200, dtype=torch.bool, device=device),
    )

    onnx_path = output_dir / "prefix_encoder.onnx"
    t0 = time.time()
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        str(onnx_path),
        opset_version=17,
        input_names=["img0", "img1", "img2", "mask0", "mask1", "mask2", "tokens", "token_masks"],
        output_names=["prefix_pad_masks", "kv_keys", "kv_values"],
        dynamic_axes=None,
    )
    elapsed = time.time() - t0

    import onnx
    from onnx.external_data_helper import convert_model_to_external_data
    log.info("  Saving prefix encoder with external data...")
    model_proto = onnx.load(str(onnx_path), load_external_data=True)
    convert_model_to_external_data(
        model_proto, all_tensors_to_one_file=True,
        location="prefix_encoder.onnx.data", size_threshold=1024,
    )
    onnx.save_model(model_proto, str(onnx_path))
    del model_proto

    size_mb = onnx_path.stat().st_size / 1e6
    data_path = output_dir / "prefix_encoder.onnx.data"
    data_mb = data_path.stat().st_size / 1e6 if data_path.exists() else 0
    log.info("  Prefix encoder exported in %.1fs → %s (%.1f MB proto + %.1f MB data)",
             elapsed, onnx_path, size_mb, data_mb)
    return onnx_path


def export_denoise_step(model, shapes, output_dir, device="cpu"):
    """Export denoise step wrapper to ONNX."""
    log.info("=== Exporting denoise step to ONNX ===")

    num_layers = shapes["num_kv_layers"]
    kv_shape = shapes["kv_shape"]
    kv_dtype = shapes["kv_dtype"]
    prefix_len = shapes["prefix_len"]

    wrapper = DenoiseStepWrapper(model, num_layers)
    wrapper.eval()

    B = 1
    dummy_inputs = (
        torch.randn(B, model.config.action_dim, device=device),
        torch.ones(B, prefix_len, dtype=torch.bool, device=device),
        torch.randn(B, model.config.action_horizon, model.config.action_dim, device=device),
        torch.tensor([1.0], device=device),
        torch.randn(num_layers, *kv_shape, dtype=kv_dtype, device=device),
        torch.randn(num_layers, *kv_shape, dtype=kv_dtype, device=device),
    )

    onnx_path = output_dir / "denoise_step.onnx"
    t0 = time.time()
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        str(onnx_path),
        opset_version=17,
        input_names=["state", "prefix_pad_masks", "x_t", "timestep", "kv_keys", "kv_values"],
        output_names=["velocity"],
        dynamic_axes=None,
    )
    elapsed = time.time() - t0
    size_mb = onnx_path.stat().st_size / 1e6
    log.info("  Denoise step exported in %.1fs → %s (%.1f MB)", elapsed, onnx_path, size_mb)
    return onnx_path


def verify_onnx(onnx_path):
    """Basic ONNX model validity check."""
    import onnx

    size_mb = onnx_path.stat().st_size / 1e6
    data_file = pathlib.Path(str(onnx_path) + ".data")
    total_mb = size_mb + (data_file.stat().st_size / 1e6 if data_file.exists() else 0)
    log.info("Verifying %s (%.1f MB proto, %.1f MB total)...", onnx_path.name, size_mb, total_mb)

    if total_mb > 2000:
        onnx.checker.check_model(str(onnx_path))
        model = onnx.load(str(onnx_path), load_external_data=False)
    else:
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model, full_check=True)

    log.info("  ONNX check passed. Inputs: %d, Outputs: %d",
             len(model.graph.input), len(model.graph.output))
    for inp in model.graph.input:
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        log.info("    input  %s: %s", inp.name, dims)
    for out in model.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        log.info("    output %s: %s", out.name, dims)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pi05_excavator_v2",
                        choices=list(ROBOT_CONFIGS.keys()),
                        help="Robot config to use (sets ACTION_DIM and default checkpoint)")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16",
                        help="Export precision (float32 for CPU verification, float16 for TRT)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint directory (overrides default from config).")
    parser.add_argument("--qat-checkpoint", type=str, default=None,
                        help="Path to a QAT-trained checkpoint directory. "
                             "When provided, ONNX export includes QuantizeLinear/DequantizeLinear ops "
                             "with learned scales so TRT skips PTQ calibration.")
    args = parser.parse_args()
    if args.checkpoint and args.qat_checkpoint:
        parser.error("Use --checkpoint or --qat-checkpoint, not both.")

    robot_cfg = ROBOT_CONFIGS[args.config_name]
    action_dim = robot_cfg["action_dim"]
    default_ckpt_dir = str(robot_cfg["checkpoint_dir"])
    log.info("Robot config: %s  ACTION_DIM=%d  default_checkpoint=%s", args.config_name, action_dim, default_ckpt_dir)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cpu"

    ckpt = args.qat_checkpoint or args.checkpoint or default_ckpt_dir
    model = load_model(device, export_dtype=args.dtype, qat_checkpoint=args.qat_checkpoint,
                       checkpoint_override=ckpt, action_dim=action_dim)
    shapes = determine_shapes(model, device)

    prefix_path = export_prefix_encoder(model, shapes, OUTPUT_DIR, device)
    denoise_path = export_denoise_step(model, shapes, OUTPUT_DIR, device)

    verify_onnx(prefix_path)
    verify_onnx(denoise_path)

    log.info("=== All ONNX exports complete! ===")
    for f in sorted(OUTPUT_DIR.iterdir()):
        log.info("  %s: %.1f MB", f.name, f.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
