#!/usr/bin/env python3
"""Standalone verification of PyTorch training script fixes.

Tests the key mechanics without the full data pipeline:
- Parameter freezing works (frozen params don't change after optimizer step)
- EMA tracking works (shadow params update correctly)
- Only trainable params are in the optimizer
- AMP autocast doesn't break the forward pass
- Loss decreases over steps

Usage (in the trt_pipeline container):
  python3 /workspace/openpi/scripts/verify_training_fixes.py
"""

import dataclasses
import logging
import sys
import pathlib
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn as nn
import numpy as np

# Mock lerobot to avoid import errors, then import from train_pytorch
_mock_lerobot = types.ModuleType("lerobot")
_mock_common = types.ModuleType("lerobot.common")
_mock_datasets = types.ModuleType("lerobot.common.datasets")
_mock_ld = types.ModuleType("lerobot.common.datasets.lerobot_dataset")
_mock_ld.LeRobotDataset = type("LeRobotDataset", (), {})
_mock_ld.LeRobotDatasetMetadata = type("LeRobotDatasetMetadata", (), {})
sys.modules["lerobot"] = _mock_lerobot
sys.modules["lerobot.common"] = _mock_common
sys.modules["lerobot.common.datasets"] = _mock_datasets
sys.modules["lerobot.common.datasets.lerobot_dataset"] = _mock_ld

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train_pytorch import apply_freeze_filter, EMATracker, tree_map_tensors

logging.basicConfig(level=logging.INFO, format="%(message)s")


class FakeConfig:
    """Minimal config for testing freeze filter."""
    class Model:
        paligemma_variant = "gemma_2b_lora"
        action_expert_variant = "gemma_300m_lora"
    model = Model()


class FakeConfigNoLora:
    class Model:
        paligemma_variant = "gemma_2b"
        action_expert_variant = "gemma_300m"
    model = Model()


class ToyModel(nn.Module):
    """Mimics the pi0.5 parameter naming convention."""
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = nn.ModuleDict({
            "paligemma": nn.ModuleDict({
                "language_model": nn.Linear(16, 16),
                "vision_tower": nn.Linear(16, 16),
            }),
            "gemma_expert": nn.ModuleDict({
                "model": nn.Linear(16, 16),
            }),
        })
        self.action_in_proj = nn.Linear(16, 16)
        self.time_mlp = nn.Linear(16, 16)
        self.lora_adapter = nn.Linear(16, 16)

    def forward(self, x):
        h = self.paligemma_with_expert["paligemma"]["language_model"](x)
        h = h + self.paligemma_with_expert["paligemma"]["vision_tower"](x)
        h = h + self.paligemma_with_expert["gemma_expert"]["model"](x)
        h = h + self.action_in_proj(x)
        h = h + self.time_mlp(x)
        h = h + self.lora_adapter(x)
        return h.mean()


def test_freeze_filter():
    """Verify parameter freezing matches JAX behavior."""
    print("=" * 60)
    print("TEST: Parameter Freezing")
    print("=" * 60)

    model = ToyModel()
    config = FakeConfig()
    trainable_params, frozen_count, trainable_count = apply_freeze_filter(model, config)

    frozen_names = [n for n, p in model.named_parameters() if not p.requires_grad]
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]

    print(f"  Frozen ({len(frozen_names)}):")
    for n in frozen_names:
        print(f"    {n}")
    print(f"  Trainable ({len(trainable_names)}):")
    for n in trainable_names:
        print(f"    {n}")

    assert "paligemma_with_expert.paligemma.language_model.weight" in frozen_names, \
        "language_model weight should be frozen"
    assert "paligemma_with_expert.paligemma.language_model.bias" in frozen_names, \
        "language_model bias should be frozen"
    assert "paligemma_with_expert.gemma_expert.model.weight" in frozen_names, \
        "gemma_expert weight should be frozen"

    assert "paligemma_with_expert.paligemma.vision_tower.weight" in trainable_names, \
        "vision_tower should be trainable (not LLM)"
    assert "action_in_proj.weight" in trainable_names, \
        "action_in_proj should be trainable"
    assert "time_mlp.weight" in trainable_names, \
        "time_mlp should be trainable"
    assert "lora_adapter.weight" in trainable_names, \
        "lora_adapter should be trainable (contains 'lora')"

    for n in frozen_names:
        p = dict(model.named_parameters())[n]
        assert p.dtype == torch.bfloat16, f"Frozen param {n} should be cast to bf16, got {p.dtype}"

    print("  PASS: Freeze filter correctly identifies frozen/trainable params")
    print("  PASS: Frozen params cast to bfloat16")
    return True


def test_freeze_no_lora():
    """With no LoRA config, nothing should be frozen."""
    print("\n" + "=" * 60)
    print("TEST: No Freeze (non-LoRA config)")
    print("=" * 60)

    model = ToyModel()
    config = FakeConfigNoLora()
    trainable_params, frozen_count, trainable_count = apply_freeze_filter(model, config)

    assert frozen_count == 0, f"Expected 0 frozen, got {frozen_count}"
    total = sum(1 for _ in model.parameters())
    assert trainable_count == total, f"Expected {total} trainable, got {trainable_count}"

    print(f"  PASS: All {total} params trainable with non-LoRA config")
    return True


def test_frozen_params_dont_update():
    """Verify that frozen params don't change after optimizer step."""
    print("\n" + "=" * 60)
    print("TEST: Frozen Params Don't Update")
    print("=" * 60)

    model = ToyModel().cuda()
    config = FakeConfig()
    trainable_params, _, _ = apply_freeze_filter(model, config)

    frozen_before = {
        n: p.data.clone() for n, p in model.named_parameters() if not p.requires_grad
    }

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
    x = torch.randn(4, 16, device="cuda")

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x)
    loss.float().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    for n, p in model.named_parameters():
        if not p.requires_grad:
            assert torch.equal(p.data, frozen_before[n]), \
                f"Frozen param {n} changed after optimizer step!"

    print("  PASS: No frozen parameters were modified")
    return True


def test_ema():
    """Verify EMA tracking works correctly."""
    print("\n" + "=" * 60)
    print("TEST: EMA Tracking")
    print("=" * 60)

    model = ToyModel().cuda()
    config = FakeConfig()
    trainable_params, _, _ = apply_freeze_filter(model, config)

    decay = 0.99
    ema = EMATracker(model, decay)

    trainable_before = {
        n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad
    }
    shadow_before = {n: v.clone() for n, v in ema.shadow.items()}

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-2)
    x = torch.randn(4, 16, device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x)
    loss.float().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    ema.update(model)

    for name in ema.shadow:
        param_now = dict(model.named_parameters())[name].data
        expected = decay * shadow_before[name] + (1 - decay) * param_now
        actual = ema.shadow[name]
        assert torch.allclose(actual, expected, atol=1e-6), \
            f"EMA mismatch for {name}: max diff {(actual - expected).abs().max()}"

    print("  PASS: EMA shadow params updated correctly (decay=0.99)")

    backup = ema.apply_to(model)
    for name in ema.shadow:
        param = dict(model.named_parameters())[name].data
        assert torch.equal(param, ema.shadow[name]), \
            f"apply_to didn't swap weights for {name}"
    ema.restore(model, backup)
    for name in backup:
        param = dict(model.named_parameters())[name].data
        assert torch.equal(param, backup[name]), \
            f"restore didn't restore weights for {name}"

    print("  PASS: EMA apply/restore cycle works")
    return True


def test_amp_autocast():
    """Verify AMP autocast works with the model."""
    print("\n" + "=" * 60)
    print("TEST: AMP Autocast (BF16)")
    print("=" * 60)

    model = ToyModel().cuda()
    x = torch.randn(4, 16, device="cuda")

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x)

    assert loss.dtype == torch.bfloat16, f"Expected bf16 loss, got {loss.dtype}"
    loss_f32 = loss.float()
    loss_f32.backward()

    print(f"  PASS: Forward pass with AMP produced bf16 output (loss={loss.item():.4f})")
    return True


def test_tree_map_tensors():
    """Verify tree_map_tensors handles nested structures."""
    print("\n" + "=" * 60)
    print("TEST: tree_map_tensors (JAX dependency removal)")
    print("=" * 60)

    device = torch.device("cuda")

    nested = {
        "images": {
            "scene": torch.zeros(1, 3, 224, 224),
            "wrist": torch.zeros(1, 3, 224, 224),
        },
        "state": torch.zeros(1, 6),
        "masks": [torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool)],
        "prompt": None,
        "np_array": np.zeros(5),
    }

    result = tree_map_tensors(lambda x: x.to(device), nested)

    assert result["images"]["scene"].device.type == "cuda"
    assert result["images"]["wrist"].device.type == "cuda"
    assert result["state"].device.type == "cuda"
    assert result["masks"][0].device.type == "cuda"
    assert result["prompt"] is None
    assert isinstance(result["np_array"], np.ndarray)

    print("  PASS: Correctly maps tensors in nested dicts, lists, handles None and numpy")
    return True


def test_loss_decreases():
    """Verify loss decreases over multiple training steps."""
    print("\n" + "=" * 60)
    print("TEST: Loss Decreases Over Steps")
    print("=" * 60)

    model = ToyModel().cuda()
    config = FakeConfig()
    trainable_params, _, _ = apply_freeze_filter(model, config)
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

    torch.manual_seed(42)
    x = torch.randn(8, 16, device="cuda")

    losses = []
    for step in range(50):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x)
        loss.float().backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())

    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f} (Δ={losses[0]-losses[-1]:.4f})")
    assert losses[-1] < losses[0], "Loss should decrease over training steps"

    param_norm = torch.sqrt(sum(p.data.float().norm() ** 2 for p in trainable_params)).item()
    print(f"  param_norm: {param_norm:.4f}")
    assert param_norm > 0, "param_norm should be positive"

    print("  PASS: Loss decreased and param_norm computed")
    return True


def main():
    print("Verifying PyTorch training script fixes")
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
    print()

    results = []
    results.append(("Freeze filter (LoRA)", test_freeze_filter()))
    results.append(("Freeze filter (no LoRA)", test_freeze_no_lora()))
    results.append(("Frozen params immutable", test_frozen_params_dont_update()))
    results.append(("EMA tracking", test_ema()))
    results.append(("AMP autocast", test_amp_autocast()))
    results.append(("tree_map_tensors", test_tree_map_tensors()))
    results.append(("Loss decreases", test_loss_decreases()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests passed!")
        print("\nNote: Full JAX vs PyTorch parity test (1K steps) requires")
        print("lerobot + full training environment. Run manually with:")
        print("  python scripts/train.py debug --exp_name jax_parity")
        print("  python scripts/train_pytorch.py debug --exp_name pt_parity")
        print("Then compare loss curves in wandb.")
    else:
        print("\nSome tests failed!")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
