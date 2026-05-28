"""Export pi-0.5 JAX model to ONNX subgraphs for TensorRT compilation.

Splits the model into two ONNX subgraphs:
  1. Prefix encoder: images + text -> KV cache
  2. Denoise step: KV cache + noisy actions + timestep -> velocity

Usage (inside the 26.02 container):
  python /workspace/openpi/scripts/export_onnx.py
"""

import os
import sys
import logging
import pathlib
import time

import numpy as np
import jax
import jax.numpy as jnp
from flax import traverse_util
import orbax.checkpoint as ocp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CHECKPOINT_DIR = pathlib.Path("/workspace/openpi/checkpoints/pi05_runC/params")
OUTPUT_DIR = pathlib.Path("/workspace/openpi/onnx_export")
sys.path.insert(0, "/workspace/openpi/src")


def load_and_merge():
    """Load checkpoint and merge LoRA weights. Returns params in model-ready format."""
    log.info("Loading checkpoint from %s", CHECKPOINT_DIR)
    t0 = time.time()

    with ocp.PyTreeCheckpointer() as ckptr:
        md = ckptr.metadata(CHECKPOINT_DIR)
        tree = md.item_metadata.tree
        restore_args = jax.tree.map(
            lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray, dtype=jnp.bfloat16),
            tree,
        )
        raw = ckptr.restore(
            CHECKPOINT_DIR,
            ocp.args.PyTreeRestore(item=tree, restore_args=restore_args),
        )

    # Extract "params" level (matches restore_params behavior)
    params = raw["params"]
    flat = traverse_util.flatten_dict(params)
    log.info("Loaded %d entries in %.1fs", len(flat), time.time() - t0)

    # Strip "value" suffix from all keys (NNX/orbax convention)
    flat = {(k[:-1] if k[-1] == "value" else k): v for k, v in flat.items()}

    # Merge LoRA: W += scale * A @ B
    # Both gemma_2b_lora (rank=16, alpha=16) and gemma_300m_lora (rank=32, alpha=32) have scale=1.0
    layers = ("PaliGemma", "llm", "layers")

    def merge_pair(base_key, a_key, b_key, label):
        if base_key not in flat:
            log.warning("  Base not found: %s", label)
            return
        w = flat[base_key].astype(np.float32)
        a = flat[a_key].astype(np.float32)
        b = flat[b_key].astype(np.float32)
        delta = np.einsum("...ij,...jk->...ik", a, b)
        flat[base_key] = (w + delta).astype(np.float16)
        log.info("  Merged %s: delta_norm=%.4f", label, np.linalg.norm(delta.reshape(-1)))

    log.info("Merging LoRA weights...")

    # Attention LoRA (Einsum): base is ".w", lora is ".lora_a" / ".lora_b"
    for suffix in ("", "_1"):  # "" = PaliGemma 2B, "_1" = action expert 300M
        for proj in ("q_einsum", "kv_einsum", "attn_vec_einsum"):
            name = f"{proj}{suffix}"
            merge_pair(
                (*layers, "attn", name, "w"),
                (*layers, "attn", name, "lora_a"),
                (*layers, "attn", name, "lora_b"),
                f"attn.{name}",
            )
        # FFN LoRA (FeedForward): base is gating_einsum/linear, lora is separate params
        mlp = f"mlp{suffix}"
        merge_pair(
            (*layers, mlp, "gating_einsum"),
            (*layers, mlp, "gating_einsum_lora_a"),
            (*layers, mlp, "gating_einsum_lora_b"),
            f"{mlp}.gating",
        )
        merge_pair(
            (*layers, mlp, "linear"),
            (*layers, mlp, "linear_lora_a"),
            (*layers, mlp, "linear_lora_b"),
            f"{mlp}.linear",
        )

    # Remove all LoRA entries
    lora_keys = [k for k in flat if any("lora" in str(x) for x in k)]
    for k in lora_keys:
        del flat[k]
    log.info("Removed %d LoRA entries", len(lora_keys))

    # Convert all params to JAX arrays (needed for tracing)
    for k in flat:
        v = flat[k]
        if hasattr(v, 'dtype') and v.dtype == jnp.bfloat16:
            flat[k] = jnp.array(v, dtype=jnp.bfloat16)
        elif isinstance(v, np.ndarray):
            flat[k] = jnp.array(v)

    total_params = sum(v.size for v in flat.values())
    log.info("Final merged params: %d entries, %d params (%.2f GB in fp16)",
             len(flat), total_params, total_params * 2 / 1e9)

    return traverse_util.unflatten_dict(flat)


def build_model(merged_params):
    """Build JAX model with non-LoRA config and load merged weights."""
    from openpi.models import pi0_config

    config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=6,
        action_horizon=11,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
    )

    log.info("Loading merged params into model...")
    t0 = time.time()
    model = config.load(merged_params)
    log.info("Model built in %.1fs", time.time() - t0)
    return model, config


def test_forward_pass(model, config):
    """Run a quick forward pass to verify the model works."""
    from openpi.models.model import Observation, preprocess_observation
    from openpi.models.pi0 import make_attn_mask

    log.info("Testing forward pass on CPU...")
    B = 1

    obs = Observation(
        images={
            "base_0_rgb": jnp.zeros((B, 224, 224, 3), dtype=jnp.float32),
            "left_wrist_0_rgb": jnp.zeros((B, 224, 224, 3), dtype=jnp.float32),
            "right_wrist_0_rgb": jnp.zeros((B, 224, 224, 3), dtype=jnp.float32),
        },
        image_masks={
            "base_0_rgb": jnp.ones((B,), dtype=jnp.bool_),
            "left_wrist_0_rgb": jnp.ones((B,), dtype=jnp.bool_),
            "right_wrist_0_rgb": jnp.ones((B,), dtype=jnp.bool_),
        },
        state=jnp.zeros((B, 6), dtype=jnp.float32),
        tokenized_prompt=jnp.zeros((B, 200), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((B, 200), dtype=jnp.bool_),
    )

    obs = preprocess_observation(None, obs, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
    log.info("  prefix_tokens: %s, prefix_mask: %s", prefix_tokens.shape, prefix_mask.shape)

    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
    )
    kv_k, kv_v = kv_cache
    log.info("  kv_cache: k=%s v=%s", kv_k.shape, kv_v.shape)

    # Test one denoise step
    import einops
    noisy_actions = jnp.zeros((B, config.action_horizon, config.action_dim))
    timestep = jnp.array([1.0])

    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
        obs, noisy_actions, timestep
    )
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask_for_suffix = einops.repeat(
        prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
    )
    full_attn_mask = jnp.concatenate([prefix_attn_mask_for_suffix, suffix_attn_mask], axis=-1)
    positions_suffix = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

    (_, suffix_out), _ = model.PaliGemma.llm(
        [None, suffix_tokens],
        mask=full_attn_mask,
        positions=positions_suffix,
        kv_cache=kv_cache,
        adarms_cond=[None, adarms_cond],
    )
    v_t = model.action_out_proj(suffix_out[:, -config.action_horizon:])
    log.info("  velocity output: %s", v_t.shape)
    log.info("  velocity sample: %s", v_t[0, 0, :3])
    log.info("Forward pass OK!")

    return {
        "kv_k_shape": kv_k.shape,
        "kv_v_shape": kv_v.shape,
        "kv_k_dtype": kv_k.dtype,
        "prefix_mask_shape": prefix_mask.shape,
        "prefix_tokens_shape": prefix_tokens.shape,
        "suffix_tokens_shape": suffix_tokens.shape,
    }


def export_onnx(model, config, shapes_info):
    """Export the two subgraphs to ONNX via jax2onnx."""
    from flax import nnx
    from openpi.models.model import Observation, preprocess_observation
    from openpi.models.pi0 import make_attn_mask
    import einops

    graphdef, state = nnx.split(model)

    B = 1
    action_horizon = config.action_horizon
    action_dim = config.action_dim

    # ── Prefix encoder ───────────────────────────────────────────────────

    def prefix_fn(images_base, images_left, images_right, tokenized_prompt, tokenized_prompt_mask):
        mdl = nnx.merge(graphdef, state)
        obs = Observation(
            images={
                "base_0_rgb": images_base,
                "left_wrist_0_rgb": images_left,
                "right_wrist_0_rgb": images_right,
            },
            image_masks={
                "base_0_rgb": jnp.ones((B,), dtype=jnp.bool_),
                "left_wrist_0_rgb": jnp.ones((B,), dtype=jnp.bool_),
                "right_wrist_0_rgb": jnp.ones((B,), dtype=jnp.bool_),
            },
            state=jnp.zeros((B, action_dim), dtype=jnp.float32),
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )
        obs = preprocess_observation(None, obs, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = mdl.embed_prefix(obs)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = mdl.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        return kv_cache[0], kv_cache[1], prefix_mask

    prefix_specs = [
        jax.ShapeDtypeStruct((B, 224, 224, 3), jnp.float32),   # images_base
        jax.ShapeDtypeStruct((B, 224, 224, 3), jnp.float32),   # images_left
        jax.ShapeDtypeStruct((B, 224, 224, 3), jnp.float32),   # images_right
        jax.ShapeDtypeStruct((B, 200), jnp.int32),             # tokenized_prompt
        jax.ShapeDtypeStruct((B, 200), jnp.bool_),             # tokenized_prompt_mask
    ]

    # ── Denoise step ─────────────────────────────────────────────────────

    kv_k_shape = shapes_info["kv_k_shape"]
    kv_v_shape = shapes_info["kv_v_shape"]
    prefix_mask_shape = shapes_info["prefix_mask_shape"]
    prefix_tokens_len = shapes_info["prefix_tokens_shape"][1]

    def denoise_step_fn(kv_cache_k, kv_cache_v, prefix_mask, noisy_actions, timestep):
        mdl = nnx.merge(graphdef, state)
        kv_cache = (kv_cache_k, kv_cache_v)

        obs_dummy = Observation(
            images={
                "base_0_rgb": jnp.zeros((B, 224, 224, 3)),
                "left_wrist_0_rgb": jnp.zeros((B, 224, 224, 3)),
                "right_wrist_0_rgb": jnp.zeros((B, 224, 224, 3)),
            },
            image_masks={
                "base_0_rgb": jnp.ones((B,), dtype=jnp.bool_),
                "left_wrist_0_rgb": jnp.ones((B,), dtype=jnp.bool_),
                "right_wrist_0_rgb": jnp.ones((B,), dtype=jnp.bool_),
            },
            state=jnp.zeros((B, action_dim), dtype=jnp.float32),
            tokenized_prompt=jnp.zeros((B, 200), dtype=jnp.int32),
            tokenized_prompt_mask=jnp.ones((B, 200), dtype=jnp.bool_),
        )

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = mdl.embed_suffix(
            obs_dummy, noisy_actions, timestep
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_for_suffix = einops.repeat(
            prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
        )
        full_attn_mask = jnp.concatenate([prefix_attn_for_suffix, suffix_attn_mask], axis=-1)
        positions = (
            jnp.sum(prefix_mask, axis=-1)[:, None]
            + jnp.cumsum(suffix_mask, axis=-1) - 1
        )
        (_, suffix_out), _ = mdl.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = mdl.action_out_proj(suffix_out[:, -action_horizon:])
        return v_t

    denoise_specs = [
        jax.ShapeDtypeStruct(kv_k_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(kv_v_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(prefix_mask_shape, jnp.bool_),
        jax.ShapeDtypeStruct((B, action_horizon, action_dim), jnp.float32),
        jax.ShapeDtypeStruct((B,), jnp.float32),
    ]

    # ── Try tracing first ────────────────────────────────────────────────

    log.info("Tracing prefix function...")
    try:
        # Use real arrays, not ShapeDtypeStruct, for make_jaxpr
        prefix_args = [jnp.ones(s.shape, s.dtype) for s in prefix_specs]
        t0 = time.time()
        jaxpr = jax.make_jaxpr(prefix_fn)(*prefix_args)
        log.info("  Prefix jaxpr traced in %.1fs: %d equations", time.time() - t0, len(jaxpr.jaxpr.eqns))
        out_avals = jaxpr.out_avals
        log.info("  Output shapes: %s", [(a.shape, a.dtype) for a in out_avals])
    except Exception as e:
        log.error("Prefix tracing failed: %s", e)
        import traceback
        traceback.print_exc()
        return False

    log.info("Tracing denoise step function...")
    try:
        denoise_args = [jnp.ones(s.shape, s.dtype) for s in denoise_specs]
        t0 = time.time()
        jaxpr_d = jax.make_jaxpr(denoise_step_fn)(*denoise_args)
        log.info("  Denoise jaxpr traced in %.1fs: %d equations", time.time() - t0, len(jaxpr_d.jaxpr.eqns))
    except Exception as e:
        log.error("Denoise tracing failed: %s", e)
        import traceback
        traceback.print_exc()
        return False

    # ── Export to ONNX ───────────────────────────────────────────────────

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("Exporting prefix encoder to ONNX...")
    try:
        from jax2onnx import to_onnx
        t0 = time.time()
        to_onnx(prefix_fn, prefix_specs, return_mode="file",
                output_path=str(OUTPUT_DIR / "prefix_encoder.onnx"))
        log.info("  Prefix encoder exported in %.1fs", time.time() - t0)
    except Exception as e:
        log.error("jax2onnx prefix export failed: %s", e)
        import traceback
        traceback.print_exc()
        return False

    log.info("Exporting denoise step to ONNX...")
    try:
        t0 = time.time()
        to_onnx(denoise_step_fn, denoise_specs, return_mode="file",
                output_path=str(OUTPUT_DIR / "denoise_step.onnx"))
        log.info("  Denoise step exported in %.1fs", time.time() - t0)
    except Exception as e:
        log.error("jax2onnx denoise export failed: %s", e)
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1+2: Load checkpoint and merge LoRA
    merged_params = load_and_merge()

    # Step 3: Build model with non-LoRA config
    model, config = build_model(merged_params)
    del merged_params

    # Step 4: Verify forward pass
    shapes_info = test_forward_pass(model, config)

    # Step 5: Export to ONNX
    ok = export_onnx(model, config, shapes_info)
    if ok:
        log.info("ONNX export complete! Files in %s", OUTPUT_DIR)
        for f in os.listdir(OUTPUT_DIR):
            fpath = OUTPUT_DIR / f
            log.info("  %s: %.1f MB", f, fpath.stat().st_size / 1e6)
    else:
        log.warning("jax2onnx export failed. See errors above.")
        log.info("Will need to use PyTorch fallback path.")


if __name__ == "__main__":
    main()
