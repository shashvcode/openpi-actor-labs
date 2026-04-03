#!/usr/bin/env python3
"""Generate JAX reference outputs for the excavator model.

Runs inside the NVIDIA JAX container on GPU, saves outputs to npz for
comparison against PyTorch/TRT outside the container.

Usage (inside NVIDIA container):
    python3 scripts/jax_reference_excavator.py \
        --jax_checkpoint checkpoints/excavator_v1_jax \
        --config_name pi05_excavator_v2 \
        --output /tmp/jax_reference.npz
"""

import argparse
import gc
import logging
import os
import sys
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_synthetic_inputs(model_cfg, rng, batch_size=1):
    B = batch_size
    H, W, C = 224, 224, 3

    images_np = {
        "base_0_rgb": rng.standard_normal((B, H, W, C)).astype(np.float32) * 0.1,
        "left_wrist_0_rgb": rng.standard_normal((B, H, W, C)).astype(np.float32) * 0.1,
        "right_wrist_0_rgb": np.zeros((B, H, W, C), dtype=np.float32),
    }
    image_masks_np = {
        "base_0_rgb": np.ones((B,), dtype=bool),
        "left_wrist_0_rgb": np.ones((B,), dtype=bool),
        "right_wrist_0_rgb": np.zeros((B,), dtype=bool),
    }
    state_np = rng.standard_normal((B, model_cfg.action_dim)).astype(np.float32) * 0.1
    actions_np = rng.standard_normal((B, model_cfg.action_horizon, model_cfg.action_dim)).astype(np.float32) * 0.1
    noise_np = rng.standard_normal((B, model_cfg.action_horizon, model_cfg.action_dim)).astype(np.float32)
    time_np = np.array([0.5] * B, dtype=np.float32)

    max_token_len = model_cfg.max_token_len
    tokenized_prompt_np = np.zeros((B, max_token_len), dtype=np.int32)
    tokenized_prompt_np[:, 0] = 2
    tokenized_prompt_mask_np = np.zeros((B, max_token_len), dtype=bool)
    tokenized_prompt_mask_np[:, 0] = True

    sample_noise_np = rng.standard_normal((B, model_cfg.action_horizon, model_cfg.action_dim)).astype(np.float32)

    return {
        "images": images_np,
        "image_masks": image_masks_np,
        "state": state_np,
        "actions": actions_np,
        "noise": noise_np,
        "time": time_np,
        "tokenized_prompt": tokenized_prompt_np,
        "tokenized_prompt_mask": tokenized_prompt_mask_np,
        "sample_noise": sample_noise_np,
    }


def run_jax_inference(model_cfg, jax_checkpoint_path, inputs):
    import jax
    import jax.numpy as jnp
    import openpi.models.model as _model
    from openpi.models.pi0 import make_attn_mask

    logger.info("JAX backend: %s | Devices: %s", jax.default_backend(), jax.devices())

    logger.info("Loading JAX model from %s ...", jax_checkpoint_path)
    t0 = time.time()

    import pathlib
    import orbax.checkpoint as ocp

    ckpt_path = pathlib.Path(jax_checkpoint_path)
    params_path = ckpt_path / "params" if (ckpt_path / "params").exists() else ckpt_path
    params_path = params_path.resolve()

    mesh = jax.sharding.Mesh(jax.devices(), ("x",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        # Handle both old orbax (dict) and new orbax (StepMetadata)
        if hasattr(metadata, "item_metadata"):
            item = {"params": metadata.item_metadata["params"]}
        else:
            item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(
                        sharding=sharding, restore_type=jax.Array, dtype=jnp.float32
                    ),
                    item,
                ),
            ),
        )["params"]

    def _unwrap_value(tree):
        """Unwrap orbax's {'value': array} nesting if present."""
        if isinstance(tree, dict):
            if list(tree.keys()) == ["value"] and not isinstance(tree["value"], dict):
                return tree["value"]
            return {k: _unwrap_value(v) for k, v in tree.items()}
        return tree

    params = _unwrap_value(params)
    logger.info("Params keys: %s (count: %d)", list(params.keys()), len(params))
    jax_model = model_cfg.load(params)
    jax_model.eval()
    logger.info("JAX model loaded in %.1fs", time.time() - t0)

    jax_observation = _model.Observation(
        images={k: jnp.array(v) for k, v in inputs["images"].items()},
        image_masks={k: jnp.array(v) for k, v in inputs["image_masks"].items()},
        state=jnp.array(inputs["state"]),
        tokenized_prompt=jnp.array(inputs["tokenized_prompt"]),
        tokenized_prompt_mask=jnp.array(inputs["tokenized_prompt_mask"]),
    )

    jax_actions = jnp.array(inputs["actions"])
    jax_noise = jnp.array(inputs["noise"])
    jax_time = jnp.array(inputs["time"])

    time_expanded = jax_time[:, None, None]
    jax_x_t = time_expanded * jax_noise + (1 - time_expanded) * jax_actions
    jax_u_t = jax_noise - jax_actions

    jax_observation_processed = _model.preprocess_observation(None, jax_observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = jax_model.embed_prefix(jax_observation_processed)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = jax_model.embed_suffix(
        jax_observation_processed, jax_x_t, jax_time
    )

    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1

    logger.info("Running single-step velocity (compile + execute)...")
    t0 = time.time()
    (prefix_out, suffix_out), _ = jax_model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions,
        adarms_cond=[None, adarms_cond]
    )
    jax_v_t = jax_model.action_out_proj(suffix_out[:, -jax_model.action_horizon:])
    jax.block_until_ready(jax_v_t)
    single_step_ms = (time.time() - t0) * 1000

    jax_loss = float(jnp.mean(jnp.square(jax_v_t - jax_u_t)))
    jax_v_t_np = np.array(jax_v_t)

    logger.info("JAX single-step: loss=%.6f, v_t mean=%.6f std=%.6f (%.1fms)",
                jax_loss, float(jnp.mean(jax_v_t)), float(jnp.std(jax_v_t)), single_step_ms)

    # Second run (XLA cache warm)
    t0 = time.time()
    (prefix_out2, suffix_out2), _ = jax_model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions,
        adarms_cond=[None, adarms_cond]
    )
    jax_v_t_2 = jax_model.action_out_proj(suffix_out2[:, -jax_model.action_horizon:])
    jax.block_until_ready(jax_v_t_2)
    single_step_warm_ms = (time.time() - t0) * 1000
    logger.info("JAX single-step (warm): %.1fms", single_step_warm_ms)

    logger.info("Running JAX sample_actions (8 denoise steps)...")
    sample_noise = jnp.array(inputs["sample_noise"])
    rng_key = jax.random.PRNGKey(42)

    # First call (XLA compile)
    t0 = time.time()
    jax_sampled_actions = jax_model.sample_actions(
        rng_key, jax_observation, noise=sample_noise, num_steps=8
    )
    jax.block_until_ready(jax_sampled_actions)
    sample_compile_ms = (time.time() - t0) * 1000

    jax_sampled_np = np.array(jax_sampled_actions)
    logger.info("JAX sample_actions (compile+exec): shape=%s, time=%.1fms, mean=%.6f std=%.6f",
                jax_sampled_np.shape, sample_compile_ms,
                jax_sampled_np.mean(), jax_sampled_np.std())

    # Warm run
    t0 = time.time()
    jax_sampled_actions_2 = jax_model.sample_actions(
        rng_key, jax_observation, noise=sample_noise, num_steps=8
    )
    jax.block_until_ready(jax_sampled_actions_2)
    sample_warm_ms = (time.time() - t0) * 1000
    logger.info("JAX sample_actions (warm): %.1fms", sample_warm_ms)

    return {
        "v_t": jax_v_t_np,
        "loss": jax_loss,
        "sampled_actions": jax_sampled_np,
        "sample_compile_ms": sample_compile_ms,
        "sample_warm_ms": sample_warm_ms,
        "single_step_compile_ms": single_step_ms,
        "single_step_warm_ms": single_step_warm_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate JAX reference outputs for excavator model")
    parser.add_argument("--jax_checkpoint", required=True, help="Path to JAX checkpoint directory")
    parser.add_argument("--config_name", default="pi05_excavator_v2", help="Training config name")
    parser.add_argument("--output", default="/tmp/jax_reference.npz", help="Output npz file")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import openpi.training.config as _config
    config = _config.get_config(args.config_name)
    model_cfg = config.model
    logger.info("Config: %s (action_dim=%d, action_horizon=%d, pi05=%s)",
                args.config_name, model_cfg.action_dim, model_cfg.action_horizon, model_cfg.pi05)

    rng = np.random.default_rng(42)
    inputs = create_synthetic_inputs(model_cfg, rng, batch_size=1)

    results = run_jax_inference(model_cfg, args.jax_checkpoint, inputs)

    np.savez(args.output,
             v_t=results["v_t"],
             loss=np.array(results["loss"]),
             sampled_actions=results["sampled_actions"],
             sample_compile_ms=np.array(results["sample_compile_ms"]),
             sample_warm_ms=np.array(results["sample_warm_ms"]),
             single_step_compile_ms=np.array(results["single_step_compile_ms"]),
             single_step_warm_ms=np.array(results["single_step_warm_ms"]),
             # Also save inputs for reproducibility
             input_state=inputs["state"],
             input_sample_noise=inputs["sample_noise"],
             input_actions=inputs["actions"],
             input_noise=inputs["noise"],
             input_time=inputs["time"],
             **{f"input_image_{k}": v for k, v in inputs["images"].items()},
             )
    logger.info("Saved JAX reference to %s", args.output)
    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
