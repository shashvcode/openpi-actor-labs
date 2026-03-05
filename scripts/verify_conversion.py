#!/usr/bin/env python3
"""
Verify JAX-to-PyTorch weight conversion by comparing model outputs numerically.

Loads both the JAX model and the converted PyTorch model, feeds identical synthetic
inputs, and compares the loss outputs. A passing result (MSE < 1e-4) confirms the
conversion preserved weight fidelity.

Usage:
    python scripts/verify_conversion.py \
        --config_name pi05_so100_lora_v3 \
        --pytorch_weight_path ./checkpoints/pi05_base_pytorch

The JAX checkpoint path is read from the config's weight_loader automatically.
"""

import argparse
import logging
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Verify JAX↔PyTorch conversion fidelity")
    parser.add_argument("--config_name", type=str, required=True, help="Training config name (e.g. pi05_so100_lora_v3)")
    parser.add_argument("--pytorch_weight_path", type=str, required=True, help="Path to converted PyTorch checkpoint dir")
    parser.add_argument("--threshold", type=float, default=1e-4, help="Max acceptable MSE between JAX and PyTorch outputs")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    import openpi.training.config as _config

    config = _config.get_config(args.config_name)
    model_cfg = config.model
    logger.info(f"Config: {args.config_name}")
    logger.info(f"  action_dim={model_cfg.action_dim}, action_horizon={model_cfg.action_horizon}, pi05={model_cfg.pi05}")

    # ── Create synthetic inputs ──────────────────────────────────────────────
    rng = np.random.default_rng(42)
    B = args.batch_size
    H, W, C = 224, 224, 3

    images_np = {
        "base_0_rgb": rng.standard_normal((B, C, H, W)).astype(np.float32) * 0.1,
        "left_wrist_0_rgb": rng.standard_normal((B, C, H, W)).astype(np.float32) * 0.1,
        "right_wrist_0_rgb": np.zeros((B, C, H, W), dtype=np.float32),
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

    # ── JAX forward ──────────────────────────────────────────────────────────
    logger.info("Loading JAX model...")
    import jax
    import jax.numpy as jnp

    import openpi.models.model as _model

    jax_checkpoint_path = None
    from openpi.training.weight_loaders import CheckpointWeightLoader
    if isinstance(config.weight_loader, CheckpointWeightLoader):
        jax_checkpoint_path = config.weight_loader.checkpoint_path
    if jax_checkpoint_path is None:
        logger.error("Could not determine JAX checkpoint path from config weight_loader")
        sys.exit(1)
    logger.info(f"JAX checkpoint: {jax_checkpoint_path}")

    jax_model = model_cfg.load(
        _model.restore_params(f"{jax_checkpoint_path}", dtype=jnp.float32)
    )
    jax_model.eval()
    logger.info("JAX model loaded")

    jax_observation = _model.Observation(
        images={k: jnp.array(v) for k, v in images_np.items()},
        image_masks={k: jnp.array(v) for k, v in image_masks_np.items()},
        state=jnp.array(state_np),
        tokenized_prompt=jnp.array(tokenized_prompt_np),
        tokenized_prompt_mask=jnp.array(tokenized_prompt_mask_np),
    )
    jax_actions = jnp.array(actions_np)
    jax_noise = jnp.array(noise_np)
    jax_time = jnp.array(time_np)

    time_expanded = jax_time[:, None, None]
    jax_x_t = time_expanded * jax_noise + (1 - time_expanded) * jax_actions
    jax_u_t = jax_noise - jax_actions

    jax_observation_processed = _model.preprocess_observation(None, jax_observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = jax_model.embed_prefix(jax_observation_processed)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = jax_model.embed_suffix(
        jax_observation_processed, jax_x_t, jax_time
    )

    from openpi.models.pi0 import make_attn_mask
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1

    (prefix_out, suffix_out), _ = jax_model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
    )
    jax_v_t = jax_model.action_out_proj(suffix_out[:, -jax_model.action_horizon:])
    jax_loss = jnp.mean(jnp.square(jax_v_t - jax_u_t))
    jax_v_t_np = np.array(jax_v_t)
    jax_loss_val = float(jax_loss)
    logger.info(f"JAX loss: {jax_loss_val:.6f}")
    logger.info(f"JAX v_t stats: mean={float(jnp.mean(jax_v_t)):.6f}, std={float(jnp.std(jax_v_t)):.6f}")

    del jax_model, prefix_tokens, suffix_tokens, prefix_out, suffix_out
    import gc
    gc.collect()

    # ── PyTorch forward ──────────────────────────────────────────────────────
    logger.info("Loading PyTorch model...")
    import torch
    import safetensors.torch
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt_model = PI0Pytorch(model_cfg).to(device)
    safetensors.torch.load_model(pt_model, f"{args.pytorch_weight_path}/model.safetensors")
    pt_model.eval()
    logger.info(f"PyTorch model loaded on {device}")

    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

    pt_images = {k: torch.from_numpy(v).to(device) for k, v in images_np.items()}
    pt_image_masks = {k: torch.tensor(v).to(device) for k, v in image_masks_np.items()}
    pt_state = torch.from_numpy(state_np).to(device)
    pt_actions = torch.from_numpy(actions_np).to(device)
    pt_noise = torch.from_numpy(noise_np).to(device)
    pt_time = torch.from_numpy(time_np).to(device)

    pt_time_expanded = pt_time[:, None, None]
    pt_x_t = pt_time_expanded * pt_noise + (1 - pt_time_expanded) * pt_actions
    pt_u_t = pt_noise - pt_actions

    pt_observation = _model.Observation(
        images=pt_images,
        image_masks=pt_image_masks,
        state=pt_state,
        tokenized_prompt=torch.from_numpy(tokenized_prompt_np).to(device),
        tokenized_prompt_mask=torch.from_numpy(tokenized_prompt_mask_np).to(device),
    )

    with torch.no_grad():
        images_list, img_masks_list, lang_tokens, lang_masks, state = pt_model._preprocess_observation(
            pt_observation, train=False
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = pt_model.embed_prefix(
            images_list, img_masks_list, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = pt_model.embed_suffix(
            state, pt_x_t, pt_time
        )

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = pt_model._prepare_attention_masks_4d(att_2d_masks)

        if prefix_embs.dtype != suffix_embs.dtype:
            suffix_embs = suffix_embs.to(dtype=prefix_embs.dtype)
            prefix_embs = prefix_embs.to(dtype=prefix_embs.dtype)

        (_, suffix_out), _ = pt_model.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        pt_suffix_out = suffix_out[:, -model_cfg.action_horizon:]
        pt_suffix_out = pt_suffix_out.to(dtype=torch.float32)
        pt_v_t = pt_model.action_out_proj(pt_suffix_out)
        pt_loss = torch.nn.functional.mse_loss(pt_u_t, pt_v_t)

    pt_v_t_np = pt_v_t.cpu().numpy()
    pt_loss_val = pt_loss.item()
    logger.info(f"PyTorch loss: {pt_loss_val:.6f}")
    logger.info(f"PyTorch v_t stats: mean={pt_v_t_np.mean():.6f}, std={pt_v_t_np.std():.6f}")

    # ── Compare ──────────────────────────────────────────────────────────────
    v_t_mse = float(np.mean((jax_v_t_np - pt_v_t_np) ** 2))
    v_t_max_diff = float(np.max(np.abs(jax_v_t_np - pt_v_t_np)))
    loss_diff = abs(jax_loss_val - pt_loss_val)

    logger.info("=" * 60)
    logger.info("CONVERSION VERIFICATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  v_t MSE (JAX vs PyTorch):  {v_t_mse:.2e}")
    logger.info(f"  v_t max absolute diff:     {v_t_max_diff:.2e}")
    logger.info(f"  Loss diff:                 {loss_diff:.2e}")
    logger.info(f"  Threshold:                 {args.threshold:.2e}")

    if v_t_mse < args.threshold:
        logger.info("  ✓ PASS — Conversion is numerically correct!")
        return 0
    else:
        logger.error(f"  ✗ FAIL — MSE {v_t_mse:.2e} exceeds threshold {args.threshold:.2e}")
        logger.error("  The conversion may have introduced errors. Inspect weight mappings.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
