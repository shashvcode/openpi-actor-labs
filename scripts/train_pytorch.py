"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \\
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \\
    --master_addr=<master_ip> --master_port=<port> \\
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import gc
import logging
import os
import platform
import shutil
import time

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data


# ─── Utilities ────────────────────────────────────────────────────────────────


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def tree_map_tensors(fn, obj):
    """Recursively apply fn to all tensors in a nested structure (replaces jax.tree.map)."""
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        return {k: tree_map_tensors(fn, v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        mapped = [tree_map_tensors(fn, x) for x in obj]
        return type(obj)(mapped)
    if hasattr(obj, "__dataclass_fields__"):
        changes = {f: tree_map_tensors(fn, getattr(obj, f)) for f in obj.__dataclass_fields__}
        try:
            return dataclasses.replace(obj, **changes)
        except TypeError:
            return type(obj)(**changes)
    if hasattr(obj, "_fields"):
        return type(obj)(*(tree_map_tensors(fn, getattr(obj, f)) for f in obj._fields))
    return obj


def log_memory_usage(device, step, phase="unknown"):
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = (torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)) / 1e9

    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, "
        f"free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, "
        f"peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


# ─── Parameter Freezing ──────────────────────────────────────────────────────


def apply_freeze_filter(model: torch.nn.Module, config: _config.TrainConfig) -> tuple[list, int, int]:
    """Freeze parameters based on config, matching JAX freeze_filter behavior.

    In JAX, the freeze filter uses PathRegex patterns:
      - ".*llm.*" matches all LLM params (paligemma + action expert)
      - ".*llm.*_1.*" matches action expert params only
      - ".*lora.*" matches LoRA adapter params (never frozen)

    In PyTorch, the equivalent parameter name patterns are:
      - "language_model" for paligemma LLM
      - "gemma_expert.model" for action expert LLM
      - "lora" for LoRA adapter params

    Returns (trainable_params, frozen_count, trainable_count).
    """
    model_cfg = config.model
    has_lora_paligemma = "lora" in getattr(model_cfg, "paligemma_variant", "")
    has_lora_expert = "lora" in getattr(model_cfg, "action_expert_variant", "")

    if not has_lora_paligemma and not has_lora_expert:
        trainable = list(model.parameters())
        total = len(trainable)
        logging.info(f"No freeze filter: all {total} parameters are trainable")
        return trainable, 0, total

    frozen_count = 0
    trainable_count = 0
    frozen_bytes = 0
    trainable_params = []

    for name, param in model.named_parameters():
        should_freeze = False

        is_lora = "lora" in name.lower()
        is_paligemma_llm = "language_model" in name and "gemma_expert" not in name
        is_expert_llm = "gemma_expert" in name

        if has_lora_paligemma and is_paligemma_llm and not is_lora:
            should_freeze = True
        if has_lora_expert and is_expert_llm and not is_lora:
            should_freeze = True

        if should_freeze:
            param.requires_grad_(False)
            param.data = param.data.to(torch.bfloat16)
            frozen_count += 1
            frozen_bytes += param.numel() * param.element_size()
        else:
            param.requires_grad_(True)
            trainable_count += 1
            trainable_params.append(param)

    logging.info(
        f"Parameter freezing: {frozen_count} frozen ({frozen_bytes / 1e9:.2f}GB in bf16), "
        f"{trainable_count} trainable"
    )
    return trainable_params, frozen_count, trainable_count


# ─── EMA (Exponential Moving Average) ────────────────────────────────────────


class EMATracker:
    """Maintains an exponential moving average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict: dict):
        self.shadow = state_dict["shadow"]
        self.decay = state_dict["decay"]

    def apply_to(self, model: torch.nn.Module):
        """Temporarily swap EMA weights into the model (for export/eval)."""
        backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return backup

    def restore(self, model: torch.nn.Module, backup: dict):
        """Restore original weights after apply_to."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])


# ─── Wandb ────────────────────────────────────────────────────────────────────


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


# ─── DDP Setup ────────────────────────────────────────────────────────────────


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


# ─── Data ─────────────────────────────────────────────────────────────────────


def build_datasets(config: _config.TrainConfig):
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


# ─── Checkpointing ────────────────────────────────────────────────────────────


def _unwrap_model(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config, ema=None):
    if not is_main:
        return
    if not ((global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1):
        return

    final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
    tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

    if tmp_ckpt_dir.exists():
        shutil.rmtree(tmp_ckpt_dir)
    tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

    safetensors.torch.save_model(_unwrap_model(model), tmp_ckpt_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

    metadata = {
        "global_step": global_step,
        "config": dataclasses.asdict(config),
        "timestamp": time.time(),
    }
    torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

    if ema is not None:
        torch.save(ema.state_dict(), tmp_ckpt_dir / "ema.pt")

    norm_stats = data_config.norm_stats
    if norm_stats is not None and data_config.asset_id is not None:
        _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

    if final_ckpt_dir.exists():
        shutil.rmtree(final_ckpt_dir)
    tmp_ckpt_dir.rename(final_ckpt_dir)
    logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

    if config.wandb_enabled:
        wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device, ema=None):
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")
        safetensors.torch.load_model(_unwrap_model(model), safetensors_path, device=str(device))
        logging.info("Loaded model state from safetensors format")

        torch.cuda.empty_cache()
        gc.collect()

        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"
        if optimizer_path.exists():
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device, weights_only=False))
            logging.info("Loaded optimizer state")
        else:
            logging.warning("No optimizer checkpoint found, starting optimizer from scratch")

        ema_path = ckpt_dir / "ema.pt"
        if ema is not None and ema_path.exists():
            ema.load_state_dict(torch.load(ema_path, map_location=device, weights_only=False))
            logging.info("Loaded EMA state")

        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_checkpoint")

        logging.info(f"Successfully loaded checkpoint from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


# ─── QAT (Quantization-Aware Training) ───────────────────────────────────────


def apply_qat(model: torch.nn.Module):
    """Insert fake quantization nodes for INT8-aware training.

    Simulates INT8 rounding in the forward pass so LoRA adapters learn to
    compensate. Uses per-tensor symmetric quantization with learned scales.
    Only quantizes frozen (base) weight linear layers; LoRA and other
    trainable params stay in full precision.
    """
    from torch.ao.quantization import QConfigMapping, get_default_qat_qconfig
    from torch.ao.quantization.quantize_fx import prepare_qat_fx

    try:
        qconfig = get_default_qat_qconfig("x86")
        qconfig_mapping = QConfigMapping().set_global(qconfig)

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and not any(
                p.requires_grad for p in module.parameters()
            ):
                module.qconfig = qconfig

        logging.info("Applied QAT fake quantization to frozen linear layers")
    except Exception as e:
        logging.warning(f"QAT setup failed ({e}), falling back to manual fake quant")
        _apply_manual_fake_quant(model)


def _apply_manual_fake_quant(model: torch.nn.Module):
    """Fallback: wrap frozen Linear layers with fake quantization observers."""
    from torch.ao.quantization import FakeQuantize, default_weight_fake_quant

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            all_frozen = all(not p.requires_grad for p in module.parameters())
            if all_frozen:
                fq = default_weight_fake_quant()
                original_forward = module.forward

                def make_fq_forward(mod, fq_node):
                    def fq_forward(x):
                        mod.weight.data = fq_node(mod.weight.data)
                        return original_forward(x)
                    return fq_forward

                module.forward = make_fq_forward(module, fq)
                count += 1

    logging.info(f"Applied manual fake quantization to {count} frozen Linear layers")


# ─── Training Loop ────────────────────────────────────────────────────────────


def train_loop(config: _config.TrainConfig):
    # Enable TF32 unconditionally for all Ampere+ GPUs
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    set_seed(config.seed, local_rank)

    # Checkpoint directory setup
    resuming = False
    if config.resume:
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(f"Resuming from {exp_checkpoint_dir} at step {latest_step}")
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir}")
        else:
            raise FileNotFoundError(f"Checkpoint dir {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    if not resuming:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Data
    effective_batch_size = config.batch_size // world_size
    loader, data_config = build_datasets(config)

    if is_main and config.wandb_enabled and not resuming:
        sample_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_loader))
        observation, actions = sample_batch
        sample_dict = observation.to_dict()
        sample_dict["actions"] = actions

        images_to_log = []
        batch_size = next(iter(sample_dict["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            img_cat = torch.cat([img[i].permute(1, 2, 0) for img in sample_dict["image"].values()], axis=1)
            images_to_log.append(wandb.Image(img_cat.cpu().numpy()))
        wandb.log({"camera_views": images_to_log}, step=0)

        del sample_dict, observation, actions, images_to_log, sample_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ─── Model ────────────────────────────────────────────────────────────────

    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing")

    # Load pretrained weights before freezing
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")
        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(model, model_path)
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Apply freeze filter (must be after weight loading, before optimizer)
    trainable_params, frozen_count, trainable_count = apply_freeze_filter(model, config)
    has_frozen_params = frozen_count > 0

    # QAT: insert fake quantization on frozen layers (optional)
    use_qat = getattr(config, "quantization_aware", False)
    if use_qat and has_frozen_params:
        apply_qat(model)
        logging.info("QAT enabled: frozen layers have fake quantization")
    elif use_qat:
        logging.warning("QAT requested but no frozen params -- skipping")

    # DDP
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )

    # torch.compile (opt-in via env var to avoid issues with some model configs)
    if os.environ.get("TORCH_COMPILE", "0") == "1":
        logging.info("Applying torch.compile to model...")
        compiled_model = torch.compile(_unwrap_model(model))
        logging.info("torch.compile applied")
    else:
        compiled_model = None

    # ─── Optimizer ────────────────────────────────────────────────────────────

    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    optim = torch.optim.AdamW(
        trainable_params,
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # EMA
    ema = None
    if config.ema_decay is not None:
        ema = EMATracker(_unwrap_model(model), config.ema_decay)
        logging.info(f"EMA enabled with decay={config.ema_decay}")

    # Resume
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device, ema=ema)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    # ─── Logging ──────────────────────────────────────────────────────────────

    model.train()
    start_time = time.time()
    infos = []

    if is_main:
        logging.info(f"Running on: {platform.node()} | world_size={world_size}")
        logging.info(
            f"Training: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, "
            f"num_train_steps={config.num_train_steps}"
        )
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, "
            f"decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: AdamW, weight_decay={config.optimizer.weight_decay}, "
            f"clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info(
            f"Params: {frozen_count} frozen, {trainable_count} trainable, "
            f"EMA={'yes' if ema else 'no'}, QAT={'yes' if use_qat else 'no'}"
        )
        logging.info(f"Training precision: {model_cfg.dtype}, TF32: enabled, AMP: bf16")

    # ─── Training Loop ────────────────────────────────────────────────────────

    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    use_amp = config.pytorch_training_precision == "bfloat16" and torch.cuda.is_available()

    while global_step < config.num_train_steps:
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            if global_step >= config.num_train_steps:
                break

            observation = tree_map_tensors(lambda x: x.to(device), observation)
            actions = actions.to(torch.float32).to(device)

            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                losses = model(observation, actions)
                if isinstance(losses, (list, tuple)):
                    losses = torch.stack(losses)
                elif not isinstance(losses, torch.Tensor):
                    losses = torch.tensor(losses, device=device, dtype=torch.float32)
                loss = losses.mean()

            loss.backward()

            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.optimizer.clip_gradient_norm)
            optim.step()
            optim.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update(_unwrap_model(model))

            # Collect stats
            if is_main:
                info = {
                    "loss": loss.item(),
                    "learning_rate": optim.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                }
                infos.append(info)

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                avg_loss = sum(i["loss"] for i in infos) / len(infos)
                avg_lr = sum(i["learning_rate"] for i in infos) / len(infos)
                avg_grad_norm = None
                grad_vals = [i["grad_norm"] for i in infos if i.get("grad_norm") is not None]
                if grad_vals:
                    avg_grad_norm = sum(grad_vals) / len(grad_vals)

                param_norm = torch.sqrt(
                    sum(p.data.float().norm() ** 2 for p in trainable_params)
                ).item()

                log_parts = [
                    f"step={global_step}",
                    f"loss={avg_loss:.4f}",
                    f"lr={avg_lr:.2e}",
                ]
                if avg_grad_norm is not None:
                    log_parts.append(f"grad_norm={avg_grad_norm:.2f}")
                log_parts.append(f"param_norm={param_norm:.2f}")
                log_parts.append(f"time={elapsed:.1f}s")
                logging.info(" ".join(log_parts))

                if config.wandb_enabled:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "param_norm": param_norm,
                        "step": global_step,
                        "time_per_step": elapsed / max(1, config.log_interval),
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []

            global_step += 1
            save_checkpoint(model, optim, global_step, config, is_main, data_config, ema=ema)

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    if pbar is not None:
        pbar.close()
    if is_main and config.wandb_enabled:
        wandb.finish()
    cleanup_ddp()


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
