# Pi-0.5 PyTorch Training — Cloud GPU Guide

This guide covers everything needed to train a native PyTorch pi-0.5 model on cloud GPUs (H100s).
The resulting `model.safetensors` can be deployed directly on the Jetson without any JAX-to-PyTorch
conversion step.

## Prerequisites

- 1–4× H100 GPUs (1 is sufficient for LoRA fine-tuning, more = faster)
- ~50 GB disk (model weights + dataset)
- Network access to GCS (`gs://openpi-assets`) and HuggingFace
- Python 3.11+

---

## Step 0: Clone & Install

```bash
git clone https://github.com/razamalik7/openpi-actor-labs-raza.git
cd openpi-actor-labs-raza
git checkout jetson-integration

# Install with uv (recommended) — handles all deps including JAX+CUDA and PyTorch
pip install uv
uv sync

# OR install manually (if uv doesn't work on your cloud image):
pip install -e ".[dev]"
# JAX with CUDA (needed only for the base checkpoint conversion step):
pip install "jax[cuda12]==0.5.3"
```

Verify the install:
```bash
python -c "import openpi; import torch; import jax; print(f'PyTorch {torch.__version__}, JAX {jax.__version__}')"
```

---

## Step 1: Convert JAX Base Checkpoint to PyTorch

This converts the pretrained pi-0.5 base model from JAX/orbax format to PyTorch safetensors.
**You only need to do this once.**

```bash
python examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir gs://openpi-assets/checkpoints/pi05_base \
    --config_name pi05_so100_lora_v3 \
    --output_path ./checkpoints/pi05_base_pytorch \
    --precision float32
```

This will:
- Download the JAX checkpoint from GCS (~6 GB)
- Convert all weights (PaliGemma vision+LM, Gemma action expert, projection layers)
- Save as `./checkpoints/pi05_base_pytorch/model.safetensors`

---

## Step 2: Verify the Conversion (Important!)

Run a numerical comparison between the JAX and PyTorch models to confirm the conversion is correct:

```bash
python scripts/verify_conversion.py \
    --config_name pi05_so100_lora_v3 \
    --pytorch_weight_path ./checkpoints/pi05_base_pytorch
```

Expected output:
```
CONVERSION VERIFICATION RESULTS
  v_t MSE (JAX vs PyTorch):  < 1e-04
  v_t max absolute diff:     < 1e-02
  ✓ PASS — Conversion is numerically correct!
```

**If this fails, do NOT proceed with training** — the conversion has a bug.

---

## Step 3: Train

### Single GPU (simplest)

```bash
python scripts/train_pytorch.py pi05_so100_lora_v3 \
    --pytorch_weight_path ./checkpoints/pi05_base_pytorch \
    --exp_name runD
```

### Multi-GPU (2–4 GPUs with DDP)

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py pi05_so100_lora_v3 \
    --pytorch_weight_path ./checkpoints/pi05_base_pytorch \
    --exp_name runD
```

### With Weights & Biases logging

W&B is enabled by default. Set your API key first:
```bash
wandb login
```

To disable:
```bash
python scripts/train_pytorch.py pi05_so100_lora_v3 \
    --pytorch_weight_path ./checkpoints/pi05_base_pytorch \
    --exp_name runD \
    --wandb_enabled=False
```

### Training Config Summary (pi05_so100_lora_v3)

| Parameter | Value |
|---|---|
| Model | pi-0.5 (3.1B params) |
| Fine-tuning | LoRA (both PaliGemma + action expert) |
| Action dim | 6 (SO-100 joystick) |
| Action horizon | 11 |
| Dataset | `verm11/runA` (auto-downloads from HuggingFace) |
| Batch size | 32 |
| Steps | 10,000 |
| LR | 2.5e-5 (cosine decay w/ 1000 step warmup) |
| Optimizer | AdamW (b1=0.9, b2=0.95, clip=1.0) |
| Precision | bf16 (AMP) |
| EMA | Disabled |
| Save interval | Every 10,000 steps (final only) |

### Expected Training Time

| GPUs | Approximate Time |
|---|---|
| 1× H100 | ~1–2 hours |
| 2× H100 | ~45 min |
| 4× H100 | ~25 min |

---

## Step 4: Collect the Output

After training completes, the checkpoint is at:

```
checkpoints/pi05_so100_lora_v3/runD/10000/
├── model.safetensors    ← this is your trained model
├── optimizer.pt
├── metadata.pt
└── assets/
    └── verm11/runA/
        └── norm_stats.json  ← normalization stats (needed for inference)
```

Copy the checkpoint to your local machine:

```bash
# From your local machine / Jetson:
scp -r <cloud-host>:/path/to/openpi-actor-labs-raza/checkpoints/pi05_so100_lora_v3/runD/10000 \
    ./checkpoints/runD_pytorch/
```

---

## Step 5: Deploy on Jetson

Once the checkpoint is on the Jetson, you can run it with the PyTorch server:

```bash
# Inside the Docker container on Jetson:
python scripts/serve_pytorch_minimal.py
# (update the checkpoint path in the script to point to your new model)
```

Or export to TensorRT for lower latency:

```bash
# Export to ONNX then compile TRT engines
python scripts/export_pytorch_onnx.py \
    --config_name pi05_so100_lora_v3 \
    --weight_path ./checkpoints/runD_pytorch/model.safetensors \
    --output_dir ./onnx_export

# Compile with trtexec (on Jetson host)
trtexec --onnx=onnx_export/prefix_encoder.onnx --saveEngine=engines/prefix_encoder_fp32.engine
trtexec --onnx=onnx_export/denoise_step.onnx --saveEngine=engines/denoise_step_fp32.engine
```

---

## Optional: Run Parallel Experiments

With 4 GPUs, you can run 2–4 experiments simultaneously to find the best hyperparameters:

```bash
# Terminal 1: baseline (10K steps)
CUDA_VISIBLE_DEVICES=0 python scripts/train_pytorch.py pi05_so100_lora_v3 \
    --pytorch_weight_path ./checkpoints/pi05_base_pytorch \
    --exp_name runD_baseline

# Terminal 2: longer training (20K steps)
CUDA_VISIBLE_DEVICES=1 python scripts/train_pytorch.py pi05_so100_lora_v3 \
    --pytorch_weight_path ./checkpoints/pi05_base_pytorch \
    --exp_name runD_20k \
    --num_train_steps=20000 --save_interval=5000

# Terminal 3: higher learning rate
CUDA_VISIBLE_DEVICES=2 python scripts/train_pytorch.py pi05_so100_lora_v3 \
    --pytorch_weight_path ./checkpoints/pi05_base_pytorch \
    --exp_name runD_lr5e5 \
    --lr_schedule.peak_lr=5e-5

# Terminal 4: with QAT for INT8 deployment
CUDA_VISIBLE_DEVICES=3 python scripts/train_pytorch.py pi05_so100_lora_v3 \
    --pytorch_weight_path ./checkpoints/pi05_base_pytorch \
    --exp_name runD_qat \
    --quantization_aware=True
```

---

## Troubleshooting

### "pytorch_weight_path is required for fresh training"
You forgot `--pytorch_weight_path`. Run Step 1 first, then pass the path.

### GCS access error downloading JAX checkpoint
Make sure `gcloud` is authenticated or `GOOGLE_APPLICATION_CREDENTIALS` is set.
Alternatively, download the checkpoint manually:
```bash
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base ./checkpoints/pi05_base_jax
# Then use --checkpoint_dir ./checkpoints/pi05_base_jax in the convert command
```

### Dataset download fails
The training script auto-downloads `verm11/runA` from HuggingFace. If this fails:
```bash
pip install huggingface_hub
huggingface-cli login  # if the dataset is private
```

### Out of memory
- Reduce batch size: `--batch_size=16`
- Gradient checkpointing is enabled by default
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
