"""Precompute 4x4 depth grids for all frames in a LeRobot dataset.

Runs DepthAnything V2 Metric Indoor on each scene image and saves a compact
(N, 16) numpy array of metric depth values (in metres, clamped to [0, 2m]).

The output file is used during training by SO100DepthGridInputs to append
spatial depth features to the joystick state vector.

Usage (run on GPU):
    uv run python scripts/compute_depth_grids.py --repo-id verm11/runA --output depth_grids.npy
"""

import argparse
import io
import logging

import numpy as np
import torch
import tqdm
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GRID_H, GRID_W = 4, 4
MIN_DEPTH_M, MAX_DEPTH_M = 0.0, 2.0


def load_depth_model(device: torch.device):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    logger.info("Loading %s on %s", model_id, device)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    if device.type == "cuda":
        model = model.half()
    logger.info("Depth model loaded.")
    return model, processor


@torch.no_grad()
def compute_depth_grid(model, processor, pil_img: Image.Image, device: torch.device) -> np.ndarray:
    """Estimate metric depth from a PIL image and return a flattened 4x4 grid (16 values in metres)."""
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if device.type == "cuda":
        inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

    depth = model(**inputs).predicted_depth.squeeze().float().cpu().numpy()
    depth = np.clip(depth, MIN_DEPTH_M, MAX_DEPTH_M)

    h, w = depth.shape
    cell_h, cell_w = h // GRID_H, w // GRID_W
    grid = np.zeros(GRID_H * GRID_W, dtype=np.float32)
    for i in range(GRID_H):
        for j in range(GRID_W):
            cell = depth[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]
            grid[i * GRID_W + j] = cell.mean()

    return grid


def decode_image(img_data) -> Image.Image:
    """Decode an image from various dataset formats."""
    if isinstance(img_data, Image.Image):
        return img_data.convert("RGB")
    if isinstance(img_data, dict):
        if "bytes" in img_data and img_data["bytes"] is not None:
            return Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
        if "path" in img_data and img_data["path"] is not None:
            return Image.open(img_data["path"]).convert("RGB")
    if isinstance(img_data, np.ndarray):
        if img_data.ndim == 3 and img_data.shape[0] == 3:
            img_data = np.transpose(img_data, (1, 2, 0))
        return Image.fromarray(img_data).convert("RGB")
    if hasattr(img_data, "numpy"):
        arr = img_data.numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        return Image.fromarray(arr).convert("RGB")
    raise ValueError(f"Cannot decode image of type {type(img_data)}")


def main():
    parser = argparse.ArgumentParser(description="Precompute depth grids for a LeRobot dataset")
    parser.add_argument("--repo-id", type=str, default="verm11/runA")
    parser.add_argument("--output", type=str, default="depth_grids.npy")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    from datasets import load_dataset

    logger.info("Loading dataset %s ...", args.repo_id)
    ds = load_dataset(args.repo_id, split="train")
    num_frames = len(ds)
    logger.info("Dataset has %d frames", num_frames)

    model, processor = load_depth_model(device)

    grids = np.zeros((num_frames, GRID_H * GRID_W), dtype=np.float32)

    for idx in tqdm.tqdm(range(num_frames), desc="Computing depth grids"):
        row = ds[idx]
        scene_data = row.get("observation.images.scene")
        if scene_data is None:
            continue

        pil_img = decode_image(scene_data)
        grids[idx] = compute_depth_grid(model, processor, pil_img, device)

    np.save(args.output, grids)
    logger.info("Saved depth grids to %s — shape %s", args.output, grids.shape)
    logger.info("Value range: [%.4f, %.4f] metres", grids.min(), grids.max())


if __name__ == "__main__":
    main()
