"""Server-side depth augmentation using DepthAnything V2 Metric Indoor.

Wraps a policy to estimate scene depth on GPU and inject it into observations
before inference. Uses metric depth with fixed [0, 2m] bounds + INFERNO colormap
for consistent colorization across frames.
"""

import logging

import cv2
import numpy as np
import torch
from openpi_client import base_policy

logger = logging.getLogger(__name__)

IMG_H, IMG_W = 480, 640
MIN_DEPTH_M = 0.0
MAX_DEPTH_M = 2.0


class DepthEstimator:
    """Metric monocular depth estimation using DepthAnything V2 Metric Indoor (ViT-S)."""

    def __init__(self, device: str = "cuda", min_depth: float = MIN_DEPTH_M, max_depth: float = MAX_DEPTH_M):
        self.device = torch.device(device)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.model = None
        self.processor = None

    def load(self):
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor

        model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
        logger.info("Loading metric depth model: %s on %s", model_id, self.device)
        logger.info("Depth bounds: [%.1f, %.1f] metres", self.min_depth, self.max_depth)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model = self.model.to(self.device).eval()
        if self.device.type == "cuda":
            self.model = self.model.half()
        logger.info("Depth model loaded.")

    @torch.no_grad()
    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        """Estimate metric depth from RGB (H,W,3 uint8), return INFERNO-colorized RGB (H,W,3 uint8).

        Depth values are in metres. Clamped to [min_depth, max_depth] then linearly
        mapped to 0-255 before applying the INFERNO colormap. This ensures the same
        physical distance always maps to the same color regardless of scene content.
        """
        from PIL import Image

        pil_img = Image.fromarray(rgb)
        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.device.type == "cuda":
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

        outputs = self.model(**inputs)
        depth = outputs.predicted_depth.squeeze().float().cpu().numpy()

        clamped = np.clip(depth, self.min_depth, self.max_depth)
        depth_norm = ((clamped - self.min_depth) / (self.max_depth - self.min_depth) * 255.0).astype(np.uint8)

        depth_bgr = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        depth_rgb = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)

        if depth_rgb.shape[0] != IMG_H or depth_rgb.shape[1] != IMG_W:
            depth_rgb = cv2.resize(depth_rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

        return depth_rgb


class DepthAugmentedPolicy(base_policy.BasePolicy):
    """Wraps a policy to add server-side scene depth estimation."""

    def __init__(self, inner_policy: base_policy.BasePolicy, device: str = "cuda"):
        self._inner = inner_policy
        self._depth = DepthEstimator(device=device)
        self._depth.load()

    @property
    def metadata(self):
        return self._inner.metadata

    def infer(self, obs: dict) -> dict:
        if "observation/image_scene" in obs and "observation/image_scene_depth" not in obs:
            scene = np.asarray(obs["observation/image_scene"])
            if scene.dtype != np.uint8:
                scene = (scene * 255).astype(np.uint8)
            obs["observation/image_scene_depth"] = self._depth.estimate(scene)
        return self._inner.infer(obs)
