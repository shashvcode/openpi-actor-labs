import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

ACTION_DIM = 6


def make_so100_example(use_depth: bool = False) -> dict:
    """Creates a random input example for the SO-100 joystick policy."""
    example = {
        "observation/state": np.random.uniform(-1, 1, size=(6,)).astype(np.float32),
        "observation/image_scene": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "Pick up the bottle and place it on the yellow outlined square.",
    }
    if use_depth:
        example["observation/image_scene_depth"] = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
    return example


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SO100Inputs(transforms.DataTransformFn):
    """Maps SO-100 joystick dataset fields to the model's expected input format.

    Joystick state (6 dims): left_x, left_y, right_x, right_y, l2_trigger, r2_trigger
    Two cameras: scene (third-person) and wrist.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image_scene"])
        wrist_image = _parse_image(data["observation/image_wrist"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SO100DepthInputs(transforms.DataTransformFn):
    """Maps SO-100 dataset fields to model input, using scene depth as the 3rd image."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image_scene"])
        wrist_image = _parse_image(data["observation/image_wrist"])

        if "observation/image_scene_depth" in data:
            depth_image = _parse_image(data["observation/image_scene_depth"])
        else:
            depth_image = np.zeros_like(base_image)

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": depth_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


DEPTH_GRID_H = 4
DEPTH_GRID_W = 4
DEPTH_GRID_DIM = DEPTH_GRID_H * DEPTH_GRID_W


def make_so100_depth_grid_example() -> dict:
    """Creates a random input example for the SO-100 policy with depth grid state."""
    return {
        "observation/state": np.random.uniform(-1, 1, size=(ACTION_DIM + DEPTH_GRID_DIM,)).astype(np.float32),
        "observation/image_scene": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "Pick up the bottle and place it on the yellow outlined square.",
    }


@dataclasses.dataclass(frozen=True)
class SO100DepthGridInputs(transforms.DataTransformFn):
    """Maps SO-100 dataset fields to model input, appending precomputed depth grid to state.

    State becomes 22-dim: [joystick(6), depth_grid(16)].
    Only scene + wrist images are used (no depth image input).
    Depth grid values are loaded from a precomputed numpy file indexed by frame.
    """

    depth_grids: np.ndarray | None = None

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image_scene"])
        wrist_image = _parse_image(data["observation/image_wrist"])

        joystick_state = np.asarray(data["observation/state"], dtype=np.float32)

        if self.depth_grids is not None and "_index" in data:
            idx = int(data["_index"])
            depth_grid = self.depth_grids[idx].astype(np.float32)
        elif "observation/depth_grid" in data:
            depth_grid = np.asarray(data["observation/depth_grid"], dtype=np.float32)
        else:
            depth_grid = np.zeros(DEPTH_GRID_DIM, dtype=np.float32)

        state = np.concatenate([joystick_state, depth_grid])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SO100Outputs(transforms.DataTransformFn):
    """Extracts SO-100 joystick actions (6 dims) from padded model output."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}
