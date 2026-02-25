import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

ACTION_DIM = 6


def make_so100_example() -> dict:
    """Creates a random input example for the SO-100 joystick policy."""
    return {
        "observation/state": np.random.uniform(-1, 1, size=(6,)).astype(np.float32),
        "observation/image_scene": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "Pick up the bottle and place it on the yellow outlined square.",
    }


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
class SO100Outputs(transforms.DataTransformFn):
    """Extracts SO-100 joystick actions (6 dims) from padded model output."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}
