import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

ACTION_DIM = 4


def make_excavator_example() -> dict:
    """Creates a random input example for the excavator joystick policy."""
    return {
        "observation/state": np.random.uniform(-1, 1, size=(ACTION_DIM,)).astype(np.float32),
        "observation/image_cab": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image_side": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "Scoop packing peanuts from large pool and dump into small pool",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class ExcavatorInputs(transforms.DataTransformFn):
    """Maps excavator joystick dataset fields to the model's expected input format.

    Joystick state (4 dims): left_x, left_y, right_x, right_y
    Two cameras: cab-mounted (csi_0_imx219) and side-mounted (usb_0).
    """

    def __call__(self, data: dict) -> dict:
        cab_image = _parse_image(data["observation/image_cab"])
        side_image = _parse_image(data["observation/image_side"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": cab_image,
                "left_wrist_0_rgb": side_image,
                "right_wrist_0_rgb": np.zeros_like(cab_image),
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
class ExcavatorOutputs(transforms.DataTransformFn):
    """Extracts excavator joystick actions (4 dims) from padded model output."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}


CAN_ACTION_DIM = 8


def make_can_excavator_example() -> dict:
    """Creates a random input example for the CAN-bus excavator policy (TB20E)."""
    return {
        "observation/image_cab_forward": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image_front_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image_front_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.uniform(-1, 1, size=(CAN_ACTION_DIM,)).astype(np.float32),
        "prompt": "Scoop packing peanuts from large pool and dump in small pool",
    }


@dataclasses.dataclass(frozen=True)
class CANExcavatorInputs(transforms.DataTransformFn):
    """Maps CAN-bus excavator (TB20E) dataset fields to the model's expected input format.

    8-dim CAN state/action: [lx, ly, rx, ry, left_track, right_track, swing, blade]
    Three cameras: cab_forward, front_left, front_right — all three model image slots used.
    """

    def __call__(self, data: dict) -> dict:
        cab_image = _parse_image(data["observation/image_cab_forward"])
        left_image = _parse_image(data["observation/image_front_left"])
        right_image = _parse_image(data["observation/image_front_right"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": cab_image,
                "left_wrist_0_rgb": left_image,
                "right_wrist_0_rgb": right_image,
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


@dataclasses.dataclass(frozen=True)
class CANExcavatorOutputs(transforms.DataTransformFn):
    """Extracts CAN-bus excavator actions (8 dims) from padded model output."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :CAN_ACTION_DIM])}
