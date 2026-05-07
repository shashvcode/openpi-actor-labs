"""Action interpolator: model 8-dim outputs -> TB20E joystick commands.

Pipeline applied to each axis, in order:
    1. EMA low-pass    (smooth model jitter; see docstring on AxisConfig.ema_alpha)
    2. invert          (negate if configured)
    3. gain            (multiplicative scale)
    4. deadzone        (set to zero if |value| < deadzone)
    5. clip            (clip to [-1, +1])
    6. slew limit      (cap step-to-step delta; default = unbounded passthrough)

Then 8 axes -> 4 commandable axes via the demux that matches the existing
Jetson executor: lx -> body, ly -> arm, rx -> bucket, ry -> boom.
The remaining four axes (left_track, right_track, swing, blade) are
observable from CAN state but are NOT commandable through the current
2-Arduino bridge — they are logged once at startup with a clear WARNING.

This module is pure-Python / numpy and has no I/O or threads. Wire it up
to a CAN sender (e.g. TakeuchiClient.send) inside the inference loop.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

CAN_ACTION_DIM = 8

AXIS_NAMES = (
    "left_stick_x",   # 0  -> body
    "left_stick_y",   # 1  -> arm
    "right_stick_x",  # 2  -> bucket
    "right_stick_y",  # 3  -> boom
    "left_track",     # 4  -> NOT COMMANDABLE
    "right_track",    # 5  -> NOT COMMANDABLE
    "swing",          # 6  -> NOT COMMANDABLE
    "blade",          # 7  -> NOT COMMANDABLE
)

COMMANDABLE_INDICES = (0, 1, 2, 3)
DROPPED_INDICES = (4, 5, 6, 7)


@dataclasses.dataclass
class AxisConfig:
    """Per-axis transform applied before slew limiting.

    ema_alpha : one-pole IIR low-pass on the RAW model output. Applied first,
                so it smooths the model's jitter before any other transform.
                ``y[t] = alpha*x[t] + (1-alpha)*y[t-1]``.
                  alpha=1.0 -> no smoothing (passthrough)
                  alpha=0.5 -> mild
                  alpha=0.2 -> ~1.6 Hz cutoff at 50 Hz tick rate
                  alpha=0.1 -> ~0.8 Hz cutoff at 50 Hz tick rate
                  alpha=0.05 -> ~0.4 Hz cutoff (very heavy)
                None or 1.0 = passthrough.
    gain     : multiplicative scale on the model output.
    invert   : negate the value (useful when joystick polarity in training
               disagrees with the bridge convention).
    deadzone : if |value| < deadzone after gain/invert, snap to 0.
    slew_per_step : maximum allowed |delta| between consecutive 50 Hz steps.
                    None = unbounded (pure passthrough).
    """
    ema_alpha: Optional[float] = None
    gain: float = 1.0
    invert: bool = False
    deadzone: float = 0.0
    slew_per_step: Optional[float] = None


@dataclasses.dataclass
class InterpolatorConfig:
    """All-axis configuration. Defaults = pure passthrough.

    The slew defaults are None (unbounded). Per the captured CAN data,
    a sensible safety value for joystick axes is ~0.5/step at 50 Hz
    (~25 axis-units/s, matches observed human max). Set via CLI.
    """
    left_stick_x: AxisConfig = dataclasses.field(default_factory=AxisConfig)
    left_stick_y: AxisConfig = dataclasses.field(default_factory=AxisConfig)
    right_stick_x: AxisConfig = dataclasses.field(default_factory=AxisConfig)
    right_stick_y: AxisConfig = dataclasses.field(default_factory=AxisConfig)
    left_track: AxisConfig = dataclasses.field(default_factory=AxisConfig)
    right_track: AxisConfig = dataclasses.field(default_factory=AxisConfig)
    swing: AxisConfig = dataclasses.field(default_factory=AxisConfig)
    blade: AxisConfig = dataclasses.field(default_factory=AxisConfig)

    def by_index(self) -> tuple[AxisConfig, ...]:
        return (
            self.left_stick_x, self.left_stick_y,
            self.right_stick_x, self.right_stick_y,
            self.left_track, self.right_track,
            self.swing, self.blade,
        )


class ActionInterpolator:
    """Stateful per-axis transform + slew limiter."""

    def __init__(self, config: Optional[InterpolatorConfig] = None):
        self.cfg = config or InterpolatorConfig()
        self._last = np.zeros(CAN_ACTION_DIM, dtype=np.float32)
        self._ema = np.zeros(CAN_ACTION_DIM, dtype=np.float32)
        self._ema_initialized = np.zeros(CAN_ACTION_DIM, dtype=bool)
        self._warned_dropped = False

    def reset(self) -> None:
        """Forget the previous output (next slew step starts from 0)."""
        self._last[:] = 0.0
        self._ema[:] = 0.0
        self._ema_initialized[:] = False

    def process(self, action_8: np.ndarray) -> np.ndarray:
        """Apply EMA smoothing + per-axis transforms + slew limit. Returns float32[8].

        The output is the value that should be SENT to the bridge for this
        timestep. The full 8 dims are returned for logging/inspection;
        callers use ``demux_to_takeuchi`` to extract the 4 commandable axes.
        """
        if action_8.shape[-1] != CAN_ACTION_DIM:
            raise ValueError(
                f"expected action of shape (..., {CAN_ACTION_DIM}), got {action_8.shape}"
            )
        a = np.asarray(action_8, dtype=np.float32).copy()
        cfgs = self.cfg.by_index()

        for i, axis_cfg in enumerate(cfgs):
            v = float(a[i])

            # 1. EMA low-pass on the raw model value (smooth high-freq jitter).
            if axis_cfg.ema_alpha is not None and 0.0 < axis_cfg.ema_alpha < 1.0:
                alpha = float(axis_cfg.ema_alpha)
                if not self._ema_initialized[i]:
                    self._ema[i] = v
                    self._ema_initialized[i] = True
                else:
                    self._ema[i] = alpha * v + (1.0 - alpha) * float(self._ema[i])
                v = float(self._ema[i])

            # 2-5. invert / gain / deadzone / clip
            if axis_cfg.invert:
                v = -v
            v *= axis_cfg.gain
            if axis_cfg.deadzone > 0.0 and abs(v) < axis_cfg.deadzone:
                v = 0.0
            v = max(-1.0, min(1.0, v))

            # 6. slew limit
            if axis_cfg.slew_per_step is not None:
                cap = float(axis_cfg.slew_per_step)
                prev = float(self._last[i])
                delta = v - prev
                if delta > cap:
                    v = prev + cap
                elif delta < -cap:
                    v = prev - cap
            a[i] = v

        self._last = a.copy()
        return a

    def warn_dropped_once(self) -> None:
        """Log a single bright warning about the 4 axes we cannot command."""
        if self._warned_dropped:
            return
        self._warned_dropped = True
        names = ", ".join(AXIS_NAMES[i] for i in DROPPED_INDICES)
        logger.warning(
            "Model outputs 8 axes but the current TB20E bridge can only command "
            "the four joystick axes. Dropped (logged only): %s", names
        )

    @staticmethod
    def demux_to_takeuchi(action_8: np.ndarray) -> dict[str, float]:
        """Map the 8-axis action to TakeuchiClient kwargs (bucket, boom, body, arm).

        Mapping is identical to actor-final-jetson-deployment/gpu_inference/executor.py:
            bucket = right_stick_x  (idx 2)
            boom   = right_stick_y  (idx 3)
            body   = left_stick_x   (idx 0)
            arm    = left_stick_y   (idx 1)
        """
        return {
            "bucket": float(action_8[2]),
            "boom":   float(action_8[3]),
            "body":   float(action_8[0]),
            "arm":    float(action_8[1]),
        }

    @staticmethod
    def neutral() -> np.ndarray:
        """Return an all-zero action vector for resets / e-stop."""
        return np.zeros(CAN_ACTION_DIM, dtype=np.float32)
