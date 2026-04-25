"""Background reader of the TB20E CAN bus exposing the live 8-axis state.

The cab broadcasts joystick state on five extended CAN IDs (0x1C50A0..A4 EB) at
~9 Hz. We sniff those frames with python-can/socketcan, run them through the
TB20E decoder, and expose the latest decoded 8-axis vector to the inference
client at any rate.

Eight axes:
    [left_stick_x, left_stick_y, right_stick_x, right_stick_y,
     left_track,   right_track,  swing,         blade]

All values are in [-1, +1] (the same normalization used by the training data).

The reader runs in a daemon thread and is safe to start once at process
init. ``get_state()`` is a non-blocking snapshot.

Requirements
------------
- python-can installed (``pip install python-can``)
- socketcan interface up at the configured bitrate, e.g.::

      sudo ip link set can0 up type can bitrate 500000

- The TB20E decoder lives outside this repo at the path passed via
  ``decoder_module_path`` (defaults to ``/home/Actor/Thor-CAN-recording``).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

CAN_ACTION_DIM = 8
DEFAULT_CHANNEL = "can0"
DEFAULT_DECODER_PATH = "/home/Actor/Thor-CAN-recording"
DEFAULT_STALE_TIMEOUT_S = 0.5


def _import_decoder(decoder_module_path: str):
    """Import the TB20EDecoder class from a directory on disk."""
    if decoder_module_path not in sys.path:
        sys.path.insert(0, decoder_module_path)
    candidate = os.path.join(decoder_module_path, "tb20e_decoder.py")
    if not os.path.isfile(candidate):
        raise FileNotFoundError(
            f"tb20e_decoder.py not found at {candidate}. "
            "Pass decoder_module_path=<dir containing tb20e_decoder.py>."
        )
    spec = importlib.util.spec_from_file_location("tb20e_decoder", candidate)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load tb20e_decoder from {candidate}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TB20EDecoder


class CANStateReader:
    """Streams the TB20E's live 8-axis joystick state from socketcan.

    Args:
        channel: socketcan interface name (default ``can0``).
        decoder_module_path: directory containing ``tb20e_decoder.py``.
        stale_timeout_s: if no frames have been received in this many seconds
            ``get_state`` still returns the last value but ``is_stale()``
            reports True so the caller can react.
        log_every_n_frames: emit an INFO log every N frames received (0 disables).
    """

    def __init__(
        self,
        channel: str = DEFAULT_CHANNEL,
        decoder_module_path: str = DEFAULT_DECODER_PATH,
        stale_timeout_s: float = DEFAULT_STALE_TIMEOUT_S,
        log_every_n_frames: int = 0,
    ):
        try:
            import can
        except ImportError as exc:
            raise ImportError(
                "python-can is required for CANStateReader. "
                "Install with: pip install python-can"
            ) from exc

        self._can = can
        self.channel = channel
        self.stale_timeout_s = float(stale_timeout_s)
        self.log_every_n_frames = int(log_every_n_frames)

        decoder_cls = _import_decoder(decoder_module_path)
        self._decoder = decoder_cls()

        self._bus: Optional[object] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        self._state = np.zeros(CAN_ACTION_DIM, dtype=np.float32)
        self._last_frame_t: float = 0.0
        self._frames_received = 0

    def start(self) -> None:
        """Open the bus and start the reader thread (idempotent)."""
        if self._running:
            return
        try:
            self._bus = self._can.interface.Bus(
                self.channel, interface="socketcan", receive_own_messages=False
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open socketcan channel '{self.channel}'. "
                "Is the interface up? Try:  sudo ip link set "
                f"{self.channel} up type can bitrate 500000"
            ) from exc
        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="CANStateReader",
            daemon=True,
        )
        self._thread.start()
        logger.info("CAN reader started on %s", self.channel)

    def stop(self) -> None:
        """Stop the reader thread and close the bus."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._bus is not None:
            try:
                self._bus.shutdown()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._bus = None
        logger.info(
            "CAN reader stopped (%d frames decoded)", self._frames_received
        )

    def __enter__(self) -> "CANStateReader":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    def get_state(self) -> np.ndarray:
        """Return the latest 8-axis state as a fresh float32 array."""
        with self._lock:
            return self._state.copy()

    def is_stale(self) -> bool:
        """True if no frames have been decoded within ``stale_timeout_s``."""
        with self._lock:
            if self._last_frame_t == 0.0:
                return True
            return (time.monotonic() - self._last_frame_t) > self.stale_timeout_s

    @property
    def frames_received(self) -> int:
        return self._frames_received

    def _reader_loop(self) -> None:
        assert self._bus is not None
        while self._running:
            msg = self._bus.recv(timeout=0.1)  # type: ignore[attr-defined]
            if msg is None:
                continue
            self._decoder.decode(msg.arbitration_id, msg.data, msg.is_extended_id)
            self._frames_received += 1
            state = self._decoder.get_state()
            arr = np.asarray(
                [
                    state.left_stick_x, state.left_stick_y,
                    state.right_stick_x, state.right_stick_y,
                    state.left_track, state.right_track,
                    state.swing, state.blade,
                ],
                dtype=np.float32,
            )
            now = time.monotonic()
            with self._lock:
                self._state = arr
                self._last_frame_t = now
            if (
                self.log_every_n_frames > 0
                and self._frames_received % self.log_every_n_frames == 0
            ):
                logger.info(
                    "CAN frames=%d  state=%s",
                    self._frames_received,
                    np.array2string(arr, precision=3, suppress_small=True),
                )
