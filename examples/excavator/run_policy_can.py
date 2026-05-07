"""Run the CAN-bus excavator VLA policy (8-dim, 3 cameras) on the Takeuchi TB20E.

Architecture
------------
                              +-------------------+
                              |   RunPod GPU pod  |
                              | serve_policy.py   |
                              | pi05_canteleop_   |
                              |     fullft        |
                              +---------+---------+
                                        ^
                                        | wss:// proxy
                                        |
   3 USB cameras (cab/front_l/front_r)  |
   socketcan can0 -> 8-axis state ------+
                                        |
                                        v
                              +-------------------+
                              | this script (Jet) |
                              | run_policy_can.py |
                              +---------+---------+
                                        |
                                        | TakeuchiClient (UDP -> 2 Arduinos)
                                        v
                            +-----------+-----------+
                            |   Excavator (CAN bus) |
                            +-----------------------+

Inputs to the model (8-dim CAN-state policy):
    observation/image_cab_forward   (224x224x3 uint8)
    observation/image_front_left    (224x224x3 uint8)
    observation/image_front_right   (224x224x3 uint8)
    observation/state               float32[8]  (decoded TB20E joystick state)
    prompt                          str

Outputs from the model:
    actions  float32[N, 8]
        [left_stick_x, left_stick_y, right_stick_x, right_stick_y,
         left_track,   right_track,  swing,         blade]

Asymmetry: 8 axes are READABLE from CAN but only 4 are WRITABLE through the
current 2-Arduino bridge (joystick X/Y for each stick). The remaining four
axes (tracks / dedicated swing / blade) are LOGGED but not actuated. A loud
WARNING is printed once at startup naming each dropped axis.

Quick start
-----------
1. Bring up the CAN bus on the Jetson::

       sudo ip link set can0 up type can bitrate 500000

2. Plug in three USB cameras (defaults: cab=2, front_left=6, front_right=0).

3. Start the policy server on RunPod (see RUNPOD_DEPLOY_CAN.md).

4. Run this script::

       sudo /home/Actor/actor-final-jetson-deployment/.venv/bin/python \\
           examples/excavator/run_policy_can.py \\
           --host wss://<POD-ID>-8000.proxy.runpod.net \\
           --prompt "Scoop packing peanuts from large pool and dump into small pool"

Runtime keys (cooked terminal):
    SPACE  E-STOP   (zero all axes immediately)
    R      Resume
    Q      Quit (graceful shutdown)
"""

from __future__ import annotations

import argparse
import logging
import os
import select
import signal
import sys
import termios
import threading
import time
import tty
import urllib.request
from typing import Optional

import cv2
import numpy as np

from action_interpolator import (
    CAN_ACTION_DIM,
    ActionInterpolator,
    AxisConfig,
    InterpolatorConfig,
)
from can_state_reader import (
    DEFAULT_CHANNEL,
    DEFAULT_DECODER_PATH,
    CANStateReader,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_policy_can")

# ---------------------------------------------------------------------------
# Constants (mirror the executor in actor-final-jetson-deployment)
# ---------------------------------------------------------------------------

MODEL_IMG_SIZE = 224
CAM_W, CAM_H = 640, 480
DEFAULT_ACTION_HZ = 50.0   # action chunks are decoded at this rate
DEFAULT_CAB_CAM_INDEX = 2
DEFAULT_FRONT_LEFT_INDEX = 6
DEFAULT_FRONT_RIGHT_INDEX = 0

DEFAULT_TAKEUCHI_REPO = "/home/Actor/takeuchi-canbus"


# ---------------------------------------------------------------------------
# Camera source — local OpenCV index OR remote HTTP JPEG
# ---------------------------------------------------------------------------

class CameraSource:
    """Unified camera grabber. Either a local OpenCV device index or HTTP URL."""

    def __init__(self, source, label: str, rotate_180: bool = False):
        self.label = label
        self.rotate_180 = rotate_180
        self._is_http = isinstance(source, str) and source.startswith("http")

        if self._is_http:
            self._url = source
            self._cap = None
            try:
                resp = urllib.request.urlopen(self._url, timeout=5)
                resp.read()
                logger.info("[cam:%s] HTTP source OK: %s", label, source)
            except Exception as exc:
                sys.exit(f"Failed to reach {label} camera at {source}: {exc}")
        else:
            self._url = None
            self._cap = cv2.VideoCapture(int(source), cv2.CAP_V4L2)
            if not self._cap.isOpened():
                sys.exit(f"Failed to open {label} camera (device index {source})")
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
            self._cap.set(cv2.CAP_PROP_FPS, 30)
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc_int = int(self._cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
            warmup_ok = 0
            for _ in range(15):
                ok, _ = self._cap.read()
                if ok:
                    warmup_ok += 1
                    if warmup_ok >= 3:
                        break
                time.sleep(0.05)
            logger.info(
                "[cam:%s] /dev/video%s %dx%d %s rotate_180=%s warmup=%d",
                label, source, actual_w, actual_h, fourcc_str, rotate_180, warmup_ok,
            )
            if warmup_ok == 0:
                logger.warning(
                    "[cam:%s] /dev/video%s opened but produced no warmup frames; "
                    "will continue but expect drops.", label, source,
                )

    def grab(self, size: int = MODEL_IMG_SIZE) -> Optional[np.ndarray]:
        if self._is_http:
            frame = self._grab_http(size)
        else:
            frame = self._grab_local(size)
        if frame is not None and self.rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    def _grab_local(self, size: int) -> Optional[np.ndarray]:
        ret, bgr = self._cap.read()  # type: ignore[union-attr]
        if not ret:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        return rgb.astype(np.uint8)

    def _grab_http(self, size: int) -> Optional[np.ndarray]:
        try:
            resp = urllib.request.urlopen(self._url, timeout=2)  # type: ignore[arg-type]
            data = resp.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
            return rgb.astype(np.uint8)
        except Exception:
            return None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()


# ---------------------------------------------------------------------------
# Background CAN command sender — drives TakeuchiClient at a fixed rate
# ---------------------------------------------------------------------------

class TakeuchiCommander:
    """Sends the latest 4-axis target to the TakeuchiClient at action_hz.

    A background thread holds the last-set axis values and re-emits them at a
    fixed cadence so the CAN bridge keeps a steady heartbeat (matching the
    ~35 Hz we measured during teleop). The main thread updates targets
    via ``set_targets``; the watchdog returns to neutral if no update arrives
    for ``watchdog_timeout`` seconds.

    Args:
        send_hz: rate at which the bridge is driven (default 50 Hz, matches
            training data).
        watchdog_timeout: seconds without a new ``set_targets`` call before
            the commander forces neutral.
        repo_path: takeuchi-canbus repo for TakeuchiClient.
        dry_run: if True, do not touch CAN — print intent only.
    """

    def __init__(
        self,
        send_hz: float = DEFAULT_ACTION_HZ,
        watchdog_timeout: float = 0.5,
        repo_path: str = DEFAULT_TAKEUCHI_REPO,
        dry_run: bool = False,
    ):
        self.send_hz = float(send_hz)
        self.watchdog_timeout = float(watchdog_timeout)
        self.dry_run = dry_run
        self._tc = None
        self._lock = threading.Lock()
        self._targets = {"bucket": 0.0, "boom": 0.0, "body": 0.0, "arm": 0.0}
        self._last_set_t = 0.0
        self._estop = False
        self._running = False
        self._packets_sent = 0
        self._thread: Optional[threading.Thread] = None

        if not dry_run:
            scripts = os.path.join(repo_path, "scripts")
            if scripts not in sys.path:
                sys.path.insert(0, scripts)
            from takeuchi_client import TakeuchiClient  # noqa: F401
            self._TakeuchiClient = TakeuchiClient
        else:
            self._TakeuchiClient = None

    def start(self) -> None:
        if self._running:
            return
        if not self.dry_run:
            self._tc = self._TakeuchiClient()  # type: ignore[misc]
            self._tc.__enter__()
        self._running = True
        self._last_set_t = time.monotonic()
        self._thread = threading.Thread(
            target=self._send_loop, name="TakeuchiCommander", daemon=True,
        )
        self._thread.start()
        logger.info(
            "TakeuchiCommander started (send_hz=%.1f, dry_run=%s)",
            self.send_hz, self.dry_run,
        )

    def stop(self) -> None:
        self._running = False
        self.set_estop()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._tc is not None:
            try:
                self._tc.neutral(count=5)
            except Exception:
                pass
            self._tc.__exit__(None, None, None)
            self._tc = None
        logger.info(
            "TakeuchiCommander stopped (%d packets sent)", self._packets_sent
        )

    def set_targets(self, *, bucket: float, boom: float, body: float, arm: float) -> None:
        with self._lock:
            self._targets["bucket"] = float(np.clip(bucket, -1.0, 1.0))
            self._targets["boom"]   = float(np.clip(boom,   -1.0, 1.0))
            self._targets["body"]   = float(np.clip(body,   -1.0, 1.0))
            self._targets["arm"]    = float(np.clip(arm,    -1.0, 1.0))
            self._last_set_t = time.monotonic()

    def set_estop(self) -> None:
        with self._lock:
            self._estop = True
            for k in self._targets:
                self._targets[k] = 0.0

    def clear_estop(self) -> None:
        with self._lock:
            self._estop = False
            self._last_set_t = time.monotonic()

    @property
    def packets_sent(self) -> int:
        return self._packets_sent

    def _send_loop(self) -> None:
        interval = 1.0 / self.send_hz
        next_tick = time.monotonic()
        while self._running:
            with self._lock:
                idle = time.monotonic() - self._last_set_t
                if idle > self.watchdog_timeout:
                    for k in self._targets:
                        self._targets[k] = 0.0
                t = dict(self._targets)
            if not self.dry_run and self._tc is not None:
                try:
                    self._tc.send(**t)
                    self._packets_sent += 1
                except Exception as exc:
                    logger.error("TakeuchiClient.send failed: %s", exc)
            next_tick += interval
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()


# ---------------------------------------------------------------------------
# Keyboard listener — SPACE / R / Q
# ---------------------------------------------------------------------------

class KeyboardController:
    def __init__(self, commander: Optional[TakeuchiCommander]):
        self._commander = commander
        self._estopped = False
        self._quit = False
        self._old_settings = None
        self._thread = threading.Thread(target=self._listen, name="kb", daemon=True)

    @property
    def estopped(self) -> bool:
        return self._estopped

    @property
    def quit_requested(self) -> bool:
        return self._quit

    def start(self) -> None:
        if not sys.stdin.isatty():
            logger.info("stdin is not a TTY; keyboard controls disabled.")
            return
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self._thread.start()
        logger.info("Keyboard:  SPACE=E-STOP   R=Resume   Q=Quit")

    def stop(self) -> None:
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None

    def _listen(self) -> None:
        try:
            while not self._quit:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    if ch == " ":
                        self._estopped = True
                        if self._commander is not None:
                            self._commander.set_estop()
                        logger.warning(">>> E-STOP — axes zeroed. Press R to resume.")
                    elif ch in ("r", "R"):
                        self._estopped = False
                        if self._commander is not None:
                            self._commander.clear_estop()
                        logger.info(">>> RESUMED.")
                    elif ch in ("q", "Q"):
                        self._quit = True
                        logger.info(">>> QUIT requested.")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Build interpolator config from CLI args
# ---------------------------------------------------------------------------

def build_interp_config(args: argparse.Namespace) -> InterpolatorConfig:
    def _slew(v: Optional[float]) -> Optional[float]:
        return None if v is None or v < 0 else float(v)

    def _ema(v: Optional[float]) -> Optional[float]:
        if v is None or v <= 0.0 or v >= 1.0:
            return None
        return float(v)

    return InterpolatorConfig(
        left_stick_x=AxisConfig(
            ema_alpha=_ema(args.smooth_lx),
            gain=args.gain_lx, invert=args.invert_lx,
            deadzone=args.deadzone_lx, slew_per_step=_slew(args.slew_lx),
        ),
        left_stick_y=AxisConfig(
            ema_alpha=_ema(args.smooth_ly),
            gain=args.gain_ly, invert=args.invert_ly,
            deadzone=args.deadzone_ly, slew_per_step=_slew(args.slew_ly),
        ),
        right_stick_x=AxisConfig(
            ema_alpha=_ema(args.smooth_rx),
            gain=args.gain_rx, invert=args.invert_rx,
            deadzone=args.deadzone_rx, slew_per_step=_slew(args.slew_rx),
        ),
        right_stick_y=AxisConfig(
            ema_alpha=_ema(args.smooth_ry),
            gain=args.gain_ry, invert=args.invert_ry,
            deadzone=args.deadzone_ry, slew_per_step=_slew(args.slew_ry),
        ),
        left_track=AxisConfig(),
        right_track=AxisConfig(),
        swing=AxisConfig(),
        blade=AxisConfig(),
    )


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    from openpi_client import websocket_client_policy as wcp

    cab_source = args.cab_cam_url if args.cab_cam_url else args.cab_cam
    fl_source = args.front_left_cam_url if args.front_left_cam_url else args.front_left_cam
    fr_source = args.front_right_cam_url if args.front_right_cam_url else args.front_right_cam
    cab_cam = CameraSource(cab_source, "cab_forward", rotate_180=args.cab_rotate_180)
    fl_cam  = CameraSource(fl_source,  "front_left",  rotate_180=args.front_left_rotate_180)
    fr_cam  = CameraSource(fr_source,  "front_right", rotate_180=args.front_right_rotate_180)

    can_reader: Optional[CANStateReader] = None
    if not args.no_can:
        can_reader = CANStateReader(
            channel=args.can_channel,
            decoder_module_path=args.decoder_path,
            stale_timeout_s=args.can_stale_timeout,
        )
        can_reader.start()
    else:
        logger.warning("--no-can set: state will be all zeros.")

    commander = TakeuchiCommander(
        send_hz=args.action_hz,
        watchdog_timeout=args.watchdog_timeout,
        repo_path=args.takeuchi_repo,
        dry_run=args.dry_run,
    )
    if not args.no_send:
        commander.start()
    else:
        logger.warning("--no-send set: actions will be computed but NOT transmitted to CAN.")

    interp = ActionInterpolator(build_interp_config(args))
    interp.warn_dropped_once()

    kb = KeyboardController(commander if not args.no_send else None)
    kb.start()

    logger.info("Connecting to policy server at %s ...", args.host)
    policy = wcp.WebsocketClientPolicy(
        host=args.host, port=args.port if not args.host.startswith("ws") else None,
    )
    metadata = policy.get_server_metadata()
    logger.info("Server metadata: %s", metadata)

    interval = 1.0 / args.action_hz
    step = 0
    shutdown = False

    def on_signal(_sig, _frame):
        nonlocal shutdown
        shutdown = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        while not shutdown and not kb.quit_requested:
            if args.max_steps and step >= args.max_steps:
                logger.info("Reached max steps (%d). Stopping.", args.max_steps)
                break

            if kb.estopped:
                interp.reset()
                time.sleep(0.05)
                continue

            cab = cab_cam.grab(MODEL_IMG_SIZE)
            fl = fl_cam.grab(MODEL_IMG_SIZE)
            fr = fr_cam.grab(MODEL_IMG_SIZE)
            if cab is None or fl is None or fr is None:
                drop_counter = locals().get("drop_counter", 0) + 1
                if drop_counter == 1 or drop_counter % 30 == 0:
                    missing = [n for n, x in (("cab", cab), ("fl", fl), ("fr", fr)) if x is None]
                    logger.warning(
                        "camera frame dropped (%s) — retry #%d", ",".join(missing), drop_counter,
                    )
                time.sleep(0.01)
                continue
            drop_counter = 0

            if can_reader is not None:
                state = can_reader.get_state()
                if can_reader.is_stale():
                    logger.warning(
                        "CAN state stale (no frames for >%.2fs). "
                        "Sending zeros as state.", can_reader.stale_timeout_s,
                    )
                    state = np.zeros(CAN_ACTION_DIM, dtype=np.float32)
            else:
                state = np.zeros(CAN_ACTION_DIM, dtype=np.float32)

            obs = {
                "observation/image_cab_forward":  cab,
                "observation/image_front_left":   fl,
                "observation/image_front_right":  fr,
                "observation/image/cab_forward":  cab,
                "observation/image/front_left":   fl,
                "observation/image/front_right":  fr,
                "observation/state":              state.astype(np.float32),
                "prompt":                         args.prompt,
            }

            t_infer = time.perf_counter()
            result = policy.infer(obs)
            infer_ms = (time.perf_counter() - t_infer) * 1000.0
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] < CAN_ACTION_DIM:
                logger.error(
                    "unexpected action shape from server: %s (expected [N, 8])",
                    actions.shape,
                )
                continue

            voted_stage = result.get("voted_stage")
            voted_label = result.get("voted_stage_label")
            raw_label   = result.get("raw_stage_label")
            cycles      = result.get("cycle_count")
            cycle_done  = bool(result.get("cycle_complete", False))
            if cycle_done:
                logger.info(
                    "*** CYCLE COMPLETE *** total cycles=%s  stage=%s",
                    cycles, voted_label,
                )

            for i in range(len(actions)):
                if shutdown or kb.quit_requested or kb.estopped:
                    break
                t0 = time.perf_counter()
                processed = interp.process(actions[i, :CAN_ACTION_DIM])
                cmd = ActionInterpolator.demux_to_takeuchi(processed)

                if not args.no_send:
                    commander.set_targets(**cmd)

                if args.verbose or i == 0:
                    if voted_label is not None:
                        stage_str = (
                            f" stage={voted_stage}:{voted_label[:14]:<14s}"
                            f" cyc={cycles if cycles is not None else '-'}"
                        )
                    else:
                        stage_str = ""
                    sys.stdout.write(
                        f"\rstep {step:6d} chunk {i+1:2d}/{len(actions):2d} "
                        f"infer {infer_ms:5.0f}ms |{stage_str} | "
                        f"lx={processed[0]:+.2f} ly={processed[1]:+.2f} "
                        f"rx={processed[2]:+.2f} ry={processed[3]:+.2f} | "
                        f"lt={processed[4]:+.2f} rt={processed[5]:+.2f} "
                        f"sw={processed[6]:+.2f} bl={processed[7]:+.2f}   "
                    )
                    sys.stdout.flush()

                step += 1
                if args.max_steps and step >= args.max_steps:
                    break

                elapsed = time.perf_counter() - t0
                sleep_for = interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
    finally:
        sys.stdout.write("\n")
        logger.info("shutting down ...")
        kb.stop()
        try:
            cab_cam.release(); fl_cam.release(); fr_cam.release()
        except Exception:
            pass
        if not args.no_send:
            commander.stop()
        if can_reader is not None:
            can_reader.stop()
        logger.info("done. executed %d steps.", step)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Run the 8-axis CAN excavator policy on the TB20E.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = p.add_argument_group("Policy server (RunPod)")
    g.add_argument("--host", required=True,
                   help="Policy server host. Use the RunPod proxy URL "
                        "wss://<POD-ID>-8000.proxy.runpod.net")
    g.add_argument("--port", type=int, default=8000,
                   help="Port (ignored when --host starts with ws/wss).")
    g.add_argument("--prompt", default="Scoop packing peanuts from large pool and dump into small pool",
                   help="Language instruction for the policy.")

    g = p.add_argument_group("Cameras")
    g.add_argument("--cab-cam", type=int, default=DEFAULT_CAB_CAM_INDEX,
                   help="OpenCV index for cab_forward camera.")
    g.add_argument("--front-left-cam", type=int, default=DEFAULT_FRONT_LEFT_INDEX,
                   help="OpenCV index for front_left camera.")
    g.add_argument("--front-right-cam", type=int, default=DEFAULT_FRONT_RIGHT_INDEX,
                   help="OpenCV index for front_right camera.")
    g.add_argument("--cab-cam-url", default=None, help="HTTP URL for cab cam (overrides index).")
    g.add_argument("--front-left-cam-url", default=None, help="HTTP URL for front_left.")
    g.add_argument("--front-right-cam-url", default=None, help="HTTP URL for front_right.")
    g.add_argument("--cab-rotate-180", action="store_true",
                   help="Rotate the cab camera 180 deg (matches existing Jetson rig).")
    g.add_argument("--front-left-rotate-180", action="store_true",
                   help="Rotate the front_left camera 180 deg (use if mounted inverted).")
    g.add_argument("--front-right-rotate-180", action="store_true",
                   help="Rotate the front_right camera 180 deg (use if mounted inverted).")

    g = p.add_argument_group("CAN state input")
    g.add_argument("--can-channel", default=DEFAULT_CHANNEL,
                   help=f"socketcan channel for the TB20E (default {DEFAULT_CHANNEL}).")
    g.add_argument("--decoder-path", default=DEFAULT_DECODER_PATH,
                   help="Directory containing tb20e_decoder.py.")
    g.add_argument("--can-stale-timeout", type=float, default=0.5,
                   help="Warn if no CAN frames have been decoded in this many seconds.")
    g.add_argument("--no-can", action="store_true",
                   help="Disable the CAN reader; send zeros as state.")

    g = p.add_argument_group("CAN command output (TakeuchiClient)")
    g.add_argument("--takeuchi-repo", default=DEFAULT_TAKEUCHI_REPO,
                   help="Path to the takeuchi-canbus repo.")
    g.add_argument("--action-hz", type=float, default=DEFAULT_ACTION_HZ,
                   help="Rate at which actions are decoded and re-sent (default 50 Hz).")
    g.add_argument("--watchdog-timeout", type=float, default=0.5,
                   help="Seconds without a new action before commander forces neutral.")
    g.add_argument("--no-send", action="store_true",
                   help="Compute actions but do not send to CAN bridge.")
    g.add_argument("--dry-run", action="store_true",
                   help="No CAN at all (no Arduino subprocesses spawned).")

    g = p.add_argument_group("Per-axis interpolator (defaults: pure passthrough)")
    for axis in ("lx", "ly", "rx", "ry"):
        g.add_argument(f"--smooth-{axis}", type=float, default=0.0,
                       help=(f"EMA low-pass alpha for {axis} (0.0 = off; 0.2 ~= 1.6Hz "
                             f"cutoff at 50Hz; 0.1 ~= 0.8Hz; 0.05 ~= 0.4Hz). "
                             f"Smooths model jitter without adding latency or "
                             f"changing magnitude. Recommended: 0.10-0.20."))
        g.add_argument(f"--gain-{axis}", type=float, default=1.0,
                       help=f"Gain for {axis} axis (default 1.0).")
        g.add_argument(f"--invert-{axis}", type=lambda x: bool(int(x)), default=False,
                       help=f"Invert {axis} axis (1/0, default 0).")
        g.add_argument(f"--deadzone-{axis}", type=float, default=0.0,
                       help=f"Deadzone for {axis} axis (default 0.0).")
        g.add_argument(f"--slew-{axis}", type=float, default=-1.0,
                       help=f"Max |delta| per step for {axis} (negative = unbounded).")

    g = p.add_argument_group("Run")
    g.add_argument("--max-steps", type=int, default=None,
                   help="Stop after this many action steps.")
    g.add_argument("--verbose", action="store_true",
                   help="Print every action in the chunk (default: only the first).")

    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
