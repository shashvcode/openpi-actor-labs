"""Run the 4-dim, 3-cam excavator dirt-scoop VLA policy on the TB20E.

Pipeline
--------
    can0 (J1939 0x1C50A0EB)        ── raw SocketCAN ──► state thread ──► observation/state[4]
    /dev/video{0,2,4} (UVC)        ── cv2.VideoCapture ──► observation/image/{arm,front,side}_cam
    Cloud GPU policy server        ◄── WSS msgpack ──► policy.infer(obs)
    TakeuchiClient (UDP JOY4)      ── ports 42101+42102 ──► usb_can_bridge ──► can1/can2 ──► cab joysticks

State convention (matches actor-display SHM and the recorder dataset):
    observation/state = [left_stick_x, left_stick_y, right_stick_x, right_stick_y]

Model output -> machine axes (per takeuchi_client.py docstring):
    action[0] = lsx -> swing
    action[1] = lsy -> arm
    action[2] = rsx -> bucket
    action[3] = rsy -> boom

Camera mapping (per actor-display config REC_POSITION_TO_CAM):
    /dev/video0 -> arm_cam
    /dev/video2 -> front_cam   (operator-facing label "top")
    /dev/video4 -> side_cam

Prereqs
-------
1. can0 must be UP at 500 kbit/s (it is by default on this rig).
2. can1 + can2 must be UP at 500 kbit/s for the takeuchi bridges. The simplest
   one-shot is `sudo /home/lob/takeuchi-canbus/run_dual_sticks.sh` once per
   boot, or pass --auto-start-bridges so this script spawns them via sudo.
3. actor-display.service should be STOPPED so it doesn't hog the cameras.

Usage
-----
    uv run --no-project \\
        --with 'websockets>=11' --with 'msgpack>=1' \\
        --with 'numpy<2' --with 'opencv-python-headless' \\
        examples/excavator/run_policy_takeuchi.py \\
        --host wss://rxz3mb39eex3qv-8000.proxy.runpod.net \\
        --prompt "Scoop dirt from the pile and dump it." \\
        --dry-run

Drop --dry-run to actually drive the machine.
"""

from __future__ import annotations

import argparse
import logging
import os
import select
import signal
import socket
import struct
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import cv2
import msgpack
import numpy as np
import websockets.sync.client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


ACTION_DIM = 4
STATE_DIM = 4
MODEL_IMG_SIZE = 224
CAM_W, CAM_H = 640, 480
CONTROL_HZ_DEFAULT = 10
WS_TIMEOUT = 300

JOYSTICK_CAN_ID = 0x1C50A0EB
JOYSTICK_SCALE = 7000.0

CAN_TELEMETRY_IFACE_DEFAULT = "can0"
CAM_DEV_ARM = "/dev/video0"
CAM_DEV_FRONT = "/dev/video2"
CAM_DEV_SIDE = "/dev/video4"

TAKEUCHI_REPO = Path("/home/lob/takeuchi-canbus")
TAKEUCHI_HOST = "127.0.0.1"
TAKEUCHI_STICK1_CAN = "can2"
TAKEUCHI_STICK2_CAN = "can1"


# ───────────────────────────── CAN state reader ─────────────────────────────


class CANStateReader:
    """Background thread that decodes the 4-axis joystick frame from SocketCAN.

    Reads CAN ID 0x1C50A0EB on the configured iface (default ``can0``),
    extracts the four signed int16 axis fields, scales by 7000, and exposes
    the latest [lsx, lsy, rsx, rsy] vector with an "age" estimate.
    """

    def __init__(self, iface: str = CAN_TELEMETRY_IFACE_DEFAULT):
        self.iface = iface
        self._lock = threading.Lock()
        self._state = np.zeros(STATE_DIM, dtype=np.float32)
        self._last_ts = 0.0
        self._frames_seen = 0
        self._running = False
        self._sock: socket.socket | None = None
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        s.bind((self.iface,))
        eff_id = JOYSTICK_CAN_ID | socket.CAN_EFF_FLAG
        eff_mask = socket.CAN_EFF_MASK | socket.CAN_EFF_FLAG
        s.setsockopt(
            socket.SOL_CAN_RAW,
            socket.CAN_RAW_FILTER,
            struct.pack("=II", eff_id, eff_mask),
        )
        s.settimeout(1.0)
        self._sock = s
        self._running = True
        self._thread.start()
        logger.info("CAN state reader started on %s (filter 0x%08X)", self.iface, JOYSTICK_CAN_ID)

    def _loop(self) -> None:
        assert self._sock is not None
        while self._running:
            try:
                frame = self._sock.recv(16)
            except (socket.timeout, OSError):
                continue
            if len(frame) < 16:
                continue
            can_id, dlc = struct.unpack_from("=IB", frame, 0)
            data = frame[8:16]
            if dlc < 8:
                continue
            rx, ry, lx, ly = struct.unpack_from("<hhhh", data, 0)
            state = np.array(
                [
                    lx / JOYSTICK_SCALE,
                    ly / JOYSTICK_SCALE,
                    rx / JOYSTICK_SCALE,
                    ry / JOYSTICK_SCALE,
                ],
                dtype=np.float32,
            )
            np.clip(state, -1.0, 1.0, out=state)
            with self._lock:
                self._state = state
                self._last_ts = time.monotonic()
                self._frames_seen += 1

    def get(self) -> tuple[np.ndarray, float]:
        with self._lock:
            return self._state.copy(), time.monotonic() - self._last_ts

    @property
    def frames_seen(self) -> int:
        return self._frames_seen

    def close(self) -> None:
        self._running = False
        try:
            if self._sock is not None:
                self._sock.close()
        except OSError:
            pass


# ────────────────────────────── Camera grabber ──────────────────────────────


class Camera:
    """Single UVC camera with non-blocking continuous capture (no AE warmup penalty)."""

    def __init__(self, dev: str | int, label: str):
        self.label = label
        self.dev = dev
        # cv2.CAP_V4L2 wants an integer device index, not a /dev/videoN string.
        if isinstance(dev, str) and dev.startswith("/dev/video"):
            try:
                idx: int | str = int(dev.removeprefix("/dev/video"))
            except ValueError:
                idx = dev
        else:
            idx = dev
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open {label} camera at {dev}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        self._cap = cap
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._running = True
        self._frames = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ret, bgr = self._cap.read()
            if not ret or bgr is None:
                time.sleep(0.005)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb, (MODEL_IMG_SIZE, MODEL_IMG_SIZE), interpolation=cv2.INTER_AREA)
            with self._lock:
                self._latest = small.astype(np.uint8)
                self._frames += 1

    def grab(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    @property
    def frames(self) -> int:
        return self._frames

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        try:
            self._cap.release()
        except Exception:
            pass


# ─────────────────────────── WebSocket policy client ────────────────────────


def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(),
                b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]),
                          shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class PolicyClient:
    def __init__(self, host: str, port: int | None = None):
        uri = host if host.startswith("ws") else f"ws://{host}"
        if port is not None and not host.startswith("ws"):
            uri = f"{uri}:{port}"
        logger.info("Connecting to policy server at %s ...", uri)
        self._conn = websockets.sync.client.connect(
            uri, compression=None, max_size=None,
            ping_timeout=WS_TIMEOUT, close_timeout=10,
        )
        self._packer = msgpack.Packer(default=_pack_array, use_bin_type=True)
        self.metadata = msgpack.unpackb(self._conn.recv(), object_hook=_unpack_array, raw=False)
        logger.info("Server metadata: %s", self.metadata)

    def infer(self, obs: dict) -> dict:
        self._conn.send(self._packer.pack(obs))
        resp = self._conn.recv(timeout=WS_TIMEOUT)
        if isinstance(resp, str):
            raise RuntimeError(f"Policy server error:\n{resp}")
        return msgpack.unpackb(resp, object_hook=_unpack_array, raw=False)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# ───────────────────────────── Keyboard control ─────────────────────────────


class KeyboardController:
    """SPACE = e-stop, R = resume, Q = quit (cbreak terminal)."""

    def __init__(self):
        self.estopped = False
        self.quit = False
        self._old_settings = None
        self._thread = threading.Thread(target=self._listen, daemon=True)

    def start(self):
        if not sys.stdin.isatty():
            logger.warning("stdin is not a TTY — keyboard control disabled.")
            return
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self._thread.start()
        logger.info("Keyboard:  SPACE=E-STOP   R=Resume   Q=Quit")

    def stop(self):
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def _listen(self):
        try:
            while not self.quit:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    if ch == " ":
                        self.estopped = True
                        logger.warning(">>> E-STOP — actions zeroed.  Press R to resume.")
                    elif ch in ("r", "R"):
                        self.estopped = False
                        logger.info(">>> RESUMED.")
                    elif ch in ("q", "Q"):
                        self.quit = True
                        logger.info(">>> QUIT.")
        except Exception:
            pass


# ─────────────────────────────── Main loop ──────────────────────────────────


def _import_takeuchi_client():
    sys.path.insert(0, str(TAKEUCHI_REPO / "scripts"))
    from takeuchi_client import TakeuchiClient  # noqa: E402
    return TakeuchiClient


def run(args: argparse.Namespace) -> int:
    can_reader = CANStateReader(iface=args.can_iface)
    can_reader.start()

    cam_arm = Camera(args.cam_arm, "arm")
    cam_front = Camera(args.cam_front, "front")
    cam_side = Camera(args.cam_side, "side")

    logger.info("Waiting up to 3s for first camera frames + first CAN frame ...")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 3.0:
        if (cam_arm.grab() is not None and cam_front.grab() is not None
                and cam_side.grab() is not None and can_reader.frames_seen > 0):
            break
        time.sleep(0.05)
    if can_reader.frames_seen == 0:
        logger.warning("No joystick CAN frames yet on %s (id 0x%08X). "
                       "State will be zeros until they arrive.",
                       args.can_iface, JOYSTICK_CAN_ID)

    policy = PolicyClient(host=args.host, port=args.port)

    tc = None
    if not args.dry_run:
        TakeuchiClient = _import_takeuchi_client()
        tc = TakeuchiClient(
            host=TAKEUCHI_HOST,
            serial_stick1=TAKEUCHI_STICK1_CAN,
            serial_stick2=TAKEUCHI_STICK2_CAN,
            auto_start=args.auto_start_bridges,
            use_sudo=True,
            repo_path=TAKEUCHI_REPO,
        )
        if not args.auto_start_bridges:
            logger.info("Assuming bridges already running (run_dual_sticks.sh).")

    kb = KeyboardController()
    kb.start()

    shutdown = False
    def _on_signal(_sig, _frame):
        nonlocal shutdown
        shutdown = True
    signal.signal(signal.SIGINT, _on_signal)

    target_dt = 1.0 / args.control_hz
    step = 0
    loop_t0 = time.perf_counter()

    try:
        while not shutdown and not kb.quit:
            if args.max_steps and step >= args.max_steps:
                logger.info("Reached --max-steps %d. Stopping.", args.max_steps)
                break

            if kb.estopped:
                if tc is not None:
                    tc.send()  # neutral keep-alive while e-stopped
                time.sleep(0.05)
                continue

            arm_img = cam_arm.grab()
            front_img = cam_front.grab()
            side_img = cam_side.grab()
            if arm_img is None or front_img is None or side_img is None:
                logger.warning("Camera frame missing, retrying...")
                time.sleep(0.01)
                continue

            state, state_age = can_reader.get()
            if state_age > 1.0:
                logger.warning("Joystick CAN state is %.2fs old — sending stale state.", state_age)

            obs = {
                "observation/image/arm_cam":   arm_img,
                "observation/image/front_cam": front_img,
                "observation/image/side_cam":  side_img,
                "observation/state": state.astype(np.float32),
                "prompt": args.prompt,
            }

            t_inf = time.perf_counter()
            result = policy.infer(obs)
            inf_ms = (time.perf_counter() - t_inf) * 1000
            actions = np.asarray(result["actions"], dtype=np.float32)

            for action_idx in range(len(actions)):
                if shutdown or kb.quit or kb.estopped:
                    break

                a = actions[action_idx]
                lsx_raw, lsy_raw, rsx_raw, rsy_raw = (float(a[i]) for i in range(ACTION_DIM))

                # Final per-axis multiplier = action_scale * scale_<axis> * sign_<axis>.
                # The default signs (+, -, -, -) reflect the rig's wiring: arm
                # (lsy), bucket (rsx), and boom (rsy) are inverted relative to
                # the cab joystick convention captured in the dataset.
                lsx = max(-1.0, min(1.0, lsx_raw * args.action_scale * args.scale_swing  * args.sign_swing))
                lsy = max(-1.0, min(1.0, lsy_raw * args.action_scale * args.scale_arm    * args.sign_arm))
                rsx = max(-1.0, min(1.0, rsx_raw * args.action_scale * args.scale_bucket * args.sign_bucket))
                rsy = max(-1.0, min(1.0, rsy_raw * args.action_scale * args.scale_boom   * args.sign_boom))

                action_t0 = time.perf_counter()
                if tc is None:
                    if action_idx == 0:
                        logger.info(
                            "[dry-run] step=%d  state=[%+.3f %+.3f %+.3f %+.3f]  "
                            "action[0]raw=[%+.3f %+.3f %+.3f %+.3f]  "
                            "sent=[swing=%+.3f arm=%+.3f bucket=%+.3f boom=%+.3f]  "
                            "infer=%.0fms",
                            step, state[0], state[1], state[2], state[3],
                            lsx_raw, lsy_raw, rsx_raw, rsy_raw,
                            lsx, lsy, rsx, rsy, inf_ms,
                        )
                else:
                    tc.send(bucket=rsx, boom=rsy, swing=lsx, arm=lsy)

                sys.stdout.write(
                    f"\r step {step:5d} | swing={lsx:+.3f} arm={lsy:+.3f} bucket={rsx:+.3f} boom={rsy:+.3f} "
                    f"(raw rsy={rsy_raw:+.3f} lsy={lsy_raw:+.3f}) "
                    f"| chunk {action_idx+1}/{len(actions)} | infer {inf_ms:5.0f}ms "
                    f"| state_age {state_age*1000:4.0f}ms   "
                )
                sys.stdout.flush()

                step += 1
                if args.max_steps and step >= args.max_steps:
                    break

                elapsed = time.perf_counter() - action_t0
                sleep_time = target_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    finally:
        sys.stdout.write("\n")
        logger.info("Shutting down ...")
        kb.stop()
        if tc is not None:
            try:
                tc.neutral(count=3)
            except Exception:
                pass
            tc.close()
        try:
            policy.close()
        except Exception:
            pass
        cam_arm.close()
        cam_front.close()
        cam_side.close()
        can_reader.close()
        wall = time.perf_counter() - loop_t0
        logger.info("Done. step=%d in %.1fs (%.1f Hz). cam frames: arm=%d front=%d side=%d.  "
                    "joystick CAN frames: %d.",
                    step, wall, step / max(wall, 1e-6),
                    cam_arm.frames, cam_front.frames, cam_side.frames,
                    can_reader.frames_seen)
    return 0


def main():
    p = argparse.ArgumentParser(
        description="Excavator VLA policy runner (4-dim, 3-cam, TB20E + TakeuchiClient).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    g = p.add_argument_group("Policy server")
    g.add_argument("--host", required=True,
                   help="ws[s]://host[:port] or bare host (e.g. wss://<pod>-8000.proxy.runpod.net)")
    g.add_argument("--port", type=int, default=None,
                   help="Override port (only used when --host is a bare host).")
    g.add_argument("--prompt", default="Scoop dirt from the pile and dump it.",
                   help="Language instruction for the policy.")

    g = p.add_argument_group("Cameras")
    g.add_argument("--cam-arm", default=CAM_DEV_ARM)
    g.add_argument("--cam-front", default=CAM_DEV_FRONT)
    g.add_argument("--cam-side", default=CAM_DEV_SIDE)

    g = p.add_argument_group("CAN")
    g.add_argument("--can-iface", default=CAN_TELEMETRY_IFACE_DEFAULT,
                   help="SocketCAN iface for machine telemetry (joystick state).")

    g = p.add_argument_group("Control")
    g.add_argument("--control-hz", type=int, default=CONTROL_HZ_DEFAULT)
    g.add_argument("--max-steps", type=int, default=None)
    g.add_argument("--dry-run", action="store_true",
                   help="Don't write actions to the machine; print the first action of each chunk.")
    g.add_argument("--auto-start-bridges", action="store_true",
                   help="Have TakeuchiClient spawn usb_can_bridge_stick{1,2}.py via sudo. "
                        "Otherwise assume run_dual_sticks.sh is already running.")

    g = p.add_argument_group("Action shaping")
    g.add_argument("--action-scale", type=float, default=1.0,
                   help="Uniform scalar applied to ALL action axes before sending. "
                        "0.5 = half magnitude, 0.3 = ~one-third, 1.0 = raw model output. Default 1.0.")
    g.add_argument("--scale-swing",  type=float, default=1.0,
                   help="Extra per-axis scale on action[0] (swing). Stacks on --action-scale. Default 1.0.")
    g.add_argument("--scale-arm",    type=float, default=1.0,
                   help="Extra per-axis scale on action[1] (arm). Stacks on --action-scale. Default 1.0.")
    g.add_argument("--scale-bucket", type=float, default=1.0,
                   help="Extra per-axis scale on action[2] (bucket). Stacks on --action-scale. Default 1.0.")
    g.add_argument("--scale-boom",   type=float, default=1.0,
                   help="Extra per-axis scale on action[3] (boom). Stacks on --action-scale. Default 1.0.")
    g.add_argument("--sign-swing",  type=float, default=+1.0,
                   help="Sign for action[0] (left_stick_x -> swing). Default +1.")
    g.add_argument("--sign-arm",    type=float, default=-1.0,
                   help="Sign for action[1] (left_stick_y -> arm). Default -1 (rig is inverted).")
    g.add_argument("--sign-bucket", type=float, default=-1.0,
                   help="Sign for action[2] (right_stick_x -> bucket). Default -1 (rig is inverted).")
    g.add_argument("--sign-boom",   type=float, default=-1.0,
                   help="Sign for action[3] (right_stick_y -> boom). Default -1 (rig is inverted).")

    args = p.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
