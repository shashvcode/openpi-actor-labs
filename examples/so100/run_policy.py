"""Run the trained SO-100 encoder policy on the physical robot arm.

The model predicts absolute servo positions (encoder values ~1000-3000),
not joystick deltas. The client reads current servo positions as state,
sends them + camera frames to the Jetson, and moves to predicted positions.

Usage:
    # Start the policy server on the Jetson first:
    #   PYTHONPATH=src:packages/openpi-client/src python3 scripts/serve_pytorch_minimal.py \
    #     --config-name pi05_so100_encoder --port 8001

    # Then on your laptop (cameras + arm connected here):
    python examples/so100/run_policy.py \
        --host 192.168.1.88 --port 8001 \
        --scene-cam 0 --wrist-cam 1 \
        --prompt "Pick up the bottle and place it on the yellow outlined square."

    # Dry run (no arm, prints predicted positions):
    python examples/so100/run_policy.py --host 192.168.1.88 --port 8001 \
        --dry-run --scene-cam 0 --wrist-cam 1

    # Detect available cameras first:
    python examples/so100/run_policy.py --detect-cameras
"""

import argparse
import glob
import logging
import signal
import sys
import time

import cv2
import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Servo registers (Feetech STS3215)
# ---------------------------------------------------------------------------
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42
ADDR_PRESENT_POSITION = 56
ADDR_LOCK = 55
ADDR_OPERATING_MODE = 33
ADDR_P_COEFFICIENT = 21
ADDR_D_COEFFICIENT = 22
ADDR_I_COEFFICIENT = 23
ADDR_MAX_TORQUE_LIMIT = 16
ADDR_PROTECTION_CURRENT = 28
ADDR_OVERLOAD_TORQUE = 36

# ---------------------------------------------------------------------------
# Motor configuration
# ---------------------------------------------------------------------------
MOTOR_IDS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}
ALL_IDS = [1, 2, 3, 4, 5, 6]

MOTOR_LIMITS = {
    1: (200, 3896),
    2: (400, 3600),
    3: (400, 3600),
    4: (400, 3600),
    5: (0, 4095),
    6: (1500, 3100),
}

TORQUE_LIMITS = {1: 700, 2: 700, 3: 700, 4: 700, 5: 700, 6: 400}
CURRENT_LIMITS = {1: 350, 2: 350, 3: 350, 4: 350, 5: 350, 6: 200}

SERVO_MID = 2048

MAX_POSITION_DELTA = 150

CONTROL_HZ = 30

# Camera
IMG_W, IMG_H = 640, 480


# ---------------------------------------------------------------------------
# ArmController
# ---------------------------------------------------------------------------
class ArmController:
    @staticmethod
    def _lobyte(w):
        return w & 0xFF

    @staticmethod
    def _hibyte(w):
        return (w >> 8) & 0xFF

    @staticmethod
    def _makeword(lo, hi):
        return (lo & 0xFF) | ((hi & 0xFF) << 8)

    def __init__(self, port: str):
        from scservo_sdk import PortHandler, protocol_packet_handler
        import inspect

        self.port_handler = PortHandler(port)
        sig = inspect.signature(protocol_packet_handler)
        if len(sig.parameters) >= 2:
            self.packet_handler = protocol_packet_handler(self.port_handler, 0)
            self._pass_port = False
        else:
            self.packet_handler = protocol_packet_handler()
            self._pass_port = True
        self.positions = {mid: float(SERVO_MID) for mid in ALL_IDS}
        self._last_sent_positions = {}
        self.torque_enabled = False
        self.sync_writer = None

    def _ph_call(self, method, *args):
        if self._pass_port:
            return method(self.port_handler, *args)
        return method(*args)

    def connect(self):
        from scservo_sdk import COMM_SUCCESS, GroupSyncWrite

        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open port {self.port_handler.port_name}")
        self.port_handler.setBaudRate(1_000_000)

        if self._pass_port:
            self.sync_writer = GroupSyncWrite(
                self.port_handler, self.packet_handler, ADDR_GOAL_POSITION, 2,
            )
        else:
            self.sync_writer = GroupSyncWrite(
                self.packet_handler, ADDR_GOAL_POSITION, 2,
            )

        for name, mid in MOTOR_IDS.items():
            _, comm, _ = self._ph_call(self.packet_handler.ping, mid)
            if comm != COMM_SUCCESS:
                raise RuntimeError(f"Motor {mid} ({name}) not responding!")
            logger.info("Motor %d (%s) OK", mid, name)

        self.read_positions()
        self._configure_servos()
        self._enable_torque()
        logger.info("Arm connected. Positions: %s", {m: int(p) for m, p in self.positions.items()})

    def read_positions(self):
        from scservo_sdk import COMM_SUCCESS

        for mid in ALL_IDS:
            data, result, _ = self._ph_call(self.packet_handler.readTxRx, mid, ADDR_PRESENT_POSITION, 2)
            if result == COMM_SUCCESS:
                self.positions[mid] = float(self._makeword(data[0], data[1]))

    def get_positions_array(self):
        """Return current servo positions as a numpy array in motor ID order [1..6]."""
        return np.array([self.positions[mid] for mid in ALL_IDS], dtype=np.float32)

    def _configure_servos(self):
        for mid in ALL_IDS:
            self._ph_call(self.packet_handler.write1ByteTxRx, mid, ADDR_LOCK, 0)
            self._ph_call(self.packet_handler.write1ByteTxRx, mid, ADDR_OPERATING_MODE, 0)
            self._ph_call(self.packet_handler.write1ByteTxRx, mid, ADDR_P_COEFFICIENT, 10)
            self._ph_call(self.packet_handler.write1ByteTxRx, mid, ADDR_I_COEFFICIENT, 0)
            self._ph_call(self.packet_handler.write1ByteTxRx, mid, ADDR_D_COEFFICIENT, 20)
            self._ph_call(self.packet_handler.write2ByteTxRx, mid, ADDR_MAX_TORQUE_LIMIT, TORQUE_LIMITS[mid])
            self._ph_call(self.packet_handler.write2ByteTxRx, mid, ADDR_PROTECTION_CURRENT, CURRENT_LIMITS[mid])
            self._ph_call(self.packet_handler.write1ByteTxRx, mid, ADDR_OVERLOAD_TORQUE, 30)
            self._ph_call(self.packet_handler.write1ByteTxRx, mid, ADDR_LOCK, 1)

    def _enable_torque(self):
        for mid in ALL_IDS:
            self._ph_call(self.packet_handler.write1ByteTxRx, mid, ADDR_TORQUE_ENABLE, 1)
        self.torque_enabled = True

    def disable_torque(self):
        for mid in ALL_IDS:
            self._ph_call(self.packet_handler.write1ByteTxRx, mid, ADDR_TORQUE_ENABLE, 0)
        self.torque_enabled = False
        logger.info("Torque disabled — arm is free.")

    def set_target_positions(self, target_positions: np.ndarray, max_delta: float = MAX_POSITION_DELTA):
        """Move toward target positions, clamping per-step delta for safety."""
        if not self.torque_enabled:
            return

        for i, mid in enumerate(ALL_IDS):
            current = self.positions[mid]
            target = float(target_positions[i])

            lo, hi = MOTOR_LIMITS[mid]
            target = max(lo, min(hi, target))

            delta = target - current
            if abs(delta) > max_delta:
                target = current + max_delta * (1.0 if delta > 0 else -1.0)

            self.positions[mid] = target

        self._send_positions()

    def _send_positions(self):
        if not self.torque_enabled or self.sync_writer is None:
            return
        int_pos = {mid: int(p) for mid, p in self.positions.items()}
        if int_pos == self._last_sent_positions:
            return
        self._last_sent_positions = int_pos.copy()

        self.sync_writer.clearParam()
        for mid in ALL_IDS:
            pos = int_pos[mid]
            self.sync_writer.addParam(mid, [self._lobyte(pos), self._hibyte(pos)])
        self.sync_writer.txPacket()
        time.sleep(0.002)

    def close(self):
        if self.torque_enabled:
            self.disable_torque()
        self.port_handler.closePort()


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------
def open_camera(index: int, label: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"Failed to open {label} camera (index {index})")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("%s camera: index %d, native %dx%d, stored as %dx%d", label, index, actual_w, actual_h, IMG_W, IMG_H)
    return cap


def grab_frame(cap: cv2.VideoCapture) -> np.ndarray | None:
    ret, bgr = cap.read()
    if not ret:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != IMG_H or rgb.shape[1] != IMG_W:
        rgb = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return rgb


# ---------------------------------------------------------------------------
# Main control loop — encoder position mode
# ---------------------------------------------------------------------------
def run(args):
    from openpi_client import websocket_client_policy as wcp

    port = args.port if not args.host.startswith("ws") else None
    logger.info("Connecting to policy server at %s:%s ...", args.host, port)
    policy = wcp.WebsocketClientPolicy(host=args.host, port=port)
    metadata = policy.get_server_metadata()
    logger.info("Server metadata: %s", metadata)

    scene_cap = open_camera(args.scene_cam, "scene")
    wrist_cap = open_camera(args.wrist_cam, "wrist")

    arm = None
    if not args.dry_run:
        arm = ArmController(args.arm_port)
        arm.connect()

    state = arm.get_positions_array() if arm else np.array([2048.0] * 6, dtype=np.float32)
    logger.info("Initial state (servo positions): [%s]",
                ", ".join(f"{v:.0f}" for v in state))

    step = 0
    target_dt = 1.0 / CONTROL_HZ

    shutdown = False
    def on_signal(_sig, _frame):
        nonlocal shutdown
        shutdown = True
    signal.signal(signal.SIGINT, on_signal)

    logger.info("Starting control loop at %d Hz (Ctrl+C to stop) ...", CONTROL_HZ)
    logger.info("Max position delta per step: %d encoder units", args.max_delta)

    try:
        while not shutdown:
            if args.max_steps and step >= args.max_steps:
                logger.info("Reached max steps (%d). Stopping.", args.max_steps)
                break

            scene_frame = grab_frame(scene_cap)
            wrist_frame = grab_frame(wrist_cap)
            if scene_frame is None or wrist_frame is None:
                logger.warning("Camera frame dropped, retrying...")
                continue

            if arm:
                arm.read_positions()
                state = arm.get_positions_array()

            obs = {
                "observation/state": state.copy(),
                "observation/image_scene": scene_frame,
                "observation/image_wrist": wrist_frame,
                "prompt": args.prompt,
            }

            infer_start = time.perf_counter()
            result = policy.infer(obs)
            infer_ms = (time.perf_counter() - infer_start) * 1000
            actions = result["actions"]

            chunk = actions[:args.chunk_size]

            if step % 10 == 0:
                logger.info(
                    "Step %d | infer %.0fms | state: [%s] | target[0]: [%s]",
                    step, infer_ms,
                    ", ".join(f"{v:.0f}" for v in state),
                    ", ".join(f"{v:.0f}" for v in chunk[0]),
                )

            for action_idx in range(len(chunk)):
                if shutdown:
                    break

                target_positions = chunk[action_idx]
                action_start = time.perf_counter()

                if args.dry_run:
                    if action_idx == 0:
                        delta = target_positions - state
                        logger.info(
                            "  [dry-run] target: [%s]  delta: [%s]",
                            ", ".join(f"{v:.0f}" for v in target_positions),
                            ", ".join(f"{v:+.0f}" for v in delta),
                        )
                else:
                    arm.set_target_positions(target_positions, max_delta=args.max_delta)

                state = np.array(target_positions, dtype=np.float32)
                step += 1

                if args.max_steps and step >= args.max_steps:
                    break

                elapsed = time.perf_counter() - action_start
                sleep_time = target_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    finally:
        logger.info("Shutting down...")
        scene_cap.release()
        wrist_cap.release()
        if arm is not None:
            arm.close()
        logger.info("Done. Executed %d steps.", step)


def detect_cameras():
    """Probe camera indices 0-9 and report which ones are available."""
    logger.info("Detecting available cameras (indices 0-9)...")
    found = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ret, _ = cap.read()
            status = "readable" if ret else "opened but no frames"
            logger.info("  Camera index %d: %dx%d (%s)", i, w, h, status)
            if ret:
                found.append(i)
            cap.release()
    if not found:
        logger.warning("No working cameras found!")
    else:
        logger.info("Working cameras: %s", found)
        if len(found) >= 2:
            logger.info("Suggested: --scene-cam %d --wrist-cam %d", found[0], found[1])
        elif len(found) == 1:
            logger.info("Only one camera found. You need two (scene + wrist).")
    return found


def main():
    parser = argparse.ArgumentParser(description="Run SO-100 encoder policy on physical arm")
    parser.add_argument("--host", type=str, default="192.168.1.88",
                        help="Policy server host (Jetson Thor IP)")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--arm-port", type=str, default=None,
                        help="Serial port for the arm (auto-detected if not set)")
    parser.add_argument("--scene-cam", type=int, default=0, help="Scene camera index")
    parser.add_argument("--wrist-cam", type=int, default=1, help="Wrist camera index")
    parser.add_argument("--prompt", type=str,
                        default="Pick up the bottle and place it on the yellow outlined square.")
    parser.add_argument("--max-delta", type=float, default=MAX_POSITION_DELTA,
                        help="Max servo position change per step (safety clamp, default: 150)")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps before stopping")
    parser.add_argument("--chunk-size", type=int, default=6,
                        help="Number of actions from each prediction to execute")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print predicted positions without sending to servos")
    parser.add_argument("--detect-cameras", action="store_true",
                        help="Detect available cameras and exit")
    args = parser.parse_args()

    if args.detect_cameras:
        detect_cameras()
        return

    if not args.dry_run and args.arm_port is None:
        candidates = glob.glob("/dev/tty.usbmodem*") + glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
        if candidates:
            args.arm_port = candidates[0]
            logger.info("Auto-detected arm port: %s", args.arm_port)
        else:
            logger.error("No arm port found. Use --arm-port or --dry-run")
            sys.exit(1)

    run(args)


if __name__ == "__main__":
    main()
