"""Run the trained SO-100 joystick policy on the physical robot arm.

Connects to a remote GPU policy server via WebSocket, captures camera frames,
sends observations, and executes predicted joystick commands on the arm.

Usage:
    # Start the policy server on RunPod first:
    #   uv run scripts/serve_policy.py policy:checkpoint --policy.config pi05_so100_lora \
    #       --policy.dir checkpoints/pi05_so100_lora/run1/4999

    # Then on the Mac (with SSH tunnel to RunPod port 8000):
    python run_policy.py \
        --host localhost --port 8000 \
        --arm-port /dev/tty.usbmodem5A7A0157861 \
        --scene-cam 1 --wrist-cam 2 \
        --prompt "Pick up the bottle and place it on the yellow outlined square."

    # Dry run (no arm, prints predicted actions):
    python run_policy.py --host localhost --port 8000 --dry-run --scene-cam 1 --wrist-cam 2
"""

import argparse
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

# Control loop parameters
SPEED = 15.0
MAX_DT = 0.05
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
        """Call a packet_handler method, prepending port_handler if needed."""
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

        self._read_positions()
        self._configure_servos()
        self._enable_torque()
        logger.info("Arm connected. Positions: %s", {m: int(p) for m, p in self.positions.items()})

    def _read_positions(self):
        from scservo_sdk import COMM_SUCCESS

        for mid in ALL_IDS:
            data, result, _ = self._ph_call(self.packet_handler.readTxRx, mid, ADDR_PRESENT_POSITION, 2)
            if result == COMM_SUCCESS:
                self.positions[mid] = float(self._makeword(data[0], data[1]))

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

    def move(self, motor_id: int, delta: float):
        if not self.torque_enabled:
            return
        lo, hi = MOTOR_LIMITS[motor_id]
        self.positions[motor_id] = max(lo, min(hi, self.positions[motor_id] + delta))

    def send_positions(self):
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
# Joystick-to-servo mapping
# ---------------------------------------------------------------------------
def apply_joystick_to_arm(arm: ArmController, joystick: np.ndarray, dt: float, speed: float):
    """Convert predicted joystick values [lx, ly, rx, ry, l2, r2] to servo deltas."""
    dt = min(dt, MAX_DT)
    scale = speed * dt * 60

    lx, ly, rx, ry, l2, r2 = joystick

    arm.move(MOTOR_IDS["shoulder_pan"], lx * scale)
    arm.move(MOTOR_IDS["elbow_flex"], -ly * scale)
    arm.move(MOTOR_IDS["shoulder_lift"], -ry * scale)
    arm.move(MOTOR_IDS["wrist_flex"], rx * scale)

    gripper_speed = scale * 0.5
    if l2 > 0.05:
        arm.move(MOTOR_IDS["gripper"], -l2 * gripper_speed)
    if r2 > 0.05:
        arm.move(MOTOR_IDS["gripper"], r2 * gripper_speed)

    arm.send_positions()


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------
def run(args):
    from openpi_client import websocket_client_policy as wcp

    port = args.port if not args.host.startswith("ws") else None
    logger.info("Connecting to policy server at %s ...", args.host)
    policy = wcp.WebsocketClientPolicy(host=args.host, port=port)
    metadata = policy.get_server_metadata()
    logger.info("Server metadata: %s", metadata)

    scene_cap = open_camera(args.scene_cam, "scene")
    wrist_cap = open_camera(args.wrist_cam, "wrist")

    arm = None
    if not args.dry_run:
        arm = ArmController(args.arm_port)
        arm.connect()

    state = np.zeros(6, dtype=np.float32)
    step = 0
    target_dt = 1.0 / CONTROL_HZ

    shutdown = False
    def on_signal(_sig, _frame):
        nonlocal shutdown
        shutdown = True
    signal.signal(signal.SIGINT, on_signal)

    logger.info("Starting control loop at %d Hz (Ctrl+C to stop) ...", CONTROL_HZ)

    try:
        while not shutdown:
            if args.max_steps and step >= args.max_steps:
                logger.info("Reached max steps (%d). Stopping.", args.max_steps)
                break

            scene_frame = grab_frame(scene_cap)
            wrist_frame = grab_frame(wrist_cap)
            if scene_frame is None or wrist_frame is None:
                logger.warning("Camera frame dropped, retrying...")
                time.sleep(0.01)
                continue

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

            if step % 10 == 0:
                logger.info(
                    "Step %d | infer %.0fms | action[0]: [%s]",
                    step, infer_ms,
                    ", ".join(f"{v:+.3f}" for v in actions[0]),
                )

            for action_idx in range(len(actions)):
                if shutdown:
                    break

                action = actions[action_idx]
                action_start = time.perf_counter()

                if args.dry_run:
                    if action_idx == 0:
                        logger.info(
                            "  [dry-run] action[%d]: [%s]",
                            action_idx,
                            ", ".join(f"{v:+.3f}" for v in action),
                        )
                else:
                    apply_joystick_to_arm(arm, action, target_dt, args.speed)

                state = np.array(action, dtype=np.float32)
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


def main():
    parser = argparse.ArgumentParser(description="Run SO-100 policy on physical arm")
    parser.add_argument("--host", type=str, default="localhost",
                        help="Policy server host (default: localhost via SSH tunnel)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--arm-port", type=str, default="/dev/tty.usbmodem5A7A0157861",
                        help="Serial port for the arm")
    parser.add_argument("--scene-cam", type=int, default=1, help="Scene camera index")
    parser.add_argument("--wrist-cam", type=int, default=2, help="Wrist camera index")
    parser.add_argument("--prompt", type=str,
                        default="Pick up the bottle and place it on the yellow outlined square.")
    parser.add_argument("--speed", type=float, default=SPEED, help="Movement speed multiplier")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps before stopping")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print predicted actions without sending to servos")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
