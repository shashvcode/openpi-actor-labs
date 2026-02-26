# SO-100 Robot Arm Control System — Complete Technical Reference

This document describes the full control pipeline: how a PS5 DualSense controller interfaces with the SO-100 6-DOF robot arm via serial communication, how cameras are connected, and how joystick positions become servo movements.

---

## Table of Contents

1. [Hardware Overview](#1-hardware-overview)
2. [Dependencies](#2-dependencies)
3. [Serial Communication — Feetech STS3215 Servos](#3-serial-communication)
4. [Servo Control Table Registers](#4-servo-control-table-registers)
5. [Motor Configuration](#5-motor-configuration)
6. [Connecting to the Arm](#6-connecting-to-the-arm)
7. [PS5 DualSense Controller Interface](#7-ps5-dualsense-controller-interface)
8. [Excavator-Style Joystick Mapping](#8-excavator-style-joystick-mapping)
9. [The Control Loop — From Joystick to Servo](#9-the-control-loop)
10. [Camera Setup](#10-camera-setup)
11. [Recording Episodes](#11-recording-episodes)
12. [Data Format](#12-data-format)
13. [Run Commands](#13-run-commands)

---

## 1. Hardware Overview

- **Robot Arm**: SO-100, 6-DOF tabletop manipulator
- **Servos**: 6x Feetech STS3215 smart serial servos (daisy-chained)
- **Communication**: USB-to-TTL serial adapter at 1,000,000 baud
- **Controller**: PS5 DualSense (Bluetooth or USB)
- **Cameras**: 2x USB webcams (scene overview + wrist close-up)
- **Computer**: macOS (Apple Silicon) — uses PyObjC for controller input

## 2. Dependencies

```toml
# pyproject.toml
dependencies = [
    "feetech-servo-sdk>=1.0.0",   # Serial protocol for STS3215 servos
    "opencv-python",               # Camera capture
    "pyobjc-framework-cocoa>=12.1",       # macOS app framework
    "pyobjc-framework-gamecontroller>=12.1", # PS5 DualSense via GameController.framework
    "lerobot>=0.4.3",             # LeRobot dataset format
    "numpy",
]
```

Install everything with:
```bash
uv sync
```

## 3. Serial Communication

The arm uses the Feetech SCS/STS serial protocol (half-duplex UART). All 6 servos are daisy-chained on one serial bus.

```python
from scservo_sdk import (
    PacketHandler,     # Reads/writes to individual servo registers
    PortHandler,       # Manages the USB serial port
    GroupSyncWrite,    # Writes to multiple servos in a single packet
    COMM_SUCCESS,      # Return code for successful communication
    SCS_MAKEWORD,      # Combines 2 bytes into a 16-bit value
    SCS_LOBYTE,        # Low byte of a 16-bit value
    SCS_HIBYTE,        # High byte of a 16-bit value
)
```

### Opening the port

```python
port_handler = PortHandler("/dev/tty.usbmodem5A7A0157861")
packet_handler = PacketHandler(0)  # Protocol version 0 (SCS)

port_handler.openPort()
port_handler.setBaudRate(1_000_000)  # 1 Mbaud
```

### Reading a register (e.g., current position)

```python
ADDR_PRESENT_POSITION = 56  # 2-byte register

data, result, error = packet_handler.readTxRx(
    port_handler, motor_id, ADDR_PRESENT_POSITION, 2  # 2 bytes
)
if result == COMM_SUCCESS:
    position = SCS_MAKEWORD(data[0], data[1])  # 0-4095
```

### Writing a register (e.g., enable torque)

```python
# 1-byte write
packet_handler.write1ByteTxRx(port_handler, motor_id, ADDR_TORQUE_ENABLE, 1)

# 2-byte write
packet_handler.write2ByteTxRx(port_handler, motor_id, ADDR_GOAL_POSITION, 2048)
```

### Sync write (all 6 motors at once)

This is the core of real-time control — writes goal positions to all motors in a single serial packet:

```python
ADDR_GOAL_POSITION = 42

sync_writer = GroupSyncWrite(
    port_handler, packet_handler,
    ADDR_GOAL_POSITION, 2,  # Start address, data length (2 bytes)
)

# Add each motor's target position
for motor_id in [1, 2, 3, 4, 5, 6]:
    pos = target_positions[motor_id]  # integer 0-4095
    sync_writer.addParam(motor_id, [SCS_LOBYTE(pos), SCS_HIBYTE(pos)])

# Send all at once
sync_writer.txPacket()
sync_writer.clearParam()
```

## 4. Servo Control Table Registers

| Address | Name | Size | Description |
|---------|------|------|-------------|
| 16 | `MAX_TORQUE_LIMIT` | 2 bytes | Maximum torque (0-1000) |
| 21 | `P_COEFFICIENT` | 1 byte | PID proportional gain |
| 22 | `D_COEFFICIENT` | 1 byte | PID derivative gain |
| 23 | `I_COEFFICIENT` | 1 byte | PID integral gain |
| 28 | `PROTECTION_CURRENT` | 2 bytes | Over-current protection threshold |
| 33 | `OPERATING_MODE` | 1 byte | 0 = position mode |
| 36 | `OVERLOAD_TORQUE` | 1 byte | Overload protection percentage |
| 40 | `TORQUE_ENABLE` | 1 byte | 0 = free, 1 = holding |
| 42 | `GOAL_POSITION` | 2 bytes | Target position (0-4095) |
| 55 | `LOCK` | 1 byte | 0 = unlock EEPROM, 1 = lock |
| 56 | `PRESENT_POSITION` | 2 bytes | Current position (0-4095, read-only) |

## 5. Motor Configuration

### Joint definitions

```python
MOTOR_IDS = {
    "shoulder_pan":  1,  # Rotates entire arm left/right
    "shoulder_lift": 2,  # Raises/lowers the main boom
    "elbow_flex":    3,  # Bends the forearm
    "wrist_flex":    4,  # Tilts gripper up/down
    "wrist_roll":    5,  # Rotates gripper (L1/R1 buttons)
    "gripper":       6,  # Opens/closes fingers
}
```

### Position limits (servo ticks, 0-4095)

```python
MOTOR_LIMITS = {
    1: (200, 3896),   # shoulder_pan — near full range
    2: (400, 3600),   # shoulder_lift
    3: (400, 3600),   # elbow_flex
    4: (400, 3600),   # wrist_flex
    5: (0, 4095),     # wrist_roll — full rotation
    6: (1500, 3100),  # gripper — 1500=open, 3100=closed
}
```

Center position for all joints: `SERVO_MID = 2048`

### PID and safety configuration

Applied at startup, before enabling torque:

```python
def configure_servos():
    torque_limits = {1: 700, 2: 700, 3: 700, 4: 700, 5: 700, 6: 400}
    current_limits = {1: 350, 2: 350, 3: 350, 4: 350, 5: 350, 6: 200}

    for motor_id in ALL_IDS:
        # Unlock EEPROM to write settings
        write1Byte(motor_id, ADDR_LOCK, 0)

        # Position control mode
        write1Byte(motor_id, ADDR_OPERATING_MODE, 0)

        # PID tuning — soft response to prevent jerky movements
        write1Byte(motor_id, ADDR_P_COEFFICIENT, 10)   # Low P = soft
        write1Byte(motor_id, ADDR_I_COEFFICIENT, 0)    # No integral
        write1Byte(motor_id, ADDR_D_COEFFICIENT, 20)   # Moderate damping

        # Safety limits
        write2Byte(motor_id, ADDR_MAX_TORQUE_LIMIT, torque_limits[motor_id])
        write2Byte(motor_id, ADDR_PROTECTION_CURRENT, current_limits[motor_id])
        write1Byte(motor_id, ADDR_OVERLOAD_TORQUE, 30)

        # Re-lock EEPROM
        write1Byte(motor_id, ADDR_LOCK, 1)
```

The gripper (motor 6) has lower torque/current limits (400/200) to avoid crushing objects.

## 6. Connecting to the Arm

### Full startup sequence

```python
class ArmController:
    def __init__(self, port: str):
        self.port_handler = PortHandler(port)
        self.packet_handler = PacketHandler(0)
        # Start with all joints at center
        self.positions = {mid: float(2048) for mid in ALL_IDS}
        self._last_sent_positions = {}
        self.torque_enabled = False

    def connect(self):
        # 1. Open serial port at 1 Mbaud
        self.port_handler.openPort()
        self.port_handler.setBaudRate(1_000_000)

        # 2. Create sync writer for efficient position commands
        self.sync_writer = GroupSyncWrite(
            self.port_handler, self.packet_handler,
            ADDR_GOAL_POSITION, 2,
        )

        # 3. Ping all 6 motors to verify connectivity
        for name, mid in MOTOR_IDS.items():
            model, comm, err = self.packet_handler.ping(self.port_handler, mid)
            if comm != COMM_SUCCESS:
                raise RuntimeError(f"Motor {mid} ({name}) not responding!")

        # 4. Read current positions (so we don't jump on startup)
        self._read_positions()

        # 5. Configure PID, torque limits, protection
        self._configure_servos()

        # 6. Enable torque — arm now holds position
        self._enable_torque()
```

Key detail: We read current positions *before* enabling torque, so the first goal position matches where the arm physically is. This prevents the startup jerk that would happen if we sent position 2048 to a servo currently at position 1500.

### Finding the USB port

The arm's USB serial adapter shows up as `/dev/tty.usbmodemXXXX` on macOS. To find it:

```bash
ls /dev/tty.usbmodem*
```

## 7. PS5 DualSense Controller Interface

We use Apple's `GameController.framework` via PyObjC. This handles Bluetooth pairing, button mapping, and analog stick values natively.

### macOS app setup (required for GameController to work)

```python
from Foundation import NSDate, NSNotificationCenter, NSRunLoop
from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
from GameController import GCController

# Must initialize NSApplication for GameController events to fire
app = NSApplication.sharedApplication()
app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
app.activateIgnoringOtherApps_(True)

# Allow background event monitoring
GCController.setShouldMonitorBackgroundEvents_(True)
```

### Detecting controller connection

```python
# Check if already connected
controllers = GCController.controllers()
if controllers and len(controllers) > 0:
    setup_controller(controllers[0])

# Listen for new connections
NSNotificationCenter.defaultCenter().addObserverForName_object_queue_usingBlock_(
    "GCControllerDidConnectNotification", None, None,
    lambda note: setup_controller(note.object()),
)
```

### Reading joystick values

The `extendedGamepad()` profile provides all DualSense inputs. We register a callback that fires on ANY input change:

```python
def setup_controller(controller):
    gamepad = controller.extendedGamepad()
    gamepad.setValueChangedHandler_(on_gamepad_change)

def on_gamepad_change(gamepad, element):
    # Analog sticks: -1.0 to +1.0
    lx = gamepad.leftThumbstick().xAxis().value()   # Left stick horizontal
    ly = gamepad.leftThumbstick().yAxis().value()   # Left stick vertical
    rx = gamepad.rightThumbstick().xAxis().value()  # Right stick horizontal
    ry = gamepad.rightThumbstick().yAxis().value()  # Right stick vertical

    # Triggers: 0.0 to 1.0 (analog)
    lt = gamepad.leftTrigger().value()   # L2
    rt = gamepad.rightTrigger().value()  # R2

    # Shoulder buttons: boolean
    l1 = gamepad.leftShoulder().isPressed()   # L1
    r1 = gamepad.rightShoulder().isPressed()  # R1

    # Face buttons: boolean
    cross    = gamepad.buttonA().isPressed()  # Cross (×)
    circle   = gamepad.buttonB().isPressed()  # Circle (○)
    square   = gamepad.buttonX().isPressed()  # Square (□)
    triangle = gamepad.buttonY().isPressed()  # Triangle (△)
```

### Deadzone filtering

Raw stick values jitter around 0 even when untouched. We apply a deadzone:

```python
DEADZONE = 0.12

def apply_deadzone(val: float) -> float:
    if abs(val) < DEADZONE:
        return 0.0
    sign = 1.0 if val > 0 else -1.0
    # Rescale so output is 0.0 at the edge of deadzone and 1.0 at full deflection
    return sign * (abs(val) - DEADZONE) / (1.0 - DEADZONE)
```

### Event loop

GameController events require the macOS run loop. We interleave it with our control loop:

```python
target_dt = 1.0 / 30  # 30 Hz arm control

while True:
    # Process macOS events (controller input, notifications)
    NSRunLoop.currentRunLoop().runUntilDate_(
        NSDate.dateWithTimeIntervalSinceNow_(target_dt * 0.5),
    )

    # Run arm control at 30 Hz
    now = time.perf_counter()
    dt = now - last_time
    last_time = now
    update_arm(dt)
```

## 8. Excavator-Style Joystick Mapping

The mapping follows real excavator conventions:

| Input | Axis/Range | Joint | Effect |
|-------|-----------|-------|--------|
| Left Stick X | `-1.0` to `+1.0` | Shoulder Pan (1) | `+1` = swing right, `-1` = swing left |
| Left Stick Y | `-1.0` to `+1.0` | Elbow Flex (3) | `+1` = retract forearm up, `-1` = extend down |
| Right Stick X | `-1.0` to `+1.0` | Wrist Flex (4) | Tilt gripper angle |
| Right Stick Y | `-1.0` to `+1.0` | Shoulder Lift (2) | `+1` = boom DOWN, `-1` = boom UP |
| L1 button | boolean | Wrist Roll (5) | Rotate gripper counter-clockwise |
| R1 button | boolean | Wrist Roll (5) | Rotate gripper clockwise |
| L2 trigger | `0.0` to `1.0` | Gripper (6) | Close (decrease position) |
| R2 trigger | `0.0` to `1.0` | Gripper (6) | Open (increase position) |

Note the inversions:
- `ly` is **negated** before applying to elbow (stick up = retract = positive direction, but servo goes negative)
- `ry` is **negated** before applying to shoulder lift (stick down = boom down, but we want intuitive "push down to go down")

### The state vector

The joystick state recorded for training is 6-dimensional:

```python
state = [lx, ly, rx, ry, l2_trigger, r2_trigger]  # float32[6]
```

L1/R1 (wrist roll) are NOT included in the state vector because they're boolean buttons, not analog axes.

## 9. The Control Loop

This is the core of the system — translating joystick positions into servo movements.

### Speed and timing

```python
SPEED = 15.0       # Movement speed multiplier
MAX_DT = 0.05      # Cap dt to prevent jumps if a frame is slow
ARM_HZ = 30        # Arm control loop runs at 30 Hz
RECORD_FPS = 10    # Recording samples at 10 Hz
```

### Converting joystick to servo deltas

Each control loop iteration (30 Hz):

```python
def update(self, dt: float):
    dt = min(dt, MAX_DT)        # Cap at 50ms to prevent huge jumps
    scale = self.speed * dt * 60  # ~45 servo ticks per step at full deflection

    # Left stick → shoulder pan and elbow
    arm.move(MOTOR_IDS["shoulder_pan"], lx * scale)
    arm.move(MOTOR_IDS["elbow_flex"], -ly * scale)    # Negated

    # Right stick → shoulder lift and wrist flex
    arm.move(MOTOR_IDS["shoulder_lift"], -ry * scale)  # Negated
    arm.move(MOTOR_IDS["wrist_flex"], rx * scale)

    # L1/R1 → wrist roll (binary, reduced speed)
    roll_speed = scale * 0.7
    if l1:
        arm.move(MOTOR_IDS["wrist_roll"], -roll_speed)
    if r1:
        arm.move(MOTOR_IDS["wrist_roll"], roll_speed)

    # L2/R2 → gripper (analog, reduced speed)
    gripper_speed = scale * 0.5
    if lt > 0.05:
        arm.move(MOTOR_IDS["gripper"], -lt * gripper_speed)  # Close
    if rt > 0.05:
        arm.move(MOTOR_IDS["gripper"], rt * gripper_speed)   # Open

    # Send all positions to servos in one packet
    arm.send_positions()
```

### The `move()` method — position update with clamping

```python
def move(self, motor_id: int, delta: float):
    if not self.torque_enabled:
        return
    lo, hi = MOTOR_LIMITS[motor_id]
    new_pos = self.positions[motor_id] + delta
    self.positions[motor_id] = max(lo, min(hi, new_pos))
```

This is a **velocity command** system: each joystick reading adds a delta to the current position. The position is clamped to the motor's physical limits. The arm holds its current position when the joystick is centered (delta = 0).

### The `send_positions()` method — efficient sync write

```python
def send_positions(self):
    if not self.torque_enabled:
        return
    int_pos = {mid: int(p) for mid, p in self.positions.items()}

    # Skip if nothing changed (avoids unnecessary serial traffic)
    if int_pos == self._last_sent_positions:
        return
    self._last_sent_positions = int_pos

    self.sync_writer.clearParam()
    for mid in ALL_IDS:
        pos = int_pos[mid]
        self.sync_writer.addParam(mid, [SCS_LOBYTE(pos), SCS_HIBYTE(pos)])
    self.sync_writer.txPacket()
    time.sleep(0.002)  # Brief pause for servo processing
```

### Numerical example

With default settings at 30 Hz:
- `dt ≈ 0.033s`, `speed = 15.0`
- `scale = 15.0 × 0.033 × 60 ≈ 30`
- Full stick deflection (`lx = 1.0`): moves shoulder_pan by ~30 ticks per iteration
- At 30 Hz: ~900 ticks/second ≈ 22% of full range per second
- Gripper speed is halved: ~15 ticks per iteration at full trigger

### Position resync

Every 60 iterations (~2 seconds), we re-read actual servo positions to correct any drift between commanded and actual positions:

```python
self._resync_counter += 1
if self._resync_counter >= 60:
    self._resync_counter = 0
    arm.resync_positions()  # Reads ADDR_PRESENT_POSITION for all 6 motors
```

## 10. Camera Setup

### Opening cameras with OpenCV

```python
import cv2

IMG_W, IMG_H = 640, 480  # Stored frame size

def open_camera(index: int, label: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"Failed to open {label} (index {index})")

    # Request high resolution from the camera (downscale later)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {label}: index {index}, native {actual_w}x{actual_h}, stored as {IMG_W}x{IMG_H}")
    return cap
```

### Capturing a frame

```python
def grab_frame(cap: cv2.VideoCapture) -> np.ndarray | None:
    ret, bgr = cap.read()
    if not ret:
        return None
    # OpenCV captures BGR, convert to RGB for storage
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # Resize to standard dimensions
    if rgb.shape[0] != IMG_H or rgb.shape[1] != IMG_W:
        rgb = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return rgb  # shape: (480, 640, 3), dtype: uint8
```

### Finding camera indices

Camera indices are system-dependent. On macOS, plug cameras in and try:

```bash
# List available cameras (in Python)
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i}: {int(cap.get(3))}x{int(cap.get(4))}')
        cap.release()
"
```

Typical setup: `--scene-cam 1 --wrist-cam 2` (index 0 is often the built-in MacBook camera).

## 11. Recording Episodes

Recording runs at 10 Hz (every 100ms), interleaved with the 30 Hz arm control loop.

### Record tick (10 Hz)

```python
def record_tick(self):
    if not self.recording:
        return

    # Capture images from all cameras
    images = {}
    for name, cap in self.cams.items():
        rgb = grab_frame(cap)
        images[name] = rgb  # numpy array (480, 640, 3)

    # Snapshot joystick state (thread-safe)
    with self._lock:
        state = np.array([lx, ly, rx, ry, lt, rt], dtype=np.float32)

    self.episode_buffer.append({"state": state, "images": images})
```

### Saving an episode

When Circle is pressed to stop recording, the buffer is converted to LeRobot format:

```python
def save_episode(self):
    n = len(self.episode_buffer)

    for i in range(n):
        state = self.episode_buffer[i]["state"]
        # Action = the NEXT frame's state (what the joystick will be)
        action = self.episode_buffer[i + 1]["state"] if i < n - 1 else state

        frame = {
            "task": "pick up the bottle and place it on the yellow outlined square",
            "observation.state": state,    # float32[6]
            "action": action.copy(),       # float32[6]
        }
        for cam_name, img in self.episode_buffer[i]["images"].items():
            frame[f"observation.images.{cam_name}"] = img  # uint8 (480,640,3)

        self.dataset.add_frame(frame)

    self.dataset.save_episode()
```

The action at time `t` is defined as the joystick state at time `t+1`. This means the action represents "what the operator will do next given this observation."

## 12. Data Format

### LeRobot v3.0 Parquet Structure

```
dataset_root/
├── data/
│   └── chunk-000/
│       └── file-000.parquet    # All frames, images embedded as binary
└── meta/
    ├── info.json               # Dataset metadata
    ├── episodes.jsonl          # Per-episode info (length, task)
    └── tasks.jsonl             # Task descriptions
```

### Parquet columns

| Column | Type | Shape | Description |
|--------|------|-------|-------------|
| `observation.state` | `list<float32>` | `[6]` | Joystick: `[lx, ly, rx, ry, l2, r2]` |
| `action` | `list<float32>` | `[6]` | Next-step joystick state |
| `observation.images.scene` | `struct{bytes, path}` | `(480,640,3)` | Scene camera JPEG |
| `observation.images.wrist` | `struct{bytes, path}` | `(480,640,3)` | Wrist camera JPEG |
| `timestamp` | `float32` | `[1]` | Time within episode |
| `frame_index` | `int64` | `[1]` | Frame number within episode |
| `episode_index` | `int64` | `[1]` | Episode number |
| `index` | `int64` | `[1]` | Global frame index |
| `task_index` | `int64` | `[1]` | Task ID (always 0) |

### info.json

```json
{
  "codebase_version": "v3.0",
  "robot_type": "so100",
  "total_episodes": 134,
  "total_frames": 25972,
  "total_tasks": 1,
  "fps": 10,
  "features": {
    "observation.state": {
      "dtype": "float32",
      "shape": [6],
      "names": [["left_x", "left_y", "right_x", "right_y", "l2_trigger", "r2_trigger"]]
    },
    "action": {
      "dtype": "float32",
      "shape": [6],
      "names": [["left_x", "left_y", "right_x", "right_y", "l2_trigger", "r2_trigger"]]
    },
    "observation.images.scene": {
      "dtype": "image",
      "shape": [3, 480, 640],
      "names": ["channels", "height", "width"]
    },
    "observation.images.wrist": {
      "dtype": "image",
      "shape": [3, 480, 640],
      "names": ["channels", "height", "width"]
    }
  }
}
```

## 13. Run Commands

### Teleoperation + recording

```bash
uv run python record_episodes.py \
  --port /dev/tty.usbmodem5A7A0157861 \
  --cam scene:1 --cam wrist:2 \
  --task "pick up the bottle and place it on the yellow outlined square"
```

### With joint position recording

```bash
uv run python record_episodes.py \
  --port /dev/tty.usbmodem5A7A0157861 \
  --cam scene:1 --cam wrist:2 \
  --joints
```

### Replay a recorded trajectory

```bash
uv run python replay_trajectory.py \
  --port /dev/tty.usbmodem5A7A0157861
```

### Execute VLM-predicted actions

```bash
uv run python vlm/run_gemini_actions.py \
  --actions vlm/gemini_prompt/predicted.json \
  --port /dev/tty.usbmodem5A7A0157861
```

### Closed-loop Gemini control

```bash
uv run python vlm/gemini_closedloop.py \
  --port /dev/tty.usbmodem5A7A0157861 \
  --scene-cam 1 --wrist-cam 2
```

---

## Appendix: Complete Control Flow Diagram

```
PS5 DualSense (Bluetooth)
        │
        ▼
GameController.framework (macOS)
        │
        ▼
on_gamepad_change() callback
  ├── lx, ly = left stick (with deadzone)
  ├── rx, ry = right stick (with deadzone)
  ├── lt, rt = L2/R2 triggers (0-1)
  └── l1, r1 = L1/R1 buttons (bool)
        │
        ▼
update() @ 30 Hz
  ├── scale = speed × dt × 60
  ├── shoulder_pan  += lx × scale
  ├── elbow_flex    += -ly × scale
  ├── shoulder_lift += -ry × scale
  ├── wrist_flex    += rx × scale
  ├── wrist_roll    += ±roll_speed (if L1/R1)
  ├── gripper       += ±lt/rt × gripper_speed
  └── clamp all to MOTOR_LIMITS
        │
        ▼
send_positions() — GroupSyncWrite
  └── Single serial packet → all 6 servos
        │
        ▼
STS3215 servos execute PID position control
  └── P=10, I=0, D=20 (soft, damped response)
```
