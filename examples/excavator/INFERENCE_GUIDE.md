# Excavator VLA Inference Pipeline — Complete Technical Guide

## Overview

This document describes exactly how the trained pi0.5 Vision-Language-Action (VLA) model controls a physical excavator in real time. The system spans four machines connected over two networks.

---

## 1. Physical Architecture

```
┌──────────────────────┐         ┌──────────────────────┐
│   Cloud GPU (RunPod) │         │   Mac Workstation    │
│                      │  WSS    │                      │
│  pi0.5 VLA Model     │◄───────►│  run_policy.py       │
│  serve_policy.py     │  proxy  │  (inference client)  │
│  Port 8000           │         │                      │
└──────────────────────┘         └──────┬───────┬───────┘
                                        │       │
                               HTTP GET │       │ UDP
                              (cameras) │       │ (joystick)
                                        │       │
                              ┌─────────▼──┐  ┌─▼────────────┐
                              │  Pi #2     │  │  Pi #1        │
                              │  Cameras   │  │  Servos       │
                              │  .93:8080  │  │  .72:9000     │
                              │            │  │               │
                              │ CSI + USB  │  │ pi_receiver.py│
                              │ cameras    │  │ PCA9685 x2    │
                              └────────────┘  └───────────────┘
```

### Machines

| Machine | Role | IP / Address | Key Software |
|---------|------|-------------|-------------|
| **RunPod GPU** | Runs the VLA model, serves predictions via WebSocket | `wss://<pod-id>-8000.proxy.runpod.net` | `serve_policy.py`, JAX, pi0.5 checkpoint |
| **Mac Workstation** | Orchestrates everything — fetches camera frames, calls model, sends servo commands | Local machine | `run_policy.py`, `openpi_client` |
| **Raspberry Pi #2** | Streams live camera frames over HTTP | `192.168.1.93:8080` | `recording_service.py` (Flask, picamera2, OpenCV) |
| **Raspberry Pi #1** | Receives UDP joystick packets, drives 4 servos | `192.168.1.72:9000` | `pi_receiver.py`, PCA9685 servo driver boards |

### Networks

- **RunPod ↔ Mac**: Internet via RunPod's WSS proxy (`wss://<pod-id>-8000.proxy.runpod.net`)
- **Mac ↔ Pi #1 / Pi #2**: Local network (`192.168.1.x`)

---

## 2. Model Details

| Parameter | Value |
|-----------|-------|
| Base model | pi0.5 (flow matching VLA) |
| Fine-tuning | LoRA (gemma_2b_lora + gemma_300m_lora) |
| Training data | `verm11/excavator_v2` (~503 episodes, ~155K frames) |
| Action dimension | 4 (`lx`, `ly`, `rx`, `ry`) — joystick axes in `[-1, +1]` |
| Action horizon | 11 (model predicts 11 future timesteps per inference) |
| State dimension | 4 (last action sent: `[lx, ly, rx, ry]`) |
| Image inputs | 2 cameras, both resized to 224×224 RGB |
| Prompt | Natural language task instruction (e.g. "Scoop white packing peanuts from large pool and dump into small pool") |
| Config name | `pi05_excavator_v2` |
| Training steps | 15,000 |
| Batch size | 32 |

### Input/Output Format

**Observation dict sent to model:**
```python
{
    "observation/state": np.float32[4],      # last action [lx, ly, rx, ry]
    "observation/image_cab": np.uint8[224, 224, 3],  # cab CSI camera (RGB)
    "observation/image_side": np.uint8[224, 224, 3], # side USB camera (RGB)
    "prompt": "Scoop white packing peanuts..."       # language instruction
}
```

**Model output:**
```python
{
    "actions": np.float32[11, 4]  # 11 timesteps × 4 joystick dims
}
```

Each action is `[lx, ly, rx, ry]` in `[-1, +1]`, identical to what the Logitech flight sticks produced during teleop data collection.

### Data Transform Pipeline

1. **RepackTransform** (in `LeRobotExcavatorDataConfig`): Maps LeRobot dataset columns to model keys:
   - `observation.images.csi_0_imx219` → `observation/image_cab`
   - `observation.images.usb_0` → `observation/image_side`
   - `observation.state` → `observation/state`
   - `action` → `actions`

2. **ExcavatorInputs** (in `excavator_policy.py`): Maps to model's internal format:
   - `observation/image_cab` → `image/base_0_rgb`
   - `observation/image_side` → `image/left_wrist_0_rgb`
   - `image/right_wrist_0_rgb` → zeros (unused, masked out)

3. **ExcavatorOutputs**: Extracts first 4 dims from padded model output.

4. **ModelTransformFactory**: Applies SigLIP image encoding (resizes to 224×224 on server side if not already), tokenization, normalization.

---

## 3. Camera System (Pi #2)

### Hardware
- **Cab camera**: IMX219 CSI module (`csi_0_imx219`) — wide-angle overhead view from the cab
- **Side camera**: Innomaker USB camera (`usb_0_innomaker_u20cam_1080p_s1__inno`) — side view of the arm

### Software
Pi #2 runs a Flask-based `recording_service.py` on port 8080. It auto-discovers cameras and serves JPEG snapshots.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /status` | GET | Returns JSON with `frame_capture` (bool), `recording` (bool), `frame_cameras` (list) |
| `POST /frames/start` | POST | Starts frame capture threads for all discovered cameras |
| `POST /frames/stop` | POST | Stops frame capture threads |
| `GET /frame/<camera_name>` | GET | Returns latest JPEG frame from specified camera |

### Frame Capture Flow
1. `POST /frames/start` → starts background threads that continuously capture frames
2. Each thread captures at native resolution (640×480), JPEG-encodes, stores latest frame
3. `GET /frame/csi_0_imx219` → returns the most recent JPEG (~200KB, ~50ms latency)
4. Client decodes JPEG, converts BGR→RGB, resizes to 224×224

### Starting Cameras from Mac
```bash
# Check status
curl http://192.168.1.93:8080/status

# Start frame capture
curl -X POST http://192.168.1.93:8080/frames/start

# Verify frames
curl -o test.jpg http://192.168.1.93:8080/frame/csi_0_imx219
```

### Client-Side Processing (`CameraSource` class in `run_policy.py`)
```
HTTP GET /frame/<name> → raw JPEG bytes
    → np.frombuffer(..., dtype=np.uint8)
    → cv2.imdecode(arr, IMREAD_COLOR) → BGR numpy array
    → cv2.cvtColor(bgr, COLOR_BGR2RGB) → RGB
    → cv2.resize(rgb, (224, 224), INTER_AREA) → 224×224 RGB uint8
```

---

## 4. Servo Control System (Pi #1)

### Hardware
- **2× PCA9685 16-channel PWM/servo driver boards** connected via I2C
  - Left board: standard I2C (SDA=2, SCL=3) — controls `lx`, `ly` axes
  - Right board: bit-banged I2C (SDA=15, SCL=14) — controls `rx`, `ry` axes
- **4 servos** controlling excavator boom, stick, bucket, and swing

### Software: `pi_receiver.py`
Runs on Pi #1, listens for UDP packets on port 9000, converts joystick values to servo angles.

### UDP Packet Format
```
{lx:.4f},{ly:.4f},{rx:.4f},{ry:.4f},{estop_bit}\n
```
Example: `0.1234,-0.5678,0.0000,0.3456,0\n`

- Values: `[-1.0, +1.0]` (joystick axes)
- `estop_bit`: `0` = normal, `1` = emergency stop (zero all servos)
- Encoding: ASCII, newline-terminated

### `pi_receiver.py` Parameters (configured in `run_policy.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--output-mode` | `servokit` | Use adafruit-servokit PCA9685 driver |
| `--udp-port` | `9000` | Listen port |
| `--watchdog-timeout` | `0.5` | Zero servos if no packet for 500ms |
| `--status-rate` | `2` | Print status every 2 seconds |
| `--invert-lx` | `1` | Invert left X axis |
| `--invert-ly` | `1` | Invert left Y axis |
| `--invert-rx` | `0` | Don't invert right X axis |
| `--invert-ry` | `1` | Invert right Y axis |
| `--channel-rx` | `0` | Right X servo on channel 0 |
| `--channel-ry` | `1` | Right Y servo on channel 1 |
| `--center-rx` | `90` | Right X center angle (degrees) |
| `--center-ry` | `85` | Right Y center angle (degrees) |
| `--right-i2c-scl` | `14` | Bit-banged I2C clock pin for right board |
| `--right-i2c-sda` | `15` | Bit-banged I2C data pin for right board |
| `--gain-lx` | `0.65` | Left X gain (scales joystick → angle range) |
| `--gain-ly` | `0.75` | Left Y gain |
| `--gain-rx` | `0.65` | Right X gain |
| `--gain-ry` | `0.85` | Right Y gain |
| `--smoothing-alpha` | `0.22` | Exponential smoothing (0=sluggish, 1=instant) |
| `--max-deg-per-sec` | `180` | Maximum servo speed |
| `--min-angle-step` | `0.3` | Minimum angle change to command |

### Conversion Chain on Pi #1
```
UDP packet [lx, ly, rx, ry] ∈ [-1, +1]
    → apply inversions (flip sign if --invert-XX is 1)
    → apply gain (multiply by --gain-XX)
    → convert to angle: center + (value × gain × max_range)
    → exponential smoothing: new = alpha × target + (1-alpha) × current
    → rate limiting: clamp change to --max-deg-per-sec × dt
    → deadband: skip if change < --min-angle-step
    → PCA9685 PWM → physical servo
```

---

## 5. `run_policy.py` — The Orchestrator

This script runs on the Mac workstation and ties everything together.

### Startup Sequence
1. Start `ServoUDPSender` background thread (50 Hz to Pi #1)
2. Connect to RunPod policy server via WebSocket
3. Open camera sources (HTTP to Pi #2)
4. Start keyboard controller (SPACE/R/Q)
5. Enter control loop

### Control Loop (10 Hz)
```
REPEAT:
  1. Check e-stop / quit
  2. HTTP GET cab frame from Pi #2 → decode → resize to 224×224 RGB
  3. HTTP GET side frame from Pi #2 → decode → resize to 224×224 RGB
  4. Build observation dict:
     {
       observation/state: [lx, ly, rx, ry]  (last action sent)
       observation/image_cab: 224×224×3 uint8
       observation/image_side: 224×224×3 uint8
       prompt: "Scoop white packing peanuts..."
     }
  5. Send observation → RunPod via WebSocket (msgpack serialized)
  6. Receive action chunk: float32[11, 4]
  7. Execute 11 actions sequentially at 10 Hz:
     - For each action [lx, ly, rx, ry]:
       - Call servo.set_axes(lx, ly, rx, ry)
       - UDP sender thread transmits at 50 Hz
       - Sleep until next 100ms tick
  8. After all 11 actions → go to step 1 (fresh camera frames)
```

### `ServoUDPSender` (Background Thread)

Runs independently at **50 Hz**, continuously sending the most recent axis values. This ensures:
- Pi #1's 500ms watchdog never triggers (gets a packet every 20ms)
- Servo commands are held steady between action updates
- Main thread can block on WebSocket inference without servo dropout

```python
# Thread loop (simplified):
while running:
    packet = f"{lx:.4f},{ly:.4f},{rx:.4f},{ry:.4f},{estop}\n"
    sock.sendto(packet.encode(), (pi_host, 9000))
    sleep(1/50)  # 20ms
```

### WebSocket Communication

Uses `openpi_client.WebsocketClientPolicy`:
- Connects to `wss://<pod-id>-8000.proxy.runpod.net`
- Sends observations as msgpack-encoded binary
- Receives action chunks as msgpack-encoded binary
- 300-second timeout for inference response

### Keyboard Controls
| Key | Action |
|-----|--------|
| SPACE | E-STOP: zero all servos, pause inference loop |
| R | Resume: clear e-stop, continue inference |
| Q | Graceful shutdown |

---

## 6. RunPod GPU Server

### Setup Commands
```bash
cd /workspace
git clone --recurse-submodules https://github.com/shashvcode/openpi-actor-labs.git openpi
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"
cd /workspace/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

### Download Model
```bash
export HF_TOKEN=<your-token>
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('verm11/excavator_v1', repo_type='model',
    local_dir='checkpoints/pi05_excavator_v2/excavator_v1')
"
```

### Serve Model
```bash
export WANDB_MODE=disabled
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_excavator_v2 \
    --policy.dir checkpoints/pi05_excavator_v2/excavator_v1
```

### JIT Warmup (Critical)
First inference triggers JAX JIT compilation (~2-5 minutes). The RunPod WSS proxy will timeout if this happens during a client call. **Always warmup locally first:**

```bash
uv run python -c "
from openpi_client import websocket_client_policy as wcp
import numpy as np
policy = wcp.WebsocketClientPolicy(host='localhost', port=8000)
obs = {
    'observation/state': np.zeros(4, dtype=np.float32),
    'observation/image_cab': np.zeros((224,224,3), dtype=np.uint8),
    'observation/image_side': np.zeros((224,224,3), dtype=np.uint8),
    'prompt': 'warmup',
}
print('Warming up JIT...')
result = policy.infer(obs)
print('WARMUP DONE!')
"
```

---

## 7. Running Inference End-to-End

### Prerequisites
- Pi #1 running `pi_receiver.py` (servo controller)
- Pi #2 running `recording_service.py` (camera server)
- RunPod serving the model (warmed up)

### Commands

**1. Start cameras (from Mac):**
```bash
curl -X POST http://192.168.1.93:8080/frames/start
```

**2. Run inference (from Mac):**
```bash
cd /Users/shashiverma/actor/openpi && .venv/bin/python examples/excavator/run_policy.py \
    --host wss://<pod-id>-8000.proxy.runpod.net \
    --cab-cam-url http://192.168.1.93:8080/frame/csi_0_imx219 \
    --side-cam-url http://192.168.1.93:8080/frame/usb_0_innomaker_u20cam_1080p_s1__inno \
    --pi-host 192.168.1.72 --pi-port 9000 \
    --no-ssh \
    --prompt "Scoop white packing peanuts from large pool and dump into small pool"
```

### Live Telemetry
The script prints a real-time single-line display:
```
  Step    42 | lx=+0.123 ly=-0.456 rx=+0.789 ry=-0.012 | chunk 3/11 | infer 1200ms
```

---

## 8. Timing Budget

| Step | Duration | Rate |
|------|----------|------|
| Camera frame fetch (HTTP) | ~50ms per camera | — |
| WebSocket inference (post-warmup) | ~1-2s | — |
| Action chunk execution | 11 × 100ms = 1.1s | 10 Hz |
| UDP servo packet | every 20ms | 50 Hz |
| **Full cycle** (fetch + infer + execute) | ~2-3s | — |

---

## 9. Training a New Model

```bash
# On RunPod:
# 1. Compute normalization stats
uv run scripts/compute_norm_stats.py --config-name pi05_excavator_v2

# 2. Train
nohup bash -c 'export WANDB_MODE=disabled && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 && \
uv run scripts/train.py pi05_excavator_v2 --exp-name=<name> --overwrite' \
> /workspace/train.log 2>&1 &

# 3. Monitor
tail -f /workspace/train.log

# 4. Upload to HuggingFace
uv run python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('verm11/<model-name>', repo_type='model', exist_ok=True)
api.upload_folder(
    folder_path='checkpoints/pi05_excavator_v2/<name>/<step>',
    repo_id='verm11/<model-name>', repo_type='model')
"
```

---

## 10. Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `HTTP 404` on camera frame | Frame capture not started | `curl -X POST http://192.168.1.93:8080/frames/start` |
| Camera frame capture stops | Idle timeout or resource conflict | Re-run `/frames/start` |
| `TimeoutError: timed out while waiting for handshake` | RunPod WSS proxy timeout during JIT | Warmup the model on RunPod first (see Section 6) |
| `ConnectionClosedError` | Proxy dropped connection during slow first inference | Same: warmup first |
| Servos not responding | `pi_receiver.py` not running on Pi #1 | SSH into Pi #1 and start it manually |
| `Config not found` | Using upstream repo, not your fork | Clone `shashvcode/openpi-actor-labs` |
| Pi #2 unreachable | IP changed (DHCP) | Check new IP, currently `192.168.1.93` |
| `uv: command not found` | Pod restarted | `export PATH="/root/.local/bin:$PATH"` or reinstall uv |
