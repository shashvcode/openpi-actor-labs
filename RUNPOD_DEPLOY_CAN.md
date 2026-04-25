# Deploy Excavator CAN-bus Pi0.5 Model on RunPod (8-axis, 3 cameras)

Step-by-step instructions to serve the trained CAN-bus excavator VLA model
(`verm11/pi05-canteleop-fullft`) on a RunPod GPU pod and run inference from
the Jetson on the excavator.

This is the **8-axis, 3-camera** model fine-tuned on direct CAN-bus teleop of
the Takeuchi TB20E. It is a sibling of `RUNPOD_DEPLOY.md` (which serves the
older 4-axis joystick model) — read this one if you are deploying the
CAN-bus stack.

---

## 1. Create a RunPod Pod

1. Go to [runpod.io](https://www.runpod.io/) -> **Pods** -> **+ GPU Pod**.
2. Pick a GPU:
   - **Minimum**: 1x A100 40 GB
   - **Recommended**: 1x A100 80 GB or 1x H100 80 GB
3. Template: **RunPod PyTorch 2.x** (or any template with CUDA).
4. Volume: **50 GB** persistent volume mounted at `/workspace`.
5. Expose ports: confirm **HTTP port 8000** is enabled (RunPod exposes this
   automatically via its proxy).
6. Click **Deploy**.

Note your **Pod ID** from the dashboard URL — it looks like `abc123def456`.
Your proxy URL will be:

```
wss://<POD-ID>-8000.proxy.runpod.net
```

For example: `wss://abc123def456-8000.proxy.runpod.net`.

---

## 2. SSH into the Pod

From the RunPod dashboard, click **Connect** and copy the SSH command (or use
the web terminal):

```bash
ssh root@<ssh-address> -p <port> -i ~/.ssh/id_ed25519
```

All commands in sections 3–6 run **on the RunPod pod**.

---

## 3. Install Dependencies

```bash
cd /workspace

git clone https://github.com/shashvcode/openpi-actor-labs.git openpi
cd openpi

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

GIT_LFS_SKIP_SMUDGE=1 uv sync
```

If the pod restarts later, you only need to re-export `PATH` and re-sync:

```bash
export PATH="/root/.local/bin:$PATH"
cd /workspace/openpi
uv sync
```

---

## 4. Download the Checkpoint from HuggingFace

The CAN-bus model lives at `verm11/pi05-canteleop-fullft`. It is a private
repo, so you need a token with read access.

```bash
cd /workspace/openpi

export HF_TOKEN=<YOUR_HF_TOKEN>

uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'verm11/pi05-canteleop-fullft',
    repo_type='model',
    local_dir='checkpoints/pi05_canteleop_fullft/run1',
)
print('Download complete.')
"
```

Verify the checkpoint exists:

```bash
ls checkpoints/pi05_canteleop_fullft/run1/
```

You should see `params/`, `assets/`, and other checkpoint files.

If `assets/` or `norm_stats.json` is missing inside the checkpoint, copy it
from the repo (assumes the asset has been committed under
`assets/pi05_canteleop_fullft/...`):

```bash
mkdir -p checkpoints/pi05_canteleop_fullft/run1/assets/verm11/CANteleop/
cp assets/pi05_canteleop_fullft/verm11/CANteleop/norm_stats.json \
   checkpoints/pi05_canteleop_fullft/run1/assets/verm11/CANteleop/ 2>/dev/null || true
```

---

## 5. Start the Policy Server

```bash
cd /workspace/openpi
export PATH="/root/.local/bin:$PATH"
export WANDB_MODE=disabled

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_canteleop_fullft \
    --policy.dir checkpoints/pi05_canteleop_fullft/run1
```

Wait until you see:

```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

The server is now listening on port 8000.

**To run in the background (survives SSH disconnect):**

```bash
nohup bash -c '
export PATH="/root/.local/bin:$PATH"
export WANDB_MODE=disabled
cd /workspace/openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_canteleop_fullft \
    --policy.dir checkpoints/pi05_canteleop_fullft/run1
' > /workspace/serve.log 2>&1 &

tail -f /workspace/serve.log
```

---

## 6. Warmup the Model (CRITICAL — do this before connecting the Jetson)

The first inference triggers JAX JIT compilation, which takes 2–5 minutes. If
you skip this step, the RunPod proxy will time out when the Jetson tries to
connect.

Run this **on the RunPod pod** while the server is running (open a second
terminal on the pod):

```bash
cd /workspace/openpi
export PATH="/root/.local/bin:$PATH"

uv run python -c "
from openpi_client import websocket_client_policy as wcp
import numpy as np

policy = wcp.WebsocketClientPolicy(host='localhost', port=8000)

obs = {
    'observation/image_cab_forward':  np.zeros((224, 224, 3), dtype=np.uint8),
    'observation/image_front_left':   np.zeros((224, 224, 3), dtype=np.uint8),
    'observation/image_front_right':  np.zeros((224, 224, 3), dtype=np.uint8),
    'observation/state':              np.zeros(8, dtype=np.float32),
    'prompt':                         'warmup',
}

print('Warming up JIT (this takes 2-5 minutes)...')
result = policy.infer(obs)
print('Actions shape:', result['actions'].shape)
print('WARMUP DONE — model is ready for remote clients.')
"
```

Do not proceed until you see **WARMUP DONE**. Action shape should be `(11, 8)`
(11-step horizon, 8 axes).

---

## 7. Run Inference from the Jetson

Everything below runs **on the Jetson** mounted on the excavator.

### 7a. Bring up the CAN bus

```bash
sudo ip link set can0 up type can bitrate 500000
ip link show can0      # confirm UP and bitrate=500000
```

### 7b. Confirm the cameras

The defaults are: `cab_forward=/dev/video2`, `front_left=/dev/video6`,
`front_right=/dev/video0`. Confirm with:

```bash
ls /dev/video*
v4l2-ctl --list-devices
```

If your indices are different, pass `--cab-cam`, `--front-left-cam`,
`--front-right-cam` to the script.

### 7c. Quick connection test (no CAN write, no cameras needed)

```bash
cd /home/Actor/openpi-actor-labs

uv run python -c "
from openpi_client import websocket_client_policy as wcp
policy = wcp.WebsocketClientPolicy(host='wss://<POD-ID>-8000.proxy.runpod.net')
print('Server metadata:', policy.get_server_metadata())
"
```

Replace `<POD-ID>` with your RunPod pod ID. If this prints metadata, the
connection works.

### 7d. Dry run (cameras + CAN read, but no actuation)

```bash
cd /home/Actor/openpi-actor-labs

sudo /home/Actor/actor-final-jetson-deployment/.venv/bin/python \
    examples/excavator/run_policy_can.py \
    --host wss://<POD-ID>-8000.proxy.runpod.net \
    --no-send \
    --prompt "Scoop packing peanuts from large pool and dump into small pool"
```

You should see action vectors scrolling, with non-zero `lx/ly/rx/ry`. The
`lt/rt/sw/bl` axes are predicted but never transmitted (the bridge does not
support them). Press `Q` to exit.

### 7e. Full run (cameras + CAN read + CAN write through Arduinos)

Make sure both Arduino bridges are physically connected and the
`/dev/serial/by-id/...` symlinks are present:

```bash
ls /dev/serial/by-id/
```

Then:

```bash
cd /home/Actor/openpi-actor-labs

sudo /home/Actor/actor-final-jetson-deployment/.venv/bin/python \
    examples/excavator/run_policy_can.py \
    --host wss://<POD-ID>-8000.proxy.runpod.net \
    --prompt "Scoop packing peanuts from large pool and dump into small pool"
```

### Runtime Controls

| Key   | Action                                             |
|-------|----------------------------------------------------|
| SPACE | E-STOP — zero all joystick axes immediately        |
| R     | Resume inference after E-STOP                       |
| Q     | Graceful shutdown (sends neutral, releases cameras) |

---

## 8. Tuning Per-Axis (CLI flags)

The default behavior is **pure passthrough**: model output goes directly to
the bridge with no gain, no invert, no deadzone, no slew limit. Adjust per
axis with the following flags (all applied in this order: invert -> gain ->
deadzone -> clip -> slew):

```
--gain-{lx|ly|rx|ry} <float>       (default 1.0)
--invert-{lx|ly|rx|ry} <0|1>       (default 0)
--deadzone-{lx|ly|rx|ry} <float>   (default 0.0)
--slew-{lx|ly|rx|ry} <float>       (default -1.0 = unbounded)
```

A safe slew value derived from the live CAN capture is `0.5` per step at
50 Hz, which corresponds to ~25 axis-units/s — matches the maximum human
slew rate measured during teleop.

Example: clamp the right stick to a safer envelope while leaving the left
stick untouched:

```bash
sudo .../python examples/excavator/run_policy_can.py \
    --host wss://<POD-ID>-8000.proxy.runpod.net \
    --gain-rx 0.7 --gain-ry 0.7 \
    --slew-rx 0.5 --slew-ry 0.5 \
    --deadzone-rx 0.05 --deadzone-ry 0.05
```

---

## 9. Asymmetry — 8 axes in, 4 axes out

The model is trained to predict 8 axes:

```
[left_stick_x, left_stick_y, right_stick_x, right_stick_y,
 left_track,   right_track,  swing,         blade]
```

The current 2-Arduino bridge (`takeuchi-canbus`) can only command the four
joystick axes (sticks 1 and 2). The remaining four axes (`left_track`,
`right_track`, `swing`, `blade`) are READABLE from the CAN bus and are
included in the state vector sent back to the model — but the model's
predictions for those axes are NOT transmitted. A loud WARNING is printed
once at startup naming each dropped axis.

If/when the bridge is extended (extra Arduinos for tracks/blade), expand
`ActionInterpolator.demux_to_takeuchi` and the `TakeuchiCommander` payload
in `examples/excavator/run_policy_can.py`.

---

## 10. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `TimeoutError: timed out while waiting for handshake` | JIT not warmed up | Run the warmup script on RunPod first (Section 6) |
| `ConnectionClosedError` during first inference | Proxy timeout during JIT | Same — warmup first |
| `Failed to open socketcan channel 'can0'` | Interface down | `sudo ip link set can0 up type can bitrate 500000` |
| `Failed to open ... camera` | Wrong /dev/video index | `v4l2-ctl --list-devices`, pass correct `--cab-cam` etc. |
| `No frames` warning, state always zero | CAN cable unplugged or cab in standby | Confirm wiring; check `cansniffer can0` shows traffic |
| `tb20e_decoder.py not found` | Decoder repo missing on Jetson | `git clone <Thor-CAN-recording>` to `/home/Actor/Thor-CAN-recording` or pass `--decoder-path <dir>` |
| `Config 'pi05_canteleop_fullft' not found` | Using upstream openpi, not your fork | Clone `shashvcode/openpi-actor-labs` |
| `serial port not found` from TakeuchiClient | Arduino not plugged in or symlink missing | `ls /dev/serial/by-id/` should show two `Arduino_*` entries |
| Very slow inference (> 5 s) | Pod under load or small GPU | Check `nvidia-smi` on the pod — needs at least A100 40 GB |
| `No space left on device` on the pod | Root filesystem full | Always work in `/workspace`; set `HF_HOME=/workspace/.hf_home` |

---

## 11. Quick Reference — Copy-Paste Commands

**On RunPod (setup + serve):**

```bash
cd /workspace/openpi
export PATH="/root/.local/bin:$PATH"
export WANDB_MODE=disabled
export HF_TOKEN=<YOUR_HF_TOKEN>

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_canteleop_fullft \
    --policy.dir checkpoints/pi05_canteleop_fullft/run1
```

**On RunPod (warmup — second terminal):**

```bash
cd /workspace/openpi && export PATH="/root/.local/bin:$PATH" && uv run python -c "
from openpi_client import websocket_client_policy as wcp; import numpy as np
p = wcp.WebsocketClientPolicy(host='localhost', port=8000)
r = p.infer({
  'observation/image_cab_forward':  np.zeros((224,224,3), dtype=np.uint8),
  'observation/image_front_left':   np.zeros((224,224,3), dtype=np.uint8),
  'observation/image_front_right':  np.zeros((224,224,3), dtype=np.uint8),
  'observation/state':              np.zeros(8, dtype=np.float32),
  'prompt': 'warmup',
})
print('READY -', r['actions'].shape)
"
```

**On the Jetson (run):**

```bash
sudo ip link set can0 up type can bitrate 500000

cd /home/Actor/openpi-actor-labs && sudo /home/Actor/actor-final-jetson-deployment/.venv/bin/python \
    examples/excavator/run_policy_can.py \
    --host wss://<POD-ID>-8000.proxy.runpod.net \
    --prompt "Scoop white packing peanuts from large pool and dump into small pool"
```

Replace `<POD-ID>` with your RunPod pod ID (e.g. `abc123def456`).
