# Deploy Excavator Pi0.5 Model on RunPod

Step-by-step instructions to serve the trained excavator VLA model on any RunPod GPU pod and run inference from your Mac.

---

## 1. Create a RunPod Pod

1. Go to [runpod.io](https://www.runpod.io/) → **Pods** → **+ GPU Pod**
2. Pick a GPU:
   - **Minimum**: 1x A100 40GB
   - **Recommended**: 1x A100 80GB or 1x H100 80GB
3. Template: **RunPod Pytorch 2.x** (or any template with CUDA)
4. Volume: **50 GB** persistent volume at `/workspace`
5. Expose ports: make sure **HTTP port 8000** is enabled (RunPod exposes this automatically via proxy)
6. Click **Deploy**

Once the pod is running, note your **Pod ID** — you'll see it in the URL or pod list. It looks like `abc123def456`.

Your proxy URL will be:

```
wss://<POD-ID>-8000.proxy.runpod.net
```

For example: `wss://abc123def456-8000.proxy.runpod.net`

---

## 2. SSH into the Pod

From the RunPod dashboard, click **Connect** → copy the SSH command, or use the web terminal.

```bash
ssh root@<ssh-address> -p <port> -i ~/.ssh/id_ed25519
```

All remaining commands in sections 3-6 run **on the RunPod pod**.

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

If the pod restarts later, you only need to re-export PATH and re-sync:

```bash
export PATH="/root/.local/bin:$PATH"
cd /workspace/openpi
uv sync
```

---

## 4. Download the 10K Checkpoint from HuggingFace

```bash
cd /workspace/openpi

export HF_TOKEN=<YOUR_HF_TOKEN>

uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'verm11/excavator_v1',
    repo_type='model',
    local_dir='checkpoints/pi05_excavator_v2/excavator_v1',
)
print('Download complete.')
"
```

Verify the checkpoint exists:

```bash
ls checkpoints/pi05_excavator_v2/excavator_v1/
```

You should see `params/`, `assets/`, and other checkpoint files.

If `assets/` or `norm_stats.json` is missing inside the checkpoint, copy it from the repo:

```bash
mkdir -p checkpoints/pi05_excavator_v2/excavator_v1/assets/verm11/excavator-teleop/
cp assets/pi05_excavator_lora/verm11/excavator-teleop/norm_stats.json \
   checkpoints/pi05_excavator_v2/excavator_v1/assets/verm11/excavator-teleop/
```

---

## 5. Start the Policy Server

```bash
cd /workspace/openpi
export PATH="/root/.local/bin:$PATH"
export WANDB_MODE=disabled

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_excavator_v2 \
    --policy.dir checkpoints/pi05_excavator_v2/excavator_v1
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
    --policy.config pi05_excavator_v2 \
    --policy.dir checkpoints/pi05_excavator_v2/excavator_v1
' > /workspace/serve.log 2>&1 &

tail -f /workspace/serve.log
```

---

## 6. Warmup the Model (Critical — Do This Before Connecting the Client)

The first inference triggers JAX JIT compilation, which takes 2-5 minutes. If you skip this step, the RunPod proxy will timeout when the Mac client tries to connect.

Run this **on the RunPod pod** while the server is running (open a second terminal):

```bash
cd /workspace/openpi
export PATH="/root/.local/bin:$PATH"

uv run python -c "
from openpi_client import websocket_client_policy as wcp
import numpy as np

policy = wcp.WebsocketClientPolicy(host='localhost', port=8000)

obs = {
    'observation/state': np.zeros(4, dtype=np.float32),
    'observation/image_cab': np.zeros((224, 224, 3), dtype=np.uint8),
    'observation/image_side': np.zeros((224, 224, 3), dtype=np.uint8),
    'prompt': 'warmup',
}

print('Warming up JIT (this takes 2-5 minutes)...')
result = policy.infer(obs)
print('Actions shape:', result['actions'].shape)
print('WARMUP DONE — model is ready for remote clients.')
"
```

Do not proceed until you see **WARMUP DONE**.

---

## 7. Run Inference from Mac

Everything below runs **on your Mac workstation**.

### 7a. Find Your Proxy URL

Your RunPod proxy URL is:

```
wss://<POD-ID>-8000.proxy.runpod.net
```

Find `<POD-ID>` from the RunPod dashboard — it's the alphanumeric string in your pod's URL or listed under the pod name. Example:

```
wss://abc123def456-8000.proxy.runpod.net
```

### 7b. Quick Connection Test (No Hardware Needed)

```bash
cd /Users/shashiverma/actor/openpi

.venv/bin/python -c "
from openpi_client import websocket_client_policy as wcp
policy = wcp.WebsocketClientPolicy(host='wss://<POD-ID>-8000.proxy.runpod.net')
meta = policy.get_server_metadata()
print('Connected! Server metadata:', meta)
"
```

If this prints metadata, the connection works.

### 7c. Dry Run (Cameras, No Servos)

Make sure the cameras on Pi #2 are streaming:

```bash
curl -X POST http://192.168.1.93:8080/frames/start
curl -s http://192.168.1.93:8080/status | python3 -m json.tool
```

Then dry-run the policy (prints predicted actions, does not send to servos):

```bash
cd /Users/shashiverma/actor/openpi

.venv/bin/python examples/excavator/run_policy.py \
    --host wss://<POD-ID>-8000.proxy.runpod.net \
    --cab-cam-url http://192.168.1.93:8080/frame/csi_0_imx219 \
    --side-cam-url http://192.168.1.93:8080/frame/usb_0_innomaker_u20cam_1080p_s1__inno \
    --no-ssh \
    --dry-run \
    --prompt "Scoop white packing peanuts from large pool and dump into small pool"
```

You should see predicted joystick values scrolling by. `Ctrl+C` to stop.

### 7d. Full Run (Cameras + Servos)

With Pi #1 already running `pi_receiver.py`:

```bash
cd /Users/shashiverma/actor/openpi

.venv/bin/python examples/excavator/run_policy.py \
    --host wss://<POD-ID>-8000.proxy.runpod.net \
    --cab-cam-url http://192.168.1.93:8080/frame/csi_0_imx219 \
    --side-cam-url http://192.168.1.93:8080/frame/usb_0_innomaker_u20cam_1080p_s1__inno \
    --pi-host 192.168.1.72 --pi-port 9000 \
    --no-ssh \
    --prompt "Scoop white packing peanuts from large pool and dump into small pool"
```

### Runtime Controls

| Key   | Action                              |
|-------|-------------------------------------|
| SPACE | E-STOP — zero all servos instantly  |
| R     | Resume inference                    |
| Q     | Graceful shutdown                   |

---

## 8. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `TimeoutError: timed out while waiting for handshake` | JIT not warmed up | Run the warmup script on RunPod first (Section 6) |
| `ConnectionClosedError` during first inference | Proxy timeout during JIT | Same — warmup first |
| `ConnectionRefusedError` from Mac | Server not running or wrong Pod ID | Verify server is running (`tail /workspace/serve.log`), double-check Pod ID |
| `uv: command not found` | Pod restarted | `export PATH="/root/.local/bin:$PATH"` |
| `Config 'pi05_excavator_v2' not found` | Using upstream OpenPI, not your fork | Clone `shashvcode/openpi-actor-labs` |
| Camera HTTP 404 | Frame capture not started on Pi #2 | `curl -X POST http://192.168.1.93:8080/frames/start` |
| Servos not moving | `pi_receiver.py` not running on Pi #1 | SSH into Pi #1 and start it |
| Very slow inference (>5s) | Pod under load or small GPU | Check `nvidia-smi` — need at least A100 40GB |
| `No space left on device` | Root filesystem full | Always work in `/workspace`, set `HF_HOME=/workspace/.hf_home` |

---

## Quick Reference — Copy-Paste Commands

**On RunPod (setup + serve):**

```bash
cd /workspace/openpi
export PATH="/root/.local/bin:$PATH"
export WANDB_MODE=disabled
export HF_TOKEN=<YOUR_HF_TOKEN>

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_excavator_v2 \
    --policy.dir checkpoints/pi05_excavator_v2/excavator_v1
```

**On RunPod (warmup — second terminal):**

```bash
cd /workspace/openpi && export PATH="/root/.local/bin:$PATH" && uv run python -c "
from openpi_client import websocket_client_policy as wcp; import numpy as np
p = wcp.WebsocketClientPolicy(host='localhost', port=8000)
r = p.infer({'observation/state': np.zeros(4, dtype=np.float32), 'observation/image_cab': np.zeros((224,224,3), dtype=np.uint8), 'observation/image_side': np.zeros((224,224,3), dtype=np.uint8), 'prompt': 'warmup'})
print('READY -', r['actions'].shape)
"
```

**On Mac (run):**

```bash
cd /Users/shashiverma/actor/openpi && .venv/bin/python examples/excavator/run_policy.py \
    --host wss://<POD-ID>-8000.proxy.runpod.net \
    --cab-cam-url http://192.168.1.93:8080/frame/csi_0_imx219 \
    --side-cam-url http://192.168.1.93:8080/frame/usb_0_innomaker_u20cam_1080p_s1__inno \
    --pi-host 192.168.1.72 --pi-port 9000 \
    --no-ssh \
    --prompt "Scoop white packing peanuts from large pool and dump into small pool"
```

Replace `<POD-ID>` with your RunPod pod ID (e.g. `abc123def456`).
