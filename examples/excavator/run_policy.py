"""Run the trained excavator VLA policy on the physical excavator.

Connects to a remote GPU policy server via WebSocket, captures camera frames,
sends observations, and streams predicted joystick commands via UDP to
pi_receiver.py on the Raspberry Pi, which drives four servos through two
PCA9685 boards.

Architecture
------------
Cloud GPU (policy server)  <--WebSocket-->  This script  --UDP-->  Pi #1 (pi_receiver.py -> servos)

The model outputs 4-dim joystick values [lx, ly, rx, ry] in [-1, +1],
identical to what JoystickHandler produced during training data collection.
These are sent over UDP to pi_receiver.py, which applies its own inversions,
gains, smoothing, and rate-limiting before commanding the PCA9685 servo
drivers.  The entire servo conversion chain is handled on the Pi side.

Camera Setup
------------
Cameras (cab CSI + side USB) are on Pi #2 (192.168.1.83).  Two options:

  Option A — Stream frames from Pi #2 over HTTP (run this script on workstation):
    # On Pi #2, start the camera server (auto-discovers cameras):
    python3 pi_camera_server.py --port 8081

    # On workstation:
    python run_policy.py \\
        --host wss://<runpod-proxy-url> \\
        --cab-cam-url http://192.168.1.83:8080/frame/csi_0_imx219 \\
        --side-cam-url http://192.168.1.83:8080/frame/usb_0 \\
        --no-ssh

  Option B — Run this script directly on Pi #2 (cameras are local):
    python run_policy.py \\
        --cab-cam 0 --side-cam 2 \\
        --host wss://<runpod-proxy-url>

Usage
-----
    # 1. Start the policy server on the cloud GPU (e.g. RunPod):
    uv run scripts/serve_policy.py policy:checkpoint \\
        --policy.config pi05_excavator_v2 \\
        --policy.dir checkpoints/pi05_excavator_v2/run1/14999

    # 2. Run inference (uses RunPod proxy URL directly — no SSH tunnel needed):
    python run_policy.py --host wss://<pod-id>-8000.proxy.runpod.net \\
        --cab-cam-url http://192.168.1.83:8080/frame/csi_0_imx219 \\
        --side-cam-url http://192.168.1.83:8080/frame/usb_0 \\
        --no-ssh

    # Dry run (no Pi connection, prints predicted actions):
    python run_policy.py --host wss://<pod-id>-8000.proxy.runpod.net \\
        --dry-run --cab-cam 0 --side-cam 1
"""

import argparse
import io
import logging
import os
import select
import signal
import socket
import subprocess
import sys
import termios
import threading
import time
import tty
import urllib.request
import wave

import cv2
import numpy as np

STRATEGIST_AVAILABLE = False
REMOTE_ESTOP_AVAILABLE = False
OVERRIDE_AVAILABLE = False
try:
    from strategist.strategy_buffer import StrategyReader, strategy_to_prompt
    STRATEGIST_AVAILABLE = True
except ImportError:
    pass
try:
    from strategist.override_buffer import OverrideReader
    OVERRIDE_AVAILABLE = True
except ImportError:
    pass
try:
    from strategist.remote_estop_buffer import RemoteEstopReader
    REMOTE_ESTOP_AVAILABLE = True
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_DIM = 4
CONTROL_HZ = 10
UDP_SEND_HZ = 50
MODEL_IMG_SIZE = 224
CAM_W, CAM_H = 640, 480

PI_HOST_DEFAULT = "192.168.1.72"
PI_PORT_DEFAULT = 9000
PI_USER_DEFAULT = "actor"

PI_RECEIVER_CMD = (
    "python3 /home/actor/pi_receiver.py"
    " --output-mode servokit"
    " --udp-port 9000"
    " --status-rate 2"
    " --watchdog-timeout 0.5"
    " --invert-lx 1"
    " --invert-ly 1"
    " --channel-rx 0"
    " --channel-ry 1"
    " --invert-rx 0"
    " --invert-ry 1"
    " --center-rx 90"
    " --center-ry 85"
    " --right-i2c-scl 14"
    " --right-i2c-sda 15"
    " --gain-lx 1.4"
    " --gain-ly 1.4"
    " --gain-rx 1.4"
    " --gain-ry 1.4"
    " --smoothing-alpha 0.22"
    " --max-deg-per-sec 180"
    " --min-angle-step 0.3"
)


# ---------------------------------------------------------------------------
# ServoUDPSender — background thread sending axis values to Pi at 50 Hz
# ---------------------------------------------------------------------------

class ServoUDPSender:
    """Continuously sends joystick-format UDP packets to pi_receiver.py.

    A background thread transmits at 50 Hz so the Pi's watchdog (0.5 s) never
    triggers, even while the main thread blocks on a WebSocket inference call.
    The main loop calls ``set_axes`` to update the target values; this thread
    handles the actual network I/O.

    Packet format (ASCII, same as the original joystick pipeline)::

        {lx:.4f},{ly:.4f},{rx:.4f},{ry:.4f},{estop_bit}\\n

    Values are in [-1, +1], matching JoystickHandler output during training.
    """

    def __init__(self, pi_host: str = PI_HOST_DEFAULT, pi_port: int = PI_PORT_DEFAULT):
        self._addr = (pi_host, pi_port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self._lock = threading.Lock()
        self._lx = 0.0
        self._ly = 0.0
        self._rx = 0.0
        self._ry = 0.0
        self._estop = False
        self._running = True
        self._packets_sent = 0
        self._thread = threading.Thread(target=self._send_loop, daemon=True)
        self._thread.start()
        logger.info("UDP sender started -> %s:%d at %d Hz", pi_host, pi_port, UDP_SEND_HZ)

    def set_axes(self, lx: float, ly: float, rx: float, ry: float):
        with self._lock:
            self._lx = float(np.clip(lx, -1.0, 1.0))
            self._ly = float(np.clip(ly, -1.0, 1.0))
            self._rx = float(np.clip(rx, -1.0, 1.0))
            self._ry = float(np.clip(ry, -1.0, 1.0))

    def set_estop(self):
        with self._lock:
            self._estop = True
            self._lx = self._ly = self._rx = self._ry = 0.0
        logger.warning("E-STOP activated — all axes zeroed")

    def clear_estop(self):
        with self._lock:
            self._estop = False

    @property
    def packets_sent(self) -> int:
        return self._packets_sent

    def _send_loop(self):
        interval = 1.0 / UDP_SEND_HZ
        while self._running:
            with self._lock:
                pkt = (
                    f"{self._lx:.4f},{self._ly:.4f},"
                    f"{self._rx:.4f},{self._ry:.4f},"
                    f"{int(self._estop)}\n"
                )
            try:
                self._sock.sendto(pkt.encode(), self._addr)
                self._packets_sent += 1
            except OSError:
                pass
            time.sleep(interval)

    def close(self):
        self.set_estop()
        time.sleep(0.15)
        self._running = False
        self._thread.join(timeout=1.0)
        self._sock.close()
        logger.info("UDP sender closed (%d packets sent total)", self._packets_sent)


# ---------------------------------------------------------------------------
# SSH launcher for pi_receiver.py
# ---------------------------------------------------------------------------

def launch_pi_receiver(pi_host: str, pi_user: str = PI_USER_DEFAULT):
    ssh_target = f"{pi_user}@{pi_host}"

    logger.info("Killing existing pi_receiver.py on %s ...", pi_host)
    subprocess.run(
        ["ssh", ssh_target, "pkill -f pi_receiver.py || true"],
        timeout=10, capture_output=True,
    )
    time.sleep(0.5)

    launch_cmd = f"nohup {PI_RECEIVER_CMD} >/tmp/pi_receiver.log 2>&1 < /dev/null &"
    logger.info("Launching pi_receiver.py on %s ...", pi_host)
    subprocess.run(
        ["ssh", ssh_target, launch_cmd],
        timeout=10, capture_output=True,
    )
    time.sleep(1.0)

    result = subprocess.run(
        ["ssh", ssh_target, "pgrep -f pi_receiver.py >/dev/null"],
        timeout=10, capture_output=True,
    )
    if result.returncode == 0:
        logger.info("pi_receiver.py is running on %s", pi_host)
    else:
        logger.error("Failed to launch pi_receiver.py on %s! Check /tmp/pi_receiver.log on the Pi.", pi_host)
        sys.exit(1)


def kill_pi_receiver(pi_host: str, pi_user: str = PI_USER_DEFAULT):
    ssh_target = f"{pi_user}@{pi_host}"
    logger.info("Stopping pi_receiver.py on %s ...", pi_host)
    subprocess.run(
        ["ssh", ssh_target, "pkill -f pi_receiver.py || true"],
        timeout=10, capture_output=True,
    )


# ---------------------------------------------------------------------------
# Speech-to-text — push-to-talk via OpenAI Whisper API
# ---------------------------------------------------------------------------

class SpeechPrompt:
    """Records microphone audio and transcribes via OpenAI Whisper API."""

    SAMPLE_RATE = 16000
    CHANNELS = 1

    def __init__(self, api_key: str):
        import openai
        self._client = openai.OpenAI(api_key=api_key)
        self._recording = False
        self._frames: list[np.ndarray] = []
        self._stream = None

    def start_recording(self):
        import sounddevice as sd
        self._frames = []
        self._recording = True
        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype="int16",
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info(">>> RECORDING — speak your command. Press T again to stop.")

    def _audio_callback(self, indata, frames, time_info, status):
        if self._recording:
            self._frames.append(indata.copy())

    def stop_and_transcribe(self) -> str:
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._frames:
            return ""

        audio_data = np.concatenate(self._frames, axis=0)
        duration = len(audio_data) / self.SAMPLE_RATE
        logger.info("Transcribing %.1fs of audio...", duration)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())
        buf.seek(0)
        buf.name = "speech.wav"

        try:
            result = self._client.audio.transcriptions.create(
                model="whisper-1",
                file=buf,
            )
            text = result.text.strip()
            logger.info('Transcribed: "%s"', text)
            return text
        except Exception as exc:
            logger.error("Whisper API error: %s", exc)
            return ""


# ---------------------------------------------------------------------------
# Keyboard listener — e-stop / resume / quit / voice prompt during inference
# ---------------------------------------------------------------------------

class KeyboardController:
    """Non-blocking keyboard listener for runtime control.

    Keys:
        SPACE  — E-STOP: zero all servos immediately
        r      — Resume: clear e-stop, continue inference
        t      — Push-to-talk: record speech, transcribe, update prompt
        q      — Quit: graceful shutdown
    """

    def __init__(self, servo: ServoUDPSender | None, initial_prompt: str,
                 openai_api_key: str | None = None):
        self._servo = servo
        self._estopped = False
        self._quit = False
        self._old_settings = None
        self._prompt = initial_prompt
        self._prompt_lock = threading.Lock()
        self._speech = SpeechPrompt(openai_api_key) if openai_api_key else None
        self._is_recording = False
        self._thread = threading.Thread(target=self._listen, daemon=True)

    @property
    def prompt(self) -> str:
        with self._prompt_lock:
            return self._prompt

    @property
    def estopped(self) -> bool:
        return self._estopped

    @property
    def quit_requested(self) -> bool:
        return self._quit

    def start(self):
        if sys.stdin.isatty():
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self._thread.start()
            voice_str = "  T=Talk" if self._speech else ""
            logger.info("Keyboard controls:  SPACE=E-STOP  R=Resume%s  Q=Quit", voice_str)
        else:
            logger.info("No TTY — keyboard controls disabled (e-stop/resume/quit). Use Operator UI or SIGINT to stop.")

    def stop(self):
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def _listen(self):
        try:
            while not self._quit:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    if ch == " ":
                        self._estopped = True
                        if self._servo:
                            self._servo.set_estop()
                        logger.warning(">>> E-STOP (space) — servos zeroed. Press R to resume.")
                    elif ch in ("r", "R"):
                        if self._is_recording:
                            continue
                        self._estopped = False
                        if self._servo:
                            self._servo.clear_estop()
                        logger.info(">>> RESUMED (r) — inference continuing.")
                    elif ch in ("t", "T"):
                        self._handle_talk()
                    elif ch in ("q", "Q"):
                        self._quit = True
                        logger.info(">>> QUIT (q) — shutting down.")
        except Exception:
            pass

    def _handle_talk(self):
        if self._speech is None:
            logger.warning("Voice control disabled — no OpenAI API key. Use --openai-api-key or set OPENAI_API_KEY.")
            return

        if not self._is_recording:
            self._estopped = True
            if self._servo:
                self._servo.set_estop()
            self._is_recording = True
            self._speech.start_recording()
        else:
            text = self._speech.stop_and_transcribe()
            self._is_recording = False
            if text:
                with self._prompt_lock:
                    self._prompt = text
                logger.info('>>> NEW PROMPT: "%s"', text)
            else:
                logger.warning("No speech detected — keeping previous prompt.")
            self._estopped = False
            if self._servo:
                self._servo.clear_estop()
            logger.info('>>> RESUMED — inferencing with: "%s"', self.prompt)


# ---------------------------------------------------------------------------
# Camera source abstraction — local OpenCV or remote HTTP
# ---------------------------------------------------------------------------

class CameraSource:
    """Unified camera access: local V4L2/USB via OpenCV, or HTTP JPEG from pi_camera_server.py."""

    def __init__(self, source, label: str):
        self.label = label
        self._is_http = isinstance(source, str) and source.startswith("http")

        if self._is_http:
            self._url = source
            self._cap = None
            try:
                resp = urllib.request.urlopen(self._url, timeout=5)
                resp.read()
                logger.info("%s camera: HTTP source %s (OK)", label, source)
            except Exception as exc:
                sys.exit(f"Failed to reach {label} camera at {source}: {exc}")
        else:
            self._url = None
            self._cap = cv2.VideoCapture(source)
            if not self._cap.isOpened():
                sys.exit(f"Failed to open {label} camera (device index: {source})")
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info("%s camera: local device %s, native %dx%d", label, source, actual_w, actual_h)

    def grab(self, size: int = MODEL_IMG_SIZE) -> np.ndarray | None:
        if self._is_http:
            return self._grab_http(size)
        return self._grab_local(size)

    def _grab_local(self, size: int) -> np.ndarray | None:
        ret, bgr = self._cap.read()
        if not ret:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        return rgb.astype(np.uint8)

    def _grab_http(self, size: int) -> np.ndarray | None:
        try:
            resp = urllib.request.urlopen(self._url, timeout=2)
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

    def release(self):
        if self._cap is not None:
            self._cap.release()


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------

def run(args):
    from openpi_client import websocket_client_policy as wcp

    if not args.dry_run and not args.no_ssh:
        launch_pi_receiver(args.pi_host, args.pi_user)

    servo = None
    if not args.dry_run:
        servo = ServoUDPSender(args.pi_host, args.pi_port)

    # Load .env file from project root if it exists
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.isfile(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    openai_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    kb = KeyboardController(servo, initial_prompt=args.prompt, openai_api_key=openai_key)

    # VLM strategist (optional — shared memory from run_strategist.py)
    strategy_reader = None
    override_reader = None
    if args.strategist:
        if not STRATEGIST_AVAILABLE:
            sys.exit("--strategist requires the 'strategist' package. "
                     "Add actor-labs-box2 to PYTHONPATH or install it.")
        strategy_reader = StrategyReader(args.shm_name)
        if OVERRIDE_AVAILABLE:
            override_reader = OverrideReader()
        logger.info("Strategist enabled: reading shared memory '%s'%s", args.shm_name,
                   " | override enabled" if override_reader else "")

    host_uri = args.host
    port = args.port if not host_uri.startswith("ws") else None
    logger.info("Connecting to policy server at %s:%s ...", host_uri, port)
    policy = wcp.WebsocketClientPolicy(host=host_uri, port=port)
    metadata = policy.get_server_metadata()
    logger.info("Server metadata: %s", metadata)

    cab_source = args.cab_cam_url if args.cab_cam_url else args.cab_cam
    side_source = args.side_cam_url if args.side_cam_url else args.side_cam
    cab_cam = CameraSource(cab_source, "cab")
    side_cam = CameraSource(side_source, "side")

    state = np.zeros(ACTION_DIM, dtype=np.float32)
    step = 0
    target_dt = 1.0 / args.control_hz

    shutdown = False
    signal_estop_action = [None]  # None | "estop" | "resume", mutable for signal handler

    def on_signal(_sig, _frame):
        nonlocal shutdown
        shutdown = True

    def on_sigusr1(_sig, _frame):
        signal_estop_action[0] = "estop"

    def on_sigusr2(_sig, _frame):
        signal_estop_action[0] = "resume"

    signal.signal(signal.SIGINT, on_signal)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, on_sigusr1)
    if hasattr(signal, "SIGUSR2"):
        signal.signal(signal.SIGUSR2, on_sigusr2)

    kb.start()

    remote_estop_reader = None
    if REMOTE_ESTOP_AVAILABLE:
        try:
            remote_estop_reader = RemoteEstopReader()
            remote_estop_reader._ensure_open(create=True)
            logger.info("Remote e-stop enabled: Operator UI or kill -USR1 <pid> = E-STOP, kill -USR2 <pid> = Resume")
        except Exception as e:
            logger.warning("Remote e-stop unavailable: %s", e)
            remote_estop_reader = None

    logger.info(
        "Control loop: %d Hz action execution, %d Hz UDP to Pi.",
        args.control_hz, UDP_SEND_HZ,
    )

    try:
        while not shutdown and not kb.quit_requested:
            if args.max_steps and step >= args.max_steps:
                logger.info("Reached max steps (%d). Stopping.", args.max_steps)
                break

            # Apply signal-based remote e-stop/resume
            if remote_estop_reader and signal_estop_action[0] == "estop":
                remote_estop_reader.write_estop(True)
                signal_estop_action[0] = None
                logger.warning(">>> E-STOP (SIGUSR1) — servos zeroed. Resume via Operator UI or kill -USR2 %d", os.getpid())
            if remote_estop_reader and signal_estop_action[0] == "resume":
                remote_estop_reader.write_estop(False)
                signal_estop_action[0] = None
                logger.info(">>> RESUMED (SIGUSR2)")

            remote_estop = remote_estop_reader.read() if remote_estop_reader else None
            effective_estop = kb.estopped or (remote_estop is True)

            # While e-stopped, keep the loop alive but don't send any actions
            if effective_estop:
                if servo:
                    servo.set_estop()
                time.sleep(0.05)
                continue

            if servo:
                servo.clear_estop()

            cab_frame = cab_cam.grab(MODEL_IMG_SIZE)
            side_frame = side_cam.grab(MODEL_IMG_SIZE)
            if cab_frame is None or side_frame is None:
                logger.warning("Camera frame dropped, retrying...")
                time.sleep(0.01)
                continue

            # Determine prompt: strategist shared memory or manual keyboard/voice
            current_prompt = kb.prompt
            if strategy_reader is not None:
                strat_state = strategy_reader.read()
                if strat_state is not None:
                    if strat_state.is_stale:
                        if servo:
                            servo.set_estop()
                        logger.warning("Strategist stale (>%.1fs since last update); e-stop",
                                       strat_state.timestamp)
                        time.sleep(0.05)
                        continue
                    if not strat_state.global_safety_ok:
                        # Operator override: if active and not expired, proceed anyway
                        override_active = False
                        if override_reader is not None:
                            ov = override_reader.read()
                            if ov is not None and ov.is_valid:
                                override_active = True
                        if not override_active:
                            if servo:
                                servo.set_estop()
                            logger.warning("SAFETY HOLD from strategist: %s",
                                           strat_state.hazard_description)
                            time.sleep(0.05)
                            continue
                        logger.info("Operator override active — proceeding despite SAFETY_HOLD")
                    if servo and not effective_estop:
                        servo.clear_estop()
                    current_prompt = strategy_to_prompt(strat_state, task_description=kb.prompt)
            obs = {
                "observation/state": state.copy(),
                "observation/image_cab": cab_frame,
                "observation/image_side": side_frame,
                "prompt": current_prompt,
            }

            infer_start = time.perf_counter()
            result = policy.infer(obs)
            infer_ms = (time.perf_counter() - infer_start) * 1000
            actions = result["actions"]

            for action_idx in range(len(actions)):
                if shutdown or kb.quit_requested or effective_estop:
                    break

                action = actions[action_idx]
                lx, ly, rx, ry = action[:ACTION_DIM]
                action_start = time.perf_counter()

                if args.dry_run:
                    if action_idx == 0:
                        logger.info(
                            "  [dry-run] action[%d/%d]: lx=%+.3f ly=%+.3f rx=%+.3f ry=%+.3f",
                            action_idx, len(actions), lx, ly, rx, ry,
                        )
                else:
                    servo.set_axes(lx, ly, rx, ry)

                state = np.array(action[:ACTION_DIM], dtype=np.float32)
                step += 1

                if args.max_steps and step >= args.max_steps:
                    break

                elapsed = time.perf_counter() - action_start
                sleep_time = target_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if step % 10 == 0:
                pkt = servo.packets_sent if servo else 0
                estop_str = " [E-STOPPED]" if effective_estop else ""
                logger.info(
                    "Step %d | infer %.0fms | horizon %d | action[0]: lx=%+.3f ly=%+.3f rx=%+.3f ry=%+.3f | pkts=%d%s",
                    step, infer_ms, len(actions),
                    *actions[0][:ACTION_DIM], pkt, estop_str,
                )

    finally:
        logger.info("Shutting down...")
        kb.stop()
        if remote_estop_reader is not None:
            remote_estop_reader.close()
        cab_cam.release()
        side_cam.release()
        if strategy_reader is not None:
            strategy_reader.close()
        if override_reader is not None:
            override_reader.close()
        if servo is not None:
            servo.close()
        if not args.dry_run and not args.no_ssh and args.kill_on_exit:
            kill_pi_receiver(args.pi_host, args.pi_user)
        logger.info("Done. Executed %d steps.", step)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run excavator VLA policy — streams predicted joystick commands to Pi servos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = parser.add_argument_group("Policy server")
    g.add_argument("--host", default="localhost",
                   help="Policy server host (localhost via SSH tunnel, or wss:// RunPod proxy URL)")
    g.add_argument("--port", type=int, default=8000)

    g = parser.add_argument_group("Pi connection")
    g.add_argument("--pi-host", default=PI_HOST_DEFAULT, help="Pi #1 IP (servo controller)")
    g.add_argument("--pi-port", type=int, default=PI_PORT_DEFAULT, help="UDP port on Pi #1")
    g.add_argument("--pi-user", default=PI_USER_DEFAULT, help="SSH user on Pi #1")
    g.add_argument("--no-ssh", action="store_true",
                   help="Don't auto-launch pi_receiver.py via SSH (assumes it's already running)")
    g.add_argument("--kill-on-exit", action="store_true",
                   help="Kill pi_receiver.py on the Pi when this script exits")

    g = parser.add_argument_group("Cameras")
    g.add_argument("--cab-cam", type=int, default=0, help="Cab camera OpenCV index")
    g.add_argument("--side-cam", type=int, default=1, help="Side camera OpenCV index")
    g.add_argument("--cab-cam-url", default=None,
                   help="Cab camera HTTP URL (overrides --cab-cam)")
    g.add_argument("--side-cam-url", default=None,
                   help="Side camera HTTP URL (overrides --side-cam)")

    g = parser.add_argument_group("Task")
    g.add_argument("--prompt", default="Scoop packing peanuts from large pool and dump into small pool",
                   help="Initial language instruction (can be changed at runtime via voice)")
    g.add_argument("--openai-api-key", default=None,
                   help="OpenAI API key for Whisper speech-to-text (or set OPENAI_API_KEY env var). "
                        "Press T during inference to speak a new prompt.")

    g = parser.add_argument_group("Strategist (VLM)")
    g.add_argument("--strategist", action="store_true",
                   help="Enable VLM strategist via shared memory (requires run_strategist.py)")
    g.add_argument("--shm-name", default="excavator_strategy_v1",
                   help="POSIX shared memory segment name (must match run_strategist.py)")

    g = parser.add_argument_group("Control")
    g.add_argument("--control-hz", type=int, default=CONTROL_HZ,
                   help="Action execution rate (default: 10 Hz)")
    g.add_argument("--max-steps", type=int, default=None, help="Stop after N action steps")
    g.add_argument("--dry-run", action="store_true",
                   help="Print predicted actions without connecting to Pi")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
