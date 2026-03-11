#!/usr/bin/env python3
"""Live frame server for VLA policy inference on Raspberry Pi.

Auto-discovers CSI and USB cameras using the same rpicam-hello / v4l2-ctl
approach as the recording manager, opens them for live capture (picamera2
for CSI, OpenCV for USB), and serves the latest frame as JPEG over HTTP.

Can be used two ways:

  1. STANDALONE — run as its own process on a separate port:

       python3 pi_camera_server.py --port 8081

     Then from run_policy.py on the workstation:

       python run_policy.py \\
           --cab-cam-url http://192.168.1.83:8081/frame/csi_0_imx219 \\
           --side-cam-url http://192.168.1.83:8081/frame/usb_0 \\
           ...

  2. INTEGRATED — import FrameCapture into your existing recorder web server
     and add a few endpoints (see "Integration" section at bottom of file).

NOTE: Frame capture and recording (rpicam-vid / ffmpeg) are mutually exclusive
— they can't both hold the camera device.  During VLA inference you're not
recording training data, so this is fine.  If you need both, stop recording
before starting frame capture.

Endpoints (standalone mode)
---------------------------
    GET /frame/<camera_name>   — latest JPEG from that camera
    GET /cameras               — JSON list of discovered cameras
    GET /health                — "ok"
"""

import argparse
import json
import logging
import re
import subprocess
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Camera discovery (mirrors your recording manager's logic)
# ---------------------------------------------------------------------------

def discover_csi_cameras() -> list[dict]:
    """Discover CSI cameras via rpicam-hello --list-cameras."""
    try:
        result = subprocess.run(
            ["rpicam-hello", "--list-cameras", "-t", "1"],
            capture_output=True, text=True, timeout=10,
        )
        output = result.stdout + result.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("rpicam-hello not available — no CSI cameras discovered")
        return []

    cameras = []
    # Parse lines like: 0 : imx219 [3280x2464 10-bit RGGB] (/base/...)
    for match in re.finditer(
        r"(\d+)\s*:\s*(\w+)\s*\[(\d+)x(\d+)", output
    ):
        idx = int(match.group(1))
        sensor = match.group(2)
        # Use a moderate resolution for low-latency capture
        cameras.append({
            "type": "csi",
            "index": idx,
            "name": f"csi_{idx}_{sensor}",
            "sensor": sensor,
            "width": 640,
            "height": 480,
        })
        logger.info("Discovered CSI camera: index=%d sensor=%s", idx, sensor)

    return cameras


def discover_usb_cameras() -> list[dict]:
    """Discover USB cameras via v4l2-ctl --list-devices."""
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True, text=True, timeout=10,
        )
        output = result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("v4l2-ctl not available — no USB cameras discovered")
        return []

    cameras = []
    usb_idx = 0
    lines = output.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for USB device headers (contain "usb" in the path)
        if "usb" in line.lower() and line.endswith(":"):
            # Next lines are device paths like /dev/video0
            i += 1
            while i < len(lines) and lines[i].startswith("\t"):
                dev_path = lines[i].strip()
                if dev_path.startswith("/dev/video"):
                    # Verify it has VIDEO_CAPTURE capability
                    try:
                        cap_result = subprocess.run(
                            ["v4l2-ctl", "-d", dev_path, "--all"],
                            capture_output=True, text=True, timeout=5,
                        )
                        if "Video Capture" in cap_result.stdout:
                            dev_index = int(dev_path.replace("/dev/video", ""))
                            cameras.append({
                                "type": "usb",
                                "index": dev_index,
                                "name": f"usb_{usb_idx}",
                                "device_path": dev_path,
                                "width": 640,
                                "height": 480,
                            })
                            logger.info("Discovered USB camera: %s (index %d)", dev_path, dev_index)
                            usb_idx += 1
                            break  # one device path per USB camera
                    except (FileNotFoundError, subprocess.TimeoutExpired):
                        pass
                i += 1
        i += 1

    return cameras


def discover_all() -> list[dict]:
    """Discover all connected cameras (CSI + USB)."""
    return discover_csi_cameras() + discover_usb_cameras()


# ---------------------------------------------------------------------------
# Capture backends
# ---------------------------------------------------------------------------

class OpenCVCapture:
    """USB camera capture via OpenCV / V4L2."""

    def __init__(self, device_index: int, width: int, height: int):
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera at /dev/video{device_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("OpenCV camera /dev/video%d: %dx%d", device_index, actual_w, actual_h)

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class Picamera2Capture:
    """CSI camera capture via picamera2."""

    def __init__(self, camera_index: int, width: int, height: int):
        try:
            from picamera2 import Picamera2
        except ImportError:
            raise RuntimeError(
                "picamera2 not installed. Install: sudo apt install python3-picamera2"
            )
        self.picam = Picamera2(camera_index)
        config = self.picam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            buffer_count=2,
        )
        self.picam.configure(config)
        self.picam.start()
        time.sleep(1.0)
        logger.info("picamera2 CSI camera %d: %dx%d", camera_index, width, height)

    def read(self):
        frame = self.picam.capture_array()
        if frame is not None:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, bgr
        return False, None

    def release(self):
        self.picam.stop()


# ---------------------------------------------------------------------------
# Background capture thread — always holds the latest frame
# ---------------------------------------------------------------------------

class CameraThread:
    """Continuously captures frames in a background thread.

    get_jpeg() always returns the most recent frame — no buffering lag.
    """

    def __init__(self, backend, name: str, jpeg_quality: int = 85):
        self.name = name
        self._backend = backend
        self._quality = jpeg_quality
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._frame_count = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._quality]
        while self._running:
            ret, frame = self._backend.read()
            if ret and frame is not None:
                ok, buf = cv2.imencode(".jpg", frame, encode_params)
                if ok:
                    with self._lock:
                        self._jpeg = buf.tobytes()
                        self._frame_count += 1
            else:
                time.sleep(0.01)

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def close(self):
        self._running = False
        self._thread.join(timeout=2.0)
        self._backend.release()
        logger.info("Camera '%s' closed (%d frames captured)", self.name, self._frame_count)


# ---------------------------------------------------------------------------
# FrameCapture — the main class to integrate into your existing server
# ---------------------------------------------------------------------------

class FrameCapture:
    """Live frame capture from all discovered cameras.

    Auto-discovers CSI + USB cameras, opens each with the appropriate backend,
    and runs background capture threads.  Call get_jpeg(camera_name) to get the
    latest frame.

    Usage — standalone::

        fc = FrameCapture()
        fc.start()
        jpeg = fc.get_jpeg("csi_0_imx219")   # or "usb_0"
        fc.stop()

    Usage — integrated into your existing recorder server::

        # In your Flask/FastAPI app:
        frame_capture = FrameCapture()

        @app.route("/frames/start")
        def start_frames():
            frame_capture.start()
            return {"cameras": frame_capture.camera_names}

        @app.route("/frames/stop")
        def stop_frames():
            frame_capture.stop()
            return {"status": "ok"}

        @app.route("/frame/<name>")
        def get_frame(name):
            jpeg = frame_capture.get_jpeg(name)
            if jpeg is None:
                abort(404)
            return Response(jpeg, mimetype="image/jpeg")
    """

    def __init__(self, jpeg_quality: int = 85, width: int = 640, height: int = 480):
        self._quality = jpeg_quality
        self._width = width
        self._height = height
        self._cameras: dict[str, CameraThread] = {}
        self._discovered: list[dict] = []

    def start(self, camera_overrides: dict[str, str] | None = None):
        """Discover cameras and start capture threads.

        Args:
            camera_overrides: optional dict mapping camera name to device spec,
                e.g. {"csi_0_imx219": "picamera2:0", "usb_0": "opencv:2"}.
                If None, auto-discovers everything.
        """
        if self._cameras:
            logger.warning("FrameCapture already running — stop() first")
            return

        if camera_overrides:
            for name, spec in camera_overrides.items():
                backend = self._make_backend_from_spec(spec)
                self._cameras[name] = CameraThread(backend, name, self._quality)
        else:
            self._discovered = discover_all()
            if not self._discovered:
                raise RuntimeError("No cameras discovered!")
            for cam in self._discovered:
                backend = self._make_backend(cam)
                self._cameras[cam["name"]] = CameraThread(backend, cam["name"], self._quality)

        logger.info("FrameCapture started: %s", list(self._cameras.keys()))

    def stop(self):
        """Stop all capture threads and release cameras."""
        for cam in self._cameras.values():
            cam.close()
        self._cameras.clear()
        logger.info("FrameCapture stopped")

    def get_jpeg(self, name: str) -> bytes | None:
        """Get the latest JPEG frame from the named camera, or None."""
        cam = self._cameras.get(name)
        return cam.get_jpeg() if cam else None

    @property
    def camera_names(self) -> list[str]:
        return list(self._cameras.keys())

    @property
    def running(self) -> bool:
        return len(self._cameras) > 0

    def _make_backend(self, cam_info: dict):
        w = cam_info.get("width", self._width)
        h = cam_info.get("height", self._height)
        if cam_info["type"] == "csi":
            return Picamera2Capture(cam_info["index"], w, h)
        return OpenCVCapture(cam_info["index"], w, h)

    def _make_backend_from_spec(self, spec: str):
        if spec.startswith("picamera2"):
            idx = int(spec.split(":")[-1]) if ":" in spec else 0
            return Picamera2Capture(idx, self._width, self._height)
        idx = int(spec.split(":")[-1]) if ":" in spec else int(spec)
        return OpenCVCapture(idx, self._width, self._height)


# ---------------------------------------------------------------------------
# Standalone HTTP server (if you don't want to integrate into existing server)
# ---------------------------------------------------------------------------

class FrameHandler(BaseHTTPRequestHandler):
    frame_capture: FrameCapture | None = None

    def do_GET(self):
        path = self.path.rstrip("/")

        if path.startswith("/frame/"):
            cam_name = path[len("/frame/"):]
            fc = self.frame_capture
            if fc is None or not fc.running:
                self.send_error(503, "Frame capture not running")
                return
            jpeg = fc.get_jpeg(cam_name)
            if jpeg is None:
                available = fc.camera_names
                self.send_error(404, f"Camera '{cam_name}' not found. Available: {available}")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(jpeg)

        elif path == "/cameras":
            fc = self.frame_capture
            names = fc.camera_names if fc and fc.running else []
            body = json.dumps({"cameras": names}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")

        else:
            self.send_error(404, "Endpoints: /frame/<name>, /cameras, /health")

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Live camera frame server for VLA inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--port", type=int, default=8081,
                        help="HTTP port (default 8081, avoids conflict with recorder on 8080)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--cam", action="append", metavar="NAME=SPEC",
                        help="Manual camera spec instead of auto-discovery, "
                             "e.g. --cam csi_0_imx219=picamera2:0 --cam usb_0=opencv:2")
    args = parser.parse_args()

    fc = FrameCapture(
        jpeg_quality=args.jpeg_quality,
        width=args.width,
        height=args.height,
    )

    overrides = None
    if args.cam:
        overrides = {}
        for entry in args.cam:
            name, spec = entry.split("=", 1)
            overrides[name] = spec

    fc.start(camera_overrides=overrides)

    FrameHandler.frame_capture = fc
    server = HTTPServer(("0.0.0.0", args.port), FrameHandler)
    logger.info("Frame server on port %d", args.port)
    for name in fc.camera_names:
        logger.info("  GET /frame/%s", name)
    logger.info("  GET /cameras")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        fc.stop()
        server.server_close()


if __name__ == "__main__":
    main()
