#!/usr/bin/env python3
"""Test connectivity to Pi #2 camera frame server.

Starts frame capture, fetches frames from each discovered camera,
validates them, measures latency, and stops frame capture.

Usage:
    python test_camera_connection.py
    python test_camera_connection.py --pi-url http://192.168.1.83:8080
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error

import cv2
import numpy as np


def http_get(url: str, timeout: float = 5.0) -> bytes:
    resp = urllib.request.urlopen(url, timeout=timeout)
    return resp.read()


def http_post(url: str, timeout: float = 10.0) -> bytes:
    req = urllib.request.Request(url, method="POST", data=b"")
    resp = urllib.request.urlopen(req, timeout=timeout)
    return resp.read()


def main():
    parser = argparse.ArgumentParser(description="Test Pi camera connectivity")
    parser.add_argument("--pi-url", default="http://192.168.1.83:8080",
                        help="Base URL of the Pi recorder/frame server")
    parser.add_argument("--num-frames", type=int, default=10,
                        help="Number of frames to fetch per camera for latency test")
    args = parser.parse_args()

    base = args.pi_url.rstrip("/")
    print(f"Testing camera connection to {base}\n")

    # --- 1. Health check ---
    print("=" * 60)
    print("1. Health check")
    print("=" * 60)
    try:
        status_data = http_get(f"{base}/status")
        status = json.loads(status_data)
        print(f"   Pi status: {json.dumps(status, indent=4)}")
    except urllib.error.URLError as e:
        print(f"   FAIL: Cannot reach {base}/status — {e}")
        print(f"\n   Is the Pi reachable? Try: ping 192.168.1.83")
        print(f"   Is the recording service running on port 8080?")
        sys.exit(1)
    except Exception:
        print(f"   /status not available, trying /health...")
        try:
            health = http_get(f"{base}/health")
            print(f"   Health: {health.decode()}")
        except Exception as e:
            print(f"   FAIL: Cannot reach Pi — {e}")
            sys.exit(1)

    # --- 2. Start frame capture ---
    print(f"\n{'=' * 60}")
    print("2. Starting frame capture")
    print("=" * 60)
    try:
        resp = http_post(f"{base}/frames/start")
        result = json.loads(resp)
        print(f"   Response: {json.dumps(result, indent=4)}")
        cameras = result.get("cameras", [])
        if not cameras:
            print("   WARNING: No cameras discovered!")
            print("   Check that cameras are connected to Pi #2")
            sys.exit(1)
        print(f"   Discovered {len(cameras)} camera(s): {cameras}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"   HTTP {e.code}: {body}")
        if "already" in body.lower() or "active" in body.lower():
            print("   Frame capture may already be running — continuing...")
            try:
                cam_data = http_get(f"{base}/cameras")
                cameras = json.loads(cam_data).get("cameras", [])
                print(f"   Active cameras: {cameras}")
            except Exception:
                print("   Could not get camera list. Trying known names...")
                cameras = ["csi_0_imx219", "usb_0"]
        else:
            sys.exit(1)
    except Exception as e:
        print(f"   FAIL: {e}")
        sys.exit(1)

    # Give cameras a moment to warm up
    time.sleep(1.0)

    # --- 3. Fetch and validate one frame from each camera ---
    print(f"\n{'=' * 60}")
    print("3. Fetching test frame from each camera")
    print("=" * 60)
    valid_cameras = []
    for cam_name in cameras:
        url = f"{base}/frame/{cam_name}"
        try:
            t0 = time.perf_counter()
            jpeg_data = http_get(url, timeout=5.0)
            latency_ms = (time.perf_counter() - t0) * 1000

            arr = np.frombuffer(jpeg_data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img is None:
                print(f"   {cam_name}: GOT JPEG ({len(jpeg_data)} bytes) but FAILED to decode!")
                continue

            h, w, c = img.shape
            print(f"   {cam_name}: OK — {w}x{h} RGB, {len(jpeg_data)} bytes JPEG, {latency_ms:.1f}ms")
            valid_cameras.append(cam_name)
        except urllib.error.HTTPError as e:
            print(f"   {cam_name}: HTTP {e.code} — {e.read().decode()}")
        except Exception as e:
            print(f"   {cam_name}: FAIL — {e}")

    if not valid_cameras:
        print("\n   No valid cameras! Cannot proceed.")
        sys.exit(1)

    # --- 4. Latency test — fetch N frames and measure timing ---
    print(f"\n{'=' * 60}")
    print(f"4. Latency test ({args.num_frames} frames per camera)")
    print("=" * 60)
    for cam_name in valid_cameras:
        url = f"{base}/frame/{cam_name}"
        latencies = []
        sizes = []
        for i in range(args.num_frames):
            t0 = time.perf_counter()
            jpeg_data = http_get(url, timeout=5.0)
            latencies.append((time.perf_counter() - t0) * 1000)
            sizes.append(len(jpeg_data))

        avg_ms = sum(latencies) / len(latencies)
        min_ms = min(latencies)
        max_ms = max(latencies)
        avg_kb = sum(sizes) / len(sizes) / 1024
        achievable_hz = 1000.0 / avg_ms if avg_ms > 0 else 0

        print(f"   {cam_name}:")
        print(f"     Latency: avg={avg_ms:.1f}ms  min={min_ms:.1f}ms  max={max_ms:.1f}ms")
        print(f"     Size:    avg={avg_kb:.1f} KB")
        print(f"     Max fetch rate: ~{achievable_hz:.0f} Hz")
        if avg_ms > 50:
            print(f"     WARNING: >50ms avg latency — may be too slow for 10Hz control")
        else:
            print(f"     OK for 10 Hz control loop (need <100ms budget per frame)")

    # --- 5. Verify both cab and side cameras are available ---
    print(f"\n{'=' * 60}")
    print("5. Policy compatibility check")
    print("=" * 60)
    has_csi = any("csi" in c for c in valid_cameras)
    has_usb = any("usb" in c for c in valid_cameras)
    csi_name = next((c for c in valid_cameras if "csi" in c), None)
    usb_name = next((c for c in valid_cameras if "usb" in c), None)

    if has_csi and has_usb:
        print(f"   PASS: Both cameras available")
        print(f"     Cab (CSI):  {csi_name}")
        print(f"     Side (USB): {usb_name}")
        print(f"\n   run_policy.py command:")
        print(f"     python run_policy.py \\")
        print(f"       --cab-cam-url {base}/frame/{csi_name} \\")
        print(f"       --side-cam-url {base}/frame/{usb_name} \\")
        print(f"       --host localhost --port 8000 \\")
        print(f"       --prompt \"Scoop packing peanuts from large pool and dump into small pool\"")
    elif has_csi:
        print(f"   WARNING: Only CSI camera found ({csi_name}), no USB camera")
    elif has_usb:
        print(f"   WARNING: Only USB camera found ({usb_name}), no CSI camera")
    else:
        print(f"   WARNING: No CSI or USB cameras matched expected naming")
        print(f"   Available cameras: {valid_cameras}")

    # --- 6. Stop frame capture ---
    print(f"\n{'=' * 60}")
    print("6. Stopping frame capture")
    print("=" * 60)
    try:
        resp = http_post(f"{base}/frames/stop")
        print(f"   {json.loads(resp)}")
    except Exception as e:
        print(f"   Warning: {e}")

    print(f"\n{'=' * 60}")
    print("DONE — camera pipeline is ready for VLA inference")
    print("=" * 60)


if __name__ == "__main__":
    main()
