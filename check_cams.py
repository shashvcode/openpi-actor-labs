"""Snapshot all available cameras to identify scene/wrist indices.

Usage: python check_cams.py
Opens cam_0.jpg, cam_1.jpg, etc. in the current directory.
"""

import cv2
import sys

found = []
for i in range(10):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        continue
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ret, frame = cap.read()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if ret:
        path = f"cam_{i}.jpg"
        cv2.imwrite(path, frame)
        print(f"Camera {i}: {w}x{h} -> saved {path}")
        found.append(i)
    else:
        print(f"Camera {i}: opened but no frame")

if not found:
    print("No cameras found!")
    sys.exit(1)

print(f"\nFound {len(found)} cameras: {found}")
print("Open the cam_*.jpg files to identify which is scene and which is wrist.")
