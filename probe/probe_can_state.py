"""Quick test: read 4-axis joystick state from can0 for ~3 seconds.

Verifies the same decoder used by run_policy_takeuchi.py.
"""

from __future__ import annotations

import socket
import struct
import time

import numpy as np

JOYSTICK_CAN_ID = 0x1C50A0EB
JOYSTICK_SCALE = 7000.0


def main(iface: str = "can0", duration: float = 3.0) -> None:
    s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    s.bind((iface,))
    eff_id = JOYSTICK_CAN_ID | socket.CAN_EFF_FLAG
    eff_mask = socket.CAN_EFF_MASK | socket.CAN_EFF_FLAG
    s.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                 struct.pack("=II", eff_id, eff_mask))
    s.settimeout(0.5)

    n = 0
    last_print = 0.0
    t_end = time.monotonic() + duration
    print(f"Listening on {iface} for ID 0x{JOYSTICK_CAN_ID:08X} ...")
    while time.monotonic() < t_end:
        try:
            frame = s.recv(16)
        except socket.timeout:
            continue
        if len(frame) < 16:
            continue
        can_id, dlc = struct.unpack_from("=IB", frame, 0)
        data = frame[8:16]
        if dlc < 8:
            continue
        rx, ry, lx, ly = struct.unpack_from("<hhhh", data, 0)
        state = np.array([lx, ly, rx, ry], dtype=np.float32) / JOYSTICK_SCALE
        np.clip(state, -1.0, 1.0, out=state)
        n += 1
        now = time.monotonic()
        if now - last_print >= 0.2:
            last_print = now
            lsx, lsy, rsx, rsy = state.tolist()
            print(
                f"  frame #{n:4d}  raw=(rx={rx:+5d} ry={ry:+5d} lx={lx:+5d} ly={ly:+5d})  "
                f"state=[lsx={lsx:+.4f} lsy={lsy:+.4f} rsx={rsx:+.4f} rsy={rsy:+.4f}]"
            )
    s.close()
    print(f"\nTotal joystick frames in {duration:.1f}s: {n} ({n/duration:.1f} Hz)")


if __name__ == "__main__":
    import sys
    iface = sys.argv[1] if len(sys.argv) > 1 else "can0"
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    main(iface, dur)
