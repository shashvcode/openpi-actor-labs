"""Drive a single TakeuchiClient axis to verify direction + mapping.

Sends a small JOY4 burst on ONE axis at a time so you can confirm the
physical machine motion that corresponds to each model action index.

Reference (per actor-display config + takeuchi_client docstring):
    action[0] = left_stick_x  -> swing   (cab rotation: + right / - left)
    action[1] = left_stick_y  -> arm     (stick:        + in    / - out)
    action[2] = right_stick_x -> bucket  (curl:         + out   / - in)
    action[3] = right_stick_y -> boom    (lift:         + up    / - down)

The test holds the value for --duration seconds, then sends a 0.5s neutral
burst on exit (the receiver's 250ms watchdog also snaps to neutral if we die).

Prereq: bridges running (run_dual_sticks.sh) OR pass --auto-start-bridges.

Examples
--------
    # +0.3 on swing (cab rotates one direction) for 2s
    python examples/excavator/test_axis.py --axis swing --value 0.3 --duration 2

    # Cycle: +value, neutral, -value, neutral
    python examples/excavator/test_axis.py --axis bucket --value 0.3 --duration 2 --cycle

Use small magnitudes (0.2-0.4) for safety while you verify directions.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

TAKEUCHI_REPO = Path("/home/lob/takeuchi-canbus")
TAKEUCHI_HOST = "127.0.0.1"
TAKEUCHI_STICK1_CAN = "can2"
TAKEUCHI_STICK2_CAN = "can1"

AXES = ("swing", "arm", "bucket", "boom")
AXIS_TO_ACTION_INDEX = {
    "swing":  (0, "left_stick_x  -> action[0]"),
    "arm":    (1, "left_stick_y  -> action[1]"),
    "bucket": (2, "right_stick_x -> action[2]"),
    "boom":   (3, "right_stick_y -> action[3]"),
}


def _import_takeuchi_client():
    sys.path.insert(0, str(TAKEUCHI_REPO / "scripts"))
    from takeuchi_client import TakeuchiClient  # noqa: E402
    return TakeuchiClient


def _hold(tc, axis: str, value: float, duration: float, rate_hz: float = 100.0) -> None:
    """Spam JOY4 packets at rate_hz with `axis` set to `value` for `duration`s."""
    interval = 1.0 / rate_hz
    end = time.monotonic() + duration
    n = 0
    while time.monotonic() < end:
        tc.send(**{axis: value})
        n += 1
        time.sleep(interval)
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--axis", required=True, choices=AXES)
    p.add_argument("--value", type=float, required=True,
                   help="Axis value in [-1, +1] (use small magnitudes 0.2-0.4 for testing).")
    p.add_argument("--duration", type=float, default=2.0,
                   help="Seconds to hold the value (default 2).")
    p.add_argument("--cycle", action="store_true",
                   help="After +value, also do -value (with neutral pauses between).")
    p.add_argument("--rate-hz", type=float, default=100.0,
                   help="JOY4 packet send rate (default 100 Hz).")
    p.add_argument("--auto-start-bridges", action="store_true",
                   help="Spawn usb_can_bridge_stick{1,2}.py via sudo. "
                        "Otherwise assume run_dual_sticks.sh is already running.")
    args = p.parse_args()

    if not -1.0 <= args.value <= 1.0:
        p.error("--value must be in [-1, +1]")

    idx, descr = AXIS_TO_ACTION_INDEX[args.axis]
    print(f"\nAxis: {args.axis}  ({descr})")
    print(f"Value: {args.value:+.3f}    Duration: {args.duration:.2f}s    "
          f"Rate: {args.rate_hz:.0f} Hz    Cycle: {args.cycle}\n")

    TakeuchiClient = _import_takeuchi_client()
    tc = TakeuchiClient(
        host=TAKEUCHI_HOST,
        serial_stick1=TAKEUCHI_STICK1_CAN,
        serial_stick2=TAKEUCHI_STICK2_CAN,
        rate_hz=args.rate_hz,
        auto_start=args.auto_start_bridges,
        use_sudo=True,
        repo_path=TAKEUCHI_REPO,
    )

    try:
        # Settle + tiny neutral hold so receivers are alive
        print("[t=0.00s] sending neutral 0.3s ...")
        _hold(tc, args.axis, 0.0, 0.3, args.rate_hz)

        print(f"[t=0.30s] +{args.value:+.3f} on {args.axis} for {args.duration:.2f}s ...")
        n_pos = _hold(tc, args.axis, args.value, args.duration, args.rate_hz)
        print(f"           sent {n_pos} packets")

        print("[t=...]   neutral 0.5s ...")
        _hold(tc, args.axis, 0.0, 0.5, args.rate_hz)

        if args.cycle:
            print(f"[t=...]   {-args.value:+.3f} on {args.axis} for {args.duration:.2f}s ...")
            n_neg = _hold(tc, args.axis, -args.value, args.duration, args.rate_hz)
            print(f"           sent {n_neg} packets")

            print("[t=...]   neutral 0.5s ...")
            _hold(tc, args.axis, 0.0, 0.5, args.rate_hz)

        print("\nDone. Closing (TakeuchiClient.__exit__ sends neutral×3).")
    finally:
        tc.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
