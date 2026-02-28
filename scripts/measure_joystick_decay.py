"""Measure the real PS5 DualSense joystick spring-back decay curve.

Hold a stick to full deflection, then release. The script captures
the release trajectory and fits an exponential decay factor.

Usage:
    python scripts/measure_joystick_decay.py
"""

import time
import numpy as np

SAMPLE_HZ = 200
DEAD_ZONE = 0.08
RELEASE_THRESHOLD = 0.5
MIN_RELEASE_FRAMES = 3


def main():
    import pygame
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No controller found. Connect your PS5 DualSense and try again.")
        return

    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Connected: {js.get_name()}")
    print(f"Axes: {js.get_numaxes()}")
    print()
    print("Instructions:")
    print("  1. Push a stick to FULL deflection and HOLD it")
    print("  2. RELEASE the stick suddenly")
    print("  3. The script will capture the spring-back curve")
    print("  4. Repeat several times for accuracy")
    print("  5. Press Ctrl+C when done")
    print()

    axis_names = ["LX", "LY", "RX", "RY", "L2", "R2"]
    num_axes = min(js.get_numaxes(), 6)

    prev_values = np.zeros(num_axes)
    release_captures = []
    capturing = [None] * num_axes
    capture_data = [[] for _ in range(num_axes)]

    dt = 1.0 / SAMPLE_HZ

    try:
        while True:
            pygame.event.pump()
            values = np.array([js.get_axis(i) for i in range(num_axes)])

            for i in range(num_axes):
                v = values[i]
                pv = prev_values[i]

                if capturing[i] is not None:
                    capture_data[i].append((time.perf_counter(), v))
                    if abs(v) < DEAD_ZONE:
                        frames = capture_data[i]
                        if len(frames) >= MIN_RELEASE_FRAMES:
                            release_captures.append((axis_names[i], list(frames)))
                            analyze_release(axis_names[i], frames, SAMPLE_HZ)
                        capturing[i] = None
                        capture_data[i] = []
                elif abs(pv) > RELEASE_THRESHOLD and abs(v) < abs(pv) * 0.7:
                    capturing[i] = time.perf_counter()
                    capture_data[i] = [(capturing[i], pv), (time.perf_counter(), v)]

            prev_values = values.copy()
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        if release_captures:
            print(f"Captured {len(release_captures)} release events.\n")
            all_factors = []
            for axis_name, frames in release_captures:
                factor = compute_decay_factor(frames, SAMPLE_HZ)
                if factor is not None:
                    all_factors.append(factor)

            if all_factors:
                avg = np.mean(all_factors)
                print(f"\n{'=' * 60}")
                print(f"AVERAGE DECAY FACTOR per frame @ {SAMPLE_HZ}Hz: {avg:.4f}")
                factor_30hz = avg ** (SAMPLE_HZ / 30.0)
                print(f"Equivalent decay factor @ 30Hz (your control rate): {factor_30hz:.4f}")
                print(f"\nAt 30Hz, a held value of 0.8 would decay like:")
                val = 0.8
                for f in range(10):
                    ms = (f + 1) * (1000 / 30)
                    val *= factor_30hz
                    print(f"  Frame {f+1:2d} ({ms:5.0f}ms): {val:.3f}")
                print(f"\nUse --decay {factor_30hz:.2f} in run_policy.py")
        else:
            print("No releases captured. Try holding a stick fully, then releasing.")
        print("=" * 60)

    pygame.quit()


def analyze_release(axis_name, frames, sample_hz):
    times = [f[0] for f in frames]
    vals = [abs(f[1]) for f in frames]
    duration_ms = (times[-1] - times[0]) * 1000

    print(f"\n  [{axis_name}] Release captured: {len(frames)} samples, {duration_ms:.0f}ms")
    print(f"    Start: {vals[0]:.3f} → End: {vals[-1]:.3f}")
    print(f"    Curve: ", end="")
    for v in vals[:15]:
        bar = "█" * int(v * 30)
        print(f"{v:.2f} {bar}")
        print(f"           ", end="")
    print()

    factor = compute_decay_factor(frames, sample_hz)
    if factor is not None:
        print(f"    Decay factor @ {sample_hz}Hz: {factor:.4f}")


def compute_decay_factor(frames, sample_hz):
    vals = [abs(f[1]) for f in frames]
    ratios = []
    for i in range(1, len(vals)):
        if vals[i - 1] > 0.02:
            ratios.append(vals[i] / vals[i - 1])
    if ratios:
        return float(np.median(ratios))
    return None


if __name__ == "__main__":
    main()
