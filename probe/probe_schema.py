"""Probe the policy server's expected obs schema.

Strategy: send progressively richer observations and watch the server's
error tracebacks (each error closes the connection, so we reconnect each round).
"""

from __future__ import annotations

import argparse
import sys

import msgpack
import numpy as np
import websockets.sync.client


def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(),
                b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]),
                          shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def _try(uri: str, label: str, obs: dict) -> tuple[str, object]:
    """Connect, send obs, return ('ok', action_dict) or ('err', traceback_str)."""
    print(f"\n=== {label} ===", flush=True)
    print("  sending keys:", sorted(obs.keys()))
    with websockets.sync.client.connect(uri, compression=None, max_size=None,
                                        ping_timeout=60, close_timeout=10) as conn:
        _ = conn.recv()  # discard metadata
        conn.send(msgpack.packb(obs, default=_pack_array, use_bin_type=True))
        resp = conn.recv(timeout=120)
    if isinstance(resp, str):
        # First/last few lines tend to identify the missing key
        last_lines = "\n".join(resp.strip().splitlines()[-12:])
        print("  ERROR (last lines):\n" + "\n".join("    " + ln for ln in last_lines.splitlines()))
        return "err", resp
    action = msgpack.unpackb(resp, object_hook=_unpack_array, raw=False)
    print("  OK. action keys:", list(action.keys()))
    if "actions" in action:
        a = action["actions"]
        if isinstance(a, np.ndarray):
            print(f"  actions: ndarray shape={a.shape} dtype={a.dtype} "
                  f"min={a.min():.3f} max={a.max():.3f}")
    if "server_timing" in action:
        print("  server_timing:", action["server_timing"])
    return "ok", action


def _img(seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()
    uri = args.host if args.host.startswith("ws") else f"ws://{args.host}"
    if args.port:
        uri = f"{uri}:{args.port}"

    candidates = [
        ("(0) empty obs",
         {"prompt": "test"}),
        ("(1) excavator 2-cam: image_cab + image_side, state=4",
         {
             "observation/image_cab": _img(1),
             "observation/image_side": _img(2),
             "observation/state": np.zeros(4, dtype=np.float32),
             "prompt": "test",
         }),
        ("(2) 3-cam custom: image_arm + image_top + image_side, state=4",
         {
             "observation/image_arm": _img(3),
             "observation/image_top": _img(4),
             "observation/image_side": _img(5),
             "observation/state": np.zeros(4, dtype=np.float32),
             "prompt": "test",
         }),
        ("(3) CAN 3-cam: cab_forward + front_left + front_right, state=4",
         {
             "observation/image_cab_forward": _img(6),
             "observation/image_front_left": _img(7),
             "observation/image_front_right": _img(8),
             "observation/state": np.zeros(4, dtype=np.float32),
             "prompt": "test",
         }),
    ]

    for label, obs in candidates:
        try:
            status, _ = _try(uri, label, obs)
        except Exception as e:
            print(f"  TRANSPORT ERROR: {e}", flush=True)
            continue
        if status == "ok":
            print("\n>>> Found schema:", label)
            return


if __name__ == "__main__":
    main()
