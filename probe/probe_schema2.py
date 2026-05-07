"""Iteratively peel off missing keys until the server accepts the obs."""

from __future__ import annotations

import argparse
import re
import sys
import traceback as tb

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


KEY_PAT = re.compile(r"KeyError:\s*'([^']+)'")


def _img(seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)


def _send(uri, obs):
    with websockets.sync.client.connect(uri, compression=None, max_size=None,
                                        ping_timeout=60, close_timeout=10) as conn:
        _ = conn.recv()
        conn.send(msgpack.packb(obs, default=_pack_array, use_bin_type=True))
        resp = conn.recv(timeout=120)
    return resp


def _build(known_keys: list[str], state_dim: int) -> dict:
    obs: dict = {"prompt": "test", "observation/state": np.zeros(state_dim, dtype=np.float32)}
    for i, k in enumerate(known_keys):
        if k.startswith("observation/image"):
            obs[k] = _img(i)
        elif k == "observation/state":
            pass
        else:
            obs[k] = np.zeros(1, dtype=np.float32)
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--state-dim", type=int, default=4)
    ap.add_argument("--max-rounds", type=int, default=8)
    args = ap.parse_args()
    uri = args.host if args.host.startswith("ws") else f"ws://{args.host}"
    if args.port:
        uri = f"{uri}:{args.port}"

    discovered: list[str] = []
    for round_idx in range(args.max_rounds):
        obs = _build(discovered, args.state_dim)
        print(f"\n--- round {round_idx} ---")
        print("  obs keys:", sorted(obs.keys()))
        try:
            resp = _send(uri, obs)
        except Exception as e:
            print(f"  TRANSPORT ERROR: {e}")
            return
        if isinstance(resp, str):
            m = KEY_PAT.search(resp)
            if not m:
                # Dump last lines for shape mismatches etc.
                print("  Non-KeyError traceback (last 20 lines):")
                for ln in resp.strip().splitlines()[-20:]:
                    print("    " + ln)
                return
            missing = m.group(1)
            print(f"  server reports missing key: {missing!r}")
            if missing in discovered:
                print("  (already in discovered list — loop?)")
                return
            discovered.append(missing)
        else:
            action = msgpack.unpackb(resp, object_hook=_unpack_array, raw=False)
            print("  OK. action keys:", list(action.keys()))
            if "actions" in action and isinstance(action["actions"], np.ndarray):
                a = action["actions"]
                print(f"  actions shape={a.shape} dtype={a.dtype} "
                      f"min={a.min():.3f} max={a.max():.3f}")
            print("\n>>> Discovered obs keys (in order):")
            for k in discovered:
                print(f"    - {k}")
            print(f"    + observation/state (shape=({args.state_dim},), float32)")
            print("    + prompt (str)")
            return

    print("\n>>> Did not converge after", args.max_rounds, "rounds.")
    print("    Discovered so far:", discovered)


if __name__ == "__main__":
    main()
