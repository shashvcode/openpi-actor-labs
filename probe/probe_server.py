"""Connect to the policy server, print the metadata it announces.

Usage:
    uv run --with 'websockets>=11' --with 'msgpack>=1' --with 'numpy<2' \
        probe/probe_server.py --host wss://rxz3mb39eex3qv-8000.proxy.runpod.net
"""

from __future__ import annotations

import argparse
import json
import sys
from pprint import pprint

import msgpack
import numpy as np
import websockets.sync.client


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def _walk_summary(node, path=""):
    """Yield (path, summary_str) for every leaf in a nested dict."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk_summary(v, f"{path}/{k}" if path else str(k))
    elif isinstance(node, np.ndarray):
        yield path, f"ndarray shape={node.shape} dtype={node.dtype}"
    elif isinstance(node, (list, tuple)):
        yield path, f"{type(node).__name__} len={len(node)}"
    else:
        yield path, f"{type(node).__name__}={node!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="ws://host:port or wss://...proxy.runpod.net")
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    uri = args.host if args.host.startswith("ws") else f"ws://{args.host}"
    if args.port:
        uri = f"{uri}:{args.port}"

    print(f"Connecting to {uri} ...", flush=True)
    with websockets.sync.client.connect(uri, compression=None, max_size=None,
                                        ping_timeout=60, close_timeout=10) as conn:
        raw = conn.recv()
        if isinstance(raw, str):
            print("Server returned a TEXT frame (likely an error):", raw)
            sys.exit(1)
        meta = msgpack.unpackb(raw, object_hook=_unpack_array, raw=False)

    print("\n=== Metadata (raw JSON-able view) ===")
    def _to_jsonable(x):
        if isinstance(x, np.ndarray):
            return {"__ndarray__": True, "shape": list(x.shape), "dtype": str(x.dtype)}
        if isinstance(x, dict):
            return {str(k): _to_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_to_jsonable(v) for v in x]
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        return x
    print(json.dumps(_to_jsonable(meta), indent=2)[:8000])

    print("\n=== Flattened summary ===")
    for p, s in _walk_summary(meta):
        print(f"  {p}: {s}")


if __name__ == "__main__":
    main()
