from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing.connection import Listener

import torch

# Allow running as a script from repo root without installing as a package
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from multiview_consistency.video_metrics import ( 
    compute_video_metrics,
    init_model_context,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VGGT model service (loads model once, computes metrics per request).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6111)
    p.add_argument("--authkey", default="vggt")
    p.add_argument("--model", default="facebook/VGGT-1B")
    p.add_argument("--device", default=None, help='e.g. "cuda" or "cpu" (default: auto)')
    p.add_argument("--dtype", default=None, help='e.g. "bfloat16", "float16", "float32" (default: auto)')
    return p.parse_args()


def _dtype_from_str(s: str | None) -> torch.dtype | None:
    if s is None:
        return None
    s = s.strip().lower()
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16", "half"}:
        return torch.float16
    if s in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown dtype: {s!r}")


def serve_forever() -> None:
    args = _parse_args()
    ctx = init_model_context(
        device=args.device,
        autocast_dtype=_dtype_from_str(args.dtype),
        model_name=args.model,
    )

    addr = (args.host, int(args.port))
    print(f"[model_service] Listening on {addr} (device={ctx.device}, dtype={ctx.autocast_dtype})")
    print("[model_service] Model loaded; ready for requests.")

    listener = Listener(addr, authkey=args.authkey.encode("utf-8"))
    try:
        while True:
            conn = listener.accept()
            try:
                msg = conn.recv()
                if not isinstance(msg, dict):
                    conn.send({"ok": False, "error": f"Expected dict, got {type(msg)}"})
                    continue

                cmd = msg.get("cmd")
                if cmd == "ping":
                    conn.send({"ok": True, "ts": time.time(), "device": ctx.device})
                    continue

                if cmd == "compute_video_metrics":
                    episode_dir = msg["episode_dir"]
                    views = msg.get("views", ["view1", "view2"])
                    stride = int(msg.get("stride", 1))
                    max_frames = msg.get("max_frames", None)
                    include_per_frame = bool(msg.get("include_per_frame", False))

                    out = compute_video_metrics(
                        ctx,
                        episode_dir=str(episode_dir),
                        views=list(views),
                        stride=stride,
                        max_frames=None if max_frames is None else int(max_frames),
                        include_per_frame=include_per_frame,
                    )
                    conn.send({"ok": True, "result": out})
                    continue

                if cmd == "shutdown":
                    conn.send({"ok": True, "message": "shutting down"})
                    return

                conn.send({"ok": False, "error": f"Unknown cmd={cmd!r}"})
            except Exception as e:
                conn.send({"ok": False, "error": repr(e)})
            finally:
                conn.close()
    finally:
        listener.close()


if __name__ == "__main__":
    serve_forever()

