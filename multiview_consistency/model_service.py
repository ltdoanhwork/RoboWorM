from __future__ import annotations

import argparse
import os
import sys
import traceback
from multiprocessing.connection import Listener
from typing import Any

# Suppress benign Triton/gcc macro-redefinition noise during torch.compile.
os.environ.setdefault("CFLAGS", "-Wno-macro-redefined")

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VGGT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _VGGT_ROOT not in sys.path:
    sys.path.insert(0, _VGGT_ROOT)

from multiview_consistency.video_metrics import (
    compute_video_metrics,
    init_model_context,
)

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VGGT model service (loads model once, computes metrics per request).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6111)
    p.add_argument("--authkey", default="vggt_omega")
    p.add_argument(
        "--omega_checkpoint",
        default=None,
        help="Path to VGGT-Omega checkpoint .pt (backend=vggt_omega).",
    )
    p.add_argument("--omega_image_resolution", type=int, default=512)
    p.add_argument("--omega_patch_size", type=int, default=16)
    p.add_argument("--dtype", default=None, help='e.g. "bfloat16", "float16", "float32" (default: auto)')
    p.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "torch.compile aggregator (mode=reduce-overhead). "
            "Use --no-compile for faster first single-episode runs (default: compile on)."
        ),
    )
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

def _handle_request(msg: dict, ctx: Any) -> tuple[dict, str | None]:
    """
    Handle one request message.

    Returns:
        (response_dict, control) where control is:
          - None: keep session open
          - "close_session": close client connection, keep server running
          - "shutdown": stop server
    """
    cmd = msg.get("cmd")

    if cmd == "ping":
        return {"ok": True}, None

    if cmd == "close_session":
        return {"ok": True, "message": "session closed"}, "close_session"

    if cmd == "shutdown":
        return {"ok": True, "message": "shutting down"}, "shutdown"

    if cmd == "compute_video_metrics":
        episode_dir = msg["episode_dir"]
        views = msg.get("views", ["view1", "view2"])
        frame_batch_size = int(msg.get("frame_batch_size", 4))

        out = compute_video_metrics(
            ctx,
            episode_dir=str(episode_dir),
            views=list(views),
            frame_batch_size=frame_batch_size,
        )
        torch.cuda.empty_cache()
        return {"ok": True, "result": out}, None

    return {"ok": False, "error": f"Unknown cmd={cmd!r}"}, None


def _serve_connection(conn, ctx: Any) -> bool:
    """
    Serve requests on one client connection until disconnect or close_session.

    Returns True if the server should shut down entirely.
    """
    try:
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                break

            if not isinstance(msg, dict):
                conn.send({"ok": False, "error": f"Expected dict, got {type(msg)}"})
                continue

            try:
                resp, control = _handle_request(msg, ctx)
                conn.send(resp)
                if control == "close_session":
                    break
                if control == "shutdown":
                    return True
            except Exception as e:
                traceback.print_exc()
                try:
                    conn.send({"ok": False, "error": repr(e)})
                except Exception:
                    break
                torch.cuda.empty_cache()
    finally:
        conn.close()

    return False

def serve_forever() -> None:
    args = _parse_args()
    ctx = init_model_context(
        autocast_dtype=_dtype_from_str(args.dtype),
        omega_checkpoint=args.omega_checkpoint,
        omega_image_resolution=args.omega_image_resolution,
        omega_patch_size=args.omega_patch_size,
        enable_compile=bool(args.compile),
    )

    addr = (args.host, int(args.port))
    print(f"[model_service] Listening on {addr}, dtype={ctx.autocast_dtype})")

    listener = Listener(addr, authkey=args.authkey.encode("utf-8"))
    try:
        while True:
            conn = listener.accept()
            print("[model_service] Client connected.", flush=True)
            if _serve_connection(conn, ctx):
                print("[model_service] Shutdown requested.", flush=True)
                break
    finally:
        listener.close()


if __name__ == "__main__":
    serve_forever()
