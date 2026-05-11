from __future__ import annotations

import argparse
import json
from multiprocessing.connection import Client


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Client to request multiview metrics from model_service.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6111)
    p.add_argument("--authkey", default="vggt")
    p.add_argument("--episode_dir", required=True)
    p.add_argument("--views", default="view1,view2", help='Comma-separated, e.g. "view1,view2"')
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--include_per_frame", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    addr = (args.host, int(args.port))
    views = [v.strip() for v in str(args.views).split(",") if v.strip()]

    conn = Client(addr, authkey=args.authkey.encode("utf-8"))
    try:
        conn.send(
            {
                "cmd": "compute_video_metrics",
                "episode_dir": args.episode_dir,
                "views": views,
                "stride": int(args.stride),
                "max_frames": None if args.max_frames is None else int(args.max_frames),
                "include_per_frame": bool(args.include_per_frame),
            }
        )
        resp = conn.recv()
    finally:
        conn.close()

    print(json.dumps(resp, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

