from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from multiprocessing.connection import Client
from pathlib import Path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VGGT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _VGGT_ROOT not in sys.path:
    sys.path.insert(0, _VGGT_ROOT)

from multiview_consistency.utils import discover_episode_dirs

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Client to request multiview metrics from model_service (single episode or batch)."
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6111)
    p.add_argument("--authkey", default="vggt_omega")
    p.add_argument("--views", default="view1,view2", help='Comma-separated, e.g. "view1,view2"')
    p.add_argument(
        "--frame_batch_size",
        type=int,
        default=8,
        help="VGGT batch size B: number of timesteps per forward (views stay in S).",
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--episode_dir",
        help="Single episode folder containing view subfolders (view1, view2, ...).",
    )
    mode.add_argument(
        "--episodes_root",
        help="Root folder whose subdirectories are episode folders.",
    )
    p.add_argument(
        "--out_csv",
        default=None,
        help="Output CSV path (default: <episodes_root>/video_metrics.csv for batch).",
    )
    return p.parse_args()


def _connect(host: str, port: int, authkey: str) -> Client:
    return Client((host, int(port)), authkey=authkey.encode("utf-8"))

def _close_session(conn: Client) -> None:
    try:
        conn.send({"cmd": "close_session"})
        resp = conn.recv()
        if isinstance(resp, dict) and not resp.get("ok"):
            print(f"[run_video_metrics] close_session warning: {resp}", file=sys.stderr)
    except Exception as e:
        print(f"[run_video_metrics] close_session warning: {e!r}", file=sys.stderr)

def _request_metrics(
    conn: Client,
    *,
    episode_dir: str,
    views: list[str],
    frame_batch_size: int,
) -> dict:
    conn.send(
        {
            "cmd": "compute_video_metrics",
            "episode_dir": episode_dir,
            "views": views,
            "frame_batch_size": int(frame_batch_size),
        }
    )
    resp = conn.recv()
    if not isinstance(resp, dict):
        raise RuntimeError(f"Unexpected response type: {type(resp)}")
    if not resp.get("ok"):
        raise RuntimeError(str(resp.get("error", "unknown error")))
    result = resp.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected result payload: {result!r}")
    return result

def _write_metrics_csv(rows: list[dict], out_csv: str) -> None:
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "geometric_consistency",
        "pose_stability",
        "num_disjoint_frames",
        "disjoint_frame_ratio"
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _run_single(args: argparse.Namespace, views: list[str]) -> None:
    conn = _connect(args.host, args.port, args.authkey)
    try:
        result = _request_metrics(
            conn,
            episode_dir=args.episode_dir,
            views=views,
            frame_batch_size=args.frame_batch_size,
        )
    finally:
        _close_session(conn)
        conn.close()

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.out_csv is not None:
        name = Path(args.episode_dir).name
        _write_metrics_csv(
            [
                {
                    "name": name,
                    "geometric_consistency": result["geometric_consistency"],
                    "pose_stability": result["pose_stability"],
                    "num_disjoint_frames": result.get("num_disjoint_frames"),
                    "disjoint_frame_ratio": result.get("disjoint_frame_ratio")
                }
            ],
            args.out_csv,
        )
        print(f"Wrote metrics to {args.out_csv}", file=sys.stderr)


def _run_batch(args: argparse.Namespace, views: list[str]) -> None:
    if args.out_csv is None:
        args.out_csv = str(Path(args.episodes_root) / "video_metrics.csv")

    episodes = discover_episode_dirs(args.episodes_root, views)
    if not episodes:
        raise ValueError(
            f"No valid episode folders found under {args.episodes_root} for views={views}"
        )

    rows: list[dict] = []
    conn = _connect(args.host, args.port, args.authkey)
    import time
    try:
        for name, episode_dir in episodes:
            try:
                time_start = time.time()
                result = _request_metrics(
                    conn,
                    episode_dir=episode_dir,
                    views=views,
                    frame_batch_size=args.frame_batch_size,
                )
                time_end = time.time()
                print(f"[run_video_metrics] {name} took {time_end - time_start:0.2f} seconds for {result.get('num_frames', '?')} frames", file=sys.stderr)
                print("----Geometric Consistency: ", float(result["geometric_consistency"]))
                print("----Pose Stability: ", float(result["pose_stability"]))
                
                rows.append(
                    {
                        "name": name,
                        "geometric_consistency": float(result["geometric_consistency"]),
                        "pose_stability": float(result["pose_stability"]),
                        "num_disjoint_frames": int(result.get("num_disjoint_frames", 0)),
                        "disjoint_frame_ratio": float(result.get("disjoint_frame_ratio", 0.0)),
                    }
                )
            except Exception as e:
                print(f"[run_video_metrics] {name} failed: {e!r}", file=sys.stderr)
    finally:
        _close_session(conn)
        conn.close()

    _write_metrics_csv(rows, args.out_csv)
    print(f"Wrote {len(rows)} episodes to {args.out_csv}", file=sys.stderr)


def main() -> None:
    args = _parse_args()
    views = [v.strip() for v in str(args.views).split(",") if v.strip()]
    if not views:
        raise ValueError("At least one view is required via --views")

    if args.episode_dir is not None:
        _run_single(args, views)
    else:
        _run_batch(args, views)


if __name__ == "__main__":
    main()
