#!/usr/bin/env python3
"""Pre-flight a directory of generated clips for the hallucination metric.

The kinescore hallucination metric reads mp4s DIRECTLY -- there is no frame
extraction step, unlike the multiview-consistency harness. What it does require
is that every clip is exactly the reader's panel size
(``registry/views.py::ViewSpec.check_frame_size``, driven by
``configs/views.yaml``), because a mismatch raises and the clip is recorded as
``error`` in ``results.jsonl`` instead of being scored.

So the analogous "extraction helper" for this metric is a *probe*: walk a
variant tree, measure every clip with ``ffprobe``, and report which ones the
reader will accept, which will be rejected and why, and how many 16-frame
segments each will contribute. Run this before burning GPU hours.

It also separates the two roles that share every directory: a prediction
``<episode>.mp4`` and its paired real ground truth ``<episode>_gt.mp4``. The
``_gt`` clips are the real-video anchor the scores are only interpretable
against (``docs/metrics/hallucination_metric.md`` §6.3).

Usage::

    # one variant, all embodiments
    python metrics/kiva/probe_clips.py --variant dense

    # every variant, as a readiness table
    python metrics/kiva/probe_clips.py \
        --variant dense pisa radial svg1 svg2 --json /tmp/ready.json

    # an explicit directory
    python metrics/kiva/probe_clips.py \
        --videos sampling_dataset/dense/humanoid/output/singleview/dreamgen/makovian \
        --embodiment humanoid
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

#: embodiment -> (view_id, (panel_width, panel_height)).
#: Mirrors ``configs/views.yaml`` on ``origin/quang/hallucination-official``
#: and the per-embodiment reader table in the branch README.
VIEWS: dict[str, tuple[str, tuple[int, int]]] = {
    "single_arm": ("sv1_4x3", (640, 480)),
    "bimanual": ("sv1_16x9", (768, 432)),
    "humanoid": ("sv1_16x9", (768, 432)),
}

#: ``violations/segments.py::SEGMENT_LEN``
SEGMENT_LEN = 16

#: ``JerkDetector.per_frame`` scores all-zero below this.
MIN_FRAMES_FOR_JERK = 4

#: The pfss ffmpeg build. /usr/bin/ffprobe dies with a gotoblas symbol error
#: once torch has been imported in the same process tree, and the pfss binary
#: needs its own lib dir on LD_LIBRARY_PATH to resolve libavdevice.
FFMPEG_PREFIX = os.environ.get(
    "HAL_FFMPEG_BIN",
    "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/ffmpeg/"
    "ffmpeg-master-latest-linux64-gpl-shared")


def _ffprobe() -> str:
    local = Path(FFMPEG_PREFIX) / "bin" / "ffprobe"
    if local.is_file():
        return str(local)
    found = shutil.which("ffprobe")
    if not found:
        raise SystemExit(
            "ffprobe not found. Set HAL_FFMPEG_BIN to an ffmpeg prefix, or put "
            "ffprobe on PATH.")
    return found


def _env() -> dict[str, str]:
    env = dict(os.environ)
    lib = Path(FFMPEG_PREFIX) / "lib"
    if lib.is_dir():
        env["LD_LIBRARY_PATH"] = f"{lib}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def probe(path: Path, ffprobe: str, env: dict[str, str]) -> dict:
    """``{width, height, n_frames, fps}`` for one clip, or ``{error}``."""
    try:
        raw = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,nb_frames,avg_frame_rate",
             # key=value, NOT positional: ffprobe emits these entries in its own
             # stream-struct order, not the order they are requested in.
             "-of", "default=nw=1", str(path)],
            capture_output=True, text=True, env=env, timeout=60, check=True,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    fields = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
    num, _, den = (fields.get("avg_frame_rate") or "0/1").partition("/")
    try:
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = float("nan")
    try:
        return {"width": int(fields["width"]), "height": int(fields["height"]),
                "n_frames": int(fields["nb_frames"]), "fps": fps}
    except (ValueError, KeyError):
        return {"error": f"ffprobe returned {fields!r}"}


def probe_dir(videos: Path, embodiment: str, ffprobe: str, env: dict[str, str],
              limit: int = 0) -> dict:
    """Probe one directory, split by role (prediction vs paired ``_gt``)."""
    view_id, (pw, ph) = VIEWS[embodiment]
    roles: dict[str, dict] = {}
    for role, select in (("pred", lambda p: not p.stem.endswith("_gt")),
                         ("gt", lambda p: p.stem.endswith("_gt"))):
        clips = sorted(p for p in videos.rglob("*.mp4") if select(p))
        if limit:
            clips = clips[:limit]
        sizes: Counter = Counter()
        fpss: Counter = Counter()
        ok = bad = short = 0
        frames = 0
        segments = 0
        errors: list[str] = []
        for clip in clips:
            info = probe(clip, ffprobe, env)
            if "error" in info:
                errors.append(f"{clip.name}: {info['error']}")
                bad += 1
                continue
            sizes[(info["width"], info["height"])] += 1
            fpss[round(info["fps"], 3)] += 1
            if (info["width"], info["height"]) != (pw, ph):
                bad += 1
                continue
            if info["n_frames"] < MIN_FRAMES_FOR_JERK:
                short += 1
                continue
            ok += 1
            frames += info["n_frames"]
            # ``segments.bounds``: consecutive 16-frame windows, last one short.
            segments += -(-info["n_frames"] // SEGMENT_LEN)
        roles[role] = {
            "n_clips": len(clips),
            "n_scorable": ok,
            "n_wrong_size": bad,
            "n_too_short": short,
            "total_frames": frames,
            "est_segments": segments,
            "sizes": {f"{w}x{h}": n for (w, h), n in sizes.most_common()},
            "fps": dict(fpss.most_common()),
            "errors": errors[:5],
        }
    return {
        "videos": str(videos),
        "embodiment": embodiment,
        "view_id": view_id,
        "required_size": f"{pw}x{ph}",
        "roles": roles,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="sampling_dataset",
                   help="Root holding <variant>/<embodiment>/output/... "
                        "(default: sampling_dataset)")
    p.add_argument("--variant", nargs="*", default=[],
                   help="Variant dir names to probe under --data-root")
    p.add_argument("--embodiment", nargs="*", default=list(VIEWS),
                   choices=list(VIEWS),
                   help=f"(default: {' '.join(VIEWS)})")
    p.add_argument("--split", nargs="*", default=["makovian", "non_makovian"],
                   help="(default: makovian non_makovian)")
    p.add_argument("--videos", default="",
                   help="Probe this one directory instead of a variant tree "
                        "(needs exactly one --embodiment)")
    p.add_argument("--limit", type=int, default=0,
                   help="Cap clips probed per role (0 = all)")
    p.add_argument("--json", default=None, help="Write the full report here")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ffprobe, env = _ffprobe(), _env()
    reports: list[dict] = []

    if args.videos:
        if len(args.embodiment) != 1:
            raise SystemExit("--videos needs exactly one --embodiment")
        reports.append(probe_dir(Path(args.videos), args.embodiment[0],
                                 ffprobe, env, args.limit))
    else:
        if not args.variant:
            raise SystemExit("pass --variant <name> ... or --videos <dir>")
        root = Path(args.data_root)
        for variant in args.variant:
            for embodiment in args.embodiment:
                for split in args.split:
                    videos = (root / variant / embodiment / "output" /
                              "singleview" / "dreamgen" / split)
                    if not videos.is_dir():
                        print(f"[probe] skip (no dir): {videos}",
                              file=sys.stderr)
                        continue
                    r = probe_dir(videos, embodiment, ffprobe, env, args.limit)
                    r.update(variant=variant, split=split)
                    reports.append(r)

    header = (f"{'variant':<12} {'embodiment':<11} {'split':<13} "
              f"{'need':<9} {'pred ok/all':>13} {'gt ok/all':>12} "
              f"{'pred segs':>10} {'ready':>6}")
    print(header)
    print("-" * len(header))
    for r in reports:
        pred, gt = r["roles"]["pred"], r["roles"]["gt"]
        ready = "yes" if pred["n_scorable"] and gt["n_scorable"] else (
            "PRED" if pred["n_scorable"] else "no")
        print(f"{r.get('variant', '-'):<12} {r['embodiment']:<11} "
              f"{r.get('split', '-'):<13} {r['required_size']:<9} "
              f"{str(pred['n_scorable']) + '/' + str(pred['n_clips']):>13} "
              f"{str(gt['n_scorable']) + '/' + str(gt['n_clips']):>12} "
              f"{pred['est_segments']:>10} {ready:>6}")
    print("")
    print("need  = the reader's required frame size; any other size is REJECTED.")
    print("pred  = <episode>.mp4;  gt = paired <episode>_gt.mp4 real-video anchor.")
    print("ready = 'yes' both roles scorable, 'PRED' prediction only (no anchor).")

    offenders = [r for r in reports
                 if r["roles"]["pred"]["n_wrong_size"]
                 or r["roles"]["pred"]["errors"]]
    if offenders:
        print("\nproblems:")
        for r in offenders:
            pred = r["roles"]["pred"]
            print(f"  {r.get('variant', '-')}/{r['embodiment']}/"
                  f"{r.get('split', '-')}: {pred['n_wrong_size']} wrong size "
                  f"(saw {pred['sizes']}, need {r['required_size']})")
            for e in pred["errors"]:
                print(f"    {e}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "segment_len": SEGMENT_LEN,
            "views": {k: {"view_id": v[0], "panel": list(v[1])}
                      for k, v in VIEWS.items()},
            "reports": reports,
        }, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
