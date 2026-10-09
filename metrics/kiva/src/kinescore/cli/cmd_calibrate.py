"""``kinescore calibrate``: fit rigidity and jerk thresholds on real clips of one robot."""
from __future__ import annotations

import argparse

NAME = "calibrate"
HELP = "fit rigidity/jerk thresholds on real clips, for one model"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from kinescore.cli._model import add_read_arguments

    add_read_arguments(parser)
    parser.add_argument("--real", required=True,
                        help="directory of real clips of the model's robot and view")
    parser.add_argument("--out", default=None,
                        help="thresholds file (default: <model>.thresholds.json)")
    parser.add_argument("--percentile", type=float, default=95.0,
                        help="threshold percentile over the real segments (default: 95)")


def run(args: argparse.Namespace) -> int:
    import json
    import os

    from kinescore.cli._model import build_scorer, find_clips, load_model

    clips = find_clips(args.real, args.pattern, args.limit)
    if not clips:
        raise SystemExit(f"no {args.pattern} under {args.real}")
    model = load_model(args.model, args.device)

    contexts = []
    for i, path in enumerate(clips, 1):
        contexts.append(model.read(path, args.frame_chunk, args.max_frames, args.seed))
        print(f"[calibrate] {i}/{len(clips)} {os.path.basename(path)}")
    scorer = build_scorer()
    scorer.calibrate(contexts, pct=args.percentile)

    out = args.out or os.path.splitext(args.model)[0] + ".thresholds.json"
    doc = {
        "model": os.path.basename(args.model),
        "robot": model.robot.name,
        "view_id": model.view.view_id,
        "real": os.path.abspath(args.real),
        "n_clips": len(clips),
        "percentile": args.percentile,
        "seed": args.seed,
        "thresholds": scorer.thresholds(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)
    for name, t in doc["thresholds"].items():
        print(f"[calibrate] {name:9s} {t['segment_threshold']:>12g} {t['units']}")
    print(f"[calibrate] wrote {out}")
    return 0
