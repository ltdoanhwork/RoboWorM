"""Compare acceleration methods on the sim2real interaction metric.

The benchmark question is not "what is method X's absolute score" but "does X keep
the physics that dense produced". So the comparison is restricted to the episodes
that are **valid for every method** - otherwise a method looks better simply because
a different subset of its episodes was measurable - and every term is reported as a
delta against dense on that shared subset.

Usage (after running compute_sim2real_batch.py per method):

    python -m metrics.sim2real.compare_methods \
        --roots dense=/tmp/s2r_batch/dense worldcache=/tmp/s2r_batch/worldcache \
        --baseline dense --out /tmp/s2r_compare
"""

import os
import sys
import json
import argparse

import numpy as np

_REPO = os.environ.get("REPO")
if _REPO and _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from metrics.sim2real.mine_violations import collect  # noqa: E402
from metrics.sim2real.interaction_probe import _json_safe  # noqa: E402

COLS = [
    ("score", ("sim2real_interaction_score",), "high"),
    ("levit_exc", ("violations", "levitation_rate_excess"), "low"),
    ("grasp_exc", ("violations", "grasp_follow_ratio_excess"), "low"),
    ("presence_def", ("violations", "object_present_deficit"), "low"),
    ("shape_exc", ("violations", "object_shape_excess"), "low"),
    ("penet_exc", ("violations", "penetration_excess"), "low"),
    ("chatter", ("violations", "contact_chatter_excess"), "low"),
    ("onset_err", ("agreement", "contact_onset_err_frames"), "low"),
    ("c_tIoU", ("agreement", "contact_temporal_iou"), "high"),
    ("traj_err", ("agreement", "obj_traj_err"), "low"),
    ("gap_err", ("agreement", "gap_curve_err"), "low"),
    ("step_pass", ("step_level", "step_pass_rate"), "high"),
]


def _get(d, path):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def index_by_episode(work_root):
    out = {}
    for r in collect(work_root):
        if r["metrics"].get("valid", True):
            out[(r["category"], r["episode"])] = r["metrics"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True,
                    help="method=work_root pairs, e.g. dense=/tmp/s2r_batch/dense")
    ap.add_argument("--baseline", default="dense")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tables = {}
    for spec in args.roots:
        name, _, root = spec.partition("=")
        if not root:
            raise SystemExit(f"expected method=work_root, got {spec!r}")
        tables[name] = index_by_episode(root)
        print(f"[compare] {name}: {len(tables[name])} valid episodes")

    shared = None
    for t in tables.values():
        keys = set(t)
        shared = keys if shared is None else (shared & keys)
    shared = sorted(shared or [])
    print(f"[compare] {len(shared)} episodes valid for ALL methods")
    if not shared:
        return

    stats = {}
    for name, t in tables.items():
        row = {}
        for col, path, _ in COLS:
            vals = [_get(t[k], path) for k in shared]
            vals = [float(v) for v in vals
                    if v is not None and np.isfinite(float(v))]
            row[col] = float(np.mean(vals)) if vals else None
            row[col + "_n"] = len(vals)
        stats[name] = row

    order = ([args.baseline] if args.baseline in stats else []) + \
        [n for n in stats if n != args.baseline]
    head = f"{'method':<12}" + "".join(f"{c[:10]:>11}" for c, _, _ in COLS)
    print("\n" + head)
    print("-" * len(head))
    for name in order:
        line = f"{name:<12}"
        for col, _, _ in COLS:
            v = stats[name][col]
            line += f"{'  --':>11}" if v is None else f"{v:>11.3f}"
        print(line)

    base = stats.get(args.baseline)
    if base:
        print(f"\ndelta vs {args.baseline} (positive = better physics agreement):")
        print(head)
        print("-" * len(head))
        for name in order:
            if name == args.baseline:
                continue
            line = f"{name:<12}"
            for col, _, direction in COLS:
                a, b = stats[name][col], base[col]
                if a is None or b is None:
                    line += f"{'  --':>11}"
                else:
                    d = (a - b) if direction == "high" else (b - a)
                    line += f"{d:>+11.3f}"
            print(line)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        dest = os.path.join(args.out, "method_comparison.json")
        with open(dest, "w") as f:
            json.dump(_json_safe({"shared_episodes": [list(k) for k in shared],
                                  "num_shared": len(shared),
                                  "per_method_valid": {k: len(v) for k, v in tables.items()},
                                  "stats": stats, "baseline": args.baseline}), f, indent=2)
        print(f"\nwrote -> {dest}")


if __name__ == "__main__":
    main()
