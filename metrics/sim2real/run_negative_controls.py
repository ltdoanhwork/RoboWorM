"""Run the negative-control suite on cached tracks and print the verdict table.

Usage (tracks.npz comes from interaction_probe.py, no GPU needed here):

    python -m metrics.sim2real.run_negative_controls \
        --tracks /tmp/s2r_ep255b/tracks.npz --out /tmp/s2r_negctrl
"""

import os
import sys
import json
import argparse

import numpy as np

_REPO = os.environ.get("REPO")
if _REPO and _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from metrics.sim2real.track_backend import load_tracks  # noqa: E402
from metrics.sim2real.interaction_probe import analyse_tracks, _json_safe  # noqa: E402
from metrics.sim2real import negative_controls as NC  # noqa: E402
from metrics.sim2real import interaction_viz  # noqa: E402


TERM_COLS = ["levitation", "grasp_follow", "object_presence", "object_shape",
             "penetration", "chatter", "contact_timing", "contact_overlap",
             "trajectory", "gap_curve"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--donor_tracks", default=None,
                    help="tracks.npz of a DIFFERENT episode, for the swap_gripper control")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--tau_gap", type=float, default=None)
    ap.add_argument("--eps_move", type=float, default=None)
    ap.add_argument("--viz", action="store_true", help="also render a panel per control")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tracks = load_tracks(args.tracks)
    params = {"tau_gap": args.tau_gap, "eps_move": args.eps_move}

    cases = {"real": lambda t: t}
    cases.update(NC.CONTROLS)
    if args.donor_tracks:
        donor = load_tracks(args.donor_tracks)
        cases["swap_gripper"] = lambda t, d=donor: NC.ctrl_swap_gripper(t, d)

    rows, results = [], {}
    for name, fn in cases.items():
        m = analyse_tracks(fn(tracks), fps=args.fps, params=params)
        results[name] = m
        ok, got = NC.verdict(name, m)
        rows.append((name, m, ok, got))
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump(_json_safe({k: v for k, v in m.items() if not k.startswith("_")}),
                      f, indent=2)
        if args.viz:
            vw = fn(tracks)["views"][m["_main_view"]]
            interaction_viz.episode_panel(
                vw, m, os.path.join(args.out, f"panel_{name}.png"))

    w = 16
    head = f"{'control':<{w}}{'score':>7}" + "".join(f"{c[:9]:>10}" for c in TERM_COLS) \
        + f"{'steps':>8}{'verdict':>9}"
    print("\n" + head)
    print("-" * len(head))
    for name, m, ok, got in rows:
        sc = m["sim2real_interaction_score"]
        line = f"{name:<{w}}" + ("  INVAL" if sc is None else f"{sc:>7.3f}")
        for c in TERM_COLS:
            n = m["score_terms"].get(c, {}).get("norm")
            line += f"{'  --':>10}" if n is None else f"{n:>10.3f}"
        s = m["step_level"]
        line += f"{s['num_passed']}/{s['num_nodes']:<6}"
        line += f"{('PASS' if ok else 'FAIL') if name in NC.EXPECT else '-':>9}"
        print(line)
    print("\n(norm: 1.0 = physically clean, 0.0 = maximally violating)")
    for name, m, ok, got in rows:
        if name in NC.EXPECT and got:
            exp = ", ".join(f"{k}{'<' if d == 'low' else '>'}{t}"
                            for k, (d, t) in NC.EXPECT[name].items())
            act = ", ".join(f"{k}={'None' if v is None else f'{v:.3f}'}"
                            for k, v in got.items())
            print(f"  {name:<16} expect {exp:<38} got {act}"
                  f"   -> {'PASS' if ok else 'FAIL'}")

    n_fail = sum(1 for _, _, ok, _ in rows if not ok and _ is not None)
    _ = n_fail
    with open(os.path.join(args.out, "verdicts.json"), "w") as f:
        json.dump(_json_safe({n: {"pass": ok, "terms": g} for n, _, ok, g in rows
                              if n in NC.EXPECT}), f, indent=2)
    print(f"\nwrote -> {args.out}")
    _ = np


if __name__ == "__main__":
    main()
