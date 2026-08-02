"""Render one violating episode next to a clean one, in a single video.

A violation score in isolation says little: the reader has no way to tell whether
0.52 means "obviously broken" or "the tracker was noisy". Playing the worst and the
best episode in lockstep, through the *same* tracker and the same overlays, is what
makes the number falsifiable by eye.

Picks the extremes automatically from a batch work root, or takes explicit episode
ids. No GPU: masks come from ``tracks.npz`` and frames are re-extracted from the
mp4s on demand.

Usage:

    python -m metrics.sim2real.render_comparison \
        --work_root /tmp/s2r_batch/dense --term score --out /tmp/s2r_compare_viz
"""

import os
import sys
import json
import shutil
import argparse

import cv2
import numpy as np

_REPO = os.environ.get("REPO")
if _REPO and _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from metrics.sim2real.track_backend import load_tracks, rebuild_frames  # noqa: E402
from metrics.sim2real.interaction_probe import analyse_tracks  # noqa: E402
from metrics.sim2real.mine_violations import collect, rank, RANK_TERMS  # noqa: E402
from metrics.sim2real import interaction_viz  # noqa: E402

BAD = (80, 90, 255)
GOOD = (120, 230, 120)


def _prepare(rec, work, fps, conditioning=None):
    tracks = load_tracks(rec["tracks"])
    # Inherit the conditioning the probe actually scored with, so the number burnt
    # into the video can never disagree with sim2real_metrics.json. Recomputing with
    # the "action" default silently redisplayed the old full score (0.834) next to a
    # physics-only headline of 0.849 on GR1 episode_002346.
    if conditioning is None:
        conditioning = rec.get("metrics", {}).get("conditioning") or "action"
    m = analyse_tracks(tracks, fps=fps, conditioning=conditioning)
    if "gt_video" not in tracks:
        raise SystemExit(
            f"{rec['episode']}: tracks.npz predates the video-path field, so its frames "
            f"cannot be regenerated. Delete it and re-run the batch for that episode.")
    rebuild_frames(tracks, tracks["gt_video"], tracks["pred_video"],
                   os.path.join(work, rec["episode"]))
    # the tracker keeps the parsed prompt; fall back to the dataset metadata when the
    # batch has not written metrics.json yet
    instr = rec.get("metrics", {}).get("instruction")
    if not instr:
        # gt_video lives in the OUTPUT episode dir; metadata.json is in the shared
        # dense INPUT tree at the same {category}/{episode} leaf
        d = os.path.dirname(tracks["gt_video"])
        cands = [os.path.join(d, "metadata.json"),
                 os.path.join(d.replace("/output/", "/input/"), "metadata.json")]
        for c in cands:
            if os.path.exists(c):
                try:
                    instr = json.load(open(c)).get("instruction")
                except Exception:
                    instr = None
                if instr:
                    break
        instr = instr or tracks.get("object_prompt")
    return tracks["views"][m["_main_view"]], m, instr


def _label(rec, m, term, value, kind, instruction=None):
    sc = m["sim2real_interaction_score"]
    s = m["step_level"]
    txt = (f"{kind}  {rec['category']}/{rec['episode']}   "
           f"score={'INVALID' if sc is None else f'{sc:.3f}'}")
    if term != "score" and value is not None:
        txt += f"   {term}={value:.3f}"
    txt += f"   steps {s['num_passed']}/{s['num_nodes']}"
    instr = instruction or rec.get("metrics", {}).get("instruction")
    if instr:
        txt += f'   "{instr[:52]}"'
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--term", default="score", choices=list(RANK_TERMS))
    ap.add_argument("--worst_ep", default=None, help="category/episode override")
    ap.add_argument("--clean_ep", default=None)
    ap.add_argument("--fps", type=float, default=5.0,
                    help="dataset fps (used for the timing metrics)")
    ap.add_argument("--out_fps", type=int, default=None,
                    help="playback fps of the rendered video (default: 2x dataset fps, "
                         "since 5 fps robot data is painfully slow to watch)")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--conditioning", default=None, choices=["action", "text"],
                    help="override; by default inherited from each episode's "
                         "sim2real_metrics.json so the rendered score matches it")
    ap.add_argument("--keep_frames", action="store_true")
    args = ap.parse_args()

    recs = collect(args.work_root)

    def by_id(spec):
        """Resolve category/episode straight from tracks.npz.

        Explicit episodes must work even before the batch has written their
        sim2real_metrics.json, since ``tracks.npz`` already holds everything needed to
        score and render.
        """
        cat, _, ep = spec.partition("/")
        for r in recs:
            if r["category"] == cat and r["episode"] == ep:
                return r
        npz = os.path.join(args.work_root, cat, ep, "tracks.npz")
        if not os.path.exists(npz):
            raise SystemExit(f"no tracks for {spec} (looked for {npz})")
        return {"category": cat, "episode": ep, "tracks": npz,
                "dir": os.path.dirname(npz), "metrics": {}}

    def pick(spec, fallback_list):
        if spec:
            return None, by_id(spec)
        if not fallback_list:
            raise SystemExit(
                f"no measurable episodes to rank on {args.term}; pass --worst_ep / "
                f"--clean_ep explicitly, or wait for the batch to write metrics")
        return fallback_list[0]

    if not recs and not (args.worst_ep and args.clean_ep):
        raise SystemExit(f"no scored episodes under {args.work_root}")

    worst_v, worst = pick(args.worst_ep, rank(recs, args.term, 1, worst=True))
    clean_v, clean = pick(args.clean_ep, rank(recs, args.term, 1, worst=False))

    work = os.path.join(args.out, "_frames")
    os.makedirs(args.out, exist_ok=True)
    vw_bad, m_bad, instr_bad = _prepare(worst, work, args.fps, args.conditioning)
    vw_ok, m_ok, instr_ok = _prepare(clean, work, args.fps, args.conditioning)
    if worst_v is None:
        worst_v = m_bad["sim2real_interaction_score"]
    if clean_v is None:
        clean_v = m_ok["sim2real_interaction_score"]

    entries = [
        {"view": vw_bad, "metrics": m_bad, "color": BAD,
         "label": _label(worst, m_bad, args.term, worst_v, "WORST ", instr_bad)},
        {"view": vw_ok, "metrics": m_ok, "color": GOOD,
         "label": _label(clean, m_ok, args.term, clean_v, "CLEAN ", instr_ok)},
    ]

    mp4 = os.path.join(args.out, f"compare_{args.term}.mp4")
    out_fps = args.out_fps or max(2, int(round(args.fps * 2)))
    interaction_viz.comparison_clip(entries, mp4, fps=out_fps, scale=args.scale)
    # mp4v is what OpenCV can write without extra codecs, but many players and the
    # browser preview want h264; re-encode when ffmpeg is available
    h264 = os.path.join(args.out, f"compare_{args.term}_h264.mp4")
    if shutil.which("ffmpeg") and os.system(
            f'ffmpeg -y -loglevel error -i "{mp4}" -c:v libx264 -pix_fmt yuv420p '
            f'-crf 20 "{h264}"') == 0 and os.path.exists(h264):
        mp4 = h264

    # a still of the most incriminating moment, for contexts that cannot play video.
    # Pick the frame that the WORST-scoring term actually points at, rather than always
    # the gap curve: a levitation episode and a vanishing-object episode fail at
    # completely different moments.
    T = len(m_bad["contact_pred"])
    flagged = m_bad.get("levitation_frames") or []
    pres = np.asarray(m_bad["_present_pred"], bool) if "_present_pred" in m_bad else None
    if flagged:
        t_bad, why = int(np.median(flagged)), "levitation"
    elif pres is not None and (~pres).any():
        # longest run of frames where the object is missing -> show the middle of it
        idx = np.flatnonzero(~pres)
        splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        run = max(splits, key=len)
        t_bad, why = int(run[len(run) // 2]), "object lost"
    else:
        gapd = np.abs(np.asarray(m_bad.get("_gap_pred", []), float)
                      - np.asarray(m_bad.get("_gap_gt", []), float))
        t_bad = int(np.nanargmax(gapd)) if np.isfinite(gapd).any() else T - 1
        why = "max gap error"
    print(f"still frame t={t_bad} chosen by: {why}")
    rows = [
        interaction_viz._episode_row(vw_bad, m_bad, min(t_bad, T - 1),
                                     entries[0]["label"], (320, 192), args.scale, BAD),
        interaction_viz._episode_row(
            vw_ok, m_ok, min(t_bad, len(m_ok["contact_pred"]) - 1),
            entries[1]["label"], (320, 192), args.scale, GOOD),
    ]
    W = max(r.shape[1] for r in rows)
    cap = np.full((26, W, 3), 14, np.uint8)
    cv2.putText(cap, f"frame t={t_bad}, selected as the worst moment by: {why}"
                     f"    (green=object, red=gripper, orange=target, "
                     f"yellow line=measured gap)",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1)
    still = np.vstack([cap, rows[0], np.full((8, W, 3), 8, np.uint8), rows[1]])
    png = os.path.join(args.out, f"compare_{args.term}_t{t_bad}.png")
    cv2.imwrite(png, still)

    for name, m in (("worst", m_bad), ("clean", m_ok)):
        interaction_viz.episode_panel(
            entries[0]["view"] if name == "worst" else entries[1]["view"], m,
            os.path.join(args.out, f"panel_{name}.png"))

    if not args.keep_frames:
        shutil.rmtree(work, ignore_errors=True)

    print(f"WORST  {worst['category']}/{worst['episode']}  "
          f"score={m_bad['sim2real_interaction_score']:.3f}")
    print(f"CLEAN  {clean['category']}/{clean['episode']}  "
          f"score={m_ok['sim2real_interaction_score']:.3f}")
    print(f"video -> {mp4}")
    print(f"still -> {png}")


if __name__ == "__main__":
    main()
