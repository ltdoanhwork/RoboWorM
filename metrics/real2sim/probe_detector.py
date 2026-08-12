"""Does the detector ever see the object at all? Sweep threshold and prompt, then look.

The `no_contact` veto established that all four candidates in both bok choy episodes go the
whole episode without touching the arm, i.e. the right object is absent from the candidate list
rather than mis-ranked. That leaves exactly two possibilities, and they call for different fixes:

* the detector **does** propose the object but below the `box_thr` cut, or outside the top-4
  `prefilter` — a threshold problem, cheap to fix;
* the detector **never** proposes it under this prompt — a prompt or model-capacity problem, and
  no amount of threshold tuning helps.

This distinguishes them by running GroundingDINO directly on frame 0 across a threshold sweep
and several phrasings, and rendering every box with its confidence so the answer is visible
rather than inferred from a scalar.

Needs the WorldScore third-party path, so run it as the probe scripts do:

    cd metrics/WorldScore && PYTHONPATH=... python -m metrics.real2sim.probe_detector
"""

from __future__ import annotations

import json
import os

import numpy as np


REPO = os.environ.get("REPO", os.getcwd())
# Absolute, because this has to be run from the WorldScore directory (that is where the
# GroundingDINO config paths resolve), so a repo-relative path would not resolve.
ROOT = f"{REPO}/metrics/sim2real_out/dense/humanoid_singleview_dreamgen/makovian"

# Phrasings to try, from the instruction's own words outwards. The point is not to hand-pick a
# winner per episode -- that would be a per-episode constant, which this repo avoids -- but to
# find out whether ANY reasonable phrasing makes the object detectable. If only a hyper-specific
# one works, the honest conclusion is that the detector is the limit.
PROMPTS = [
    "green bok choy .",
    "bok choy .",
    "green leafy vegetable .",
    "green vegetable .",
    "vegetable .",
    "lettuce . cabbage . bok choy . green leafy vegetable .",
    "bok choy . corn . apple . avocado . grapes . banana .",
]
THRESHOLDS = [0.05, 0.10, 0.15, 0.25]


def frame0(ep: str, upscale: int = 3) -> tuple[str, np.ndarray]:
    """Frame 0 of the GT video, upscaled exactly as the tracker upscales it.

    The tracker detects on a 3x LANCZOS upscale, so probing at native resolution would be
    measuring a different input than the one that failed.
    """
    import cv2

    z = np.load(f"{ROOT}/{ep}/tracks.npz", allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    cap = cv2.VideoCapture(meta["gt_video"])
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read {meta['gt_video']}")
    rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    up = cv2.resize(rgb, (W * upscale, H * upscale), interpolation=cv2.INTER_LANCZOS4)
    path = f"/tmp/det_probe_{ep}.jpg"
    cv2.imwrite(path, cv2.cvtColor(up, cv2.COLOR_RGB2BGR))
    return path, up


def main() -> None:
    import cv2

    from metrics.sim2real.track_backend import Grounder

    ckpt = f"{REPO}/checkpoints/worldscore"
    g = Grounder(
        "worldscore/benchmark/metrics/third_party/groundingdino/config/"
        "GroundingDINO_SwinT_OGC.py",
        f"{ckpt}/groundingdino_swint_ogc.pth", "cuda")

    os.makedirs(f"{REPO}/metrics/sim2real_out/r2s", exist_ok=True)
    for ep in ("episode_000829", "episode_003151"):
        path, img = frame0(ep)
        print(f"\n===== {ep} =====")
        print(f"{'prompt':52s} {'thr':>5s} {'n':>3s}  confs")
        best = {}
        for pr in PROMPTS:
            for thr in THRESHOLDS:
                dets, hw = g.detect(path, pr, box_thr=thr)
                confs = sorted((float(d["conf"]) for d in dets), reverse=True)[:6]
                print(f"{pr:52s} {thr:5.2f} {len(dets):3d}  "
                      f"{' '.join(f'{c:.2f}' for c in confs)}")
                if dets:
                    best.setdefault(pr, (thr, dets, hw))
        _ = best

        # Render the widest sweep for the most specific prompt plus the multi-object prompt:
        # a picture is the only way to tell "found it, ranked low" from "never found it".
        panels = []
        for pr in ("green bok choy .", "bok choy . corn . apple . avocado . grapes . banana ."):
            dets, hw = g.detect(path, pr, box_thr=0.05)
            vis = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3))
            for d in sorted(dets, key=lambda x: -x["conf"])[:12]:
                x0, y0, x1, y1 = [int(v / 3) for v in d["xyxy"]]
                cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 220, 0), 1)
                cv2.putText(vis, f"{d['phrase'][:14]} {d['conf']:.2f}", (x0, max(y0 - 3, 9)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, f"{ep[-6:]} {pr[:34]} thr0.05 n={len(dets)}",
                        (5, vis.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        (0, 255, 255), 1, cv2.LINE_AA)
            panels.append(vis)
        out = f"{REPO}/metrics/sim2real_out/r2s/det_{ep}.png"
        cv2.imwrite(out, cv2.cvtColor(np.concatenate(panels, 1), cv2.COLOR_RGB2BGR))
        print(f"-> {out}")


if __name__ == "__main__":
    main()
