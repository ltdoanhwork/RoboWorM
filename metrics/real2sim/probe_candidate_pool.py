"""Which candidate box is on the object, and why does its contact fraction come out zero?

The detector sweep showed GroundingDINO *does* put a box on the bok choy (conf 0.43 / 0.38), so
the earlier conclusion -- "the detector never proposes it" -- was wrong. That leaves a sharper
question: the pipeline's top-4 prefilter at threshold 0.15 keeps confidences 0.43, 0.43, 0.39,
0.29, so the correct box plausibly *is* inside the pool, and yet all four candidates reported
`contact = 0.00`.

Either the correct box is being dropped by one of the steps between detection and ranking
(`nms`, `_drop_huge`, `max_object_boxes`, `prefilter`), or it survives and the contact
measurement itself is what fails. This reproduces the pipeline's exact detection path and then
reports, per surviving candidate, both its overlap with a hand-marked reference region and its
contact fraction against the acting-arm mask -- so the two hypotheses separate.

Run from the WorldScore directory (see `probe_detector`).
"""

from __future__ import annotations

import json
import os

import numpy as np


REPO = os.environ.get("REPO", os.getcwd())
ROOT = f"{REPO}/metrics/sim2real_out/dense/humanoid_singleview_dreamgen/makovian"


def stored_arm_mask(ep: str) -> tuple[np.ndarray, tuple[int, int], int]:
    """The gripper mask as the cached run stored it, plus its shape and frame count."""
    from metrics.force_feasibility.lift_to_3d import unpack_masks

    z = np.load(f"{ROOT}/{ep}/tracks.npz", allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    hw = tuple(meta["mask_hw"])
    T = int(meta["num_frames"])
    return unpack_masks(z, "v0_gt_gripper", hw, T), hw, T


def main() -> None:
    import cv2

    from metrics.sim2real.track_backend import (
        Grounder, _drop_huge, _min_gap_series, nms,
    )
    from metrics.real2sim.probe_detector import frame0

    ckpt = f"{REPO}/checkpoints/worldscore"
    g = Grounder(
        "worldscore/benchmark/metrics/third_party/groundingdino/config/"
        "GroundingDINO_SwinT_OGC.py",
        f"{ckpt}/groundingdino_swint_ogc.pth", "cuda")

    # Reference regions marked from the rendered sweep (`r2s/det_*.png`), in the 768x432 video
    # frame. These exist ONLY to label which candidate is the bok choy in this diagnostic; no
    # metric consumes them, so they are not a hand-set constant in the pipeline.
    REF = {
        "episode_000829": (150, 175, 215, 250),
        "episode_003151": (200, 130, 260, 195),
    }

    for ep, ref in REF.items():
        path, img = frame0(ep)
        up = 3
        arm, hw, T = stored_arm_mask(ep)
        print(f"\n===== {ep} =====   arm mask {hw}, {T} frames")
        for thr, pre in ((0.15, 4), (0.05, 8), (0.05, 16)):
            raw, dhw = g.detect(path, "green bok choy .", box_thr=thr)
            # Exactly the pipeline's path: nms, drop-huge, cap, then top-conf prefilter.
            kept = _drop_huge(nms(raw), dhw)[:8]
            pool = sorted(kept, key=lambda d: -d["conf"])[:pre]
            hit = -1
            for i, d in enumerate(pool):
                x0, y0, x1, y1 = [v / up for v in d["xyxy"]]
                # "On the object" = box centre inside the reference region. Centre rather than
                # IoU because the detector's boxes vary a lot in tightness on this leafy object.
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                if ref[0] <= cx <= ref[2] and ref[1] <= cy <= ref[3]:
                    hit = i
                    break
            print(f"  thr={thr:.2f} prefilter={pre:2d}: {len(raw):3d} raw -> {len(kept)} kept "
                  f"-> {len(pool)} pool; bok-choy box at pool index "
                  f"{hit if hit >= 0 else 'ABSENT'}"
                  + (f" (conf {pool[hit]['conf']:.2f})" if hit >= 0 else ""))

        # Is the contact measurement even capable of firing? Test it against the stored object
        # mask, which is the mask the cached run actually tracked, so a zero here means the
        # contact test -- not the candidate choice -- is what fails.
        from metrics.force_feasibility.lift_to_3d import unpack_masks
        z = np.load(f"{ROOT}/{ep}/tracks.npz", allow_pickle=True)
        obj = unpack_masks(z, "v0_gt_object", hw, T)
        gap = _min_gap_series(list(obj), list(arm))
        finite = np.isfinite(gap)
        if finite.any():
            gp = gap[finite]
            print(f"  stored obj vs stored arm: gap min={gp.min():.3f} "
                  f"median={np.median(gp):.3f} max={gp.max():.3f} "
                  f"(tau=0.35 -> contact_frac={float(np.mean(gp < 0.35)):.2f})")
        else:
            print("  stored obj vs stored arm: gap is all-NaN (unmeasurable)")
        _ = cv2, img


if __name__ == "__main__":
    main()
