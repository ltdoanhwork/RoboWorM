"""Visualize object_control: draw GroundingDINO boxes on sampled frames so the
detection-rate numbers are inspectable (which frames lose the arm / the box).

Emits, per clip, a few annotated panels chosen to cover the interesting cases:
best (both objects found), partial (one found), worst (nothing found).
"""
import os
import sys
import json
import argparse

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quality_control_probe import Detector, decode  # noqa: E402


def annotate(img_bgr, boxes, labels, colors):
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()
    for (cx, cy, bw, bh), lab, col in zip(boxes, labels, colors):
        x0, y0 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x1, y1 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 2)
        cv2.putText(out, lab, (x0 + 2, max(14, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return out


@torch.no_grad()
def boxes_and_phrases(det, path, caption):
    """Same as Detector.phrases but also returns boxes."""
    import groundingdino.datasets.transforms as T  # noqa: F401
    pil = Image.open(path).convert("RGB")
    t, _ = det.tr(pil, None)
    cap = caption.lower().strip()
    if not cap.endswith("."):
        cap += "."
    out = det.model(t[None].to(det.device), captions=[cap])
    logits = out["pred_logits"].cpu().sigmoid()[0]
    bx = out["pred_boxes"].cpu()[0]
    keep = logits.max(dim=1)[0] > det.box_thr
    logits, bx = logits[keep], bx[keep]
    tok = det.model.tokenizer(cap)
    phr, scr = [], []
    for lg in logits:
        phr.append(det._gp(lg > det.txt_thr, tok, det.model.tokenizer))
        scr.append(float(lg.max().item()))
    return bx.numpy(), phr, scr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--objects", required=True)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--gdino_cfg", required=True)
    ap.add_argument("--gdino_ckpt", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    objects = [o.strip().lower() for o in args.objects.split(",") if o.strip()]
    caption = ", ".join(objects)
    det = Detector(args.gdino_cfg, args.gdino_ckpt, device)
    PALETTE = [(60, 220, 60), (60, 160, 255), (255, 200, 60), (200, 60, 255)]

    manifest = {}
    for spec in args.clips:
        tag, video = spec.split("=", 1)
        work = os.path.join(args.out, f"_{tag}_fr")
        frames = decode(video, work, stride=args.stride)
        print(f"=== {tag}: {len(frames)} frames", flush=True)

        rates = []
        for f in frames:
            r, _ = det.rate(f, objects)
            rates.append(r)
        rates = np.array(rates)

        # pick representative frames: a full hit, a partial, and the worst
        picks = []
        for want, name in ((1.0, "both detected"), (0.5, "one missing"), (0.0, "nothing detected")):
            idx = np.where(np.isclose(rates, want))[0]
            if len(idx):
                picks.append((int(idx[len(idx) // 2]), name, want))

        entries = []
        for i, name, want in picks:
            bx, phr, scr = boxes_and_phrases(det, frames[i], caption)
            img = cv2.imread(frames[i])
            labs = [f"{p} {s:.2f}" for p, s in zip(phr, scr)]
            cols = [PALETTE[k % len(PALETTE)] for k in range(len(labs))]
            ann = annotate(img, bx, labs, cols)
            # letterbox-free downscale for the canvas
            H = 240
            ann = cv2.resize(ann, (int(ann.shape[1] * H / ann.shape[0]), H))
            fn = f"{tag}_{name.replace(' ', '_')}.jpg"
            cv2.imwrite(os.path.join(args.out, fn), ann, [cv2.IMWRITE_JPEG_QUALITY, 88])
            entries.append({"file": fn, "case": name, "rate": want,
                            "frame_eval_idx": i,
                            "pct_of_clip": round(100 * i / max(1, len(frames) - 1), 1),
                            "phrases": phr, "scores": [round(s, 3) for s in scr]})
            print(f"  {name:18s} eval-frame {i:3d} ({entries[-1]['pct_of_clip']}%) -> {phr}", flush=True)

        manifest[tag] = {
            "video": video, "n_eval": len(frames),
            "mean_rate": float(rates.mean()),
            "n_zero": int((rates == 0).sum()),
            "n_partial": int(((rates > 0) & (rates < 1)).sum()),
            "n_full": int((rates == 1).sum()),
            "panels": entries,
        }
        import shutil
        shutil.rmtree(work, ignore_errors=True)

    with open(os.path.join(args.out, "object_control_viz.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("\nWrote", os.path.join(args.out, "object_control_viz.json"))


if __name__ == "__main__":
    main()
