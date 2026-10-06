#!/usr/bin/env python3
"""Measure -- not assume -- how a DreamGen prediction maps onto its source episode.

Six grid cells have predictions but no ground truth, so GT has to be rendered from the
LeRobot source video. Rendering needs two unknowns: the spatial transform and the
TEMPORAL map from prediction frame index to source frame index. The batch_input_*.json
files that would have recorded this are stale (several point at /tmp or at an older
checkpoint dir), and the chunking formula in prepare_singleview_dreamgen_batch.py
disagrees with the predictions that actually exist -- it predicts 6 chunks for a
bimanual episode whose prediction is 10 chunks long. So neither source is trustworthy.

This probe recovers the map from the pixels instead. For a handful of prediction frames
it searches EVERY source frame for the best match, which yields (pred_idx, src_idx)
pairs with no prior assumption about the rate. A least-squares line through the origin
then gives the rate, and its residual says whether a single linear rate explains the
data at all.

Why this is trustworthy for DreamGen specifically: prediction frame 0 is the
conditioning frame, copied from source frame 0 and merely rescaled. So the probe has a
built-in control -- if the recovered map does not send pred 0 to src 0 with a high
PSNR, the cell's pairing is wrong and must not be scored. That is the same signal that
caught 33 mislabelled hallucinate clips (frame 0 matched, frame 46 did not).

Usage:
    python scripts/metrics/gt_align_probe.py --cell dreamgen_humanoid_sv --episodes 3
    python scripts/metrics/gt_align_probe.py --all --episodes 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gt_cells import CELLS, list_clips  # noqa: E402


def decode_small(path: Path, size=(96, 54), limit: int | None = None) -> np.ndarray:
    """All frames as small float32 greyscale -- enough to rank matches, cheap to scan."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'cannot open {path}')
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.append(cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.float32))
        if limit is not None and len(out) >= limit:
            break
    cap.release()
    if not out:
        raise RuntimeError(f'no frames in {path}')
    return np.stack(out)


def probe_dims(path: Path) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'cannot open {path}')
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return n, w, h, fps


def best_match(pred_frame: np.ndarray, src_stack: np.ndarray) -> tuple[int, float]:
    """Source index minimising MSE against one prediction frame, plus its PSNR."""
    diff = src_stack - pred_frame[None]
    mse = (diff * diff).mean(axis=(1, 2))
    j = int(np.argmin(mse))
    m = float(mse[j])
    psnr = 99.0 if m <= 1e-9 else 10.0 * float(np.log10(255.0 * 255.0 / m))
    return j, psnr


def fit_rate(pairs: list[tuple[int, int]]) -> tuple[float, float]:
    """Least-squares rate through the origin, and the worst residual in source frames."""
    pts = [(i, j) for i, j in pairs if i > 0]
    if not pts:
        return float('nan'), float('nan')
    num = sum(i * j for i, j in pts)
    den = sum(i * i for i, _ in pts)
    rate = num / den if den else float('nan')
    resid = max(abs(j - rate * i) for i, j in pts)
    return rate, resid


def probe_clip(cell: dict, clip: dict, n_probe: int = 6) -> dict:
    pred = clip['pred']
    n_pred, w_pred, h_pred, fps_pred = probe_dims(pred)
    src_small, n_src, w_src, h_src, fps_src = cell['load_probe'](clip)

    pred_small = decode_small(pred)
    n_pred = len(pred_small)

    idxs = sorted({int(round(k * (n_pred - 1) / (n_probe - 1))) for k in range(n_probe)})
    pairs, rows = [], []
    for i in idxs:
        j, psnr = best_match(pred_small[i], src_small)
        pairs.append((i, j))
        rows.append({'pred_idx': i, 'src_idx': j, 'psnr': round(psnr, 2)})

    rate, resid = fit_rate(pairs)
    _, psnr0 = best_match(pred_small[0], src_small[:1])   # the control: pred 0 vs src 0

    return {
        'clip': clip['key'],
        'n_pred': n_pred, 'pred_wh': [w_pred, h_pred], 'pred_fps': round(fps_pred, 3),
        'n_src': len(src_small), 'src_wh': [w_src, h_src], 'src_fps': round(fps_src, 3),
        'len_ratio': round(len(src_small) / n_pred, 4),
        'fitted_rate': None if np.isnan(rate) else round(rate, 4),
        'max_residual_frames': None if np.isnan(resid) else round(resid, 2),
        'psnr_pred0_vs_src0': round(psnr0, 2),
        'matches': rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', action='append', default=[], choices=sorted(CELLS))
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--split', default='makovian')
    ap.add_argument('--probe-frames', type=int, default=6)
    ap.add_argument('--out', default='results/reports/gt_align_probe.json')
    args = ap.parse_args()

    names = sorted(CELLS) if args.all else args.cell
    if not names:
        ap.error('pass --cell NAME (repeatable) or --all')

    report = {}
    for name in names:
        cell = CELLS[name]
        clips = list_clips(cell, args.split, method='dense')
        print(f'\n=== {name} / {args.split}: {len(clips)} dense clips', flush=True)
        if not clips:
            report[name] = {'error': 'no clips'}
            continue
        step = max(1, len(clips) // max(1, args.episodes))
        chosen = clips[::step][:args.episodes]
        results = []
        for clip in chosen:
            try:
                r = probe_clip(cell, clip, args.probe_frames)
            except Exception as exc:                      # noqa: BLE001
                r = {'clip': clip['key'], 'error': f'{type(exc).__name__}: {exc}'}
                print(f'  {clip["key"]}: {r["error"]}', flush=True)
                results.append(r)
                continue
            results.append(r)
            print(f'  {r["clip"]}', flush=True)
            print(f'    pred {r["n_pred"]:5d}f {r["pred_wh"]} @{r["pred_fps"]}  '
                  f'src {r["n_src"]:5d}f {r["src_wh"]} @{r["src_fps"]}  '
                  f'len_ratio={r["len_ratio"]}', flush=True)
            print(f'    fitted_rate={r["fitted_rate"]}  max_resid={r["max_residual_frames"]}f  '
                  f'PSNR(pred0,src0)={r["psnr_pred0_vs_src0"]} dB', flush=True)
            print('    matches (pred_idx -> src_idx @ psnr): '
                  + '  '.join(f'{m["pred_idx"]}->{m["src_idx"]}@{m["psnr"]}' for m in r['matches']),
                  flush=True)
        report[name] = results

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f'\nwrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
