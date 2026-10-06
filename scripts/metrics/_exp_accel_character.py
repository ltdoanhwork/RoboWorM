"""E7b - Which synthetic artifact do the REAL accel trees actually resemble?

E7 showed AMC penalises frame-duplication (dup2, -39) and blocking (-4.1) but is
FOOLED by spatial blur (+3.6) and temporal smoothing (+7.8). Whether that matters
depends on which artifact each accel family really produces. Measured in pixel space,
no GPU, on paired dense/accel clips:

  sharpness  = variance of Laplacian   -> lower than dense means blur-like
  temporal   = mean |frame_t - frame_t-1| -> lower than dense means smoothing/reuse-like
  dup_rate   = fraction of adjacent frames that are near-identical -> dup2-like

Usage:
  python scripts/metrics/_exp_accel_character.py --methods dicache worldcache sito pisa --n 24
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def stats(path, max_frames=40):
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(frames) < 4:
        return None
    a = np.stack(frames).astype(np.float32)
    sharp = float(np.mean([cv2.Laplacian(f, cv2.CV_64F).var() for f in frames]))
    d = np.abs(np.diff(a, axis=0))
    per_pair = d.reshape(len(d), -1).mean(axis=1)
    return dict(sharp=sharp, temporal=float(per_pair.mean()),
                dup_rate=float((per_pair < 0.5).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+',
                    default=['dicache', 'worldcache', 'fastercache', 'sito', 'itm',
                             'pisa', 'radial', 'svg1', 'svg2'])
    ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--out', type=Path,
                    default=ROOT / 'metrics/WorldScore/results/exp_accel_character.json')
    args = ap.parse_args()

    dense_root = ROOT / 'sampling_dataset/dense'
    # Use the clips the v3.2 run already scored, so the sample is the same population
    # the NR numbers come from.
    run = ROOT / 'metrics/WorldScore/results/dense_nr_v3_div30/results'
    seen, dense_clips = set(), []
    for p in sorted(run.glob('*.json')):
        r = json.loads(p.read_text())
        if r['method'] != 'dense' or r['model'].startswith('gt'):
            continue
        if r['path'] in seen:
            continue
        seen.add(r['path'])
        dense_clips.append(r)

    summary = {}
    for method in args.methods:
        tree = ROOT / 'sampling_dataset' / method
        if not tree.exists():
            print(f'{method}: tree missing, skipped')
            continue
        rows = []
        for r in dense_clips:
            rel = Path(r['path']).relative_to(dense_root)
            cand = tree / rel
            if not cand.exists():
                continue
            a, b = stats(r['path']), stats(cand)
            if not a or not b:
                continue
            rows.append((a, b))
            if len(rows) >= args.n:
                break
        if len(rows) < 4:
            print(f'{method}: only {len(rows)} pairs, skipped')
            continue
        ds = np.median([b['sharp'] - a['sharp'] for a, b in rows])
        dt = np.median([b['temporal'] - a['temporal'] for a, b in rows])
        dd = np.median([b['dup_rate'] - a['dup_rate'] for a, b in rows])
        rs = np.median([b['sharp'] / max(a['sharp'], 1e-6) for a, b in rows])
        rt = np.median([b['temporal'] / max(a['temporal'], 1e-6) for a, b in rows])
        summary[method] = dict(n=len(rows), d_sharp=float(ds), d_temporal=float(dt),
                               d_dup_rate=float(dd), ratio_sharp=float(rs),
                               ratio_temporal=float(rt))

    print('\nPaired (accel - dense) in pixel space. Negative sharpness = blur-like;')
    print('negative temporal = smoothing/reuse-like; positive dup_rate = duplication.\n')
    print('  %-13s %4s %11s %11s %11s %10s %10s' % (
        'method', 'n', 'd_sharp', 'd_temporal', 'd_dup_rate', 'sharp_x', 'temporal_x'))
    for m, s in summary.items():
        print('  %-13s %4d %+11.2f %+11.4f %+11.4f %10.3f %10.3f' % (
            m, s['n'], s['d_sharp'], s['d_temporal'], s['d_dup_rate'],
            s['ratio_sharp'], s['ratio_temporal']))

    print('\nRisk to AMC (which is fooled by blur AND by temporal smoothing):')
    for m, s in summary.items():
        risk = []
        if s['ratio_sharp'] < 0.95:
            risk.append('blur-like')
        if s['ratio_temporal'] < 0.95:
            risk.append('smoothing-like')
        if s['d_dup_rate'] > 0.02:
            risk.append('duplication (AMC handles this correctly)')
        print(f'  {m:<13} ' + (', '.join(risk) if risk else 'no blur/smoothing signature'))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
