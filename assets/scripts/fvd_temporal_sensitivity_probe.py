#!/usr/bin/env python3
"""Does the Kinetics-400 I3D feature space that FVD lives in actually respond to DYNAMICS?

Companion to fvd_kinetics_domain_audit.py. That script shows the feature space is the
pre-softmax Kinetics-400 class-logit layer and that robot video occupies a ~19-dimensional
corner of it. This script asks the question that matters for a physics benchmark: within
that corner, how strongly does the metric react to destroying temporal structure compared
with how strongly it reacts to a real model difference?

DESIGN -- one variable, matched n, matched clips
    Real GT clips of one cell are split into two disjoint halves A and B.
      floor      = FVD(A, B)                     both real, both unmodified
      scramble   = FVD(A, shuffle_frames(B))     identical pixels, frame order destroyed
      reverse    = FVD(A, reverse(B))            smooth but causally impossible
      generated  = FVD(A, dense predictions)     the difference the benchmark reports
    Every comparison uses the same number of windows per side, so the FVD estimator's
    finite-sample bias (fvd_sample_size_probe.json) is a common offset and the CONTRASTS
    are interpretable even though the absolute values at small n are inflated.

Frames are shuffled WITHIN each 16-frame I3D window, so every window is built from exactly
the same 16 frames as its unmodified counterpart. Appearance, scene, framing, lighting and
the marginal distribution of single frames are therefore held exactly fixed, and the only
thing that changes is the order -- i.e. all motion and all physics.

Reading the result: if `scramble` sits near `floor` while `generated` sits far above it,
the column is ordering appearance/framing, not temporal plausibility, and that is a
limitation of FVD in this benchmark rather than of the models. Generated features are read
from the cache written by extract_perview_fvd_feats.py (identical preprocessing).

Run (CPU is fine, ~2 s per window; use --max-windows to bound the cost):
    models/dreamgen/.venv/bin/python assets/scripts/fvd_temporal_sensitivity_probe.py \
        --embodiment bimanual --split makovian --label 'CW-4view MV' --max-windows 256

Writes results/reports/fvd_temporal_sensitivity.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '32')

import numpy as np                                                       # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'assets' / 'scripts'))
sys.path.insert(0, str(REPO / 'scripts' / 'metrics'))

from export_dr_grid_xlsx import model_label                              # noqa: E402
from export_dr_grid_xlsx_vs_real import ref_identity                     # noqa: E402
from fvd_pool_common import (I3D_SIZE, I3D_WINDOW, TARGET_FPS,           # noqa: E402
                             resample_fps, split_mosaic)

from metrics.fvd import I3D_CKPT                                         # noqa: E402
FEAT_CACHE = REPO / 'metrics/WorldScore/results/fvd_perview/feats'
OUT = REPO / 'results/reports/fvd_temporal_sensitivity.json'


def probe_fps(path: str) -> float:
    import subprocess
    try:
        o = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
             'stream=r_frame_rate', '-of', 'json', path],
            capture_output=True, text=True, timeout=60).stdout
        n, d = json.loads(o)['streams'][0]['r_frame_rate'].split('/')
        return float(n) / float(d)
    except Exception:                                                # noqa: BLE001
        return float('nan')


def sqrtm_psd(a: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh((a + a.T) / 2)
    return (v * np.sqrt(np.clip(w, 0, None))) @ v.T


def fvd(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    dmu = x.mean(0) - y.mean(0)
    cx, cy = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    return float(dmu @ dmu + np.trace(cx) + np.trace(cy)
                 - 2 * np.trace(sqrtm_psd(cx @ cy)))


def windows(tile: np.ndarray) -> list[np.ndarray]:
    """Consecutive non-overlapping 16-frame windows, exactly as the pooling protocol cuts."""
    return [tile[s:s + I3D_WINDOW]
            for s in range(0, len(tile) - I3D_WINDOW + 1, I3D_WINDOW)]


def feats_of(wins: list[np.ndarray], model, batch: int = 8) -> np.ndarray:
    """I3D Kinetics-400 logits for a list of (16,h,w,3) uint8 windows."""
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(wins), batch):
            chunk = wins[i:i + batch]
            t = torch.stack([torch.from_numpy(np.ascontiguousarray(w)) for w in chunk])
            t = t.permute(0, 1, 4, 2, 3).float() / 255           # B,T,3,h,w
            b, tt = t.shape[:2]
            t = torch.nn.functional.interpolate(
                t.reshape(b * tt, *t.shape[2:]), size=(I3D_SIZE, I3D_SIZE),
                mode='bilinear', align_corners=False).reshape(b, tt, 3, I3D_SIZE, I3D_SIZE)
            t = (t * 2 - 1).permute(0, 2, 1, 3, 4)               # B,3,T,H,W
            out.append(model(t, rescale=False, resize=False,
                             return_features=True).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 400))


def cached_dense(embodiment: str, label: str, split: str, cam_idx: int,
                 limit: int) -> np.ndarray:
    """Dense predictions of the same cell from the GPU-extracted cache."""
    acc = []
    for f in sorted(FEAT_CACHE.glob('*.npz')):
        try:
            d = np.load(f, allow_pickle=True)
            m = json.loads(str(d['meta']))
        except Exception:                                        # noqa: BLE001
            continue
        if (m['embodiment'], m.get('label'), m.get('split'), m.get('method')) != \
           (embodiment, label, split, 'dense'):
            continue
        key = f'pred_{cam_idx}'
        if key in d.files:
            acc.append(np.asarray(d[key], np.float64))
        if sum(len(a) for a in acc) >= limit:
            break
    return np.concatenate(acc)[:limit] if acc else np.zeros((0, 400))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--embodiment', default='bimanual')
    ap.add_argument('--split', default='makovian')
    ap.add_argument('--label', default='CW-4view MV')
    ap.add_argument('--camera-index', type=int, default=0,
                    help='tile position in the mosaic (0 = cam_high for both CW layouts)')
    ap.add_argument('--max-windows', type=int, default=256,
                    help='windows PER SIDE; total I3D forwards is ~3x this')
    ap.add_argument('--run', default='metrics/WorldScore/results/dr_full')
    args = ap.parse_args()

    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 32)))
    from compute_dr_metrics import read_video

    man = json.loads((REPO / args.run / 'manifest.json').read_text())
    rows = man['rows'] if isinstance(man, dict) else man

    pick: dict[str, str] = {}
    for r in rows:
        if r.get('ref_kind') != 'vs_gt' or 'multiview' not in r.get('relative', ''):
            continue
        if r.get('embodiment') != args.embodiment or r.get('split') != args.split:
            continue
        if model_label(r.get('relative', '')) != args.label:
            continue
        pick.setdefault(ref_identity(r['ref']), r['ref'])
    clips = [pick[k] for k in sorted(pick)]
    if not clips:
        raise SystemExit(f'no GT clips for {args.embodiment}/{args.split}/{args.label}')
    print(f'{len(clips)} unique GT clips for {args.embodiment}/{args.split}/{args.label}')

    # Interleave so halves A and B draw from the same tasks (episode ids are sorted, and
    # tasks are contiguous in that order; a front/back cut would compare task sets).
    half_a, half_b = clips[0::2], clips[1::2]

    model = torch.jit.load(str(I3D_CKPT)).eval()
    rng = np.random.default_rng(0)

    def collect(paths, variant: str) -> np.ndarray:
        acc, t0 = [], time.time()
        for p in paths:
            vid = read_video(p)
            tiles = split_mosaic(vid)
            if not tiles or args.camera_index >= len(tiles):
                continue
            tile = resample_fps(tiles[args.camera_index], probe_fps(p), TARGET_FPS)
            wins = windows(tile)
            if variant == 'scramble':
                wins = [w[rng.permutation(len(w))] for w in wins]
            elif variant == 'reverse':
                wins = [w[::-1] for w in wins]
            acc.append(feats_of(wins, model))
            n = sum(len(a) for a in acc)
            if n >= args.max_windows:
                break
        f = np.concatenate(acc)[:args.max_windows]
        print(f'  {variant:9s} {len(f):4d} windows in {time.time() - t0:6.1f}s')
        return f

    print('extracting features (CPU)...')
    a_real = collect(half_a, 'real')
    b_real = collect(half_b, 'real')
    b_scram = collect(half_b, 'scramble')
    b_rev = collect(half_b, 'reverse')
    n = min(len(a_real), len(b_real), len(b_scram), len(b_rev))
    gen = cached_dense(args.embodiment, args.label, args.split, args.camera_index, n)

    res = {
        'floor_real_vs_real': fvd(a_real[:n], b_real[:n]),
        'scramble_frames_within_window': fvd(a_real[:n], b_scram[:n]),
        'reverse_time': fvd(a_real[:n], b_rev[:n]),
    }
    if len(gen) >= n // 2:
        res['dense_generated_vs_real'] = fvd(a_real[:n], gen[:min(n, len(gen))])
    floor = res['floor_real_vs_real']

    rep = dict(
        cell=dict(embodiment=args.embodiment, split=args.split, label=args.label,
                  camera_index=args.camera_index),
        n_windows_per_side=int(n), target_fps=TARGET_FPS,
        note='absolute values are inflated by finite-sample bias at this n; the '
             'comparisons are valid because every pair uses the same n and the same '
             'clip partition, so only the stated manipulation differs',
        fvd=res,
        excess_over_floor={k: v - floor for k, v in res.items()},
        ratio_to_floor={k: v / floor if floor else None for k, v in res.items()},
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2))

    print(f'\n=== {args.embodiment}/{args.split}/{args.label} cam{args.camera_index}, '
          f'n={n} windows per side ===')
    for k, v in res.items():
        print(f'  {k:32s} {v:9.1f}   {v / floor:5.2f}x floor')
    print(f'\nwrote {OUT.relative_to(REPO)}')


if __name__ == '__main__':
    main()
