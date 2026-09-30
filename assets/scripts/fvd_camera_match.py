#!/usr/bin/env python3
"""Which tile of model A is the same physical camera as which tile of model B?

A shared real pool keyed on (embodiment, camera, regime) needs camera IDENTITY, and the
stored names cannot supply it: humanoid Ctrl-World annotations carry no `view_keys`, so
the inference script fell back to the bimanual defaults and metrics.json reports
`cam_high/cam_left_wrist/cam_right_wrist` for cameras that are not those at all.

So identity is recovered from the data. For every pair of models in a cell this computes
the full FVD matrix between their GT tiles, on the SAME episodes at a common fps, and
solves the optimal 1-1 assignment. The matching is trustworthy only if it is unambiguous:
the assigned cost must be far below the alternatives (margin), and at/below the
within-model split-half floor.

Output feeds `fvd_camera_map.json`, which the extractor uses to name tiles consistently.

Usage:
  models/dreamgen/.venv/bin/python assets/scripts/fvd_camera_match.py --clips 40
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import numpy as np                                                       # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'assets' / 'scripts'))
sys.path.insert(0, str(REPO / 'scripts'))

from export_dr_grid_xlsx import model_label                              # noqa: E402
from export_dr_grid_xlsx_vs_real import frechet, ref_identity            # noqa: E402
from fvd_pool_common import (TARGET_FPS, ep_key, i3d_features,           # noqa: E402
                            moments, resample_fps, split_mosaic)

CELLS = [('bimanual', 'makovian'), ('bimanual', 'non_makovian'),
         ('humanoid', 'makovian'), ('humanoid', 'non_makovian'),
         ('single_arm', 'makovian'), ('single_arm', 'non_makovian')]


def probe_fps(path: str) -> float:
    try:
        o = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
             'stream=r_frame_rate', '-of', 'json', path],
            capture_output=True, text=True, timeout=60).stdout
        n, d = json.loads(o)['streams'][0]['r_frame_rate'].split('/')
        return float(n) / float(d)
    except Exception:                                                # noqa: BLE001
        return float('nan')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', type=int, default=40)
    ap.add_argument('--cells', default='',
                    help='comma list of emb/split to restrict to, e.g. humanoid/makovian')
    ap.add_argument('--out', default='metrics/reports/fvd_camera_map.json')
    args = ap.parse_args()

    import torch                                                    # noqa: F401
    from scipy.optimize import linear_sum_assignment
    from compute_dr_metrics import get_models, read_video

    run = REPO / 'metrics/WorldScore/results/dr_full'
    man = json.loads((run / 'manifest.json').read_text())
    rows = man['rows'] if isinstance(man, dict) else man

    by_cell = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        if r.get('ref_kind') != 'vs_gt' or 'multiview' not in r.get('relative', ''):
            continue
        lab = model_label(r.get('relative', ''))
        k = ep_key(r['ref'])
        if lab is None or k is None:
            continue
        by_cell[(r['embodiment'], r['split'])][lab].setdefault(k, r['ref'])

    m = get_models('cuda')
    report = {}

    want = {c.strip() for c in args.cells.split(',') if c.strip()}
    for cell in CELLS:
        if want and f'{cell[0]}/{cell[1]}' not in want:
            continue
        per_model = by_cell.get(cell)
        if not per_model or len(per_model) < 2:
            print(f'=== {cell}: only {len(per_model or {})} model(s), skip ===')
            continue
        common = None
        for lab in per_model:
            s = set(per_model[lab])
            common = s if common is None else (common & s)
        common = sorted(common or [])[:args.clips]
        fps = {lab: probe_fps(next(iter(per_model[lab].values()))) for lab in per_model}
        print(f'\n=== {cell[0]}/{cell[1]} — {len(per_model)} models, '
              f'{len(common)} shared episodes ===')
        for lab in sorted(per_model):
            print(f'    {lab:16s} {len(per_model[lab]):4d} eps  fps={fps[lab]:.0f}')
        if not common:
            print('    !! no shared episodes — cannot match cameras')
            report['|'.join(cell)] = dict(error='no shared episodes')
            continue

        feats = collections.defaultdict(list)          # (model, tile_idx) -> features
        bad = collections.Counter()
        for lab in sorted(per_model):
            for k in common:
                try:
                    vid = read_video(Path(per_model[lab][k]))
                except Exception as e:                               # noqa: BLE001
                    bad[f'{lab}:{type(e).__name__}'] += 1
                    continue
                tiles = split_mosaic(vid)
                if tiles is None:
                    print(f'    !! unknown mosaic {vid.shape} for {lab}')
                    break
                for ti, t in enumerate(tiles):
                    t = resample_fps(t, fps[lab], TARGET_FPS)
                    feats[(lab, ti)].extend(i3d_features(t, m['i3d'], 'cuda'))
        if bad:
            print(f'    unreadable clips: {dict(bad)}')
        wc = {f'{k[0]}|t{k[1]}': len(v) for k, v in sorted(feats.items())}
        print(f'    windows/tile: {wc}')
        thin = [k for k, v in feats.items() if len(v) < 64]
        if thin:
            print(f'    !! {len(thin)} tile(s) under the 64-window guard: '
                  f'{[f"{a}|t{b}" for a, b in thin]} — raise --clips for this cell')

        # split-half floor per (model, tile): the yardstick for "same distribution"
        rng = np.random.default_rng(0)
        floor = {}
        for key, v in feats.items():
            x = np.asarray(v, dtype=np.float64)
            if len(x) < 128:
                continue
            idx = rng.permutation(len(x))
            h = len(x) // 2
            floor[key] = frechet(*moments(x[idx[:h]]), *moments(x[idx[h:2 * h]]))

        models = sorted({k[0] for k in feats})
        cell_rec = dict(models=models, episodes=len(common),
                        fps={k: fps[k] for k in fps}, pairs=[],
                        floor={f'{k[0]}|{k[1]}': v for k, v in floor.items()})
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                A = sorted({k[1] for k in feats if k[0] == models[i]})
                B = sorted({k[1] for k in feats if k[0] == models[j]})
                A = [a for a in A if len(feats[(models[i], a)]) >= 64]
                B = [b for b in B if len(feats[(models[j], b)]) >= 64]
                if not A or not B:
                    continue
                M = np.full((len(A), len(B)), np.nan)
                for ia, a in enumerate(A):
                    for ib, b in enumerate(B):
                        M[ia, ib] = frechet(*moments(feats[(models[i], a)]),
                                            *moments(feats[(models[j], b)]))
                print(f'\n    FVD matrix  {models[i]} (rows) x {models[j]} (cols)')
                print('        ' + ''.join(f'{b:>9d}' for b in B))
                for ia, a in enumerate(A):
                    print(f'    t{a:<3d}' + ''.join(f'{M[ia, ib]:9.1f}'
                                                    for ib in range(len(B))))
                # rectangular assignment handles 3-vs-4 tiles: the extra tile is left out
                ri, ci = linear_sum_assignment(M)
                assign, margins = [], []
                for a_i, b_i in zip(ri, ci):
                    row = np.delete(M[a_i], b_i)
                    col = np.delete(M[:, b_i], a_i)
                    alt = np.min(np.concatenate([row, col])) if (len(row) or len(col)) \
                        else np.inf
                    marg = alt / M[a_i, b_i] if M[a_i, b_i] > 0 else np.inf
                    fl = max(floor.get((models[i], A[a_i]), 0),
                             floor.get((models[j], B[b_i]), 0))
                    assign.append(dict(tile_a=int(A[a_i]), tile_b=int(B[b_i]),
                                       fvd=float(M[a_i, b_i]), alt_best=float(alt),
                                       margin=float(marg), floor=float(fl),
                                       ratio_to_floor=float(M[a_i, b_i] / fl)
                                       if fl > 0 else None))
                    margins.append(marg)
                    print(f'      match t{A[a_i]} <-> t{B[b_i]}: '
                          f'FVD={M[a_i, b_i]:7.1f}  next-best={alt:7.1f}  '
                          f'margin={marg:5.1f}x  floor={fl:6.1f}')
                worst = min(margins) if margins else 0.0
                verdict = ('UNAMBIGUOUS' if worst >= 3 else
                           'WEAK' if worst >= 1.5 else 'AMBIGUOUS')
                print(f'      -> worst margin {worst:.1f}x = {verdict}')
                cell_rec['pairs'].append(dict(a=models[i], b=models[j],
                                              assign=assign, worst_margin=float(worst),
                                              verdict=verdict))
        report['|'.join(cell)] = cell_rec
        feats.clear()

    p = REPO / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2))
    print(f'\nwrote {p}')


if __name__ == '__main__':
    main()
