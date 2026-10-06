#!/usr/bin/env python3
"""Is the shared-reference FVD measuring quality, or residual framing mismatch?

After pooling per camera at 5 fps, dense DreamGen MV scores 232.6 on bimanual/makovian
while dense Ctrl-World MV scores 39.3. That gap must be justified before it goes in a
paper, because there are two candidate explanations and they demand opposite conclusions:

  (a) real quality difference   -> the number is publishable as-is
  (b) residual framing mismatch -> DreamGen's own GT sits further from the shared pool
                                   than Ctrl-World's does, and the "gap" is an artefact

The discriminator is each model's OWN real videos against the shared pool. A model's GT is
real video, so if the protocol were clean, every model's GT would score ~= the pool's
split-half floor. Any excess is that model's framing penalty, paid by its predictions too.

Reports per (embodiment, camera, split):
  floor        split-half noise of the shared pool
  gt_<model>   that model's real videos vs the shared pool  <-- the framing penalty
  dense_<model> that model's dense predictions vs the shared pool

Verdict: if gt_X >> floor for one model only, its dense FVD is inflated by that amount and
the cross-model comparison is not yet clean.

Usage:
  models/dreamgen/.venv/bin/python assets/scripts/verify_shared_pool_fairness.py
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import numpy as np                                                       # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'assets' / 'scripts'))

from export_dr_grid_xlsx import METHOD_LABEL                             # noqa: E402
from export_dr_grid_xlsx_vs_real import frechet                          # noqa: E402
from fvd_pool_common import moments                                      # noqa: E402
from pool_perview_fvd import RESTRICT_CAMERAS, TILE_CAMERA               # noqa: E402

MIN_W = 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', default='metrics/WorldScore/results/fvd_perview')
    ap.add_argument('--json', default='results/reports/shared_pool_fairness.json')
    args = ap.parse_args()

    gt = collections.defaultdict(list)       # (emb,cam,split,label) -> real feats
    dense = collections.defaultdict(list)    # (emb,cam,split,label) -> dense pred feats
    seen = set()

    files = sorted((REPO / args.feats / 'feats').glob('*.npz'))
    print(f'reading {len(files)} files', flush=True)
    for i, f in enumerate(files, 1):
        try:
            z = np.load(f, allow_pickle=False)
            meta = json.loads(str(z['meta']))
        except Exception:                                            # noqa: BLE001
            continue
        label, emb, split = meta['label'], meta['embodiment'], meta['split']
        cam_of = TILE_CAMERA.get(label)
        if cam_of is None:
            continue
        keep = RESTRICT_CAMERAS.get((emb, split))
        is_dense = METHOD_LABEL.get(meta['method']) == 'Dense'
        for k in z.files:
            if k == 'meta':
                continue
            tag, ti = k.rsplit('_', 1)
            cam = cam_of.get(int(ti))
            if cam is None or (keep is not None and cam not in keep):
                continue
            arr = z[k]
            if not len(arr):
                continue
            key = (emb, cam, split, label)
            if tag == 'ref':
                rk = (key, meta['ref_identity'])
                if rk in seen:
                    continue
                seen.add(rk)
                gt[key].append(arr)
            elif is_dense:
                dense[key].append(arr)
        if i % 4000 == 0:
            print(f'  {i}/{len(files)}', flush=True)

    gt = {k: np.concatenate(v).astype(np.float64) for k, v in gt.items() if v}
    dense = {k: np.concatenate(v).astype(np.float64) for k, v in dense.items() if v}

    # shared pool per (emb,cam,split) = union of all models' GT, which is what
    # pool_perview_fvd.py uses as the reference
    pool = collections.defaultdict(list)
    for (emb, cam, split, _lab), v in gt.items():
        pool[(emb, cam, split)].append(v)
    pool = {k: np.concatenate(v) for k, v in pool.items()}

    rng = np.random.default_rng(0)
    rows = []
    print(f'\n{"group":30s} {"floor":>7s} | per-model  gt_vs_pool -> dense_vs_pool')
    for g in sorted(pool):
        P = pool[g]
        if len(P) < 2 * MIN_W:
            continue
        idx = rng.permutation(len(P))
        h = len(P) // 2
        floor = frechet(*moments(P[idx[:h]]), *moments(P[idx[h:2 * h]]))
        mP = moments(P)
        line = f'{"|".join(g):30s} {floor:7.1f} |'
        rec = dict(group='|'.join(g), floor=floor, n_pool=len(P), models={})
        for lab in sorted({k[3] for k in gt if k[:3] == g}):
            G = gt.get((*g, lab))
            D = dense.get((*g, lab))
            gv = frechet(*moments(G), *mP) if G is not None and len(G) >= MIN_W else None
            dv = frechet(*moments(D), *mP) if D is not None and len(D) >= MIN_W else None
            rec['models'][lab] = dict(
                gt_vs_pool=gv, dense_vs_pool=dv,
                n_gt=int(len(G)) if G is not None else 0,
                n_dense=int(len(D)) if D is not None else 0,
                gt_over_floor=(gv / floor) if (gv and floor > 0) else None,
                dense_minus_gt=(dv - gv) if (gv is not None and dv is not None) else None)
            line += (f'  {lab}: gt={gv:6.1f}' if gv is not None else f'  {lab}: gt=  n/a')
            line += (f'->dns={dv:7.1f}' if dv is not None else '->dns=    n/a')
        print(line)
        rows.append(rec)

    print('\n=== framing penalty (a model GT should sit AT the floor) ===')
    worst = []
    for r in rows:
        for lab, v in r['models'].items():
            if v['gt_over_floor'] is not None:
                worst.append((v['gt_over_floor'], r['group'], lab,
                              v['gt_vs_pool'], r['floor'], v['dense_vs_pool']))
    worst.sort(reverse=True)
    for ratio, g, lab, gv, fl, dv in worst[:12]:
        flag = 'ARTEFACT RISK' if ratio > 3 else 'ok'
        print(f'  {g:28s} {lab:16s} gt/floor={ratio:6.1f}x '
              f'(gt={gv:7.1f} floor={fl:6.1f} dense={dv if dv is None else round(dv, 1)})'
              f'  {flag}')
    clean = sum(1 for w in worst if w[0] <= 3)
    print(f'\n  {clean}/{len(worst)} (group,model) pairs have GT within 3x the floor')

    p = REPO / args.json
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(min_windows=MIN_W, groups=rows), indent=2))
    print(f'wrote {p}')


if __name__ == '__main__':
    main()
