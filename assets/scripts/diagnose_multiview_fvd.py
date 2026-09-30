#!/usr/bin/env python3
"""Why is multiview FVD so much larger than singleview FVD?

Two candidate explanations, and they are separable from the data already on disk:

  A. MIXTURE REFERENCE. The pooled real set is keyed by (embodiment, view, split) and so
     merges the GT of every model in that group. For singleview that is harmless -- every
     model renders the same single camera. For multiview it is not: DreamGen writes a 2x2
     grid while Ctrl-World writes its own view stack, at a different resolution. Pooling
     them makes the reference a MIXTURE of two visually different layouts, and no single
     model can sit close to the mean of a bimodal set.

  B. SMALL-SAMPLE BIAS. FVD is biased upward when the covariance is estimated from few
     windows, and the multiview groups have far fewer windows than the singleview ones.

The test that separates them: compute FVD between the REAL sets of two different models
inside the SAME group. Real video against real video should be near the noise floor. If
it is instead huge for multiview and small for singleview, the reference is a mixture (A)
and the multiview numbers are measuring layout difference, not generation quality.

Usage:
  models/dreamgen/.venv/bin/python assets/scripts/diagnose_multiview_fvd.py
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'assets' / 'scripts'))

from export_dr_grid_xlsx import FvdCell, model_label                    # noqa: E402
from export_dr_grid_xlsx_vs_real import frechet, ref_identity, view_of  # noqa: E402

MIN_WIN = 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='metrics/WorldScore/results/dr_full')
    ap.add_argument('--json', default='metrics/reports/multiview_fvd_diagnosis.json')
    args = ap.parse_args()
    run = REPO / args.run

    # real features, split by (group, contributing model), deduplicated by GT identity
    per_model = collections.defaultdict(FvdCell)
    pooled = collections.defaultdict(FvdCell)
    seen = set()
    geom = collections.defaultdict(collections.Counter)

    n = 0
    with os.scandir(run / 'results') as it:
        for e in it:
            if not e.name.endswith('.json'):
                continue
            try:
                rec = json.loads(Path(e.path).read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if rec.get('ref_kind') != 'vs_gt' or rec.get('skipped'):
                continue
            if not rec.get('i3d_ref'):
                continue
            rel = rec.get('relative', '')
            label, view = model_label(rel), view_of(rel)
            if label is None or view is None:
                continue
            g = (rec['embodiment'], view, rec['split'])
            ident = ref_identity(rec['ref'])
            if (g, ident) in seen:
                continue
            seen.add((g, ident))
            n += 1
            per_model[(g, label)].add('ref', rec['i3d_ref'])
            pooled[g].add('ref', rec['i3d_ref'])
            wh = rec.get('ref_wh') or rec.get('wh') or rec.get('resolution')
            if wh:
                geom[(g, label)][str(wh)] += 1

    print(f'{n} distinct real videos\n')
    print('=== real-vs-real FVD BETWEEN models inside the same group ===')
    print('(real video against real video: should be near the noise floor;')
    print(' a large value means the pooled reference is a MIXTURE of layouts)\n')
    print(f'{"group":40s} {"model A":16s} {"model B":16s} {"win A":>6s} {"win B":>6s} '
          f'{"FVD(A,B)":>9s}')
    rows = []
    for g in sorted(pooled):
        labels = sorted({lbl for (gg, lbl) in per_model if gg == g})
        mom = {}
        for lbl in labels:
            acc = per_model[(g, lbl)]
            if acc.n['ref'] >= MIN_WIN:
                mom[lbl] = (acc.moments('ref'), acc.n['ref'])
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                if a not in mom or b not in mom:
                    continue
                (ma, ca), na = mom[a]
                (mb, cb), nb = mom[b]
                f = frechet(ma, ca, mb, cb)
                print(f'{"|".join(g):40s} {a:16s} {b:16s} {na:6d} {nb:6d} {f:9.1f}')
                rows.append(dict(group='|'.join(g), model_a=a, model_b=b,
                                 win_a=na, win_b=nb, fvd_real_vs_real=f,
                                 view=g[1]))

    mv = [r['fvd_real_vs_real'] for r in rows if r['view'] == 'multiview']
    sv = [r['fvd_real_vs_real'] for r in rows if r['view'] == 'singleview']
    print()
    if mv:
        print(f'  multiview  real-vs-real FVD: min {min(mv):.1f}  '
              f'median {sorted(mv)[len(mv)//2]:.1f}  max {max(mv):.1f}   (n={len(mv)})')
    if sv:
        print(f'  singleview real-vs-real FVD: min {min(sv):.1f}  '
              f'median {sorted(sv)[len(sv)//2]:.1f}  max {max(sv):.1f}   (n={len(sv)})')

    print('\n=== window counts per group (small-sample bias check) ===')
    for g in sorted(pooled):
        print(f'  {"|".join(g):40s} windows={pooled[g].n["ref"]:6d}  '
              f'models={sorted({l for (gg,l) in per_model if gg==g})}')

    out = REPO / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        n_real_videos=n,
        real_vs_real=rows,
        windows={'|'.join(g): pooled[g].n['ref'] for g in sorted(pooled)},
    ), indent=2))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
