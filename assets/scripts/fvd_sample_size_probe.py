#!/usr/bin/env python3
"""How much of our FVD is sample-size artefact rather than quality?

This decides what the FVD column may claim. FVD is a biased estimator: it is computed from
a 400x400 covariance, so with few windows the covariance is poorly conditioned and the
distance inflates. Our groups are very unequal (182 to 5089 windows), so if FVD moves a
lot with n, then comparing two cells with different n measures coverage, not quality.

The experiment: for each dense cell, recompute FVD against its own reference while
subsampling BOTH the reference and the prediction windows to a common budget k. If the
ranking between cells flips or the magnitudes move by more than the differences we intend
to report, cross-cell FVD is not publishable and must be restricted to within-cell
comparisons (dense versus accelerated), which share n exactly.

Usage:
  models/dreamgen/.venv/bin/python assets/scripts/fvd_sample_size_probe.py
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

import numpy as np                                                      # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'assets' / 'scripts'))

from export_dr_grid_xlsx import METHOD_LABEL, model_label               # noqa: E402
from export_dr_grid_xlsx_vs_real import frechet, ref_identity, view_of  # noqa: E402

BUDGETS = (128, 256, 512, 1024)


def moments(x: np.ndarray):
    mu = x.mean(0)
    xc = x - mu
    return mu, xc.T.dot(xc) / max(len(x) - 1, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='metrics/WorldScore/results/dr_full')
    ap.add_argument('--json', default='metrics/reports/fvd_sample_size_probe.json')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    run = REPO / args.run

    # collect RAW windows (not streamed moments) so we can subsample
    ref_w = collections.defaultdict(list)
    pred_w = collections.defaultdict(list)
    seen = set()

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
            rel = rec.get('relative', '')
            label = model_label(rel)
            method = METHOD_LABEL.get(rec.get('method'))
            view = view_of(rel)
            if label is None or method is None or view is None:
                continue
            g = (rec['embodiment'], view, rec['split'], label)
            if rec.get('i3d_ref'):
                ident = ref_identity(rec['ref'])
                if (g, ident) not in seen:
                    seen.add((g, ident))
                    ref_w[g].extend(rec['i3d_ref'])
            if method == 'Dense' and rec.get('i3d_pred'):
                pred_w[g].extend(rec['i3d_pred'])

    rng = np.random.default_rng(args.seed)
    print(f'{"cell":40s} {"nRef":>6s} {"nPred":>6s} {"full":>8s} ' +
          ' '.join(f'k={k:<6d}' for k in BUDGETS))
    rows = []
    for g in sorted(ref_w):
        if g not in pred_w:
            continue
        R = np.asarray(ref_w[g], dtype=np.float64)
        P = np.asarray(pred_w[g], dtype=np.float64)
        if len(R) < 64 or len(P) < 64:
            continue
        full = frechet(*moments(P), *moments(R))
        vals = {}
        for k in BUDGETS:
            if len(R) < k or len(P) < k:
                vals[k] = None
                continue
            # average a few draws so the number is not one lucky subsample
            got = []
            for _ in range(3):
                ri = rng.choice(len(R), k, replace=False)
                pi = rng.choice(len(P), k, replace=False)
                got.append(frechet(*moments(P[pi]), *moments(R[ri])))
            vals[k] = float(np.mean(got))
        cell = '|'.join(g)
        shown = ' '.join('   ---  ' if vals[k] is None else f'{vals[k]:8.1f}'
                         for k in BUDGETS)
        print(f'{cell:40s} {len(R):6d} {len(P):6d} {full:8.1f} {shown}')
        rows.append(dict(cell=cell, n_ref=len(R), n_pred=len(P), fvd_full=full,
                         **{f'fvd_k{k}': vals[k] for k in BUDGETS}))

    print('\n=== does a common budget change the RANKING between configurations? ===')
    for k in (None, *BUDGETS):
        key = 'fvd_full' if k is None else f'fvd_k{k}'
        have = [r for r in rows if r.get(key) is not None]
        order = [r['cell'] for r in sorted(have, key=lambda r: r[key])]
        tag = 'full' if k is None else f'k={k}'
        print(f'  {tag:7s} ({len(order):2d} cells) best->worst: '
              + ' > '.join(o.split("|")[0] + "/" + o.split("|")[3] for o in order[:6]))

    out = REPO / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(budgets=list(BUDGETS), cells=rows), indent=2))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
