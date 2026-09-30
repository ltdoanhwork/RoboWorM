#!/usr/bin/env python3
"""Test whether single-view FVD may share a real reference keyed (embodiment, split).

VERDICT (measured 2026-09-24): only bimanual may share. Do NOT use the pooled numbers this
script prints for humanoid or single_arm -- those cells must keep a per-configuration
reference. The decisive measurement is cross-model real-vs-real FVD (each model's own GT
against the other's), compared against the pool's split-half noise floor:

    bimanual   makovian / non_makovian :   7.6 /   8.3  vs floor  86.2 / 45.0 -> 0.09x/0.19x OK
    humanoid   makovian / non_makovian : 222.4 / 413.3  vs floor 101.9 /189.4 -> 2.18x/2.18x NO
    single_arm makovian / non_makovian : 417.0 / 408.9  vs floor  58.9 / 35.4 -> 7.08x/11.6x NO

Causes, confirmed: DreamDojo renders 640x480 at 10 fps (15 for single_arm) while DreamGen
renders 768x432 at 16 fps, and on single_arm the two configurations share ZERO episodes
(120 vs 119/120, intersection 0). So their real videos are genuinely different
distributions, and pooling them would charge each model the other's framing/coverage
difference as if it were a quality defect. Note this is NOT the fps confound that the
multi-view path fixes by decimating to 5 fps: bimanual has the same 10-vs-16 fps and
640x480-vs-768x432 mismatch yet still measures 7.6, so equalising fps alone does not make
humanoid/single_arm poolable -- the episode sets and scene content differ too.

Consequence for the paper: multi-view uses the shared per-camera pool (pool_perview_fvd.py),
single-view keeps the per-configuration reference (export_dr_grid_xlsx_vs_real.py). Each
table panel therefore carries exactly one FVD protocol; see the caption of
tab:dense_baselines.

Every cell in the output carries `poolable` so a downstream exporter cannot silently
consume an invalid one. Schema otherwise matches pool_perview_fvd.py.

Usage:
  models/dreamgen/.venv/bin/python assets/scripts/pool_singleview_fvd.py
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

from export_dr_grid_xlsx import METHOD_LABEL, FvdCell, model_label       # noqa: E402
from export_dr_grid_xlsx_vs_real import frechet, ref_identity, view_of   # noqa: E402

FVD_MIN_WINDOWS = 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='metrics/WorldScore/results/dr_full')
    ap.add_argument('--json', default='metrics/reports/fvd_singleview_pooled.json')
    args = ap.parse_args()
    run = REPO / args.run

    cells = collections.defaultdict(FvdCell)      # (emb,label,split,method)
    pool = collections.defaultdict(FvdCell)       # (emb,split)  <- NO model term
    pa, pb = collections.defaultdict(FvdCell), collections.defaultdict(FvdCell)
    # per-model GT, to check the same fairness condition the multiview path checks: a
    # model's own real videos must sit at the pool's floor, else its framing penalty is
    # charged to its predictions and the cross-model comparison is an artefact.
    gt_model = collections.defaultdict(FvdCell)   # (emb,split,label)
    seen = set()
    audit = collections.Counter()

    n = 0
    with os.scandir(run / 'results') as it:
        for e in it:
            if not e.name.endswith('.json'):
                continue
            n += 1
            if n % 10000 == 0:
                print(f'  scanned {n}', flush=True)
            try:
                rec = json.loads(Path(e.path).read_text())
            except (json.JSONDecodeError, OSError):
                audit['unreadable'] += 1
                continue
            if rec.get('ref_kind') != 'vs_gt' or rec.get('skipped'):
                continue
            if view_of(rec.get('relative', '')) != 'singleview':
                continue
            label = model_label(rec.get('relative', ''))
            method = METHOD_LABEL.get(rec.get('method'))
            if label is None or method is None:
                audit['unmapped'] += 1
                continue
            emb, split = rec['embodiment'], rec['split']
            if rec.get('i3d_pred'):
                cells[(emb, label, split, method)].add('pred', rec['i3d_pred'])
            if rec.get('i3d_ref'):
                ident = ref_identity(rec['ref'])
                rk = ((emb, split), ident)
                if rk in seen:                 # one GT is hard-linked into 10 method trees
                    audit['dedup_skipped'] += 1
                    continue
                seen.add(rk)
                pool[(emb, split)].add('ref', rec['i3d_ref'])
                gt_model[(emb, split, label)].add('ref', rec['i3d_ref'])
                bit = hashlib.md5(ident.encode()).digest()[0] & 1
                (pa if bit == 0 else pb)[(emb, split)].add('ref', rec['i3d_ref'])
                audit['real_videos'] += 1
    print(f'  scanned {n} files; audit={dict(audit)}')

    ref = {}
    print('\n=== shared single-view real pools (embodiment, split) ===')
    for k in sorted(pool):
        nw = pool[k].n['ref']
        print(f'  {"|".join(k):28s} n_windows={nw:6d}')
        if nw >= FVD_MIN_WINDOWS:
            ref[k] = pool[k].moments('ref')

    floor = {}
    for k in pool:
        a, b = pa.get(k), pb.get(k)
        if a and b and min(a.n['ref'], b.n['ref']) >= FVD_MIN_WINDOWS:
            floor[k] = frechet(*a.moments('ref'), *b.moments('ref'))

    rows = {}
    for key in sorted(cells):
        emb, label, split, method = key
        rk = (emb, split)
        acc = cells[key]
        if rk not in ref or acc.n['pred'] < FVD_MIN_WINDOWS:
            continue
        rows['|'.join(key)] = dict(
            fvd=frechet(*acc.moments('pred'), *ref[rk]),
            per_camera=None, n_cameras=1, n_windows=acc.n['pred'],
            n_ref_windows=pool[rk].n['ref'], floor=floor.get(rk))

    # FAIRNESS: each model's own GT against the shared pool. Single-view models render the
    # same camera, so this should be at or below the floor; a large excess would mean their
    # framing still differs (different FOV/crop) and the pooled comparison is not clean.
    print('\n=== fairness: model GT vs shared pool (must be <= ~3x floor) ===')
    fair = {}
    worst = 0.0
    for (emb, split, label), acc in sorted(gt_model.items()):
        rk = (emb, split)
        if rk not in ref or acc.n['ref'] < FVD_MIN_WINDOWS:
            continue
        gv = frechet(*acc.moments('ref'), *ref[rk])
        fl = floor.get(rk)
        ratio = (gv / fl) if fl else None
        fair[f'{emb}|{split}|{label}'] = dict(gt_vs_pool=gv, floor=fl, ratio=ratio,
                                              n_gt=acc.n['ref'])
        flag = 'ok' if (ratio is None or ratio <= 3) else 'ARTEFACT RISK'
        if ratio:
            worst = max(worst, ratio)
        print(f'  {emb:11s} {split:13s} {label:15s} gt={gv:8.1f} floor='
              f'{"n/a" if fl is None else f"{fl:7.1f}"} '
              f'ratio={"n/a" if ratio is None else f"{ratio:5.2f}x"}  {flag}')
    print(f'  worst ratio = {worst:.2f}x '
          f'({"POOLABLE" if worst <= 3 else "NOT poolable — do not share the reference"})')

    # DIRECT test: the two models' real videos against each other. This is the decisive
    # measurement -- both models render the same physical camera, so if their GT differs
    # here, the "shared" pool is a mixture of two distributions and every model pays a
    # framing penalty. Reference values from the mosaic-level diagnosis: bimanual
    # single-view scored 7.6 / 8.3, which is what "one distribution" looks like.
    print('\n=== DIRECT cross-model real-vs-real (decisive; compare to floor) ===')
    cross = {}
    for (emb, split) in sorted(pool):
        labs = sorted({lab for (e, s, lab) in gt_model if (e, s) == (emb, split)
                       and gt_model[(e, s, lab)].n['ref'] >= FVD_MIN_WINDOWS})
        fl = floor.get((emb, split))
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                a = gt_model[(emb, split, labs[i])]
                b = gt_model[(emb, split, labs[j])]
                d = frechet(*a.moments('ref'), *b.moments('ref'))
                ratio = (d / fl) if fl else None
                verdict = ('SAME distribution' if ratio is not None and ratio <= 1
                           else 'DIFFERENT — do not pool')
                cross[f'{emb}|{split}|{labs[i]}|{labs[j]}'] = dict(
                    fvd=d, floor=fl, ratio=ratio)
                print(f'  {emb:11s} {split:13s} {labs[i]:14s} vs {labs[j]:14s} '
                      f'FVD={d:8.1f}  floor={"n/a" if fl is None else f"{fl:7.1f}"}  '
                      f'ratio={"n/a" if ratio is None else f"{ratio:5.2f}x"}  {verdict}')

    # Stamp poolability per setting onto every cell. A setting is poolable only if EVERY
    # cross-model real-vs-real pair sits at or below the pool's own noise floor; anything
    # above means the pool is a mixture of distributions and the cell's FVD would encode a
    # framing/coverage penalty rather than generation quality.
    poolable = {}
    for (emb, split) in pool:
        rs = [v['ratio'] for k, v in cross.items()
              if tuple(k.split('|')[:2]) == (emb, split)]
        poolable[(emb, split)] = bool(rs) and all(r is not None and r <= 1.0 for r in rs)
    for key, row in rows.items():
        emb, _, split, _ = key.split('|')
        row['poolable'] = poolable.get((emb, split), False)
    n_bad = sum(1 for r in rows.values() if not r['poolable'])
    print(f'\n  {n_bad}/{len(rows)} cells are NOT poolable -> they must keep a '
          f'per-configuration reference (export_dr_grid_xlsx_vs_real.py)')

    print('\n=== dense single-view FVD (shared reference) ===')
    for k, v in sorted(rows.items()):
        if k.split('|')[3] == 'Dense':
            print(f'  {k:44s} FVD={v["fvd"]:8.1f}  n={v["n_windows"]:5d}  '
                  f'floor={v["floor"] if v["floor"] is None else round(v["floor"], 1)}  '
                  f'{"USABLE" if v["poolable"] else "NOT POOLABLE - use per-config"}')

    out = dict(protocol='DIAGNOSTIC ONLY for humanoid/single_arm. Shared real pool keyed '
                        '(embodiment, split). Valid only where cross_model_real ratio <= 1 '
                        '(bimanual: 0.09-0.19x). humanoid (2.18x) and single_arm '
                        '(7.1-11.6x) are NOT one distribution -- different fps/resolution '
                        'and, on single_arm, zero shared episodes -- so those cells keep a '
                        'per-configuration reference. See each cell\'s "poolable" flag.',
               min_windows=FVD_MIN_WINDOWS,
               real_pool_windows={'|'.join(k): pool[k].n['ref'] for k in pool},
               floor={'|'.join(k): v for k, v in floor.items()},
               poolable={'|'.join(k): v for k, v in poolable.items()},
               fairness=fair, cross_model_real=cross, cells=rows)
    p = REPO / args.json
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f'\nwrote {p}')


if __name__ == '__main__':
    main()
