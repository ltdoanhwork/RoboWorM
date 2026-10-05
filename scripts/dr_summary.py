#!/usr/bin/env python3
"""Pooled DR summary: median PSNR / SSIM / LPIPS per (method, reference), plus FVD.

Medians, not means: a single corrupt-ish clip can move a PSNR mean by whole dB, and the
whole DR table is read as "is this method worse than dense", which a median answers
robustly. Counts are printed beside every cell so a thin cell is visible rather than
implied.
"""
import collections
import json
import os
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN = Path(sys.argv[1] if len(sys.argv) > 1
           else REPO / 'metrics/WorldScore/results/dr_full')
ORDER = ['dense', 'fastercache', 'dicache', 'worldcache',
         'pisa', 'radial', 'svg1', 'svg2', 'sito', 'itm']

agg = collections.defaultdict(lambda: collections.defaultdict(list))
skip = collections.Counter()
n = 0
with os.scandir(RUN / 'results') as it:
    for e in it:
        if not e.name.endswith('.json'):
            continue
        try:
            d = json.loads(Path(e.path).read_text())
        except Exception:
            continue
        n += 1
        if d.get('skipped'):
            skip[d['skipped']] += 1
            continue
        for k in ('psnr', 'ssim', 'lpips'):
            if d.get(k) is not None:
                agg[(d['method'], d['ref_kind'])][k].append(d[k])

print(f'records {n}   skipped {dict(skip)}')
for ref, note in (('vs_dense', 'PhyFid-DR: reference = this episode\'s OWN dense clip. '
                               'The generative error cancels, so what is left is the '
                               'acceleration. THIS is what an accel claim rests on.'),
                  ('vs_gt', 'reference = ground truth. Mixes "model cannot predict the '
                            'future" with "acceleration degraded it" -- descriptive only, '
                            'cannot rank accelerations.')):
    print(f'\n=== {ref} ===')
    print(f'  {note}')
    print(f'  {"method":12s}{"PSNR":>8s}{"SSIM":>8s}{"LPIPS":>8s}{"n":>7s}')
    for m in ORDER:
        v = agg.get((m, ref))
        if not v:
            continue
        print(f'  {m:12s}{st.median(v["psnr"]):8.2f}{st.median(v["ssim"]):8.4f}'
              f'{st.median(v["lpips"]):8.4f}{len(v["psnr"]):7d}')

fvd = RUN / 'fvd.json'
if fvd.exists():
    d = json.loads(fvd.read_text())
    cells = d['cells']
    ok = [c for c in cells if c['fvd'] is not None]
    print(f'\n=== FVD ({len(ok)}/{len(cells)} cells have >= {d["min_clips"]} windows) ===')
    # Pool per (method, ref) by taking the median over that method's cells: FVD is not
    # additive, so averaging the numbers would be meaningless -- the median of the
    # per-cell FVDs is the honest one-number summary.
    per = collections.defaultdict(list)
    for c in ok:
        per[(c['method'], c['ref_kind'])].append(c['fvd'])
    for ref in ('vs_dense', 'vs_gt'):
        print(f'  -- {ref}')
        print(f'  {"method":12s}{"FVD (median of cells)":>24s}{"cells":>8s}')
        for m in ORDER:
            vs = per.get((m, ref))
            if not vs:
                continue
            print(f'  {m:12s}{st.median(vs):24.2f}{len(vs):8d}')
else:
    print(f'\n(no {fvd} yet -- run --fvd-only)')
