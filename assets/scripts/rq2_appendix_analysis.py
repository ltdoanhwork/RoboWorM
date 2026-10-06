#!/usr/bin/env python3
"""Numbers for the appendix analysis of the acceleration trade-off figures.

Answers three questions the main-text scalar cannot:
  1. How much do the two backbones of one panel DISAGREE about the same method?
     (this is why the figure prints a mean per model instead of one averaged dot)
  2. How much does the 4-metric mean DILUTE FVD, the only sensitive column?
  3. Does any method's ranking flip between the two backbones of a panel?

Reads the per-embodiment dumps written by dump_speedup_figure_data.py.
"""
from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EMB = ['bimanual', 'humanoid', 'single_arm']
PANELS = {'a_multiview': ['Dreamgen MV', 'Ctrl-World MV'],
          'b_singleview': ['Dreamgen SV', 'DreamDojo SV']}
FAMILY = {'World-cache': 'caching', 'Di-cache': 'caching', 'Faster-cache': 'caching',
          'sito': 'pruning', 'itm': 'pruning', 'PISA': 'sparse-attn',
          'Radial Attention': 'sparse-attn', 'SVG1': 'sparse-attn',
          'SVG2': 'sparse-attn'}


def load():
    rows = []
    for e in EMB:
        p = REPO / 'results/reports' / f'speedup_figure_data_{e}.csv'
        rows += list(csv.DictReader(p.open()))
    return [r for r in rows if r['split'] == 'makovian']


def f(r, k):
    v = r.get(k, '')
    return None if v in ('', 'None') else float(v)


def main():
    rows = load()

    print('=== 1. within-panel disagreement on the SAME method ===')
    print('    (|mean of model A - mean of model B|, makovian)')
    spreads = []
    for emb in EMB:
        for panel, tracks in PANELS.items():
            ix = {(r['track'], r['method']): r for r in rows
                  if r['embodiment'] == emb and r['panel'] == panel}
            for m in FAMILY:
                a, b = ix.get((tracks[0], m)), ix.get((tracks[1], m))
                if not a or not b:
                    continue
                va, vb = f(a, 'mean_dense'), f(b, 'mean_dense')
                if va is None or vb is None:
                    continue
                spreads.append((abs(va - vb), emb, panel, m, va, vb))
    spreads.sort(reverse=True)
    for d, emb, panel, m, va, vb in spreads[:6]:
        tag = 'MV' if panel.startswith('a') else 'SV'
        print(f'  {emb:11s} {tag} {m:17s} {va:.3f} vs {vb:.3f}   spread {d:.3f}')
    print(f'  median spread over {len(spreads)} (method, panel) pairs: '
          f'{st.median(d for d, *_ in spreads):.3f}')
    print(f'  pairs where the two backbones straddle 0.95: '
          f'{sum(1 for d, e, p, m, va, vb in spreads if min(va, vb) < 0.95 <= max(va, vb))}')

    print('\n=== 2. the mean DILUTES FVD (per family, makovian+non_mak pooled) ===')
    allrows = []
    for e in EMB:
        allrows += list(csv.DictReader(
            (REPO / 'results/reports' / f'speedup_figure_data_{e}.csv').open()))
    print(f'  {"family":12s}{"FVD loss":>10}{"mean loss":>11}{"dilution":>10}')
    for fam in ('caching', 'sparse-attn', 'pruning'):
        fv = [1 - f(r, 'ratio_dense_FVD') for r in allrows
              if FAMILY.get(r['method']) == fam and f(r, 'ratio_dense_FVD') is not None]
        mn = [1 - f(r, 'mean_dense') for r in allrows
              if FAMILY.get(r['method']) == fam and f(r, 'mean_dense') is not None]
        print(f'  {fam:12s}{100*st.mean(fv):9.1f}%{100*st.mean(mn):10.1f}%'
              f'{st.mean(fv)/st.mean(mn):9.1f}x')

    print('\n=== 3. does the family ordering hold on BOTH backbones of a panel? ===')
    for emb in EMB:
        for panel, tracks in PANELS.items():
            tag = 'MV' if panel.startswith('a') else 'SV'
            line = []
            for t in tracks:
                per = {}
                for m, fam in FAMILY.items():
                    r = next((r for r in rows if r['embodiment'] == emb
                              and r['panel'] == panel and r['track'] == t
                              and r['method'] == m), None)
                    if r and f(r, 'mean_dense') is not None:
                        per.setdefault(fam, []).append(f(r, 'mean_dense'))
                if per:
                    order = sorted(per, key=lambda k: -st.mean(per[k]))
                    line.append(f'{t.split()[0]:9s}: ' + ' > '.join(
                        f'{k}({st.mean(per[k]):.2f})' for k in order))
            print(f'  {emb:11s} {tag}  ' + '   |   '.join(line))


if __name__ == '__main__':
    main()
