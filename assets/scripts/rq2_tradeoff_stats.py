#!/usr/bin/env python3
"""Recompute every number quoted in the RQ2 (acceleration trade-off) subsection.

The RQ2 text quotes ~30 numbers. This script is the single place they come from, so
the prose can be audited without re-running any generation: it reads the dumped
figure data (which is itself produced by make_nr_circular_figures.py alongside the
c12 figures) and prints / writes every statistic with the exact label used in the
paper.

Run:
    models/dreamgen/.venv/bin/python assets/scripts/rq2_tradeoff_stats.py

Writes results/reports/rq2_tradeoff.json and prints a readable report.

Protocol notes that matter for reading the output:

* Damage is always measured against the DENSE cell of the SAME (cell, split). Dense
  is the ceiling for the reference-free tracks, so "acceleration beats dense" is not
  a quality win, it is evidence the column is being gamed (see the PhyWorld rule).
* NR mean(4) is the mean of the four adapted NR ratios, each already expressed as a
  ratio to real video, with Actor Activity folded through `closeness` so it shares
  an orientation with the others. This is the LEGACY scalar; the benchmark now
  replaces Actor Activity and Motion Continuity with PSNR and FVD (see below).
* The CURRENT metric set is Visual Integrity + PSNR + Text Alignment + FVD, read
  from `results/reports/speedup_figure_data_{embodiment}.csv` (produced by
  `assets/scripts/dump_speedup_figure_data.py`). Both sets are reported side by
  side under `per_nr_column` (legacy) and `per_metric_column_current`, because the
  RQ2 prose must not quote per-column numbers from the superseded definition while
  the dense-baselines table uses the new one. FVD there is already inverted so all
  four components point the same way.
* KIVA is a violation RATE in percent: LOWER is better, and deltas are in points
  (pp). RIGIS is a 0..1 score: HIGHER is better. They cover disjoint tracks
  (single-view vs multi-view), so they are never averaged together.
* Speedup is invariant to regime in this benchmark (verified: 0 of 111 cell-method
  pairs differ between makovian and non_makovian). So "fastest on N of 24
  (cell, split) pairs" double-counts 12 cells; `fastest_per_cell` is the honest
  denominator and is reported alongside the legacy `fastest_per_cell_split`.
"""
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'assets/paper_figures/nr_circular/real_video/data'
OUT = ROOT / 'results/reports/rq2_tradeoff.json'
SPLITS = ('makovian', 'non_makovian')
CACHING = ('World-cache', 'Di-cache', 'Faster-cache')
NR_COLS = ('Visual Integrity', 'Text Alignment', 'Actor Activity',
           'Motion Continuity')
# The CURRENT metric set: PSNR and FVD replace Actor Activity and Motion Continuity.
CUR_COLS = ('Visual Integrity', 'PSNR', 'Text Alignment', 'FVD')
CUR_DATA = ROOT / 'results/reports'
EMBODIMENTS = ('bimanual', 'humanoid', 'single_arm')
FAMILY_OF = {'World-cache': 'caching', 'Di-cache': 'caching',
             'Faster-cache': 'caching', 'PISA': 'sparse-attn',
             'Radial Attention': 'sparse-attn', 'SVG1': 'sparse-attn',
             'SVG2': 'sparse-attn', 'sito': 'pruning', 'itm': 'pruning'}


def num(v):
    """float() that treats '', 'None' and dash spellings as missing, not as 0.0."""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {'none', 'nan'} or set(s) <= set('-\u2013\u2014'):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load():
    rows = []
    for sp in SPLITS:
        p = DATA / ('circular_%s_real_video.csv' % sp)
        with open(p) as f:
            rows += [dict(r, split=sp) for r in csv.DictReader(f)]
    if not rows:
        raise SystemExit('no rows: regenerate figures first (--only c12 writes data/)')
    return rows


def load_current():
    """Rows of the CURRENT metric set, already ratio'd against Dense.

    `dump_speedup_figure_data.py` has done the two things that are easy to get wrong:
    FVD is inverted (so all four components point up) and each panel is normalised
    inside its own FVD protocol. Here we only need the per-metric ratios.
    """
    rows = []
    for emb in EMBODIMENTS:
        p = CUR_DATA / ('speedup_figure_data_%s.csv' % emb)
        if not p.exists():
            raise SystemExit('missing %s -- run dump_speedup_figure_data.py first' % p)
        with open(p) as f:
            rows += [dict(r, embodiment=emb) for r in csv.DictReader(f)]
    return rows


def main():
    rows = load()
    # Dense reference per (cell, split).
    dense = {(r['cell'], r['split']): (num(r['nr_mean4']), num(r['kiva_pct']),
                                       num(r['rigis']))
             for r in rows if r['method'] == 'Dense'}
    dense_cols = {(r['cell'], r['split']): {c: num(r.get(c)) for c in NR_COLS}
                  for r in rows if r['method'] == 'Dense'}
    cells = sorted({r['cell'] for r in rows})
    fam = {}
    per = defaultdict(lambda: defaultdict(list))
    fam_agg = defaultdict(lambda: defaultdict(list))
    per_col = defaultdict(lambda: defaultdict(list))
    audit_dir = defaultdict(lambda: [0, 0])      # family -> [better, total]
    coverage = defaultdict(set)
    backbone = defaultdict(lambda: defaultdict(list))

    for r in rows:
        coverage[r['method']].add(r['cell'])
        if r['method'] == 'Dense':
            continue
        fam[r['method']] = r['family']
        d = dense.get((r['cell'], r['split']))
        sp = num(r['speedup'])
        if sp is None or d is None:
            continue
        bb = r['cell'].split('_')[0].replace('ctrlworld4', 'ctrlworld')
        per[r['method']]['speedup'].append(sp)
        fam_agg[r['family']]['speedup'].append(sp)
        backbone[bb][r['family']].append(sp)
        nr, kiva, rigis = num(r['nr_mean4']), num(r['kiva_pct']), num(r['rigis'])
        if nr is not None and d[0]:
            loss_pct = 100.0 * (d[0] - nr) / d[0]
            per[r['method']]['nr_loss_pct'].append(loss_pct)
            fam_agg[r['family']]['nr_loss_pct'].append(loss_pct)
            # Speed bought per 1% of quality given up. Floor the denominator so a
            # ~0 loss does not produce an infinite ratio.
            per[r['method']]['gain_per_pct'].append((sp - 1.0) / max(loss_pct, 0.05))
            fam_agg[r['family']]['speedup_paired'].append(sp)
        if kiva is not None and d[1] is not None:
            per[r['method']]['d_kiva_pp'].append(kiva - d[1])
            fam_agg[r['family']]['d_kiva_pp'].append(kiva - d[1])
            audit_dir[r['family']][0] += (kiva - d[1]) < 0      # lower = better
            audit_dir[r['family']][1] += 1
        if rigis is not None and d[2] is not None:
            per[r['method']]['d_rigis'].append(rigis - d[2])
            fam_agg[r['family']]['d_rigis'].append(rigis - d[2])
            audit_dir[r['family']][0] += (rigis - d[2]) > 0     # higher = better
            audit_dir[r['family']][1] += 1
        # Per-column loss, because a pooled scalar can hide one catastrophic column:
        # this is what shows pruning damages ALL FOUR while the other two damage none
        # by more than 5%.
        for c in NR_COLS:
            v, dv = num(r.get(c)), dense_cols.get((r['cell'], r['split']), {}).get(c)
            if v is not None and dv:
                per_col[r['family']][c].append(100.0 * (dv - v) / dv)

    def pearson(xs, ys):
        if len(xs) < 3:
            return None
        mx, my = st.mean(xs), st.mean(ys)
        den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
        return (sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / den) if den else None

    # Fastest method per (cell, split) -- the "is one method the default?" question.
    # NOTE speedup does not depend on split in this benchmark, so this denominator
    # (2 x n_cells) double-counts. `wins_cell` is the honest one; both are reported
    # so the legacy figure quoting 22/24 can still be traced.
    wins = defaultdict(int)
    for c in cells:
        for sp in SPLITS:
            cand = [r for r in rows if r['cell'] == c and r['split'] == sp
                    and r['method'] != 'Dense' and num(r['speedup'])]
            if cand:
                wins[max(cand, key=lambda r: num(r['speedup']))['method']] += 1
    wins_cell = defaultdict(int)
    split_varies = 0
    for c in cells:
        cand = [r for r in rows if r['cell'] == c and r['method'] != 'Dense'
                and num(r['speedup'])]
        if cand:
            wins_cell[max(cand, key=lambda r: num(r['speedup']))['method']] += 1
        by_m = defaultdict(dict)
        for r in cand:
            by_m[r['method']][r['split']] = num(r['speedup'])
        split_varies += sum(1 for v in by_m.values()
                            if len(v) == 2 and v[SPLITS[0]] != v[SPLITS[1]])

    report = {
        'generated_by': 'assets/scripts/rq2_tradeoff_stats.py',
        'source': str(DATA.relative_to(ROOT)),
        'protocol': {
            'damage_reference': 'dense cell of the same (cell, split)',
            'nr_mean4': 'mean of the 4 adapted NR ratios (each vs real video) -- '
                        'LEGACY: Visual Integrity, Text Alignment, Actor Activity, '
                        'Motion Continuity',
            'current_metric_set': 'Visual Integrity, PSNR, Text Alignment, FVD '
                                  '(PSNR/FVD replace Actor Activity/Motion '
                                  'Continuity); ratios vs the Dense row of the same '
                                  'cell+split, FVD inverted',
            'kiva': 'violation rate in percent, LOWER better, deltas in pp',
            'rigis': '0..1 score, HIGHER better; disjoint tracks from KIVA',
            'speedup_depends_on_split': False,
            'n_cells': len(cells), 'n_rows': len(rows),
        },
        'per_method': {}, 'per_family': {}, 'per_backbone': {},
        'per_nr_column': {}, 'per_metric_column_current': {},
        'coverage': {}, 'fastest_per_cell_split': dict(wins),
        'fastest_per_cell': dict(wins_cell),
    }

    print('=== per method (both splits pooled) ===')
    hdr = '%-17s %-12s %7s %7s %9s %9s %9s %7s'
    print(hdr % ('method', 'family', 'spd', 'max', 'NRloss%', 'dKIVApp', 'dRIGIS',
                 'gain/1%'))
    for m in sorted(per, key=lambda m: -st.mean(per[m]['speedup'])):
        p = per[m]
        rec = {
            'family': fam[m],
            'n_cells': len(coverage[m]),
            'speedup_mean': round(st.mean(p['speedup']), 3),
            'speedup_max': round(max(p['speedup']), 3),
            'nr_loss_pct_mean': round(st.mean(p['nr_loss_pct']), 2) if p['nr_loss_pct'] else None,
            'nr_loss_pct_median': round(st.median(p['nr_loss_pct']), 2) if p['nr_loss_pct'] else None,
            'd_kiva_pp_mean': round(st.mean(p['d_kiva_pp']), 2) if p['d_kiva_pp'] else None,
            'd_rigis_mean': round(st.mean(p['d_rigis']), 4) if p['d_rigis'] else None,
            'gain_per_pct_median': round(st.median(p['gain_per_pct']), 3) if p['gain_per_pct'] else None,
        }
        report['per_method'][m] = rec
        g = lambda v, f='%9.2f': (f % v) if v is not None else '       --'
        print(hdr % (m, fam[m], '%6.2fx' % rec['speedup_mean'],
                     '%6.2fx' % rec['speedup_max'], g(rec['nr_loss_pct_mean']),
                     g(rec['d_kiva_pp_mean']), g(rec['d_rigis_mean'], '%9.4f'),
                     '%6.2f' % rec['gain_per_pct_median'] if rec['gain_per_pct_median'] else '    --'))

    print('\n=== per family, and the CLIFF test ===')
    print('%-12s %7s %9s %9s %9s %8s %8s' % ('family', 'spd', 'NRloss%med',
                                             'p90', 'dKIVApp', 'dRIGIS', 'r(spd,loss)'))
    for fa in ('caching', 'sparse-attn', 'pruning'):
        p = fam_agg[fa]
        ys = p['nr_loss_pct']
        r = pearson(p['speedup_paired'], ys)
        rec = {
            'speedup_mean': round(st.mean(p['speedup']), 3),
            'nr_loss_pct_median': round(st.median(ys), 2),
            'nr_loss_pct_p90': round(sorted(ys)[max(int(0.9 * len(ys)) - 1, 0)], 2),
            'd_kiva_pp_mean': round(st.mean(p['d_kiva_pp']), 2) if p['d_kiva_pp'] else None,
            'd_rigis_mean': round(st.mean(p['d_rigis']), 4) if p['d_rigis'] else None,
            # r ~ 0 is the evidence for "cliff, not slope": within a family, buying
            # more speed does NOT predict more damage -- the damage is set by the
            # mechanism and by whether a per-cell config crossed its own knee.
            'pearson_speedup_vs_loss': round(r, 3) if r is not None else None,
            'n': len(ys),
            'audit_better_than_dense': audit_dir[fa][0],
            'audit_n': audit_dir[fa][1],
        }
        report['per_family'][fa] = rec
        print('%-12s %6.2fx %9.2f %9.2f %9s %8s %8s'
              % (fa, rec['speedup_mean'], rec['nr_loss_pct_median'],
                 rec['nr_loss_pct_p90'],
                 ('%+.1f' % rec['d_kiva_pp_mean']) if rec['d_kiva_pp_mean'] is not None else '--',
                 ('%+.4f' % rec['d_rigis_mean']) if rec['d_rigis_mean'] is not None else '--',
                 ('%+.2f' % rec['pearson_speedup_vs_loss'])))

    print('\n=== mean %% loss per NR column, LEGACY set (positive = worse than dense) ===')
    print('%-12s %17s %15s %15s %18s %9s' % ('family', *NR_COLS, 'cols>5%'))
    for fa in ('caching', 'sparse-attn', 'pruning'):
        vals = {c: (round(st.mean(per_col[fa][c]), 2) if per_col[fa][c] else None)
                for c in NR_COLS}
        n_bad = sum(1 for c in NR_COLS if vals[c] is not None and vals[c] > 5)
        report['per_nr_column'][fa] = dict(vals, n_columns_over_5pct=n_bad)
        print('%-12s %17s %15s %15s %18s %9s'
              % (fa, *[('%+.1f%%' % vals[c]) if vals[c] is not None else '--'
                       for c in NR_COLS], '%d/4' % n_bad))

    # ---- CURRENT metric set: PSNR and FVD in place of Actor Activity / Motion Cont.
    cur = load_current()
    cur_col = defaultdict(lambda: defaultdict(list))
    cur_mean = defaultdict(list)
    for r in cur:
        fa = FAMILY_OF.get(r['method'])
        if fa is None:                                   # Dense is the reference
            continue
        for c in CUR_COLS:
            v = num(r.get('ratio_dense_' + c))
            if v is not None:
                cur_col[fa][c].append(100.0 * (1.0 - v))
        mv = num(r.get('mean_dense'))
        if mv is not None:
            cur_mean[fa].append(100.0 * (1.0 - mv))

    print('\n=== mean %% loss per metric column, CURRENT set '
          '(VI / PSNR / TextAlign / FVD) ===')
    print('%-12s %13s %13s %13s %13s %9s %11s'
          % ('family', *CUR_COLS, 'cols>5%', 'mean loss%'))
    for fa in ('caching', 'sparse-attn', 'pruning'):
        vals = {c: (round(st.mean(cur_col[fa][c]), 2) if cur_col[fa][c] else None)
                for c in CUR_COLS}
        med = {c: (round(st.median(cur_col[fa][c]), 2) if cur_col[fa][c] else None)
               for c in CUR_COLS}
        n_bad = sum(1 for c in CUR_COLS if vals[c] is not None and vals[c] > 5)
        report['per_metric_column_current'][fa] = dict(
            mean={c: vals[c] for c in CUR_COLS},
            median={c: med[c] for c in CUR_COLS},
            n_columns_over_5pct=n_bad,
            mean_loss_pct_mean=(round(st.mean(cur_mean[fa]), 2)
                                if cur_mean[fa] else None),
            mean_loss_pct_median=(round(st.median(cur_mean[fa]), 2)
                                  if cur_mean[fa] else None),
            n=len(cur_mean[fa]))
        print('%-12s %13s %13s %13s %13s %9s %11s'
              % (fa, *[('%+.1f%%' % vals[c]) if vals[c] is not None else '--'
                       for c in CUR_COLS], '%d/4' % n_bad,
                 ('%+.1f%%' % st.median(cur_mean[fa])) if cur_mean[fa] else '--'))
    print('  (last column = MEDIAN loss of the 4-metric mean; FVD is the sensitive '
          'one and it moves the verdict vs the legacy set)')

    print('\n=== speedup by backbone x family (the Amdahl story) ===')
    print('%-11s %9s %12s %9s' % ('backbone', 'caching', 'sparse-attn', 'pruning'))
    for bb in ('dreamgen', 'dreamdojo', 'ctrlworld'):
        rec = {fa: round(st.mean(v), 3) for fa, v in backbone[bb].items() if v}
        report['per_backbone'][bb] = rec
        c = lambda fa: ('%8.2fx' % rec[fa]) if fa in rec else '      --'
        print('%-11s %9s %12s %9s' % (bb, c('caching'), c('sparse-attn'), c('pruning')))

    print('\n=== coverage (distinct model-cells out of %d) ===' % len(cells))
    for m, cs in sorted(coverage.items(), key=lambda kv: -len(kv[1])):
        miss = sorted(set(cells) - cs)
        report['coverage'][m] = {'n': len(cs), 'missing': miss}
        print('  %-17s %2d/%d%s' % (m, len(cs), len(cells),
                                    ('   missing: ' + ', '.join(miss)) if miss else ''))

    print('\n=== fastest method per cell (out of %d cells) ===' % len(cells))
    for m, n in sorted(wins_cell.items(), key=lambda kv: -kv[1]):
        print('  %-17s %d/%d' % (m, n, len(cells)))
    print('  cell-methods whose speedup differs by split: %d (so the '
          '"N of %d (cell, split)" framing double-counts)'
          % (split_varies, 2 * len(cells)))
    print('  legacy per-(cell, split) tally: %s'
          % ', '.join('%s=%d' % kv for kv in sorted(wins.items(), key=lambda k: -k[1])))

    OUT.write_text(json.dumps(report, indent=1) + '\n')
    print('\nwrote %s' % OUT.relative_to(ROOT))


if __name__ == '__main__':
    main()
