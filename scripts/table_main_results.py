#!/usr/bin/env python
"""MAIN RESULTS matrix: rows = (model, embodiment, view_type), cols = acceleration method.

Why this table exists next to table_dense_nr.py's T3/T3b:

  T3b (pooled per method)  answers "how much does method X cost, overall" -- one row per
                           method, maximum power, the headline claim.
  T3  (per method x model x embodiment x MOUNT) is the breakdown, but mount-level cells
                           are too thin to read as a matrix and mount is not the axis a
                           reader cares about.
  THIS table              is the paper's main results grid: the reader scans DOWN a
                           column to see whether a method behaves consistently across
                           embodiments/models, and ACROSS a row to rank methods on one
                           configuration. Aggregating over mount (not over model or
                           embodiment) is what makes the cells thick enough: 77/88
                           non-empty cells reach MIN_PAIRS_FOR_CLAIM.

Every gate, delta and `resolved` decision is imported from table_dense_nr.py rather than
reimplemented, so this table can never disagree with T3b about the same pairs.

Aggregating over mount has one cost that is reported, not hidden: CONFOUNDED_MOUNTS marks
AMC and VisualIntegrity as motion-confounded on `head` mounts, so a cell mixing head with
other mounts inherits a share of that confound. Each cell records its head-clip fraction
and cells above HEAD_FLAG_FRAC are flagged.

Usage:
  python scripts/table_main_results.py --run metrics/WorldScore/results/dense_nr_v3_pooled
"""
import argparse
import collections
import csv
import importlib.util
import json
from pathlib import Path

# Single source of truth for gates/stats: load the sibling table script as a module.
_spec = importlib.util.spec_from_file_location(
    'table_dense_nr', Path(__file__).parent / 'table_dense_nr.py')
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

# Fraction of a cell's clips that must be head-mount before the cell carries the
# CONFOUNDED_MOUNTS caveat marker.
HEAD_FLAG_FRAC = 0.34

# Display names, and the family they belong to (for row grouping in the paper).
MODEL_DISPLAY = {
    'ctrlworld': ('Ctrl-World', 'per-view'),
    'ctrlworld_4view_grid': ('Ctrl-World', '4-view grid'),
    'dreamgen': ('DreamGen', ''),
    'dreamdojo': ('DreamDojo', ''),
}
EMB_DISPLAY = {'single_arm': 'Single-arm', 'bimanual': 'Bimanual', 'humanoid': 'Humanoid'}
VIEW_DISPLAY = {'multiview': 'MV', 'singleview': 'SV'}
METHOD_DISPLAY = {
    'dicache': 'DiCache', 'worldcache': 'WorldCache', 'fastercache': 'FasterCache',
    'pisa': 'PISA', 'radial': 'Radial', 'svg1': 'SVG1', 'svg2': 'SVG2',
    'sito': 'SiTo', 'itm': 'ITM',
}
# Column order: caching, then sparse attention, then token pruning -- grouping by
# mechanism is what makes the "caching beats pruning" result visible in one glance.
METHOD_ORDER = ['dicache', 'worldcache', 'fastercache',
                'pisa', 'radial', 'svg1', 'svg2', 'sito', 'itm']
METHOD_GROUPS = [('Caching', 3), ('Sparse attention', 4), ('Token pruning', 2)]


def view_type(r):
    p = r['relative']
    if '/multiview/' in p:
        return 'multiview'
    if '/singleview/' in p:
        return 'singleview'
    raise ValueError(f'cannot determine view type from {p!r}')


def load(run):
    rows = [json.loads(p.read_text()) for p in (run / 'results').glob('*.json')]
    if not rows:
        raise SystemExit(f'no results in {run / "results"}')
    return rows


def head_fracs(rows):
    """(model, embodiment, view_type) -> fraction of accel clips on a head mount."""
    tot = collections.Counter()
    head = collections.Counter()
    for r in rows:
        if r.get('method') in (None, 'gt', 'dense'):
            continue
        k = (r['model'], r['embodiment'], view_type(r))
        tot[k] += 1
        if r.get('camera_mount') == 'head':
            head[k] += 1
    return {k: head[k] / n for k, n in tot.items() if n}


def build(rows):
    """cell (method, model, embodiment, view_type) -> col -> stats dict."""
    groups, unpaired, incomparable = T.accel_deltas(
        rows, cell_of=lambda r: (r['method'], r['model'], r['embodiment'], view_type(r)))
    out = collections.defaultdict(dict)
    for cell, cols in groups.items():
        for col, deltas in cols.items():
            if not deltas:
                continue
            lo, hi = T.boot_ci(deltas)
            p = T.wilcoxon_p(deltas)
            out[cell][col] = dict(
                n=len(deltas), delta=T.median(deltas), lo=lo, hi=hi, p=p,
                resolved=T.resolved_flag(deltas, lo, hi, p, col, vs_dense=True),
                artifact=T.artifact_positive(deltas, lo, hi, p, col, vs_dense=True))
    return out, unpaired, incomparable


def row_keys(cells):
    """Config rows, ordered by family then embodiment then view."""
    keys = {c[1:] for c in cells}
    order = {'multiview': 0, 'singleview': 1}
    fam_order = {'Ctrl-World': 0, 'DreamGen': 1, 'DreamDojo': 2}
    emb_order = {'single_arm': 0, 'bimanual': 1, 'humanoid': 2}
    return sorted(keys, key=lambda k: (
        fam_order.get(MODEL_DISPLAY.get(k[0], (k[0], ''))[0], 9),
        MODEL_DISPLAY.get(k[0], (k[0], ''))[1],
        emb_order.get(k[1], 9), order.get(k[2], 9)))


def row_label(k, tex=False):
    fam, variant = MODEL_DISPLAY.get(k[0], (k[0], ''))
    emb = EMB_DISPLAY.get(k[1], k[1])
    view = VIEW_DISPLAY.get(k[2], k[2])
    name = f'{fam} ({variant})' if variant else fam
    return f'{name} / {emb} / {view}' if not tex else f'{name} & {emb} & {view}'


def fmt(st, digits, plain=False):
    """Cell body. Bold = resolved. dagger = below the claim floor. § = artifact-positive
    (statistically solid but in a direction that cannot be an improvement)."""
    if st is None:
        return '--'
    d = f'{st["delta"]:+.{digits}f}'
    if st['n'] < T.MIN_PAIRS_FOR_CLAIM:
        return f'{d}*' if plain else rf'\textcolor{{gray}}{{{d}$^\dagger$}}'
    if st.get('artifact'):
        return f'{d}!' if plain else rf'\textit{{{d}}}$^{{\S}}$'
    if st['resolved']:
        return f'[{d}]' if plain else rf'\textbf{{{d}}}'
    return d


def write_csv(path, cells, rkeys, col, digits):
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'embodiment', 'view_type', 'method',
                    'n_pairs', 'delta_median', 'ci_lo', 'ci_hi', 'p', 'resolved',
                    'artifact_positive'])
        for rk in rkeys:
            for me in METHOD_ORDER:
                st = cells.get((me,) + rk, {}).get(col)
                if st is None:
                    continue
                # boot_ci returns (None, None) below n=3; keep the row, blank the CI.
                num = lambda v: '' if v is None else f'{v:.{digits + 2}f}'
                w.writerow([rk[0], rk[1], rk[2], me, st['n'],
                            num(st['delta']), num(st['lo']), num(st['hi']),
                            '' if st['p'] is None else f'{st["p"]:.3g}', st['resolved'],
                            st.get('artifact', False)])


def write_tex(path, cells, rkeys, col, label, digits, higher, heads):
    lines = [
        r'% generated by scripts/table_main_results.py -- do not edit by hand',
        r'\begin{table}[t]', r'\centering', r'\small',
        rf'\caption{{{label}: paired delta (method $-$ dense) per configuration. '
        rf'{"Higher is better" if higher else "Direction: see schema"}; negative = '
        rf'degradation vs dense. \textbf{{Bold}} = resolved '
        rf'(95\% CI excludes 0, $p<{T.ALPHA}$, $n\ge{T.MIN_PAIRS_FOR_CLAIM}$). '
        rf'$\dagger$ = $n<{T.MIN_PAIRS_FOR_CLAIM}$, shown but not claimable. '
        rf'{r"$\S$ = solid positive delta on a column where that direction is an artifact, not an improvement. " if col in T.NO_IMPROVEMENT_VS_DENSE else ""}'
        rf'-- = method not run on that configuration.}}',
        rf'\label{{tab:main-{col.replace("_", "-")}}}',
        r'\begin{tabular}{lll' + 'r' * len(METHOD_ORDER) + '}', r'\toprule',
    ]
    grp = ' & '.join(rf'\multicolumn{{{n}}}{{c}}{{{g}}}' for g, n in METHOD_GROUPS)
    lines.append(rf'\multicolumn{{3}}{{c}}{{Configuration}} & {grp} \\')
    lines.append(r'\cmidrule(lr){1-3} '
                 + ' '.join(rf'\cmidrule(lr){{{4 + s}-{3 + s + n}}}'
                            for s, n in zip([0, 3, 7], [n for _, n in METHOD_GROUPS])))
    lines.append('Model & Embodiment & View & '
                 + ' & '.join(METHOD_DISPLAY[m] for m in METHOD_ORDER) + r' \\')
    lines.append(r'\midrule')
    for rk in rkeys:
        body = ' & '.join(fmt(cells.get((me,) + rk, {}).get(col), digits)
                          for me in METHOD_ORDER)
        mark = ''
        if col in T.CONFOUNDED_MOUNTS and heads.get(rk, 0) >= HEAD_FLAG_FRAC:
            mark = r'$^{\ddagger}$'
        lines.append(f'{row_label(rk, tex=True)}{mark} & {body} ' + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    notes = []
    if col in T.NO_IMPROVEMENT_VS_DENSE:
        notes.append(r'$\S$ ' + T.NO_IMPROVEMENT_VS_DENSE[col].replace('_', r'\_'))
    if col in T.CONFOUNDED_MOUNTS:
        notes.append(rf'$\ddagger$ cell contains $\ge${HEAD_FLAG_FRAC * 100:.0f}\% '
                     rf'head-mount clips, where this column is motion-confounded '
                     rf'({"; ".join(T.CONFOUNDED_MOUNTS[col].values())}).')
    if col in T.ACCEL_CAVEATS:
        notes.append(T.ACCEL_CAVEATS[col].replace('_', r'\_'))
    if col in T.CEILING_FAILED:
        notes.append('This column failed the GT-is-ceiling audit, so it may not rank '
                     'MODELS; the accel-vs-dense delta below is valid because the bias '
                     'cancels between two clips of the same model.')
    if col in T.PAIRED_ONLY:
        notes.append('Paired deltas only -- absolute values carry a per-caption bias.')
    if col in T.NOT_CROSS_EMBODIMENT:
        notes.append('Do not compare ABSOLUTE values across embodiments; the deltas '
                     'here are within-embodiment and therefore valid.')
    if notes:
        lines.append(r'\begin{flushleft}\footnotesize')
        lines += [n + r'\\' for n in notes]
        lines.append(r'\end{flushleft}')
    lines.append(r'\end{table}')
    path.write_text('\n'.join(lines) + '\n')


def console(cells, rkeys, heads):
    w = 13
    for col, label, higher, digits in T.COLUMNS:
        print(f'\n=== {label} '
              f'({"higher better" if higher else "descriptive"}) ===')
        print(' ' * 44 + ''.join(f'{METHOD_DISPLAY[m][:9]:>10s}' for m in METHOD_ORDER))
        for rk in rkeys:
            mark = '#' if (col in T.CONFOUNDED_MOUNTS
                           and heads.get(rk, 0) >= HEAD_FLAG_FRAC) else ' '
            line = f'{row_label(rk)[:42]:42s}{mark} '
            for me in METHOD_ORDER:
                line += f'{fmt(cells.get((me,) + rk, {}).get(col), digits, plain=True):>10s}'
            print(line)
        print('  [x] = resolved,  x* = n<8,  x! = artifact-positive,  -- = not run' +
              ('  # = head-mount confound' if col in T.CONFOUNDED_MOUNTS else ''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=None,
                    help='default: <run>/tables')
    args = ap.parse_args()
    out = args.out or (args.run / 'tables')
    out.mkdir(parents=True, exist_ok=True)

    rows = load(args.run)
    cells, unpaired, incomparable = build(rows)
    rkeys = row_keys(cells)
    heads = head_fracs(rows)
    print(f'{len(rows)} records -> {len(cells)} (method x config) cells, '
          f'{len(rkeys)} config rows x {len(METHOD_ORDER)} methods')
    if incomparable:
        print(f'  dropped by comparable_pair gate: {dict(incomparable)}')
    if unpaired:
        print(f'  dropped for missing dense partner: {dict(unpaired)}')

    for col, label, higher, digits in T.COLUMNS:
        write_csv(out / f'main_{col}.csv', cells, rkeys, col, digits)
        write_tex(out / f'main_{col}.tex', cells, rkeys, col, label, digits, higher, heads)
    console(cells, rkeys, heads)

    n_cells = sum(1 for rk in rkeys for me in METHOD_ORDER
                  if cells.get((me,) + rk, {}).get('visual_integrity'))
    n_ok = sum(1 for rk in rkeys for me in METHOD_ORDER
               if (cells.get((me,) + rk, {}).get('visual_integrity') or {}).get('n', 0)
               >= T.MIN_PAIRS_FOR_CLAIM)
    print(f'\nVisual Integrity coverage: {n_cells} non-empty cells, '
          f'{n_ok} at n>={T.MIN_PAIRS_FOR_CLAIM} '
          f'({100 * n_ok / max(n_cells, 1):.0f}%)')
    print(f'wrote {2 * len(T.COLUMNS)} files to {out}')


if __name__ == '__main__':
    main()
