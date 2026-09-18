#!/usr/bin/env python
"""Store BOTH views of the same scored records: UNPAIRED absolute and PAIRED delta.

This is deliberately two tables built from ONE set of per-clip JSON records, because
scoring is already unpaired -- `run_dense_nr.py` scores every clip independently and
writes one record per clip. Pairing is nothing but a join on (pair_key, model) plus a
subtraction, done here at table time. So keeping both views costs ZERO extra GPU, and
the two can never disagree about a clip's value: they read the identical number.

Layout follows the field standard for acceleration papers (PAB / FasterCache ICLR'25
Tab.1 / TeaCache CVPR'25 Tab.1), which pairs a reference-free column with a
compare-to-dense column:

    Method | Speedup | VBench (reference-free)   | LPIPS / SSIM / PSNR (vs dense)
    ours   |  1.62x  | 78.46%                    | 0.0835 / 0.8932 / 27.03

Their VBench column is an unpaired mean over prompts; their LPIPS/SSIM/PSNR are
computed between the accelerated output and the ORIGINAL MODEL's output on the same
prompt -- i.e. already paired, then averaged, with `-` on the dense row. Our mapping:

    their VBench            -> our absolute NR medians   (`unpaired` table here)
    their PSNR-vs-dense     -> our paired NR deltas      (`paired` table here)

We add what those tables lack: a bootstrap CI, a Wilcoxon p and an n>=8 floor, so a
0.3-point move can be separated from noise instead of being printed bare.

WHY THE UNPAIRED TABLE IS PER-CONFIG AND NEVER POOLED FOR RANKING
-----------------------------------------------------------------
Measured on this run (3,571 records), Visual Integrity, ranking the 9 methods:

    unpaired absolute, pooled over ALL configs   Spearman +0.883 vs the paired ranking
    unpaired absolute, pooled over the 6 configs
      that all 9 methods share                   Spearman +0.833

Neither reproduces the paired ranking, and balancing the config mix makes it WORSE.
The reason is measurable: restricting to the common 6 configs halves n (252 -> 108 for
dicache), and the noise that adds exceeds the config-mix bias it removes -- i.e. the
dominant error in an unpaired comparison is between-EPISODE variance inside a single
config, not the config mix. Concretely the pooled-over-all view promotes svg1 to best
(absolute 29.30) when its paired delta is only 7th (-0.48), and the common-6 view
demotes radial from best (-0.07) to 5th.

So: pooled unpaired numbers are emitted (they are what a VBench-style table shows) but
carry `pooled_unpaired_is_confounded=True`, and the per-config rows are the ones meant
to be read. The paired table remains the only ranking/claim surface.

The config mix that forces this, for the record: dicache/worldcache cover 13 configs,
fastercache only 8 (no ctrlworld_4view_grid at all), itm/sito skip every dreamdojo
config. Only 6 configs are shared by all 9 methods.

Usage:
  python scripts/table_paper_main.py --run metrics/WorldScore/results/dense_nr_v3_pooled
"""
import argparse
import collections
import csv
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    'table_dense_nr', Path(__file__).parent / 'table_dense_nr.py')
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

METHOD_ORDER = ['dicache', 'worldcache', 'fastercache',
                'pisa', 'radial', 'svg1', 'svg2', 'sito', 'itm']
METHOD_DISPLAY = {'dicache': 'DiCache', 'worldcache': 'WorldCache',
                  'fastercache': 'FasterCache', 'pisa': 'PISA', 'radial': 'Radial',
                  'svg1': 'SVG1', 'svg2': 'SVG2', 'sito': 'SiTo', 'itm': 'ITM'}
# Pooled per-method speedup from scripts/make_speedup_summary.py. Kept as a separate
# join (not recomputed) so this table cannot drift from the timing report.
SPEEDUP_CSV = Path('metrics/reports/speedup_summary_by_method.csv')


def view_type(r):
    if '/multiview/' in r['relative']:
        return 'multiview'
    if '/singleview/' in r['relative']:
        return 'singleview'
    raise ValueError(f'cannot determine view type from {r["relative"]!r}')


def config_of(r):
    """(model, embodiment, view_type), with GT records folded onto the model they anchor.

    A GT record carries model `gt_<model>` (or bare `gt`, the legacy ctrlworld anchor),
    so without this fold the GT ceiling row would land in its own phantom config and
    never appear next to the model it is the ceiling for.
    """
    m = r['model']
    if m.startswith('gt_'):
        m = m[3:]
    elif m == 'gt':
        m = 'ctrlworld'
    return (m, r['embodiment'], view_type(r))


def load(run):
    rows = [json.loads(p.read_text()) for p in (run / 'results').glob('*.json')]
    if not rows:
        raise SystemExit(f'no results in {run / "results"}')
    return rows


def speedups():
    if not SPEEDUP_CSV.exists():
        return {}
    inv = {v: k for k, v in METHOD_DISPLAY.items()}
    out = {}
    with SPEEDUP_CSV.open() as f:
        for row in csv.DictReader(f):
            k = inv.get(row['method_display'])
            if k:
                out[k] = (float(row['mean_speedup']),
                          int(row['n_cells_with_speedup']), int(row['n_cells']))
    return out


def unpaired(rows, cell_of):
    """cell -> col -> {n, median, lo, hi}. Every clip counts; no partner required.

    PAIRED_ONLY columns are skipped exactly as in T.stratum_table: an absolute
    CLIPScore median mixes per-caption bias across different prompts, so it is not a
    comparable number no matter how many clips back it.
    """
    buckets = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        cell = cell_of(r)
        if cell is None:
            continue
        for col, _, _, _ in T.COLUMNS:
            if col in T.PAIRED_ONLY:
                continue
            v = T.gated(r, col)
            if v is not None:
                buckets[cell][col].append(v)
    out = collections.defaultdict(dict)
    for cell, cols in buckets.items():
        for col, vals in cols.items():
            lo, hi = T.boot_ci(vals)
            out[cell][col] = dict(n=len(vals), median=T.median(vals), lo=lo, hi=hi)
    return out


def paired(rows, cell_of):
    """cell -> col -> {n, delta, lo, hi, p, resolved, artifact}, accel minus own dense."""
    groups, unp, incomp = T.accel_deltas(rows, cell_of=cell_of)
    out = collections.defaultdict(dict)
    for cell, cols in groups.items():
        for col, d in cols.items():
            if not d:
                continue
            lo, hi = T.boot_ci(d)
            p = T.wilcoxon_p(d)
            out[cell][col] = dict(
                n=len(d), delta=T.median(d), lo=lo, hi=hi, p=p,
                resolved=T.resolved_flag(d, lo, hi, p, col, vs_dense=True),
                artifact=T.artifact_positive(d, lo, hi, p, col, vs_dense=True))
    return out, unp, incomp


def num(v, digits):
    return '' if v is None else f'{v:.{digits}f}'


def write_unpaired_csv(path, cells, pooled, confounded_note):
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scope', 'model', 'embodiment', 'view_type', 'method', 'metric',
                    'n_clips', 'median', 'ci_lo', 'ci_hi',
                    'pooled_unpaired_is_confounded', 'note'])
        for (model, emb, vt, method), cols in sorted(cells.items()):
            for col, _, _, digits in T.COLUMNS:
                st = cols.get(col)
                if st is None:
                    continue
                w.writerow(['per_config', model, emb, vt, method, col, st['n'],
                            num(st['median'], digits + 2), num(st['lo'], digits + 2),
                            num(st['hi'], digits + 2), False, ''])
        for method, cols in sorted(pooled.items()):
            for col, _, _, digits in T.COLUMNS:
                st = cols.get(col)
                if st is None:
                    continue
                w.writerow(['pooled_all_configs', '', '', '', method, col, st['n'],
                            num(st['median'], digits + 2), num(st['lo'], digits + 2),
                            num(st['hi'], digits + 2), True, confounded_note])


def write_paired_csv(path, cells, pooled):
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scope', 'model', 'embodiment', 'view_type', 'method', 'metric',
                    'n_pairs', 'delta_median', 'ci_lo', 'ci_hi', 'p',
                    'resolved', 'artifact_positive'])
        def emit(scope, key, cols):
            for col, _, _, digits in T.COLUMNS:
                st = cols.get(col)
                if st is None:
                    continue
                w.writerow(list(key) + [col, st['n'], num(st['delta'], digits + 2),
                                        num(st['lo'], digits + 2),
                                        num(st['hi'], digits + 2),
                                        '' if st['p'] is None else f'{st["p"]:.3g}',
                                        st['resolved'], st['artifact']])
        for (method, model, emb, vt), cols in sorted(cells.items()):
            emit('per_config', ('per_config', model, emb, vt, method), cols)
        for method, cols in sorted(pooled.items()):
            emit('pooled', ('pooled_all_configs', '', '', '', method), cols)


FAMILY_OF = {'ctrlworld': 'Ctrl-World', 'ctrlworld_4view_grid': 'Ctrl-World',
             'dreamgen': 'DreamGen', 'dreamdojo': 'DreamDojo'}
FAMILY_ORDER = ['Ctrl-World', 'DreamGen', 'DreamDojo']
SPEEDUP_FULL_CSV = Path('metrics/reports/speedup_summary_full.csv')


def speedup_by_family():
    """(method, family) -> (mean speedup, n cells). Only cells whose timing was actually
    measured contribute; `speedup` is blank for the rest (`timing_source=unavailable`),
    so a missing entry is reported as `--` instead of being back-filled from a
    different model family.
    """
    if not SPEEDUP_FULL_CSV.exists():
        return {}
    inv = {v: k for k, v in METHOD_DISPLAY.items()}
    acc = collections.defaultdict(list)
    with SPEEDUP_FULL_CSV.open() as f:
        for row in csv.DictReader(f):
            k = inv.get(row['method_display'])
            if k and row['speedup']:
                acc[(k, row['family_display'])].append(float(row['speedup']))
    return {k: (sum(v) / len(v), len(v)) for k, v in acc.items()}


def method_model_grid(rows, view):
    """rows = (method, model family), cols = every metric. `view` is 'paired' or 'absolute'.

    Families, not raw model fields, because `ctrlworld` and `ctrlworld_4view_grid` are the
    same model under two view layouts and the reader wants one Ctrl-World line.

    Pooling over embodiment / mount / view_type inside a family is safe for the PAIRED
    view and only for it: each delta is accel minus its OWN dense clip, so it contains no
    absolute level and NOT_CROSS_EMBODIMENT (humanoid AMC sits ~18 points below the other
    embodiments) cannot leak in. The same pooling in the ABSOLUTE view does mix those
    levels, which is why the absolute variant is written with the confound flag.
    """
    if view == 'paired':
        cells, unp, incomp = paired(
            rows, lambda r: (r['method'], FAMILY_OF.get(r['model'], r['model'])))
        return cells, unp, incomp
    cells = unpaired(
        rows, lambda r: ((r.get('method'), FAMILY_OF.get(config_of(r)[0], config_of(r)[0]))
                         if r.get('method') else None))
    return cells, collections.Counter(), collections.Counter()


def cell_str(st, digits, view):
    """One cell. `*` = resolved, `!` = artifact-positive, `~` = n below the claim floor."""
    if st is None:
        return '--'
    if view == 'absolute':
        return f'{st["median"]:.{digits}f}' if st['median'] is not None else '--'
    d = f'{st["delta"]:+.{digits}f}'
    if st['artifact']:
        return d + '!'
    if st['resolved']:
        return d + '*'
    if st['n'] < T.MIN_PAIRS_FOR_CLAIM:
        return d + '~'
    return d


def print_method_model(cells, sp_fam, view, dense_abs=None):
    cols = [(c, lab, dg) for c, lab, _, dg in T.COLUMNS
            if not (view == 'absolute' and c in T.PAIRED_ONLY)]
    short = {'visual_integrity': 'VisInteg', 'text_condition_alignment': 'TextAlign',
             'actor_motion_raw': 'ActorAct', 'actor_motion_active_rate': 'ActRate',
             'active_motion_continuity': 'AMC'}
    head = f'{"method":13s}{"model":12s}{"speedup":>9s}' \
        + ''.join(f'{short[c]:>11s}' for c, _, _ in cols) + f'{"n":>6s}'
    print(head)
    print('-' * len(head))
    if view == 'absolute' and dense_abs:
        for fam in FAMILY_ORDER:
            st = dense_abs.get(('dense', fam))
            if not st:
                continue
            body = ''.join(cell_str(st.get(c), dg, view).rjust(11) for c, _, dg in cols)
            nmax = max((st[c]['n'] for c, _, _ in cols if c in st), default=0)
            print(f'{"dense":13s}{fam:12s}{"1.00x":>9s}{body}{nmax:6d}')
        print('-' * len(head))
    for m in METHOD_ORDER:
        first = True
        for fam in FAMILY_ORDER:
            st = cells.get((m, fam))
            if not st:
                continue
            s = sp_fam.get((m, fam))
            body = ''.join(cell_str(st.get(c), dg, view).rjust(11) for c, _, dg in cols)
            nmax = max((st[c]['n'] for c, _, _ in cols if c in st), default=0)
            print(f'{METHOD_DISPLAY[m] if first else "":13s}{fam:12s}'
                  f'{(f"{s[0]:.2f}x" if s else "--"):>9s}{body}{nmax:6d}')
            first = False
        print()


def write_method_model_csv(path, cells, sp_fam, view):
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        stat = 'median' if view == 'absolute' else 'delta_median'
        w.writerow(['method', 'model_family', 'speedup', 'speedup_n_cells', 'metric',
                    'n', stat, 'ci_lo', 'ci_hi']
                   + ([] if view == 'absolute'
                      else ['p', 'resolved', 'artifact_positive']))
        for m in METHOD_ORDER:
            for fam in FAMILY_ORDER:
                st = cells.get((m, fam))
                if not st:
                    continue
                s = sp_fam.get((m, fam))
                for col, _, _, digits in T.COLUMNS:
                    c = st.get(col)
                    if c is None:
                        continue
                    row = [m, fam, '' if not s else f'{s[0]:.4f}',
                           '' if not s else s[1], col, c['n'],
                           num(c[stat.replace('delta_median', 'delta')
                                 if view != 'absolute' else 'median'], digits + 2),
                           num(c['lo'], digits + 2), num(c['hi'], digits + 2)]
                    if view != 'absolute':
                        row += ['' if c['p'] is None else f'{c["p"]:.3g}',
                                c['resolved'], c['artifact']]
                    w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=None, help='default: <run>/tables')
    args = ap.parse_args()
    out = args.out or (args.run / 'tables')
    out.mkdir(parents=True, exist_ok=True)

    rows = load(args.run)
    sp = speedups()

    # ---- UNPAIRED: per (config, method), plus dense and GT anchor rows.
    def cell_u(r):
        m = r.get('method')
        if not m:
            return None
        return config_of(r) + (m,)
    u_cells = unpaired(rows, cell_u)

    # Pooled-unpaired: emitted for completeness, flagged as confounded (see docstring).
    note = ('config mix differs per method (only 6 of 13 configs are shared by all 9 '
            'methods); pooled unpaired ranking is Spearman +0.883 vs the paired '
            'ranking, and AMC absolutes are not cross-embodiment comparable')
    u_pooled = unpaired(rows, lambda r: r.get('method') or None)

    # ---- PAIRED: per (method, config) and pooled per method.
    p_cells, unp, incomp = paired(
        rows, lambda r: (r['method'],) + config_of(r))
    p_pooled, _, _ = paired(rows, lambda r: r['method'])

    write_unpaired_csv(out / 'paper_unpaired.csv', u_cells, u_pooled, note)
    write_paired_csv(out / 'paper_paired.csv', p_cells, p_pooled)

    # ---- The method x model-family grid: every metric as a column.
    sp_fam = speedup_by_family()
    mm_paired, _, _ = method_model_grid(rows, 'paired')
    mm_abs, _, _ = method_model_grid(rows, 'absolute')
    write_method_model_csv(out / 'paper_by_method_model_paired.csv',
                           mm_paired, sp_fam, 'paired')
    write_method_model_csv(out / 'paper_by_method_model_absolute.csv',
                           mm_abs, sp_fam, 'absolute')

    print(f'{len(rows)} records')
    print(f'  unpaired : {len(u_cells)} (config x method) cells, '
          f'{len(u_pooled)} pooled rows  -> paper_unpaired.csv')
    print(f'  paired   : {len(p_cells)} (method x config) cells, '
          f'{len(p_pooled)} pooled rows  -> paper_paired.csv')
    if incomp:
        print(f'  comparable_pair rejected: {dict(incomp)}')
    if unp:
        print(f'  no dense partner: {dict(unp)}')

    # ---- Console view: the method x model grid FIRST (this is the paper table).
    print('\n' + '=' * 100)
    print('TABLE  rows = method x model family,  cols = speedup + every metric')
    print('PAIRED delta (method - its own dense clip).  Negative = worse than dense.')
    print('=' * 100)
    print_method_model(mm_paired, sp_fam, 'paired')
    print('  * = resolved (CI excludes 0, p<0.05, n>=8)   ~ = n<8, shown but not claimable')
    print('  ! = artifact-positive: solid gain in a direction that cannot be an '
          'improvement (AMC over-smoothing)')
    print('  n = largest per-metric pair count in the row;  -- = method not run on '
          'that family / speedup not measured')

    print('\n' + '=' * 100)
    print('SAME GRID, ABSOLUTE (unpaired) values -- the VBench-style column.')
    print('CONFOUNDED: pools embodiments/mounts, and AMC absolutes are not '
          'cross-embodiment comparable.')
    print('=' * 100)
    print_method_model(mm_abs, sp_fam, 'absolute', dense_abs=mm_abs)
    print('  TextAlign withheld: absolute CLIPScore is not comparable across captions.')

    # ---- Pooled-per-method view in the field-standard layout.
    for col, label, higher, digits in T.COLUMNS:
        print(f'\n=== {label} ===')
        skip_abs = col in T.PAIRED_ONLY
        print(f'{"method":13s}{"speedup":>9s}{"absolute (unpaired)":>30s}'
              f'{"delta vs dense (paired)":>32s}')
        def abs_str(st):
            if st is None or st['lo'] is None:
                return '--'
            return (f'{st["median"]:.{digits}f} '
                    f'[{st["lo"]:.{digits}f}, {st["hi"]:.{digits}f}] n={st["n"]}')

        dn = u_pooled.get('dense', {}).get(col)
        if dn and not skip_abs:
            print(f'{"dense":13s}{"1.00x":>9s}{abs_str(dn):>30s}{"--":>32s}')
        for m in METHOD_ORDER:
            a = u_pooled.get(m, {}).get(col)
            d = p_pooled.get(m, {}).get(col)
            s = f'{sp[m][0]:.2f}x' if m in sp else '--'
            abs_s = '--' if skip_abs else abs_str(a)
            if d is None or d['lo'] is None:
                del_s = '--' if d is None else f'{d["delta"]:+.{digits}f} n={d["n"]}'
            else:
                mark = '!' if d['artifact'] else ('*' if d['resolved'] else ' ')
                del_s = (f'{d["delta"]:+.{digits}f} '
                         f'[{d["lo"]:+.{digits}f}, {d["hi"]:+.{digits}f}]{mark} n={d["n"]}')
            print(f'{METHOD_DISPLAY[m]:13s}{s:>9s}{abs_s:>30s}{del_s:>32s}')
        if skip_abs:
            print('  absolute column withheld: ' + T.PAIRED_ONLY[col].split('.')[0])
        print('  * = resolved (CI excludes 0, p<0.05, n>=8)   ! = artifact-positive')
        print('  absolute column is CONFOUNDED by config mix -- read per-config rows '
              'in paper_unpaired.csv')


if __name__ == '__main__':
    main()
