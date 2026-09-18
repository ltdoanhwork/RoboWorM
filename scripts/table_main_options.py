#!/usr/bin/env python
"""Render SEVERAL candidate designs for the single main-results table slot, so the
scale choice can be made by looking at real numbers instead of arguing in the abstract.

The problem being solved: the 5 raw columns live on incompatible scales
(VisualIntegrity 13-60, AMC 19-87, TextAlign 16-36, ActorActivity 0-27 px/frame,
ActiveRate 0-1), so a reader cannot compare "-1.22 VI" against "-0.99 AMC" against
"-0.005 AR", and 5 separate matrices do not fit one table slot.

Normalisation options that were REJECTED by measurement first:

  * GT-headroom ("% of the dense->GT gap this method throws away"). Attractive because
    we uniquely have both anchors, but measured per config the gap is 0.2-2.6 points
    and OFTEN THE WRONG SIGN (VI dense is ABOVE GT by +0.53 on ctrlworld/bimanual;
    AMC by +3.93 on ctrlworld/humanoid), and 5 of 13 configs have no GT anchor at all.
    A near-zero, sign-flipping denominator makes the ratio meaningless.
  * WorldScore-style empirical bounds with z-score rescaling so models land in 25-75
    (their Supp. C.8 recipe for metrics without natural bounds). That is designed for
    ranking MODELS on absolute scores; here every claim is a PAIRED delta against the
    method's own dense counterpart, so the bounds would be an extra arbitrary layer on
    top of a difference that is already self-normalising.

What survives: normalise by the DENSE LEVEL of the same cell (retention), or drop
units entirely (win rate / effect size). Both are computed below.

Usage:
  python scripts/table_main_options.py --run metrics/WorldScore/results/dense_nr_v3_pooled
"""
import argparse
import collections
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    'table_dense_nr', Path(__file__).parent / 'table_dense_nr.py')
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
_spec2 = importlib.util.spec_from_file_location(
    'table_main_results', Path(__file__).parent / 'table_main_results.py')
M = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(M)

# The directional quality columns. ActorActivity / ActiveRate are DESCRIPTIVE (no
# "better" direction) so they must never enter a quality composite -- a method that
# moves the arm more is not better. They are reported as a separate behaviour check.
QUALITY = ['visual_integrity', 'active_motion_continuity', 'text_condition_alignment']
BEHAVIOUR = ['actor_motion_raw', 'actor_motion_active_rate']


def load(run):
    return [json.loads(p.read_text()) for p in (run / 'results').glob('*.json')]


def per_pair(rows):
    """(method, config) -> col -> list of (dense_value, accel_value) for comparable pairs.

    Kept as PAIRS, not deltas, because retention needs the dense level of the very same
    pairs that produced the delta -- using a cell-wide dense median instead would mix
    denominators from clips that never entered the delta.
    """
    dense, accel = {}, collections.defaultdict(list)
    for r in rows:
        if r.get('method') == 'dense':
            dense[(r['pair_key'], r['model'])] = r
        elif r.get('method') not in (None, 'gt'):
            accel[(r['pair_key'], r['model'])].append(r)
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for key, variants in accel.items():
        base = dense.get(key)
        if base is None:
            continue
        for r in variants:
            if not T.comparable_pair(r, base):
                continue
            cell = (r['method'], r['model'], r['embodiment'], M.view_type(r))
            for col, _, _, _ in T.COLUMNS:
                a, b = T.gated(base, col), T.gated(r, col)
                if a is not None and b is not None:
                    out[cell][col].append((a, b))
    return out


def stats(pairs):
    """cell -> col -> dict of every candidate statistic, computed from the same pairs."""
    out = collections.defaultdict(dict)
    for cell, cols in pairs.items():
        for col, ps in cols.items():
            if not ps:
                continue
            d = [b - a for a, b in ps]
            lo, hi = T.boot_ci(d)
            p = T.wilcoxon_p(d)
            # Retention: median of the per-pair ratio, so a cell-wide outlier in the
            # dense level cannot dominate. Guard the near-zero denominators that
            # ActorActivity/ActiveRate can produce.
            rat = [b / a for a, b in ps if abs(a) > 1e-6]
            worse = sum(1 for x in d if x < 0)
            better = sum(1 for x in d if x > 0)
            out[cell][col] = dict(
                n=len(d), delta=T.median(d), lo=lo, hi=hi, p=p,
                dense=T.median([a for a, _ in ps]),
                retention=100 * T.median(rat) if rat else None,
                # Cliff's delta on the SIGN of the paired difference = win/loss rate.
                # Unitless and bounded [-1, 1]; immune to every scale problem.
                cliff=(better - worse) / len(d),
                worse_frac=worse / len(d),
                resolved=T.resolved_flag(d, lo, hi, p, col, vs_dense=True),
                artifact=T.artifact_positive(d, lo, hi, p, col, vs_dense=True))
    return out


def composite(cell_stats, cols=QUALITY):
    """Mean retention over the directional quality columns present in this cell.

    Mean of RETENTIONS (not of raw deltas) is the whole point: each column is already
    expressed as a fraction of its own dense level, so they are commensurable. Columns
    missing from a cell (TextAlignment on DreamDojo, which is action-conditioned and has
    no prompt) are skipped rather than imputed, and the count is returned so a cell built
    from fewer columns is never silently compared to a full one.

    WARNING measured 2026-09-17: this composite is valid PER CELL but must NOT be
    averaged over cells to rank methods. Doing so gives Spearman +0.883 against the
    pooled T3b ordering, not +1.0 -- it moves radial from best (T3b dVI -0.07) to 6th,
    because methods are not run on the same set of configs and small-n cells get equal
    weight. Rank methods with T3b (pooled over PAIRS); use this grid to show WHERE a
    method breaks.
    """
    vals = [cell_stats[c]['retention'] for c in cols
            if c in cell_stats and cell_stats[c]['retention'] is not None]
    if not vals:
        return None, 0, False, False, 0
    n_res = sum(1 for c in cols if c in cell_stats and cell_stats[c]['resolved'])
    n_art = sum(1 for c in cols if c in cell_stats and cell_stats[c]['artifact'])
    # Smallest n across the averaged columns: the cell is only as trustworthy as its
    # thinnest column, and 11 of 37 unresolved cells are unresolved purely for lack of
    # data. Printing those as "no effect" would be an unfalsifiable claim.
    n_min = min(cell_stats[c]['n'] for c in cols
                if c in cell_stats and cell_stats[c]['retention'] is not None)
    return sum(vals) / len(vals), len(vals), n_res > 0, n_art > 0, n_min


def grid(rkeys, methods, cellfn, width=9, head=44):
    lines = [' ' * head + ''.join(f'{M.METHOD_DISPLAY[m][:width - 1]:>{width}s}'
                                  for m in methods)]
    for rk in rkeys:
        lines.append(f'{M.row_label(rk)[:head - 2]:{head - 2}s}  '
                     + ''.join(f'{cellfn(rk, m):>{width}s}' for m in methods))
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    rows = load(args.run)
    S = stats(per_pair(rows))
    rkeys = M.row_keys(S)
    meths = M.METHOD_ORDER

    def cs(rk, m, cols=QUALITY):
        c = S.get((m,) + rk)
        return composite(c, cols) if c else (None, 0, False, False, 0)

    print(f'{len(rows)} records, {len(S)} cells, {len(rkeys)} configs x {len(meths)} methods')

    print('\n' + '=' * 128)
    print('OPTION 1 - Quality Retention %  (mean over VI/AMC/TextAlign of median '
          'per-pair accel/dense ratio).  dense = 100.0')
    print('  bold-equivalent [x] = at least one column resolved;  x! = artifact-positive;'
          '  superscript = #columns averaged (3 = full, 2 = DreamDojo has no TextAlign)')
    print('=' * 128)

    def f1(rk, m):
        v, k, res, art, n = cs(rk, m)
        if v is None:
            return '--'
        s = f'{v:.1f}'
        if art:
            s += '!'
        elif res:
            s = f'[{s}]'
        elif n < T.MIN_PAIRS_FOR_CLAIM:
            s += '*'
        return s + ('' if k == 3 else f'^{k}')
    print(grid(rkeys, meths, f1))

    print('\n' + '=' * 128)
    print('OPTION 2 - Loss Index: 100 - retention, i.e. "% of dense quality lost". '
          '0 = free, larger = worse')
    print('=' * 128)

    def f2(rk, m):
        v, k, res, art, n = cs(rk, m)
        if v is None:
            return '--'
        s = f'{100 - v:+.1f}'
        if art:
            s += '!'
        elif res:
            s = f'[{s}]'
        elif n < T.MIN_PAIRS_FOR_CLAIM:
            s += '*'
        return s
    print(grid(rkeys, meths, f2))

    print('\n' + '=' * 128)
    print('OPTION 3 - Win/Loss rate: % of paired episodes where the method is WORSE '
          'than dense, averaged over VI/AMC/TextAlign.')
    print('  Completely unitless -- no scale problem can exist. 50 = coin flip '
          '(indistinguishable), 100 = worse on every episode.')
    print('=' * 128)

    def f3(rk, m):
        c = S.get((m,) + rk)
        if not c:
            return '--'
        w = [c[x]['worse_frac'] for x in QUALITY if x in c]
        if not w:
            return '--'
        res = any(c[x]['resolved'] for x in QUALITY if x in c)
        s = f'{100 * sum(w) / len(w):.0f}'
        return f'[{s}]' if res else s
    print(grid(rkeys, meths, f3, width=8))

    print('\n' + '=' * 128)
    print('OPTION 4 - Two-row cell: the two full-coverage directional columns as '
          'retention %.  VI on top, AMC below.')
    print('=' * 128)
    print(' ' * 44 + ''.join(f'{M.METHOD_DISPLAY[m][:8]:>9s}' for m in meths))
    for rk in rkeys:
        for col, tag in (('visual_integrity', 'VI '), ('active_motion_continuity', 'AMC')):
            lab = M.row_label(rk)[:38] if tag == 'VI ' else ''
            line = f'{lab:38s} {tag:4s}'
            for m in meths:
                c = (S.get((m,) + rk) or {}).get(col)
                if not c or c['retention'] is None:
                    line += f'{"--":>9s}'
                    continue
                s = f'{c["retention"]:.1f}'
                if c['artifact']:
                    s += '!'
                elif c['resolved']:
                    s = f'[{s}]'
                line += f'{s:>9s}'
            print(line)

    print('\n' + '=' * 128)
    print('OPTION 5 - Severity tiers on the Loss Index (only resolved cells get a '
          'filled tier; ns = not distinguishable from dense)')
    print('  ...=lost >5%  ..=2-5%  .=0.5-2%  ~=<0.5% but resolved  ns=tested, no effect  '
          '?=n<8 UNTESTED  !=over-smoothing artifact')
    print('=' * 128)

    def f5(rk, m):
        v, k, res, art, n = cs(rk, m)
        if v is None:
            return '--'
        if art:
            return '!'
        if not res:
            # n<8 is NOT "no effect" -- 11 of 37 unresolved cells are unresolved only
            # for lack of data. Printing both as "ns" asserts a null that was never
            # tested, so thin cells get their own symbol.
            return '?' if n < T.MIN_PAIRS_FOR_CLAIM else 'ns'
        loss = 100 - v
        if loss <= 0:
            return '+'
        return '...' if loss > 5 else '..' if loss > 2 else '.' if loss > 0.5 else '~'
    print(grid(rkeys, meths, f5, width=6))

    print('\n' + '=' * 128)
    print('OPTION 6 - TRANSPOSED (methods as rows, configs as columns) on the Loss '
          'Index.  9 rows x 12 cols fits a page better than 12 x 9.')
    print('=' * 128)
    short = {rk: f'{M.MODEL_DISPLAY.get(rk[0], (rk[0], ""))[0][:4]}'
                 f'{"4v" if rk[0].endswith("grid") else "  "}/'
                 f'{rk[1][:3]}/{M.VIEW_DISPLAY[rk[2]]}' for rk in rkeys}
    print(' ' * 13 + ''.join(f'{short[rk]:>15s}' for rk in rkeys))
    for m in meths:
        line = f'{M.METHOD_DISPLAY[m]:12s} '
        for rk in rkeys:
            v, k, res, art, n = cs(rk, m)
            s = '--' if v is None else (f'{100 - v:+.1f}' + ('!' if art else ''))
            if v is not None and res and not art:
                s = f'[{s}]'
            elif v is not None and not art and n < T.MIN_PAIRS_FOR_CLAIM:
                s += '*'
            line += f'{s:>15s}'
        print(line)

    print('\n' + '=' * 128)
    print('BEHAVIOUR CHECK (never folded into the quality composite): ActorActivity and '
          'ActiveRate retention %.')
    print('  ~100 = the method preserves how much the robot moves. Far below 100 = the '
          'motion itself was destroyed, which is a different failure from blur.')
    print('=' * 128)
    print(' ' * 44 + ''.join(f'{M.METHOD_DISPLAY[m][:8]:>9s}' for m in meths))
    for rk in rkeys:
        for col, tag in (('actor_motion_raw', 'Act'), ('actor_motion_active_rate', 'Rate')):
            lab = M.row_label(rk)[:38] if tag == 'Act' else ''
            line = f'{lab:38s} {tag:4s}'
            for m in meths:
                c = (S.get((m,) + rk) or {}).get(col)
                if not c or c['retention'] is None:
                    line += f'{"--":>9s}'
                    continue
                s = f'{c["retention"]:.0f}'
                if c['resolved']:
                    s = f'[{s}]'
                line += f'{s:>9s}'
            print(line)


if __name__ == '__main__':
    main()
