#!/usr/bin/env python3
"""Build the two-panel Speedup x RIGIS/KIVA figure for one embodiment.

Reads `metrics/reports/speedup_figure_data_<embodiment>.json` (source of truth --
never retype the numbers) and writes a full-width `.pdf` + `.png` pair.

Panel (a) = multi-view tracks, middle axis RIGIS (higher is better).
Panel (b) = single-view tracks, middle axis KIVA % (lower is better).
Bars = speedup vs the dense rollout of the SAME backbone. The boxed value above each
bar is that backbone's aggregate retention Q: the mean of Visual Integrity, PSNR,
Text Alignment and FVD, each relative to its own dense run, with FVD inverted so
every component points up. Q = 1 is dense; Q > 1 is admissible.

Q is printed PER BACKBONE, not averaged over the panel's two models: the two
disagree by a median of 0.10 and by up to 0.39 (FasterCache on humanoid SV), so an
averaged value would erase the disagreement rather than summarise it.

Missing entries stay missing: FasterCache has no Ctrl-World MV run, SiTo/ITM have no
DreamDojo SV run, and DreamDojo has no Text Alignment (its Q is over 3 metrics).
Never substitute 0 -- a zero bar reads as "measured and slow" rather than "not run".

Usage:
  models/dreamgen/.venv/bin/python assets/scripts/make_speedup_audit_figure.py \
      --embodiment humanoid [--split makovian] [--print-only]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]

# Workbook method label -> figure category label, in figure order.
METHODS = [('Dense', 'Dense'), ('World-cache', 'WorldCache'), ('Di-cache', 'DiCache'),
           ('Faster-cache', 'FasterCache'), ('sito', 'SiTo'), ('itm', 'ITM'),
           ('PISA', 'PISA'), ('Radial Attention', 'Radial'), ('SVG1', 'SVG1'),
           ('SVG2', 'SVG2')]
CATS = [c for _m, c in METHODS]

PANELS = {
    'a_multiview': dict(tracks=['Dreamgen MV', 'Ctrl-World MV'],
                        labels=['DreamGen MV', 'Ctrl-World MV'],
                        metric='RIGIS', fmt='{:.3f}'),
    'b_singleview': dict(tracks=['Dreamgen SV', 'DreamDojo SV'],
                         labels=['DreamGen SV', 'DreamDojo SV'],
                         metric='KIVA %', fmt='{:.1f}'),
}
EMB_TITLE = {'bimanual': 'Bimanual', 'humanoid': 'Humanoid', 'single_arm': 'Single-arm'}

BLUE, PURPLE, MAROON = '#2f4b6e', '#5e2d91', '#8b2e2e'
BLUE_L, PURPLE_L = '#8ba3c4', '#b39ad0'
BAR = ['#5a5a5a', '#2e8b57', '#2e8b57', '#2e8b57', '#cc5500', '#cc5500',
       '#3b5ba5', '#3b5ba5', '#3b5ba5', '#3b5ba5']


def series(rows: list[dict], panel: str, split: str = 'makovian') -> dict:
    """Pull the arrays a panel needs, keyed by figure order.

    Every series is indexed by the SAME (track, split) pair. The earlier version of this
    figure indexed the second mean series by split instead of by track, so the hatched
    column showed Ctrl-World bars/RIGIS next to DreamGen-non_makovian means -- which
    understated Ctrl-World DiCache (0.83 shown vs 1.008 actual) and Radial (0.82 vs
    1.000). `_assert_aligned` below now makes that class of mix-up impossible.
    """
    ix = {(r['track'], r['method']): r for r in rows
          if r['panel'] == panel and r['split'] == split}
    tracks = PANELS[panel]['tracks']
    out = dict(bars=[[], []], mid=[[], []], means=[[], []], src=[[], []])
    for m, _c in METHODS:
        for t_i, t in enumerate(tracks):
            r = ix.get((t, m))
            g = lambda k: (float(r[k]) if r and r.get(k) not in (None, '') else None)
            out['bars'][t_i].append(g('speedup'))
            out['mid'][t_i].append(g('audit_value'))
            out['means'][t_i].append(g('mean_dense'))
            out['src'][t_i].append((r['track'], r['split'], r['dr_track'],
                                    r['n_used_dense']) if r else None)
    return out


def _assert_aligned(S, panel, split):
    """Every value in column i must come from the panel's track i at `split`.

    A mismatch here is the bug that produced the first draft of this figure; it is
    silent in the plot, so it is checked rather than eyeballed.
    """
    for t_i, t in enumerate(PANELS[panel]['tracks']):
        for i, s in enumerate(S['src'][t_i]):
            if s is None:
                continue
            assert s[0] == t and s[1] == split, (
                f'{panel} col{t_i} {CATS[i]}: got ({s[0]}, {s[1]}), want ({t}, {split})')
        # A bar with no mean (or vice versa) means one source has the cell and the other
        # does not -- draw it, but say so, never silently pair it with a neighbour.
        for i, (b, mv) in enumerate(zip(S['bars'][t_i], S['means'][t_i])):
            if (b is None) != (mv is None):
                print(f'  note: {panel} {t} {CATS[i]}: speedup={b} mean={mv} '
                      '(partial cell, drawn as-is)')


def audit_real(rows: list[dict], panel: str) -> float | None:
    for r in rows:
        if r['panel'] == panel and r.get('audit_real') not in (None, ''):
            return float(r['audit_real'])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--embodiment', default='humanoid',
                    choices=sorted(EMB_TITLE))
    ap.add_argument('--data', type=Path, default=None)
    ap.add_argument('--split', default='makovian',
                    choices=['makovian', 'non_makovian'])
    ap.add_argument('--out-dir', type=Path,
                    default=REPO / 'assets/paper_figures/speedup_audit')
    ap.add_argument('--print-only', action='store_true',
                    help='print the extracted arrays and exit (for verification)')
    args = ap.parse_args()

    src = args.data or (REPO / 'metrics/reports'
                        / f'speedup_figure_data_{args.embodiment}.json')
    rows = json.loads(src.read_text())['rows']
    A = series(rows, 'a_multiview', args.split)
    B = series(rows, 'b_singleview', args.split)
    _assert_aligned(A, 'a_multiview', args.split)
    _assert_aligned(B, 'b_singleview', args.split)
    rv_a, rv_b = audit_real(rows, 'a_multiview'), audit_real(rows, 'b_singleview')

    def show(tag, S, panel, fmt):
        print(f'--- {tag} ({PANELS[panel]["metric"]}) ---')
        t = PANELS[panel]['labels']
        g = lambda v, f='{:.2f}': '  --  ' if v is None else f.format(v)
        print(f'{"method":13s}|{"spd "+t[0][:9]:13s}|{"spd "+t[1][:9]:13s}|'
              f'{"mid "+t[0][:9]:13s}|{"mid "+t[1][:9]:13s}|'
              f'{"mean "+t[0][:8]:13s}|{"mean "+t[1][:8]:13s}| dr_track / n')
        for i, c in enumerate(CATS):
            s0, s1 = S['src'][0][i], S['src'][1][i]
            prov = ' , '.join('--' if s is None else f'{s[2]}/n{s[3]}'
                              for s in (s0, s1))
            print(f'{c:13s}|{g(S["bars"][0][i]):13s}|{g(S["bars"][1][i]):13s}|'
                  f'{g(S["mid"][0][i], fmt):13s}|{g(S["mid"][1][i], fmt):13s}|'
                  f'{g(S["means"][0][i], "{:.4f}"):13s}|'
                  f'{g(S["means"][1][i], "{:.4f}"):13s}| {prov}')

    show(f'{args.embodiment} panel (a)', A, 'a_multiview', '{:.4f}')
    show(f'{args.embodiment} panel (b)', B, 'b_singleview', '{:.2f}')
    print(f'real-video levels: RIGIS={rv_a}  KIVA={rv_b}')
    if args.print_only:
        return

    from figure_speedup_audit_draw import draw           # noqa: E402
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f'fig_speedup_rigis_kiva_{args.embodiment}'
    draw(A, B, EMB_TITLE[args.embodiment], rv_a, rv_b, stem,
         CATS, BAR, PANELS, MAROON, BLUE, BLUE_L, PURPLE, PURPLE_L)
    print(f'wrote {stem}.pdf\nwrote {stem}.png')


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
