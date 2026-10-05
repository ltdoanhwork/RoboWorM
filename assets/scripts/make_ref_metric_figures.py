"""Qualitative intuition figures for the reference-based metrics and FVD.

Two figures, one per family, both built from measurements already in the repo:

  fig_ref_intuition_psnr  -- the WorldCache quality ladder of one episode: five rungs
      from dense to collapsed, each row a frame strip with that rung's PSNR / SSIM /
      LPIPS against the dense rollout. The point of the figure is the BOTTOM TWO ROWS:
      the collapsed rung scores ABOVE the merely degraded one on all three metrics
      while being visibly destroyed, which is the failure mode that motivates the
      reference-free family. Source: the staged ladder
      `sampling_dataset/data_for_web/qualitative/worldcache_quality_ladder/`
      (`tmp/20261005_refactor/web/assets_scripts/stage_worldcache_ladder_web.py`), whose `info.json` carries the
      fidelity numbers measured by `scripts/_score_vs_dense.py`.

  fig_ref_intuition_fvd   -- FVD against the number of 16-frame windows, per dense
      multi-view cell, from `metrics/reports/fvd_sample_size_probe.json`
      (`assets/scripts/fvd_sample_size_probe.py`). Shows why FVD carries no per-cell
      claim: the same cell against the same reference moves by an order of magnitude
      with n alone, and cells cross, so an unmatched-n comparison can invert.

Usage (needs cv2 + matplotlib -- use the dreamgen venv):
  models/dreamgen/.venv/bin/python assets/scripts/make_ref_metric_figures.py
  ... --only psnr        # or fvd
"""
import argparse
import json

import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_fig_style as style  # noqa: E402

LADDER = ROOT / 'sampling_dataset/data_for_web/qualitative/worldcache_quality_ladder'
PROBE = ROOT / 'metrics/reports/fvd_sample_size_probe.json'

# Which ladder episode to draw. 000829 is the episode whose collapse is unambiguous to
# the eye (the bok choy dissolves), so the metric inversion below it cannot be argued
# away as "the frames look fine"; the other two staged episodes work the same way.
CELL = 'humanoid_singleview_dreamgen'
EPISODE = 'episode_000829'
N_FRAMES = 5
FIG_W = 5.5           # inches = the ICLR \textwidth, so \linewidth does NOT rescale
                      # the figure: at 7in everything was silently shrunk to 79% and the
                      # 8pt labels printed at 6.3pt.


def ladder_rows():
    """The staged rungs in order, each with its frames and fidelity-vs-dense numbers."""
    cell = LADDER / CELL
    if not cell.is_dir():
        sys.exit(f'{cell}: staged ladder not found')
    rows = []
    for d in sorted(cell.iterdir()):
        info = d / 'info.json'
        clip = d / f'{EPISODE}.mp4'
        if not info.exists() or not clip.exists():
            continue
        meta = json.loads(info.read_text())
        rows.append(dict(
            rung=meta['rung'].split('_', 1)[1],
            speedup=meta['speedup_vs_dense'],
            fid=meta.get('fidelity_vs_dense') or {},
            desc=meta['description'],
            clip=clip,
        ))
    if len(rows) < 3:
        sys.exit(f'{cell}: only {len(rows)} rungs staged')
    return rows


def strip(clip, n):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(clip))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        sys.exit(f'{clip}: no frames decoded')
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int)
    return [frames[i] for i in idx]


def draw_psnr(out_dir, formats):
    import matplotlib
    matplotlib.use('Agg')
    style.apply(matplotlib)
    import matplotlib.pyplot as plt

    rows = ladder_rows()
    strips = [strip(r['clip'], N_FRAMES) for r in rows]
    ncol = max(len(s) for s in strips)
    h, w = strips[0][0].shape[:2]
    label_w = 0.245
    cell_w = FIG_W * (1 - label_w) / ncol
    head_in = 0.26                     # title only; the explanation is the LaTeX caption
    fig_h = len(rows) * cell_w * h / w + head_in
    fig = plt.figure(figsize=(FIG_W, fig_h), dpi=300)
    top = 1 - head_in / fig_h
    row_h = top / len(rows)

    fig.text(0.004, 1 - 0.115 / fig_h,
             'WorldCache pushed from its operating point to collapse',
             fontsize=8.0, va='center', ha='left', color='#111111')
    fig.text(0.996, 1 - 0.115 / fig_h, 'PSNR / SSIM / LPIPS vs. dense',
             fontsize=7.0, va='center', ha='right', color='#555555')

    # Only the inversion is coloured: the destroyed rung (last) against the merely
    # degraded one it outscores. Colouring by rung index would imply a smooth decline,
    # which is exactly the reading the figure exists to refute.
    scored = [i for i, r in enumerate(rows) if r['fid']]
    worst = max(scored, key=lambda i: rows[i]['fid']['mean_lpips'])
    last = len(rows) - 1
    for r, (row, frames) in enumerate(zip(rows, strips)):
        y0 = top - (r + 1) * row_h
        for j, fr in enumerate(frames):
            ax = fig.add_axes([label_w + j * (1 - label_w) / ncol, y0,
                               (1 - label_w) / ncol, row_h])
            ax.imshow(fr)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor('white')
                s.set_linewidth(0.6)

        # HIGH = the reference rung, LOW = the collapsed one. The `degraded` rung is
        # highlighted without a tag because it is neither end: it is the rung the
        # collapsed one outscores, which is the comparison the figure is about.
        tag = 'HIGH' if r == 0 else ('LOW' if r == last else None)
        col = ('#1e6f4a' if r == 0 else
               '#a93226' if r == last else
               '#b7791f' if r == worst else '#222222')
        ax = fig.add_axes([0.004, y0, label_w - 0.012, row_h])
        ax.axis('off')
        if r in (0, last, worst):
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                       facecolor=col, alpha=0.06, lw=0))
        fid = row['fid']
        head = f'{tag}   {row["rung"]}' if tag else row['rung']
        blocks = [(f'{head}   {row["speedup"]:.2f}$\\times$', 7.6, col, 'bold')]
        if fid:
            blocks.append((f'PSNR {fid["mean_psnr"]:.2f}   SSIM {fid["mean_ssim"]:.3f}\n'
                           f'LPIPS {fid["mean_lpips"]:.3f}', 6.0, col, 'normal'))
        else:
            blocks.append(('the row the three metrics\nare measured against',
                           5.9, '#666666', 'normal'))
        if r == last:
            blocks.append(('outscores the row above\non all three',
                           5.6, col, 'normal'))
        row_pt = row_h * fig_h * 72
        y_pt = 3.0
        for txt, size, c, weight in blocks:
            ax.text(0.05, 1 - y_pt / row_pt, txt, fontsize=size, color=c, va='top',
                    ha='left', fontweight=weight, linespacing=1.25,
                    transform=ax.transAxes)
            y_pt += (txt.count('\n') + 1) * size * 1.25 + 3.5

    save(fig, out_dir, 'fig_ref_intuition_psnr', formats)


def draw_fvd(out_dir, formats):
    import matplotlib
    matplotlib.use('Agg')
    style.apply(matplotlib)
    import matplotlib.pyplot as plt
    import numpy as np

    probe = json.loads(PROBE.read_text())
    budgets = probe['budgets']
    # A cell enters only if every budget resolved, so panel (a)'s curves and panel (b)'s
    # ranking are over the SAME set -- otherwise (b) would rank cells (a) never showed.
    K = 512
    ks = [k for k in budgets if k <= K]
    cells = [c for c in probe['cells']
             if all(c.get(f'fvd_k{k}') for k in ks) and c.get('fvd_full')]
    if len(cells) < 4:
        sys.exit('probe has too few cells with a complete budget ladder')
    cells.sort(key=lambda c: c['fvd_full'])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(FIG_W, 2.6), dpi=300,
                                   gridspec_kw=dict(width_ratios=[1.15, 1.0],
                                                    wspace=0.32))

    infl = []
    for c in cells:
        xs = ks + [c['n_ref']]
        ys = [c[f'fvd_k{k}'] for k in ks] + [c['fvd_full']]
        axA.plot(xs, ys, marker='o', ms=1.8, lw=0.7, color='#4a5568', alpha=0.45)
        infl.append(c[f'fvd_k{ks[0]}'] / c['fvd_full'])
    axA.set_xscale('log')
    axA.set_yscale('log')
    axA.set_xlabel(f'16-frame windows $n$ (log); leftmost point is $n$ = {ks[0]}',
                   fontsize=6.6)
    axA.set_ylabel('FVD (log)', fontsize=7)
    axA.set_title(f'(a) {len(cells)} dense cells: same reference, only $n$ changes',
                  fontsize=7.4, loc='left', pad=4)
    axA.tick_params(labelsize=6)
    axA.grid(alpha=0.22, lw=0.4)
    axA.text(0.03, 0.05,
             f'median inflation at $n$={ks[0]}:  '
             f'{np.median(infl):.1f}$\\times$ the full-$n$ value',
             transform=axA.transAxes, fontsize=6.2, color='#a93226', va='bottom',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.6))

    # (b) ranking at full n against ranking at a matched budget: crossing lines are the
    # reason an unmatched-n FVD comparison cannot carry a per-cell claim.
    full = sorted(cells, key=lambda c: c['fvd_full'])
    matched = sorted(cells, key=lambda c: c[f'fvd_k{K}'])
    pos_full = {c['cell']: i for i, c in enumerate(full)}
    pos_match = {c['cell']: i for i, c in enumerate(matched)}
    moved = 0
    for c in cells:
        a, b = pos_full[c['cell']], pos_match[c['cell']]
        moved += a != b
        axB.plot([0, 1], [a, b], lw=1.0 if a != b else 0.7, marker='o', ms=2.4,
                 color='#a93226' if a != b else '#9aa3b0',
                 alpha=0.95 if a != b else 0.45)
    axB.set_xlim(-0.3, 1.3)
    axB.set_xticks([0, 1])
    axB.set_xticklabels([f'rank at full $n$\n($n$ = {min(c["n_ref"] for c in cells)}'
                         f'--{max(c["n_ref"] for c in cells)})',
                         f'rank at matched\n$n$ = {K}'], fontsize=6)
    axB.invert_yaxis()
    axB.set_ylabel('rank (1 = lowest FVD)', fontsize=7)
    axB.set_yticks(range(0, len(cells), 2))
    axB.set_yticklabels([str(i + 1) for i in range(0, len(cells), 2)], fontsize=6)
    axB.set_title(f'(b) {moved} of {len(cells)} cells change rank', fontsize=7.4,
                  loc='left', pad=4)
    axB.grid(alpha=0.22, lw=0.4, axis='y')
    for s in ('top', 'right'):
        axA.spines[s].set_visible(False)
        axB.spines[s].set_visible(False)

    save(fig, out_dir, 'fig_ref_intuition_fvd', formats)
    print(f'  (a) median inflation at n={ks[0]}: {np.median(infl):.2f}x; '
          f'(b) {moved}/{len(cells)} cells change rank at matched n={K}')


def save(fig, out_dir, stem, formats):
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        p = out_dir / f'{stem}.{ext}'
        fig.savefig(p, bbox_inches='tight', pad_inches=0.012)
        try:
            print(f'wrote {p.relative_to(ROOT)}')
        except ValueError:
            print(f'wrote {p}')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'assets/paper/figures')
    ap.add_argument('--formats', nargs='+', default=['pdf'],
                    help='pdf is what the paper includes; add png for a quick preview')
    ap.add_argument('--only', nargs='+', choices=['psnr', 'fvd'],
                    default=['psnr', 'fvd'])
    a = ap.parse_args()
    if 'psnr' in a.only:
        draw_psnr(a.out, a.formats)
    if 'fvd' in a.only:
        draw_fvd(a.out, a.formats)


if __name__ == '__main__':
    main()
