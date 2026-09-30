#!/usr/bin/env python3
"""Drawing half of the Speedup x RIGIS/KIVA figure.

Kept separate from the data extraction so the style spec lives in one place and can be
reused by any embodiment without duplicating the matplotlib.

Layout rules below are not cosmetic -- each fixes a misreading that actually happened:

* Each metric axis spans the METRIC'S OWN range (RIGIS [0, 1]; KIVA [0, cap <= 100]).
  Separation from the bars is bought on the BAR axis, never by compressing or shifting
  the metric axis: a 150% tick for a percentage, or a negative floor, misrepresents the
  axis to solve a layout problem.
* Value labels are placed by COLLISION SEARCH (`_place`), not fixed offsets. Each label
  tries an ordered list of offsets and takes the first that overlaps nothing already
  drawn. A `label placement fallback` warning means that panel needs one more candidate.
* Axis limits are finalised BEFORE any label is measured. Extents are in display space,
  so measuring while the axes still carry autoscaled limits measures the wrong pixels
  and every collision test silently passes -- this hid several "fixes" that did nothing.
* Mean-box candidates are VERTICAL ONLY. With 10 groups x 2 models the boxes are ~1
  bar-width apart, so a sideways dodge makes a box read as belonging to the neighbouring
  bar (SVG1/SVG2 looked swapped).
* The two backbones of a panel are distinguished by COLOUR (dark/light) AND marker
  (`_` vs `|`), never by linestyle alone.
* Every marker gets a fat white marker underneath and every label a white bbox, so
  nothing disappears into a bar.
* Legend and panel title are FIGURE-level, anchored off `ax.get_position()`; inside the
  axes they collide with the bars and with each other.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox                                     # noqa: E402

MEAN_LABEL = ('aggregate retention Q per backbone (Visual Int., PSNR, Text Align., FVD; '
              '3 for DreamDojo SV)')


def _extent(art, fig, pad=3.0):
    """Display-space bbox of a drawn artist, padded a little.

    Used for collision-aware label placement; requires the figure to have been drawn at
    least once so the renderer knows the text metrics.
    """
    bb = art.get_window_extent(fig.canvas.get_renderer())
    return bb.expanded(1.0 + pad / max(bb.width, 1), 1.0 + pad / max(bb.height, 1))


def _place(ax, fig, pts, fmt, color, bb_style, occupied, cands, values=None,
           zorder=7):
    """Draw each value label at the first candidate offset that hits nothing.

    The mean boxes sit on the bar tops (as in the reference figure) and the mid-axis line
    sweeps through those same heights, so a single fixed offset always collides on some
    method. Trying a short ordered list of offsets per point -- and testing against what
    is already on the canvas -- keeps the requested layout without distorting any axis.
    Falls back to the first candidate if every option is blocked, so a label is never
    dropped silently.

    `values` lets the anchor differ from the printed number (mean boxes are anchored at
    the bar top but print the mean).
    """
    for i, (xi, v) in enumerate(pts):
        txt = fmt.format(v if values is None else values[i])
        placed = None
        for dx, dy, ha, va in cands:
            t = ax.annotate(txt, (xi, v), textcoords='offset points',
                            xytext=(dx, dy), ha=ha, va=va, fontsize=19, color=color,
                            fontweight='bold', bbox=bb_style, zorder=zorder)
            fig.canvas.draw()
            e = _extent(t, fig)
            if not any(e.overlaps(o) for o in occupied):
                placed = (t, e)
                break
            t.remove()
        if placed is None:
            # Every candidate is blocked. Keep the label (never drop data) at the first
            # offset, but say so -- a reported fallback means the panel needs one more
            # candidate offset, not that the number is wrong.
            dx, dy, ha, va = cands[0]
            print(f'  label placement fallback: {txt} at x={xi:.2f}')
            t = ax.annotate(txt, (xi, v), textcoords='offset points',
                            xytext=(dx, dy), ha=ha, va=va, fontsize=19, color=color,
                            fontweight='bold', bbox=bb_style, zorder=zorder)
            fig.canvas.draw()
            placed = (t, _extent(t, fig))
        occupied.append(placed[1])


def _panel(ax, cats, x, bar_colors, b1, b2, mid1, mid2, m1, m2, mid_color, mid_color_l,
           mid_label, fmt, left_ylim, mid_ylim, mid_ticks, maroon,
           bar_text_y=0.42):
    ax2 = ax.twinx()
    # The mid-axis value labels are the readout a reader cares about; keep the mid axis
    # composited above the bars' axis and transparent so bars still show through.
    ax.set_zorder(2)
    ax2.set_zorder(3)
    ax2.patch.set_visible(False)
    ax2.spines['right'].set_color(mid_color)
    ax2.tick_params(axis='y', colors=mid_color, labelsize=20)
    ax2.set_ylabel(mid_label, color=mid_color, fontsize=24)
    ax.tick_params(axis='y', labelsize=20)

    w = 0.46
    # Mean is annotated per model in a boxed label sitting ON each bar's top (matching the
    # reference figure), not as a third y-axis: the two models' means differ enough (e.g.
    # Ctrl-World DiCache 1.01 vs DreamGen 0.86) that one averaged line hides which model
    # degraded. Their bboxes are collected so the mid-axis labels can dodge them.
    mbox = dict(boxstyle='round,pad=0.22', fc='white', ec=maroon, lw=1.6)
    occupied, mean_pts, bar_art = [], [], []
    for i in range(len(cats)):
        for xi, bv, mv, alpha, hatch, tcol in (
                (x[i] - w / 2, b1[i], m1[i], 1.0, None, 'white'),
                (x[i] + w / 2, b2[i], m2[i], 0.45, '///', '#333')):
            if bv is not None:
                ax.bar(xi, bv, w, color=bar_colors[i], alpha=alpha, hatch=hatch,
                       edgecolor='white', zorder=2)
                # Anchored at the BASE of the bar, not its middle: the mid-axis line runs
                # through the bars, and a centred "1.95x" collides with it.
                # Start ABOVE the mid-axis markers, not at the axis floor: the markers
                # are drawn inside the bars and a floor-anchored "1.12x" ends up under
                # one of them. `bar_text_y` is a fraction of the bar's own height so a
                # short bar still shows its label.
                bar_art.append(ax.text(
                    xi, bv * bar_text_y, f'{bv:.2f}x', ha='center', va='bottom',
                    color=tcol, fontsize=18, rotation=90, fontweight='bold',
                    zorder=3, clip_on=True))
            if mv is not None:
                mean_pts.append((xi, bv if bv is not None else 0.0, mv))

    def split(xs, ys):
        f = [(a, b) for a, b in zip(xs, ys) if b is not None]
        return [a for a, _ in f], [b for _, b in f]

    xs1, yy1 = split([xi - 0.22 for xi in x], mid1)
    xs2, yy2 = split([xi + 0.22 for xi in x], mid2)
    ax2.plot(xs1, yy1, '-', color=mid_color, lw=3.2, zorder=4)
    ax2.plot(xs1, yy1, ls='none', marker='_', ms=40, mew=10, color='white', zorder=5)
    ax2.plot(xs1, yy1, ls='none', marker='_', ms=32, mew=5.5, color=mid_color, zorder=6)
    ax2.plot(xs2, yy2, ':', color=mid_color_l, lw=3.2, zorder=4)
    ax2.plot(xs2, yy2, ls='none', marker='|', ms=40, mew=10, color='white', zorder=5)
    ax2.plot(xs2, yy2, ls='none', marker='|', ms=32, mew=5.5, color=mid_color_l,
             zorder=6)
    # Axis limits/ticks MUST be finalised before any label is measured: extents are in
    # display space, so placing labels while the axes still carry their autoscaled limits
    # measures the wrong pixels and every collision test silently passes.
    ax.set_ylim(*left_ylim)
    ax2.set_ylim(*mid_ylim)
    ax2.set_yticks(mid_ticks)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=20)
    # Widen past the last group: the mid-axis value labels are centred on x+-0.22 and
    # the outermost ones ("0.748", "46.4") get clipped by the axes edge otherwise --
    # they render as "748" / "46." which reads as a different number.
    ax.set_xlim(x[0] - 1.25, x[-1] + 1.25)
    ax.set_ylabel('Speedup', fontsize=24)
    ax.axhline(1.0, color='black', ls='--', lw=1.8, zorder=1)
    ax.grid(axis='y', alpha=0.22, zorder=0)

    bb = dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85)
    fig = ax.figure
    fig.canvas.draw()
    occupied += [_extent(a, fig) for a in bar_art]
    # The x tick labels are a no-go zone too: the low series can run near the floor, and a
    # "below" placement there lands on "WorldCache".
    occupied += [_extent(t, fig) for t in ax.get_xticklabels()]
    # The mid-axis MARKERS are obstacles as well, not just the labels: they are ~32pt wide
    # dashes/bars drawn inside the plot, and a mean box placed at a bar top can land right
    # on one (seen on humanoid PISA/SVG1). Approximated by a small box at each point.
    for xs, ys in ((xs1, yy1), (xs2, yy2)):
        for xi, v in zip(xs, ys):
            px, py = ax2.transData.transform((xi, v))
            occupied.append(Bbox.from_bounds(px - 22, py - 14, 44, 28))

    # Mean boxes go down FIRST (they are the headline number and sit on the bar tops as in
    # the reference figure), dodging the markers/bar text already registered above; the
    # mid-axis labels then dodge the boxes. Drawing them in the other order left the boxes
    # sitting under the markers.
    #
    # Their candidates are VERTICAL ONLY. A sideways dodge breaks the column association --
    # with 10 groups x 2 models the boxes are ~1 bar-width apart, so a box shifted 34pt
    # left reads as belonging to the neighbouring bar (SVG1/SVG2 looked swapped).
    # Candidates in preference order. Straight up/down first (reads best), then diagonals
    # and sideways, then progressively further out. The two series prefer opposite
    # directions so a crossing pair separates naturally.
    up, dn = (0, 20, 'center', 'baseline'), (0, -20, 'center', 'top')
    cand_hi = [up, dn, (-30, 14, 'right', 'baseline'), (30, 14, 'left', 'baseline'),
               (-32, 0, 'right', 'center'), (32, 0, 'left', 'center'),
               (0, 44, 'center', 'baseline'), (0, -44, 'center', 'top'),
               (-52, 0, 'right', 'center'), (52, 0, 'left', 'center'),
               (0, 68, 'center', 'baseline'), (0, -68, 'center', 'top')]
    cand_lo = [dn, up, (30, -14, 'left', 'top'), (-30, -14, 'right', 'top'),
               (32, 0, 'left', 'center'), (-32, 0, 'right', 'center'),
               (0, -44, 'center', 'top'), (0, 44, 'center', 'baseline'),
               (52, 0, 'left', 'center'), (-52, 0, 'right', 'center'),
               (0, -68, 'center', 'top'), (0, 68, 'center', 'baseline')]
    # The mid-axis LINE segments block the MEAN boxes (not the mid-axis labels, which are
    # allowed to sit on their own line): a mean box dodging upward can otherwise land on
    # the line, as on humanoid panel (b) where the DreamDojo series crosses the bar tops.
    line_boxes = []
    for xs, ys in ((xs1, yy1), (xs2, yy2)):
        for j in range(len(xs) - 1):
            for f in np.linspace(0, 1, 9):
                sx = xs[j] + f * (xs[j + 1] - xs[j])
                sy = ys[j] + f * (ys[j + 1] - ys[j])
                px, py = ax2.transData.transform((sx, sy))
                line_boxes.append(Bbox.from_bounds(px - 7, py - 7, 14, 14))

    _place(ax2, fig, list(zip(xs1, yy1)), fmt, mid_color, bb, occupied, cand_hi)
    _place(ax2, fig, list(zip(xs2, yy2)), fmt, mid_color_l, bb, occupied, cand_lo)

    # Mean boxes LAST, so they dodge the mid-axis labels rather than covering them: in
    # panel (b) the KIVA line runs right through the bar tops where the boxes want to sit,
    # and the mid-axis value is the one that must stay legible (it is the panel's metric).
    # Candidates are VERTICAL ONLY -- with 10 groups x 2 models the boxes are ~1 bar-width
    # apart, so a sideways dodge makes a box read as belonging to the neighbouring bar
    # (SVG1/SVG2 looked swapped).
    _place(ax, fig, [(xi, top) for xi, top, _mv in mean_pts], '{:.2f}', maroon, mbox,
           occupied + line_boxes,
           [(0, 10 + 22 * k, 'center', 'baseline') for k in range(9)],
           values=[mv for _xi, _t, mv in mean_pts], zorder=8)


def _legend(solid, hatch, mid_lab, mid_color, mid_color_l, note, maroon):
    l1 = f'{mid_lab} \u2014 {solid.split()[0]}'
    l2 = f'{mid_lab} \u2014 {hatch.split()[0]}'
    return ([Patch(facecolor='#555'),
             Patch(facecolor='#aaa', hatch='///', edgecolor='white'),
             Line2D([0], [0], color=mid_color, ls='-', marker='_', ms=24, mew=5.5,
                    lw=3.2),
             Line2D([0], [0], color=mid_color_l, ls=':', marker='|', ms=20, mew=5,
                    lw=3.2),
             Patch(facecolor='white', edgecolor=maroon, lw=1.6),
             Line2D([0], [0], color='none')],
            [solid, hatch, l1, l2, MEAN_LABEL, note])



def _pct_cap(vmax):
    """Round cap >= vmax for a percentage axis, with a matching tick step.

    Returns (cap, step). The cap never exceeds 100 because KIVA is a percentage, and it
    is the smallest round value above the data, so the line fills the band without the
    axis claiming range the metric cannot reach.
    """
    for cap, step in ((5, 1), (10, 2), (15, 5), (20, 5), (25, 5), (50, 10),
                      (75, 25), (100, 25)):
        if vmax <= cap:
            return cap, step
    return 100, 25


def _nice_ticks(lo, hi, step, n_max=4):
    """Ticks on multiples of `step` inside (lo, hi).

    linspace across a data-driven band produces values like 0.59 / 0.66 / 0.73, which
    read as arbitrary; RIGIS/KIVA axes are expected on round steps. Falls back to a
    coarser multiple of `step` if the band would carry too many ticks.
    """
    import math
    for mult in (1, 2, 4, 5, 10):
        s = step * mult
        t = [round(k * s, 10) for k in range(int(math.ceil(lo / s)),
                                             int(math.floor(hi / s)) + 1)]
        t = [v for v in t if lo < v < hi]
        if len(t) <= n_max:
            return t
    return t


def draw(A, B, emb_title, rv_a, rv_b, stem: Path, cats, bar_colors, panels,
         maroon, blue, blue_l, purple, purple_l) -> None:
    plt.rcParams.update({'font.family': 'serif', 'font.size': 23,
                         'axes.spines.top': False})
    x = np.arange(len(cats)) * 1.3
    fig, axes = plt.subplots(2, 1, figsize=(26, 22), gridspec_kw={'hspace': 0.42})
    plt.tight_layout(rect=[0, 0.01, 1, 0.93])

    # Panel (a). RIGIS is a 0-1 score, so its axis is the full honest [0, 1] -- no
    # data-driven floor to invent. That alone puts the line in the panel's upper band; the
    # bar axis is then scaled so the tallest bar AND its mean box finish below the LOWEST
    # RIGIS point, which is what keeps the line clearly "on top" instead of threading
    # through the bar tops.
    mid_a = [v for v in A['mid'][0] + A['mid'][1] if v is not None]
    a_lo, a_hi = 0.0, 1.0
    ticks_a = [0.25, 0.50, 0.75, 1.00]
    bars_a = [v for v in A['bars'][0] + A['bars'][1] if v is not None]
    # Bars occupy the fraction of the panel below the lowest RIGIS point, so the whole
    # RIGIS band sits above them. Clamped to [0.55, 0.72]: unclamped, a low-RIGIS cell
    # (humanoid min 0.683) demands a 4x speedup axis for 2.3x bars and the bars become a
    # sliver, while a high-RIGIS cell would let them touch the line.
    lo_frac_a = min(max(min(mid_a) - 0.12, 0.55), 0.72)
    _panel(axes[0], cats, x, bar_colors, A['bars'][0], A['bars'][1],
           A['mid'][0], A['mid'][1], A['means'][0], A['means'][1], blue, blue_l,
           panels['a_multiview']['metric'], '{:.3f}',
           (0, max(bars_a) / lo_frac_a), (a_lo, a_hi), ticks_a,
           maroon)

    mid_b = [v for v in B['mid'][0] + B['mid'][1] if v is not None]
    # KIVA is a PERCENTAGE: the band starts at true 0 and caps at a round value <= 100.
    # Do NOT inflate the top (150% for a quantity that cannot exceed 100) or drop the
    # floor below 0 to lift the line off the bars -- both misrepresent the axis to fix a
    # layout problem. Separation is bought on the BAR axis instead, exactly as in panel (a).
    b_lo = 0.0
    b_hi, step_b = _pct_cap(max(mid_b + ([rv_b] if rv_b else [])))
    ticks_b = [0] + _nice_ticks(0, b_hi, step_b) + [b_hi]
    # Bars occupy the fraction of the panel below the LOWEST KIVA point, so both series sit
    # above them -- same rule as panel (a). Clamped to [0.42, 0.68]: humanoid's lowest KIVA
    # is 4.7% of a 15% cap (0.31), which unclamped squashes 1.95x bars onto a 6x axis.
    bars_b = [v for v in B['bars'][0] + B['bars'][1] if v is not None]
    lo_frac_b = min(max(min(mid_b) / b_hi - 0.06, 0.42), 0.68)
    _panel(axes[1], cats, x, bar_colors, B['bars'][0], B['bars'][1],
           B['mid'][0], B['mid'][1], B['means'][0], B['means'][1], purple, purple_l,
           panels['b_singleview']['metric'], '{:.1f}',
           (0, max(bars_b) / lo_frac_b), (b_lo, b_hi), ticks_b,
           maroon)

    pa = panels['a_multiview']['labels']
    pb = panels['b_singleview']['labels']
    hA, lA = _legend(pa[0], pa[1], 'RIGIS', blue, blue_l,
                     f'real video RIGIS = {rv_a:.3f}' if rv_a else '', maroon)
    hB, lB = _legend(pb[0], pb[1], 'KIVA %', purple, purple_l,
                     f'real video KIVA = {rv_b:.1f}%' if rv_b else '', maroon)
    for ax, (h, l), tag, sub in ((axes[0], (hA, lA), 'a', 'multi-view \u2014 RIGIS'),
                                 (axes[1], (hB, lB), 'b', 'single-view \u2014 KIVA')):
        p = ax.get_position()
        fig.legend(h, l, loc='lower left', bbox_to_anchor=(p.x0, p.y1 + 0.006),
                   ncol=3, frameon=False, fontsize=18.5, borderaxespad=0,
                   handlelength=2.2, columnspacing=2.2)
        fig.text(p.x0, p.y1 + 0.062, f'({tag}) {emb_title} {sub}',
                 fontsize=27, fontweight='bold', ha='left', va='bottom')

    fig.savefig(f'{stem}.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(f'{stem}.png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
