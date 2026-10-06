"""Twelve figures for the six NR metrics, from the cache `nr_viz_cache.py` builds.

Every panel carries the metric's POLICY STATUS, because the six columns are not
interchangeable: one failed the GT-is-ceiling audit, one is paired-only, one is a
diagnostic, one is dropped outright. A figure that plots them side by side without
saying so invites exactly the comparison the audits forbid.

  Tier A -- metric literacy
    F1  metric card sheet        distribution + unit + polarity + status + rule
    F2  coverage waterfall       why every column has a different n
    F3  redundancy               Spearman 6x6 + clustering (Active Rate is redundant)
  Tier B -- results
    F4  grid heatmap x6          embodiment-block x (model x method) x split
    F5  forest accel pooled      delta vs dense, 95% boot CI, Wilcoxon p, resolved
    F6  forest per-cell          same deltas per (model x embodiment x mount)
    F7  ceiling test             pred minus its own co-located GT
    F8  speed-quality Pareto     measured speedup vs NR delta
  Tier C -- per-frame anatomy
    F9  motion anatomy           per-pair flow traces, dense vs accel
    F10 continuity anatomy       copy_mse vs vfi_mse per triple + the weight floor
    F11 confound panel           metric vs motion per mount, |rho| >= 0.35 bar
    F12 flicker autopsy          why Actor Flicker is DROPPED

Usage (needs matplotlib -- use the dreamgen venv):
  models/dreamgen/.venv/bin/python assets/scripts/make_nr_metric_figures.py \
      --run metrics/WorldScore/results/dense_nr_full
  ... --only f1 f3      # a subset, by id
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nr_viz_common as C  # noqa: E402

FIGURES = {}


def figure(fid, title):
    """Register a figure builder. Each gets (ctx) and returns nothing; ctx.save() writes."""
    def deco(fn):
        FIGURES[fid] = (title, fn)
        return fn
    return deco


class Ctx:
    """Plot context: the cache, the output dir, and the save/CSV helpers."""

    def __init__(self, cache, out, formats=('pdf', 'svg', 'png')):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        self.plt, self.np, self.mpl = plt, np, matplotlib
        self.cache, self.out = cache, Path(out)
        self.formats = formats
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / 'data').mkdir(exist_ok=True)
        plt.rcParams.update({
            'font.family': 'DejaVu Sans', 'font.size': 8,
            'pdf.fonttype': 42, 'ps.fonttype': 42,
            'axes.spines.top': False, 'axes.spines.right': False,
            'axes.titlesize': 9, 'axes.titleweight': 'bold',
            'figure.dpi': 110, 'savefig.bbox': 'tight',
            'legend.frameon': False, 'grid.alpha': 0.25,
        })

    def save(self, fig, stem, note=None, tight=True):
        """Write every format + stamp provenance so a figure can never be quoted
        without its coverage state.

        `tight=False` for figures whose geometry is set by an explicit gridspec: a
        `bbox_inches='tight'` crop recomputes the layout and undoes the reserved
        margins, which is what re-collides a suptitle with the panels.
        """
        stamp = self.cache.stamp()
        if note:
            stamp = f'{stamp}  |  {note}'
        fig.text(0.005, 0.002, stamp, fontsize=5.5, color='#666666', va='bottom')
        if self.cache.partial:
            fig.text(0.5, 0.5, 'PARTIAL', fontsize=70, color='#d0d0d0',
                     alpha=0.16, ha='center', va='center', rotation=28,
                     zorder=-10, fontweight='bold')
        for ext in self.formats:
            fig.savefig(self.out / f'{stem}.{ext}', dpi=300,
                        bbox_inches='tight' if tight else None)
        self.plt.close(fig)
        print(f'  wrote {stem}.{"/".join(self.formats)}')

    def csv(self, stem, rows):
        """The exact numbers a figure drew. Non-negotiable: a figure nobody can audit
        is a claim nobody can check."""
        if not rows:
            return
        fields = list(dict.fromkeys(k for r in rows for k in r))
        path = self.out / 'data' / f'{stem}.csv'
        with path.open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    # --- shared drawing helpers -------------------------------------------------
    def badge(self, ax, key, loc='upper right'):
        """The status chip. Same colours as the dashboard (nr_viz_common.STATUS_STYLE)."""
        short, _, colour = C.STATUS_STYLE[self.cache.status(key)]
        x, ha = (0.985, 'right') if 'right' in loc else (0.015, 'left')
        y, va = (0.965, 'top') if 'upper' in loc else (0.035, 'bottom')
        ax.text(x, y, short, transform=ax.transAxes, ha=ha, va=va, fontsize=6,
                fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.28', fc=colour, ec='none', alpha=0.92))

    def panel_title(self, ax, key, extra=''):
        cache = self.cache
        # Two separate texts, not a two-line title: matplotlib cannot mix font sizes
        # inside one title, and `pad` is what keeps the unit line off the axes.
        ax.set_title(f'{cache.label(key)}{extra}', loc='left', fontsize=9,
                     fontweight='bold', pad=13)
        ax.text(0.0, 1.012, f'{cache.unit(key)}  |  {cache.polarity_arrow(key)}',
                transform=ax.transAxes, fontsize=6.2, color='#555555', va='bottom')
        ax.set_xlabel('')

    def rule_footer(self, ax, key):
        ax.text(0.0, -0.165, C.STATUS_RULE[self.cache.status(key)],
                transform=ax.transAxes, fontsize=6, color='#444444', va='top')

    def shade_grey(self, ax, why):
        """Mark a panel whose column may not be read as a result."""
        ax.set_facecolor('#f2f2f2')
        for sp in ax.spines.values():
            sp.set_color('#bbbbbb')
        ax.text(0.5, 0.5, why, transform=ax.transAxes, ha='center', va='center',
                fontsize=7, color='#999999', style='italic', zorder=6)


# ===========================================================================
# Tier A -- metric literacy
# ===========================================================================

@figure('f1', 'Metric card sheet: what each of the six columns is')
def f1_metric_cards(ctx):
    """One card per metric: the pooled distribution split by camera mount, plus unit,
    polarity, status badge and the read/do-not-read rule.

    Split by MOUNT rather than pooled flat because three of the six are per-mount
    quantities (Actor Activity is computed from a different flow field on static /
    head / wrist), so a single pooled violin would be a mixture of three definitions.
    """
    cache, plt, np = ctx.cache, ctx.plt, ctx.np
    mounts = [m for m in C.MOUNT_ORDER
              if any(c.get('camera_mount') == m for c in cache.clips)]
    fig, axs = plt.subplots(2, 3, figsize=(13.2, 7.4))
    rows = []
    for ax, key in zip(axs.flat, cache.keys):
        data, labels = [], []
        for mount in mounts:
            vals = cache.values(key, camera_mount=mount)
            if len(vals) >= 5:
                data.append(vals)
                labels.append(f'{mount}\nn={len(vals)}')
                rows.append({'metric': key, 'mount': mount, 'n': len(vals),
                             'median': C_median(vals), 'p10': pct(np, vals, 10),
                             'p90': pct(np, vals, 90)})
        status = cache.status(key)
        if data:
            parts = ax.violinplot(data, showextrema=False, widths=0.82)
            colour = C.STATUS_STYLE[status][2]
            for body in parts['bodies']:
                body.set_facecolor(colour)
                body.set_alpha(0.30 if status in ('DROPPED', 'DIAGNOSTIC') else 0.42)
                body.set_edgecolor('none')
            bp = ax.boxplot(data, widths=0.16, showfliers=False, patch_artist=True,
                            medianprops=dict(color='black', lw=1.3),
                            boxprops=dict(facecolor='white', ec='#555555', lw=0.7),
                            whiskerprops=dict(color='#555555', lw=0.7),
                            capprops=dict(color='#555555', lw=0.7))
            del bp
            ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=6.5)
            for i, vals in enumerate(data, 1):
                ax.text(i, ax.get_ylim()[1], f'{C_median(vals):.{cache.digits(key)}f}',
                        ha='center', va='top', fontsize=6, color='#333333')
        else:
            ax.text(0.5, 0.5, 'no valid observations', ha='center', va='center',
                    transform=ax.transAxes, fontsize=7, color='#999999')
        ctx.panel_title(ax, key)
        ctx.badge(ax, key)
        ctx.rule_footer(ax, key)
        ax.grid(axis='y')
        if status in ('DROPPED', 'DIAGNOSTIC'):
            ax.set_facecolor('#fafafa')
    fig.suptitle('F1  The six NR metrics: distribution by camera mount, with the policy '
                 'status that governs how each may be read', y=0.995, fontsize=11)
    fig.text(0.005, 0.965,
             'Split by mount because Actor Activity / Active Rate / AMC are computed '
             'from a mount-specific flow field (static: fg p95, wrist: whole-frame '
             'median, head: ego-removed fg p95) -- a pooled violin would mix three '
             'definitions.', fontsize=6.5, color='#555555')
    fig.tight_layout(rect=(0, 0.012, 1, 0.955))
    ctx.csv('f1_metric_cards', rows)
    ctx.save(fig, 'f1_metric_cards')


def C_median(vals):
    v = sorted(vals)
    n = len(v)
    return None if not n else (v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2]))


def pct(np, vals, q):
    return float(np.percentile(vals, q)) if vals else None


# Verbatim gate reasons -> a small fixed set of families, so the F2 reason panel can
# carry ONE shared legend instead of unreadable in-bar labels. The cache deliberately
# stores the verbatim string (auditable); this is presentation only.
REASON_FAMILY = [
    ('actor_validity=', 'no robot detected'),
    ('tracked_frac', 'tracking below 50% of pairs'),
    ('too few valid motion pairs', 'too few valid motion pairs'),
    ('weight sum <', 'below AMC activity / weight floor'),
    ('stored weight sum', 'stored AMC weight below floor'),
    ('stored value missing', 'stored value missing'),
    ('not computed for this clip', 'not computed for this clip'),
    ('recompute returned None', 'recompute returned None'),
]
REASON_COLORS = ['#4a7fb5', '#e0a13a', '#8e5fa8', '#5aa68a', '#c4622d', '#7a7a7a',
                 '#b8b8b8', '#a32b2b', '#444444']


def reason_family(text):
    for needle, label in REASON_FAMILY:
        if needle in (text or ''):
            return label
    return 'other'


@figure('f2', 'Coverage and validity waterfall: why every column has a different n')
def f2_coverage(ctx):
    """Left: the shared pipeline stages, then per-metric survival. Right: the COMPOSITION
    of what each gate removed, as a 100% stacked bar.

    Normalised rather than absolute because the counts differ by two orders of magnitude
    between columns (Visual Integrity loses ~nothing, AMC loses thousands); on a shared
    linear axis the small columns collapse to invisible slivers and the figure answers
    nothing. The absolute count is printed at the end of each bar.
    """
    cache, plt, np = ctx.cache, ctx.plt, ctx.np
    cov = cache.agg['coverage']
    stages, per = cov['stages'], cov['per_metric']
    keys = cache.keys
    fig = plt.figure(figsize=(13.4, 6.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.5], hspace=0.55, wspace=0.26,
                          bottom=0.13, top=0.9)

    # --- shared stages (log y: manifest is ~60x the scored count while partial)
    ax = fig.add_subplot(gs[0, 0])
    order = ['manifest_clips', 'result_files', 'in_protocol']
    labels = ['manifest\n(expected)', 'result files\n(scored)', 'in protocol\n(v3.2)']
    vals = [stages.get(k) or 0 for k in order]
    bars = ax.bar(range(len(vals)), vals, color=['#c9c9c9', '#7ea8c4', '#3977a8'],
                  width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f'{v:,}', ha='center', va='bottom',
                fontsize=7)
    ax.set_yscale('log')
    ax.set_ylim(top=max(vals) * 3.2 if max(vals) else 1)
    ax.set_xticks(range(len(labels)), labels, fontsize=7)
    ax.set_ylabel('view-clips (log)')
    ax.set_title('Shared pipeline', loc='left')
    ax.text(0.02, 0.97, f'dropped here: {stages.get("wrong_protocol", 0)} other-protocol,'
                        f' {stages.get("unreadable", 0)} unreadable',
            transform=ax.transAxes, ha='left', va='top', fontsize=6, color='#777777')
    ax.grid(axis='y')

    # --- per-metric survival
    ax2 = fig.add_subplot(gs[1, 0])
    passed = [per[k]['gate_passed'] for k in keys]
    gated = [per[k]['gated_out'] for k in keys]
    y = np.arange(len(keys))
    ax2.barh(y, passed, color=[C.STATUS_STYLE[cache.status(k)][2] for k in keys],
             alpha=0.85, height=0.62)
    ax2.barh(y, gated, left=passed, color='#e6e6e6', height=0.62)
    for i, (p, g) in enumerate(zip(passed, gated)):
        ax2.text(p + g, i, f'  {p:,}', va='center', fontsize=6.5)
    ax2.set_yticks(y, [cache.label(k) for k in keys], fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(1, len(cache.clips)) * 1.16)
    ax2.set_xlabel('view-clips in protocol (coloured = valid)')
    ax2.set_title('Per-metric validity gate', loc='left')
    ax2.grid(axis='x')

    # --- reason composition, normalised
    ax3 = fig.add_subplot(gs[:, 1])
    families, rows_csv = [], []
    comp = {}
    for k in keys:
        total = per[k]['gated_out']
        agg_fam = {}
        for why, n in (per[k]['reasons'] or {}).items():
            fam = reason_family(why)
            agg_fam[fam] = agg_fam.get(fam, 0) + n
            rows_csv.append({'metric': k, 'reason_verbatim': why, 'family': fam,
                             'n_clips': n, 'n_gated_total': total})
        comp[k] = agg_fam
        for fam in agg_fam:
            if fam not in families:
                families.append(fam)
    colour_of = {f: REASON_COLORS[i % len(REASON_COLORS)]
                 for i, f in enumerate(families)}
    for i, k in enumerate(keys):
        total = sum(comp[k].values())
        if not total:
            ax3.text(0.002, i, 'every clip valid -- nothing gated out', va='center',
                     fontsize=6.8, color='#2f7d4f')
            continue
        left = 0.0
        for fam, n in sorted(comp[k].items(), key=lambda kv: -kv[1]):
            frac = n / total
            ax3.barh(i, frac, left=left, height=0.6, color=colour_of[fam],
                     edgecolor='white', lw=0.6)
            if frac >= 0.11:
                ax3.text(left + frac / 2, i, f'{100 * frac:.0f}%', ha='center',
                         va='center', fontsize=6, color='white', fontweight='bold')
            left += frac
        ax3.text(1.004, i, f'  {total:,} clips gated '
                           f'({100.0 * total / max(1, per[k]["in_protocol"]):.1f}%)',
                 va='center', fontsize=6.5, color='#444444')
    ax3.set_yticks(range(len(keys)), [cache.label(k) for k in keys], fontsize=7)
    ax3.invert_yaxis()
    ax3.set_xlim(0, 1.0)
    ax3.set_xticks([0, .25, .5, .75, 1.0], ['0', '25%', '50%', '75%', '100%'],
                   fontsize=7)
    ax3.set_xlabel('composition of the clips each metric lost (normalised per row)')
    ax3.set_title('Why clips were gated out', loc='left')
    ax3.grid(axis='x')
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colour_of[f]) for f in families]
    ax3.legend(handles, families, fontsize=6.2, ncol=2, loc='upper center',
               bbox_to_anchor=(0.5, -0.12))
    g = cache.policy['gates']
    fig.text(0.62, 0.028,
             f"gates re-applied at table time: tracked_frac >= {g['min_tracked_frac']}, "
             f"AMC weight_sum >= {g['min_weight_sum']}, active_pairs >= "
             f"{g['min_active_pairs']} (and >= {g['min_active_frac']} of pairs), "
             f"head bg_cov >= {g['min_bg_cov_pct']}%, copy_err_floor = "
             f"{g['copy_err_floor']}",
             ha='center', fontsize=6, color='#666666')
    fig.suptitle('F2  Coverage and validity: the six columns do NOT share a denominator',
                 y=0.985, fontsize=11)
    rows_csv += [{'metric': k, 'reason_verbatim': '(gate passed)', 'family': 'valid',
                  'n_clips': per[k]['gate_passed'],
                  'n_gated_total': per[k]['gated_out']} for k in keys]
    ctx.csv('f2_coverage', rows_csv)
    ctx.save(fig, 'f2_coverage')


@figure('f3', 'Redundancy: Spearman between the six metrics, pooled and per mount')
def f3_redundancy(ctx):
    """The measured case for Active Rate being NOT_REPORTED, and for Actor Flicker
    being motion-confounded -- both re-measured on this wave, not quoted."""
    cache, plt, np = ctx.cache, ctx.plt, ctx.np
    red = cache.agg['redundancy']
    keys = cache.keys
    short = {k: cache.label(k).replace('Active Motion Continuity', 'AMC')
             for k in keys}

    def build(matrix):
        m = np.full((len(keys), len(keys)), np.nan)
        for i, a in enumerate(keys):
            for j, b in enumerate(keys):
                if a == b:
                    m[i, j] = 1.0
                    continue
                cell = matrix.get(f'{a}|{b}') or matrix.get(f'{b}|{a}')
                if cell and cell.get('rho') is not None:
                    m[i, j] = cell['rho']
        return m

    mounts = [mt for mt in C.MOUNT_ORDER if mt in (red.get('by_mount') or {})]
    ncols = 1 + len(mounts)
    fig, axs = plt.subplots(1, ncols, figsize=(3.3 * ncols, 4.6))
    axs = np.atleast_1d(axs)
    rows = []
    panels = [('pooled', red['pooled'])] + [(mt, red['by_mount'][mt]) for mt in mounts]
    for pi, (ax, (name, matrix)) in enumerate(zip(axs, panels)):
        m = build(matrix)
        im = ax.imshow(m, cmap='RdBu_r', vmin=-1, vmax=1)
        for i in range(len(keys)):
            for j in range(len(keys)):
                if np.isnan(m[i, j]):
                    ax.text(j, i, 'n/a', ha='center', va='center', fontsize=5.5,
                            color='#999999')
                    continue
                strong = abs(m[i, j]) >= 0.7
                ax.text(j, i, f'{m[i, j]:+.2f}'.replace('+1.00', '1'), ha='center',
                        va='center', fontsize=6,
                        color='white' if strong else '#222222',
                        fontweight='bold' if strong and i != j else 'normal')
                if i < j:
                    rows.append({'scope': name, 'a': keys[i], 'b': keys[j],
                                 'rho': round(float(m[i, j]), 4),
                                 'n': (matrix.get(f'{keys[i]}|{keys[j]}')
                                       or {}).get('n')})
        ax.set_xticks(range(len(keys)),
                      [C.wrap(short[k], 12) for k in keys], fontsize=5.6, rotation=42,
                      ha='right')
        # Only the leftmost panel carries y labels: repeating them puts a 2-line label
        # on top of the neighbouring heatmap.
        if pi == 0:
            ax.set_yticks(range(len(keys)), [C.wrap(short[k], 16) for k in keys],
                          fontsize=6)
        else:
            ax.set_yticks(range(len(keys)), [''] * len(keys))
        n_here = max((c.get('n') or 0) for c in matrix.values()) if matrix else 0
        ax.set_title(('pooled (all mounts)' if name == 'pooled'
                      else f'mount = {name}') + f'\nn <= {n_here:,} clips',
                     loc='left', fontsize=8.5)
        ax.set_xticks(np.arange(-.5, len(keys), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(keys), 1), minor=True)
        ax.grid(which='minor', color='white', lw=1.1)
        ax.tick_params(which='minor', length=0)
        # Ring the one pair that decided a policy.
        ia, ib = keys.index('actor_motion_raw'), keys.index('actor_motion_active_rate')
        for (r, c) in ((ia, ib), (ib, ia)):
            ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False,
                                       ec='#111111', lw=1.6))
    fig.colorbar(im, ax=list(axs), fraction=0.02, pad=0.015,
                 label='Spearman rho')
    why = (cache.policy.get('not_reported') or {}).get('actor_motion_active_rate', '')
    fig.suptitle('F3  Redundancy between the six metrics. Boxed cell = the pair that '
                 'demoted Active Rate to a diagnostic', y=0.995, fontsize=11)
    fig.text(0.005, 0.045, C.wrap('Why Active Rate is DIAGNOSTIC, not a column: '
                                  + why.split('It REMAINS')[0].strip(), 175),
             fontsize=6, color='#555555', va='bottom')
    ctx.csv('f3_redundancy', rows)
    ctx.save(fig, 'f3_redundancy')


# ===========================================================================
# Tier B -- results
# ===========================================================================

def diverging_norm(np, plt, values, center=None):
    """Symmetric colour scale around a centre, robust to the long tails these
    distributions have (p2/p98 rather than min/max)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    if center is None:
        lo, hi = np.percentile(vals, 2), np.percentile(vals, 98)
        return plt.Normalize(vmin=lo, vmax=hi)
    span = max(abs(np.percentile(vals, 2) - center),
               abs(np.percentile(vals, 98) - center)) or 1.0
    return plt.Normalize(vmin=center - span, vmax=center + span)


@figure('f4', 'Grid heatmap: every metric across embodiment x model x method x split')
def f4_grid(ctx):
    """The paper grid, as six heatmaps. READ DOWN A COLUMN ONLY.

    One panel per metric. Rows are (model family, method) inside an embodiment block,
    columns are Markovian | Non-Markovian. A cell pools every camera mount present in
    that (embodiment, model, split) -- exactly as `table_paper_grid.py` does -- which is
    what makes methods within one block like-for-like and comparisons ACROSS blocks
    invalid (Actor Activity is per-mount; AMC is barred cross-embodiment outright).

    Colour is therefore normalised PER EMBODIMENT BLOCK, not globally: a single global
    scale would render the forbidden cross-block comparison as a visual invitation.
    """
    cache, plt, np = ctx.cache, ctx.plt, ctx.np
    grid = cache.agg['grid_absolute']
    by = {}
    for r in grid:
        by[(r['embodiment'], r['model'], r['split'], r['method'], r['metric'])] = r
    # The Real-Video anchor pools GT over every model tree, exactly as
    # `table_paper_grid.emit('Real Video', ...)` does -- the GT clip of a ctrlworld
    # episode and of a dreamgen episode are both just reality. `grid_absolute` keys GT
    # per model (base_model('gt_ctrlworld') == 'ctrlworld'), so the pooled row is
    # recomputed here from the clips rather than by averaging medians.
    gt_pooled = {}
    for emb in {r['embodiment'] for r in grid}:
        for split in {r['split'] for r in grid}:
            for key in cache.keys:
                vals = cache.values(key, embodiment=emb, split=split, method='gt')
                if vals:
                    gt_pooled[(emb, split, key)] = {'n': len(vals),
                                                    'median': C_median(vals),
                                                    'model': 'Real Video'}
    embs = [e for e in C.EMB_ORDER if any(r['embodiment'] == e for r in grid)]
    # Row layout, shared by all six panels so they can be read against each other.
    # (embodiment, model, method, kind, show_model_label) -- `model` is always the real
    # model so a cell lookup never has to guess; `show_model_label` only controls whether
    # the y tick repeats the family name.
    rows_spec = []
    for emb in embs:
        rows_spec.append((emb, None, 'gt', 'gt', False))
        for model in C.sort_models({r['model'] for r in grid
                                    if r['embodiment'] == emb
                                    and r['model'] not in ('gt',)}):
            methods = [m for m in C.METHOD_ORDER if m != 'gt'
                       and any((emb, model, s, m, k) in by
                               for s in C.SPLIT_LABEL for k in cache.keys)]
            for i, method in enumerate(methods):
                rows_spec.append((emb, model, method, 'row', i == 0))
        rows_spec.append((emb, None, None, 'gap', False))
    rows_spec = rows_spec[:-1]
    splits = [s for s in ('makovian', 'non_makovian')
              if any(r['split'] == s for r in grid)]
    ncol = len(splits)

    fig, axs = plt.subplots(1, 6, figsize=(13.0, 0.148 * len(rows_spec) + 2.1),
                            sharey=True,
                            gridspec_kw=dict(left=0.145, right=0.995, wspace=0.09,
                                             top=0.945, bottom=0.048))
    csv_rows = []
    for ax, key in zip(axs, cache.keys):
        status = cache.status(key)
        readable = status not in ('DROPPED', 'DIAGNOSTIC')
        # Per-block normalisation (see docstring).
        norms = {}
        for emb in embs:
            vals = [by[k]['median'] for k in by
                    if k[0] == emb and k[4] == key and by[k]['median'] is not None
                    and by[k]['n'] >= cache.policy['min_pairs_for_claim']
                    and by[k]['method'] != 'gt']
            vals += [gt_pooled[k]['median'] for k in gt_pooled
                     if k[0] == emb and k[2] == key
                     and gt_pooled[k]['median'] is not None]
            norms[emb] = diverging_norm(np, plt, vals)
        cmap = plt.get_cmap('viridis' if readable else 'Greys')
        for ri, (emb, model, method, kind, _show) in enumerate(rows_spec):
            if kind == 'gap':
                continue
            for ci, split in enumerate(splits):
                rec = (gt_pooled.get((emb, split, key)) if kind == 'gt'
                       else by.get((emb, model, split, method, key)))
                if rec is None:
                    ax.add_patch(plt.Rectangle((ci - .5, ri - .5), 1, 1, fc='#ffffff',
                                               ec='#e8e8e8', lw=0.4, hatch='///',
                                               zorder=2))
                    continue
                n, med = rec['n'], rec['median']
                claim = n >= cache.policy['min_pairs_for_claim']
                if med is None or not claim:
                    ax.add_patch(plt.Rectangle((ci - .5, ri - .5), 1, 1, fc='#f6f6f6',
                                               ec='#dddddd', lw=0.4, zorder=2))
                    ax.text(ci, ri, f'?({n})' if n else '-', ha='center', va='center',
                            fontsize=4.6, color='#aaaaaa', zorder=4)
                    continue
                norm = norms[emb]
                colour = cmap(norm(med)) if norm else (0.8, 0.8, 0.8, 1.0)
                ax.add_patch(plt.Rectangle((ci - .5, ri - .5), 1, 1, fc=colour,
                                           ec='white', lw=0.5, zorder=2,
                                           alpha=0.55 if not readable else 1.0))
                lum = 0.299 * colour[0] + 0.587 * colour[1] + 0.114 * colour[2]
                ax.text(ci, ri, f'{med:.{min(cache.digits(key), 2)}f}', ha='center',
                        va='center', fontsize=4.8, zorder=4,
                        color='white' if lum < 0.55 and readable else '#222222')
                csv_rows.append({'metric': key, 'embodiment': emb,
                                 'model': rec['model'], 'method': method,
                                 'split': split, 'n': n, 'median': med})
        ax.set_xlim(-0.5, ncol - 0.5)
        ax.set_ylim(len(rows_spec) - 0.5, -0.5)
        ax.set_xticks(range(ncol), ['Mark.', 'Non-Mark.'][:ncol], fontsize=6.5)
        ax.set_title(f'{cache.label(key)}', loc='left', fontsize=8.5,
                     fontweight='bold', pad=17)
        ax.text(0.0, 1.028, cache.unit(key), transform=ax.transAxes, fontsize=5.8,
                color='#555555', va='bottom')
        short, _, colour = C.STATUS_STYLE[status]
        ax.text(1.0, 1.028, short, transform=ax.transAxes, ha='right', va='bottom',
                fontsize=5.6, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.22', fc=colour, ec='none'))
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    # y labels on the leftmost panel only
    ylabels = []
    for emb, model, method, kind, show in rows_spec:
        if kind == 'gap':
            ylabels.append('')
        elif kind == 'gt':
            ylabels.append(f'{C.EMB_LABEL.get(emb, emb)}  |  Real Video')
        else:
            prefix = f'{C.model_label(model)}  ' if show else '     '
            ylabels.append(f'{prefix}{C.method_label(method)}')
    axs[0].set_yticks(range(len(rows_spec)), ylabels, fontsize=4.6)
    for tick, spec in zip(axs[0].get_yticklabels(), rows_spec):
        if spec[3] == 'gt':
            tick.set_fontweight('bold')
    fig.suptitle('F4  Read DOWN a column only. Colour is normalised WITHIN each '
                 'embodiment block -- comparing numbers across blocks is invalid '
                 '(Actor Activity is per-mount, AMC is barred cross-embodiment)',
                 y=0.996, fontsize=9)
    fig.text(0.005, 0.012,
             f'hatched = method never rendered for that cell; "?(n)" = fewer than '
             f'{cache.policy["min_pairs_for_claim"]} valid clips, so no number is '
             f'printed. Each cell is a median over every camera mount present in that '
             f'(embodiment, model, split). Grey panels are not results.',
             fontsize=5.6, color='#555555')
    ctx.csv('f4_grid', csv_rows)
    ctx.save(fig, 'f4_grid', tight=False)


def forest_panel(ctx, ax, key, records, label_of, n_of='_n_pairs', centre=0.0,
                 show_p=True):
    """One forest: median delta + 95% bootstrap CI per row, with the claim verdict.

    Marker vocabulary, which is the whole point of the figure:
      filled + solid CI  RESOLVED -- CI excludes 0, n >= MIN_PAIRS_FOR_CLAIM, and the
                         Wilcoxon p agrees. This is the only form that carries a claim.
      hollow square      ARTIFACT-POSITIVE -- a solid positive delta on a column where
                         positive cannot be an improvement (NO_IMPROVEMENT_VS_DENSE);
                         `resolved_flag` refuses it and `artifact_positive` reports it.
      open + faint CI    not resolved: either the CI spans 0 or n is below the floor.
      grey band          n < MIN_PAIRS_FOR_CLAIM: no claim is possible at any effect size.
    """
    cache = ctx.cache
    min_n = cache.policy['min_pairs_for_claim']
    no_improve = cache.policy.get('no_improvement_vs_dense') or {}
    rows_out = []
    ys, drawn = [], 0
    for i, r in enumerate(records):
        n = r.get(f'{key}{n_of}') or 0
        med = r.get(f'{key}_delta_median')
        lo, hi = r.get(f'{key}_ci_lo'), r.get(f'{key}_ci_hi')
        p = r.get(f'{key}_p')
        resolved = r.get(f'{key}_resolved')
        y = -i
        ys.append(y)
        if not n or med is None:
            ax.text(centre, y, '  no comparable pairs', va='center', fontsize=5.5,
                    color='#bbbbbb')
            continue
        drawn += 1
        # An `artifact_positive` cell: solid positive on a column where positive is the
        # artifact. Recomputed from the stored fields (the cache stores `resolved` only).
        artifact = (key in no_improve and lo is not None and lo > 0
                    and n >= min_n and (p is None or p < cache.policy['alpha']))
        colour = C.family_color(r.get('accel_method') or r.get('method') or 'dense')
        if n < min_n:
            ax.axhspan(y - 0.42, y + 0.42, color='#f0f0f0', zorder=0)
        if lo is not None and hi is not None:
            ax.plot([lo, hi], [y, y], lw=2.2 if resolved else 1.0, color=colour,
                    alpha=0.95 if resolved else 0.38,
                    solid_capstyle='butt', zorder=3)
        if artifact:
            ax.plot(med, y, marker='s', ms=5.2, mfc='none', mec='#a32b2b', mew=1.5,
                    zorder=4)
        else:
            ax.plot(med, y, marker='o', ms=4.6, color=colour, zorder=4,
                    mfc=colour if resolved else 'white', mew=1.1)
        tag = ''
        if artifact:
            tag = ' artifact+'
        elif resolved:
            tag = ' *'
        pstr = ('' if not show_p or p is None
                else f'  p={p:.0e}' if p < 0.001 else f'  p={p:.3f}')
        # Annotate in AXES coordinates, just outside the right spine: a data-x anchored
        # label lands on top of another row's CI as soon as two rows differ in scale.
        ax.annotate(f'n={n}{pstr}{tag}', xy=(1.005, y),
                    xycoords=('axes fraction', 'data'), va='center', fontsize=5.2,
                    color='#a32b2b' if artifact else ('#111111' if resolved else '#888888'),
                    fontweight='bold' if (resolved or artifact) else 'normal',
                    annotation_clip=False)
        rows_out.append({'metric': key, 'row': label_of(r), 'n_pairs': n,
                         'delta_median': med, 'ci_lo': lo, 'ci_hi': hi, 'p': p,
                         'resolved': bool(resolved), 'artifact_positive': artifact})
    ax.axvline(centre, color='#333333', lw=0.9, ls='-', zorder=1)
    ax.set_yticks(ys, [label_of(r) for r in records], fontsize=5.8)
    ax.set_ylim(min(ys) - 0.7 if ys else -1, 0.7)
    ax.grid(axis='x')
    return rows_out, drawn


@figure('f5', 'Forest: acceleration minus its own dense clip, pooled per method')
def f5_forest_pooled(ctx):
    """The headline acceleration result, per metric.

    Each delta is WITHIN one episode+view+geometry (accel vs its own dense clip), which
    is why pooling over (model x embodiment x mount) is legitimate here even for AMC,
    whose absolute level may not be compared across embodiments -- a within-clip delta
    contains no level. The caveat that pooling does NOT remove is differing coverage:
    the methods share zero clips, so `common_support` (F5's right-hand annotation)
    re-ranks on the cells every method covers.
    """
    cache, plt = ctx.cache, ctx.plt
    pooled = cache.agg['t3b_accel_pooled']
    if not pooled:
        raise SystemExit('no acceleration pairs in this cache yet')
    pooled = sorted(pooled, key=lambda r: (C.METHOD_ORDER.index(r['accel_method'])
                                           if r['accel_method'] in C.METHOD_ORDER
                                           else 99))
    # The bottom row's per-panel "common support" note sits at -0.125 in axes fraction, and
    # below it go two legend lines plus the AMC caveat. `bottom` has to clear all four or
    # they collide (they did).
    fig, axs = plt.subplots(2, 3, figsize=(14.6, 0.30 * len(pooled) * 2 + 4.1),
                            gridspec_kw=dict(left=0.078, right=0.985, wspace=0.52,
                                             hspace=0.46, top=0.90, bottom=0.20))
    axs = axs.flat
    csv_rows = []
    common = cache.agg.get('common_support') or {}
    for ax, key in zip(axs, cache.keys):
        status = cache.status(key)
        got = [r for r in pooled if r.get(f'{key}_n_pairs')]
        if not got:
            ax.set_yticks([])
            ctx.shade_grey(ax, f'{cache.label(key)}\nnot compared vs dense\n'
                               f'({status})')
            ax.set_title(cache.label(key), loc='left', fontsize=8.5, pad=14)
            continue
        rows, _ = forest_panel(
            ctx, ax, key, pooled,
            label_of=lambda r: C.method_label(r['accel_method']))
        csv_rows += rows
        ax.set_title(cache.label(key), loc='left', fontsize=8.5, fontweight='bold',
                     pad=16)
        ax.text(0.0, 1.035, f'delta vs dense, {cache.unit(key)}',
                transform=ax.transAxes, fontsize=5.8, color='#555555', va='bottom')
        short, _, colour = C.STATUS_STYLE[status]
        ax.text(1.0, 1.035, short, transform=ax.transAxes, ha='right', va='bottom',
                fontsize=5.4, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', fc=colour, ec='none'))
        cs = common.get(key)
        if cs and cs.get('cells'):
            moved = {m: v for m, v in (cs.get('moved') or {}).items() if v}
            msg = (f"common support: {cs['cells']} cells shared by all "
                   f"{len(cs['methods'])} methods")
            if moved:
                msg += '; rank moves there: ' + ', '.join(
                    f'{C.method_label(m)} {v:+d}' for m, v in
                    sorted(moved.items(), key=lambda kv: -abs(kv[1]))[:3])
            ax.text(0.0, -0.125, C.wrap(msg, 66), transform=ax.transAxes, fontsize=5.2,
                    color='#777777', va='top')
    fig.suptitle('F5  Acceleration minus its OWN dense clip, pooled over '
                 '(model x embodiment x mount). Filled marker + bold CI = RESOLVED '
                 '(claim permitted);\nhollow red square = statistically solid but the '
                 'direction is an artifact, not an improvement',
                 y=0.985, fontsize=9.5)
    legend_bits = [
        ('filled circle, thick CI', 'RESOLVED: CI excludes 0, n >= '
         f'{cache.policy["min_pairs_for_claim"]}, Wilcoxon p < {cache.policy["alpha"]}'),
        ('open circle, thin CI', 'not distinguishable from dense at this n '
         '(a valid benchmark result, not a failure)'),
        ('hollow red square', 'positive delta where positive IS the artifact '
         '(over-smoothing) -- reported, never claimed'),
        ('grey band', f'n < {cache.policy["min_pairs_for_claim"]}: no claim possible'),
    ]
    fig.text(0.078, 0.115, '   |   '.join(f'{a}: {b}' for a, b in legend_bits[:2]),
             fontsize=5.8, color='#444444')
    fig.text(0.078, 0.097, '   |   '.join(f'{a}: {b}' for a, b in legend_bits[2:]),
             fontsize=5.8, color='#444444')
    caveat = (cache.policy.get('accel_caveats') or {}).get('active_motion_continuity')
    if caveat:
        fig.text(0.078, 0.030, C.wrap('AMC caveat (why a small positive AMC delta is '
                                      'not a win): ' + caveat, 215), fontsize=5.2,
                 color='#8a6a14', va='bottom')
    ctx.csv('f5_forest_pooled', csv_rows)
    ctx.save(fig, 'f5_forest_pooled', tight=False)


@figure('f6', 'Forest per cell: where a pooled acceleration effect actually comes from')
def f6_forest_cells(ctx):
    """The same paired deltas as F5, split by (model, embodiment, mount).

    F5 pools because the per-cell grid has no power at an affordable budget -- this
    figure is the audit that pooling did not hide a method that helps one cell and hurts
    another. Rows below `MIN_PAIRS_FOR_CLAIM` are banded grey: at that n no effect size
    can be claimed, which is the expected state for most cells and NOT a failure.

    One panel per metric, one small-multiple column per acceleration method.
    """
    cache, plt = ctx.cache, ctx.plt
    cells = cache.agg['t3_accel_cell']
    if not cells:
        raise SystemExit('no acceleration pairs in this cache yet')
    methods = C.sort_methods({r['accel_method'] for r in cells})
    # Keep the metrics that actually have per-cell data; a 6x9 grid of empty panels is
    # noise.
    keys = [k for k in cache.keys
            if any(r.get(f'{k}_n_pairs') for r in cells)]
    cell_rows = sorted({(r['model'], r['embodiment'], r['camera_mount'])
                        for r in cells},
                       key=lambda c: (C.EMB_ORDER.index(c[1])
                                      if c[1] in C.EMB_ORDER else 9,
                                      C.MODEL_ORDER.index(c[0])
                                      if c[0] in C.MODEL_ORDER else 9, c[2]))
    by = {(r['accel_method'], r['model'], r['embodiment'], r['camera_mount']): r
          for r in cells}
    nrow, ncol = len(keys), len(methods)
    fig, axs = plt.subplots(nrow, ncol, sharey=True,
                            figsize=(1.46 * ncol + 3.6,
                                     (0.148 * len(cell_rows) + 0.62) * nrow + 1.3),
                            gridspec_kw=dict(left=0.245, right=0.995, top=0.945,
                                             bottom=0.035, wspace=0.14, hspace=0.52))
    axs = axs.reshape(nrow, ncol)
    csv_rows = []
    min_n = cache.policy['min_pairs_for_claim']
    for ri, key in enumerate(keys):
        # One x scale per metric row, so cells are comparable across methods.
        lims = [v for m in methods for c in cell_rows
                for v in ((by.get((m,) + c) or {}).get(f'{key}_ci_lo'),
                          (by.get((m,) + c) or {}).get(f'{key}_ci_hi'))
                if v is not None]
        span = max(abs(min(lims)), abs(max(lims))) * 1.1 if lims else 1.0
        for ci, method in enumerate(methods):
            ax = axs[ri, ci]
            recs = [by.get((method,) + c) for c in cell_rows]
            for yi, (rec, cell) in enumerate(zip(recs, cell_rows)):
                y = -yi
                if rec is None:
                    continue
                n = rec.get(f'{key}_n_pairs') or 0
                med = rec.get(f'{key}_delta_median')
                lo, hi = rec.get(f'{key}_ci_lo'), rec.get(f'{key}_ci_hi')
                if not n or med is None:
                    continue
                if n < min_n:
                    ax.axhspan(y - 0.4, y + 0.4, color='#f1f1f1', zorder=0)
                resolved = rec.get(f'{key}_resolved')
                colour = C.family_color(method)
                if lo is not None:
                    ax.plot([lo, hi], [y, y], lw=1.9 if resolved else 0.8,
                            color=colour, alpha=0.95 if resolved else 0.33, zorder=3)
                ax.plot(med, y, 'o', ms=3.4 if resolved else 2.6, color=colour,
                        mfc=colour if resolved else 'white', mew=0.8, zorder=4)
                if rec.get(f'{key}_confound_warning'):
                    ax.plot(med, y, marker='x', ms=4.6, color='#b8860b', mew=0.9,
                            zorder=5)
                csv_rows.append({'metric': key, 'method': method, 'model': cell[0],
                                 'embodiment': cell[1], 'mount': cell[2], 'n_pairs': n,
                                 'delta_median': med, 'ci_lo': lo, 'ci_hi': hi,
                                 'resolved': bool(resolved),
                                 'confound_warning':
                                     rec.get(f'{key}_confound_warning')})
            ax.axvline(0, color='#333333', lw=0.7, zorder=1)
            ax.set_xlim(-span, span)
            ax.set_ylim(-len(cell_rows) + 0.4, 0.6)
            ax.tick_params(labelsize=4.6)
            ax.grid(axis='x')
            if ri == 0:
                ax.set_title(C.method_label(method), fontsize=6.5,
                             color=C.family_color(method), pad=14)
            elif ri:
                # Repeat the method name on every metric row: with 6 stacked rows the
                # top title is off-screen by the time a reader is looking at row 4.
                ax.set_title(C.method_label(method), fontsize=5.4,
                             color=C.family_color(method), pad=13)
            if ci == 0:
                ax.set_yticks([-i for i in range(len(cell_rows))],
                              [f'{C.model_label(m)} / {e[:4]} / {mo}'
                               for m, e, mo in cell_rows], fontsize=4.6)
                status = cache.status(key)
                short, _, badge_colour = C.STATUS_STYLE[status]
                # Row header ABOVE the method titles (which sit at pad=13/14), in axes
                # coords, so it can never land inside the y tick labels.
                ax.text(0.0, 1.16, f'{cache.label(key)}   [{short}]',
                        transform=ax.transAxes, fontsize=7, fontweight='bold',
                        va='bottom', ha='left', color=badge_colour)
                ax.text(1.0, 1.16, f'delta vs dense, {cache.unit(key)}',
                        transform=axs[ri, -1].transAxes, fontsize=5.2,
                        color='#666666', va='bottom', ha='right')
    fig.suptitle('F6  The F5 deltas, per (model / embodiment / mount) cell. Grey band = '
                 f'n < {min_n}, where no claim is possible at any effect size; '
                 'x = the column is motion-confounded on that mount', y=0.985,
                 fontsize=8.5)
    ctx.csv('f6_forest_cells', csv_rows)
    ctx.save(fig, 'f6_forest_cells', tight=False)


@figure('f7', 'Ceiling test: does the model beat reality? (pred minus its own GT)')
def f7_ceiling(ctx):
    """Dense prediction minus the co-located GT clip of the same episode+view+geometry.

    This is the audit that decided the CEILING_FAILED status. A no-reference quality
    metric must rank real video above generated video; a cell where pred BEATS GT means
    the metric is measuring something other than realism. Visual Integrity fails exactly
    that test, which is why it may not rank models even though its accel-vs-dense delta
    is claimable (the bias cancels inside a pair of generated clips).

    Bars pointing the WRONG way for the metric's polarity are red: that is a ceiling
    violation. `gt_better_frac` (printed) is the fraction of individual pairs where GT
    wins -- a median near 0 with frac near 0.5 is a metric that cannot tell them apart.
    """
    cache, plt = ctx.cache, ctx.plt
    paired = cache.agg['t2_paired_gt']
    if not paired:
        raise SystemExit('no pred-vs-GT pairs in this cache yet')
    paired = sorted(paired, key=lambda r: (C.EMB_ORDER.index(r['embodiment'])
                                           if r['embodiment'] in C.EMB_ORDER else 9,
                                           C.MODEL_ORDER.index(r['model'])
                                           if r['model'] in C.MODEL_ORDER else 9,
                                           r['camera_mount']))
    keys = [k for k in cache.keys if any(r.get(f'{k}_n_pairs') for r in paired)]
    fig, axs = plt.subplots(1, len(keys),
                            figsize=(2.45 * len(keys) + 1.6,
                                     0.26 * len(paired) + 3.2),
                            sharey=True,
                            gridspec_kw=dict(left=0.155, right=0.965, top=0.83,
                                             bottom=0.155, wspace=0.30))
    axs = ctx.np.atleast_1d(axs)
    csv_rows = []
    min_n = cache.policy['min_pairs_for_claim']
    violations = {}
    for ax, key in zip(axs, keys):
        higher = cache.polarity(key)
        for yi, r in enumerate(paired):
            y = -yi
            n = r.get(f'{key}_n_pairs') or 0
            med = r.get(f'{key}_delta_median')
            lo, hi = r.get(f'{key}_ci_lo'), r.get(f'{key}_ci_hi')
            if not n or med is None:
                continue
            # pred - GT. For a higher-is-better column, pred > GT (positive) is the
            # violation. For a descriptive column there is no ceiling to violate.
            wrong = higher is not None and ((med > 0) if higher else (med < 0))
            solid = (lo is not None and (lo > 0 or hi < 0) and n >= min_n)
            colour = '#a32b2b' if (wrong and solid) else (
                '#e0a13a' if wrong else '#3977a8')
            if n < min_n:
                ax.axhspan(y - 0.42, y + 0.42, color='#f1f1f1', zorder=0)
            ax.barh(y, med, height=0.62, color=colour, alpha=0.9, zorder=2)
            if lo is not None:
                ax.plot([lo, hi], [y, y], color='#333333', lw=0.9, zorder=3)
            frac = r.get(f'{key}_gt_better_frac')
            ax.annotate(f'n={n}' + (f'  GT wins {100 * frac:.0f}%' if frac is not None
                                    else ''),
                        xy=(1.008, y), xycoords=('axes fraction', 'data'),
                        va='center', fontsize=4.8, annotation_clip=False,
                        color='#a32b2b' if (wrong and solid) else '#777777')
            if wrong and solid:
                violations.setdefault(key, []).append(
                    f"{C.model_label(r['model'])}/{r['embodiment']}/{r['camera_mount']}")
            csv_rows.append({'metric': key, 'model': r['model'],
                             'embodiment': r['embodiment'],
                             'mount': r['camera_mount'], 'n_pairs': n,
                             'delta_median_pred_minus_gt': med, 'ci_lo': lo,
                             'ci_hi': hi, 'p': r.get(f'{key}_p'),
                             'gt_better_frac': frac,
                             'resolved': bool(r.get(f'{key}_resolved')),
                             'ceiling_violation': bool(wrong and solid)})
        ax.axvline(0, color='#333333', lw=0.9, zorder=4)
        ax.set_ylim(-len(paired) + 0.4, 0.6)
        ax.grid(axis='x')
        ax.tick_params(labelsize=5.2)
        status = cache.status(key)
        short, _, badge = C.STATUS_STYLE[status]
        ax.set_title(f'{cache.label(key)}', loc='left', fontsize=8, fontweight='bold',
                     pad=16)
        ax.text(0.0, 1.022, f'pred - GT, {cache.unit(key)}', transform=ax.transAxes,
                fontsize=5.4, color='#555555', va='bottom')
        ax.text(1.0, 1.022, short, transform=ax.transAxes, ha='right', va='bottom',
                fontsize=5.2, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.18', fc=badge, ec='none'))
        if higher is not None:
            ax.text(0.5, -0.055, f'<- GT better        pred better ->' if higher
                    else '<- pred better        GT better ->',
                    transform=ax.transAxes, ha='center', va='top', fontsize=5,
                    color='#777777')
    axs[0].set_yticks([-i for i in range(len(paired))],
                      [f"{C.model_label(r['model'])} / {r['embodiment']} / "
                       f"{r['camera_mount']}" for r in paired], fontsize=5)
    fig.suptitle('F7  Ceiling test: dense prediction minus its OWN co-located GT clip. '
                 'Red = a resolved violation (the model beats reality, so the metric is '
                 'not measuring realism)', y=0.975, fontsize=9)
    lines = []
    for key, cells in violations.items():
        lines.append(f'{cache.label(key)}: resolved violations in '
                     f'{len(cells)} cell(s) -- ' + ', '.join(cells[:6]))
    why = (cache.policy.get('ceiling_failed') or {}).get('visual_integrity')
    if why:
        lines.append('Recorded verdict -- ' + why)
    if lines:
        fig.text(0.008, 0.008, C.wrap('  |  '.join(lines[:1]) if len(lines) == 1
                                      else lines[0], 230)
                 + ('\n' + C.wrap(lines[-1], 230) if len(lines) > 1 else ''),
                 fontsize=5.2, color='#8a2b2b', va='bottom')
    ctx.csv('f7_ceiling', csv_rows)
    ctx.save(fig, 'f7_ceiling', tight=False)


# knee_curves panel -> the (model, embodiment) cell of this benchmark. Only these four
# cells were timed under the matched single-GPU protocol, so F8 can only place methods
# that appear in one of them; everything else has an NR delta but no measured speedup.
KNEE_PANEL_CELL = {
    'DreamGen 2B, bimanual MV': ('dreamgen', 'bimanual'),
    'DreamGen 2B, humanoid MV': ('dreamgen', 'humanoid'),
    'DreamDojo 2B, humanoid SV': ('dreamdojo', 'humanoid'),
    'Ctrl-World U-Net, humanoid MV': ('ctrlworld', 'humanoid'),
}
# knee_curves uses display names; the NR records use the run-script method ids.
KNEE_METHOD = {'PISA': 'pisa', 'SVG': 'svg1', 'SVG2': 'svg2', 'Radial': 'radial',
               'DiCache': 'dicache', 'WorldCache': 'worldcache',
               'FasterCache': 'fastercache', 'SiTo': 'sito', 'ITM': 'itm'}


@figure('f8', 'Speed vs NR quality: is a speedup free?')
def f8_pareto(ctx):
    """Measured speedup (x) against the paired NR delta vs dense (y), per method.

    The two axes come from DIFFERENT protocols and that is stated on the figure: speedup
    is the tuning sweep's single-episode matched timing (`results/reports/knee_curves.json`,
    the shipped operating point of each method), while the NR delta is the median over
    every comparable clip in this wave. They are joined on (model, embodiment, method),
    and only the four cells that were timed can be placed at all.

    The Pareto frontier is drawn on the sign the column's polarity defines: up-and-right
    for a higher-is-better column. A point at x > 1 and y ~ 0 is the honest goal -- faster
    at no measurable cost. Descriptive and non-result columns are omitted: a "Pareto
    frontier" on a column with no polarity would be meaningless.
    """
    cache, plt, np = ctx.cache, ctx.plt, ctx.np
    knee_path = Path('results/reports/knee_curves.json')
    if not knee_path.exists():
        raise SystemExit(f'{knee_path} missing; F8 needs the measured speedups')
    knee = json.loads(knee_path.read_text())
    # (model, embodiment, method) -> best measured speedup at the SHIPPED operating point
    speedup = {}
    for panel in knee['panels']:
        cell = KNEE_PANEL_CELL.get(panel['panel'])
        if not cell:
            continue
        for pt in panel['points']:
            mid = KNEE_METHOD.get(pt['method'])
            if not mid or not pt.get('is_chosen') or pt.get('speedup') is None:
                continue
            speedup[cell + (mid,)] = {'speedup': pt['speedup'],
                                      'config': pt.get('config'),
                                      'quality_note': pt.get('note'),
                                      'panel': panel['panel']}
    cells = cache.agg['t3_accel_cell']
    # Only columns with a polarity AND permitted to carry an accel claim.
    keys = [k for k in cache.keys
            if cache.polarity(k) is not None
            and cache.status(k) not in ('DROPPED', 'DIAGNOSTIC')]
    fig, axs = plt.subplots(1, len(keys), figsize=(3.9 * len(keys) + 0.8, 4.5),
                            gridspec_kw=dict(left=0.055, right=0.995, top=0.80,
                                             bottom=0.155, wspace=0.22))
    axs = np.atleast_1d(axs)
    csv_rows = []
    min_n = cache.policy['min_pairs_for_claim']
    placed_total = 0
    for ax, key in zip(axs, keys):
        higher = cache.polarity(key)
        pts = []
        for r in cells:
            sp = speedup.get((r['model'], r['embodiment'], r['accel_method']))
            n = r.get(f'{key}_n_pairs') or 0
            med = r.get(f'{key}_delta_median')
            if not sp or not n or med is None or n < min_n:
                continue
            pts.append((sp['speedup'], med, r, sp))
        placed_total += len(pts)
        # Annotate only the points a reader should chase: the frontier, plus anything
        # that is BOTH faster and resolved. 40+ overlapping labels answer nothing.
        def better(a, b):
            return a > b if higher else a < b

        front_ids = set()
        front = []
        for x, y, r, _sp in sorted(pts, key=lambda p: -p[0]):
            if all(not (px >= x and better(py, y)) for px, py, _, _ in front):
                front.append((x, y, r, None))
                front_ids.add(id(r))
        for x, y, r, sp in pts:
            method = r['accel_method']
            colour = C.family_color(method)
            resolved = r.get(f'{key}_resolved')
            on_front = id(r) in front_ids
            ax.plot(x, y, 'o', ms=8.5 if on_front else 5.5, color=colour,
                    alpha=0.95 if on_front else 0.5,
                    mfc=colour if resolved else 'white', mew=1.4, zorder=4 if on_front
                    else 3)
            lo, hi = r.get(f'{key}_ci_lo'), r.get(f'{key}_ci_hi')
            if lo is not None:
                ax.plot([x, x], [lo, hi], color=colour, lw=1.0 if on_front else 0.6,
                        alpha=0.5 if on_front else 0.22, zorder=2)
            if on_front:
                ax.annotate(f"{C.method_label(method)}  {x:.2f}x\n"
                            f"{C.model_label(r['model'])[:9]}/{r['embodiment'][:4]}/"
                            f"{r['camera_mount'][:5]}",
                            (x, y), textcoords='offset points', xytext=(8, 4),
                            fontsize=5.2, color='#222222', fontweight='bold')
            csv_rows.append({'metric': key, 'method': method, 'model': r['model'],
                             'embodiment': r['embodiment'],
                             'mount': r['camera_mount'], 'speedup': x,
                             'speedup_config': sp['config'],
                             'speedup_panel': sp['panel'],
                             'nr_delta_median': y, 'ci_lo': lo, 'ci_hi': hi,
                             'n_pairs': r.get(f'{key}_n_pairs'),
                             'resolved': bool(resolved),
                             'on_pareto_frontier': on_front})
        # dense reference: speedup 1.0, delta 0 by construction
        ax.plot(1.0, 0.0, marker='*', ms=13, color='#111111', zorder=5)
        ax.annotate('dense\n(reference)', (1.0, 0.0), textcoords='offset points',
                    xytext=(-4, -20), fontsize=5.4, ha='right', color='#111111')
        ax.axhline(0, color='#333333', lw=0.8, zorder=1)
        ax.axvline(1.0, color='#999999', lw=0.8, ls=':', zorder=1)
        if pts and len(front) > 1:
            front.sort(key=lambda p: p[0])
            ax.plot([p[0] for p in front], [p[1] for p in front], '-',
                    color='#888888', lw=1.1, alpha=0.75, zorder=1)
        ax.set_xlabel('measured speedup vs dense (shipped operating point)', fontsize=6)
        ax.set_ylabel(f'{cache.label(key)} delta vs dense ({cache.unit(key)})',
                      fontsize=6)
        status = cache.status(key)
        short, _, badge = C.STATUS_STYLE[status]
        ax.set_title(cache.label(key), loc='left', fontsize=8.5, fontweight='bold',
                     pad=15)
        ax.text(1.0, 1.02, short, transform=ax.transAxes, ha='right', va='bottom',
                fontsize=5.2, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.18', fc=badge, ec='none'))
        ax.text(0.0, 1.02, 'up-and-right is better' if higher
                else 'down-and-right is better', transform=ax.transAxes, fontsize=5.4,
                color='#555555', va='bottom')
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.22)
        if not pts:
            ctx.shade_grey(ax, 'no method has BOTH a measured speedup\nand a resolvable '
                               'NR delta in a timed cell')
    if not placed_total:
        raise SystemExit('F8: no (timed cell x method) overlap yet -- rerun when more '
                         'of the wave has scored')
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=f)
               for f, c in C.FAMILY_COLOR.items() if f in ('pruning', 'caching',
                                                           'sparse-attn')]
    handles.append(plt.Line2D([], [], marker='*', ls='', color='#111111',
                              label='dense reference'))
    handles.append(plt.Line2D([], [], marker='o', ls='', mfc='white', mec='#555555',
                              color='#555555', label='delta not resolved (CI spans 0)'))
    handles.append(plt.Line2D([], [], ls='-', color='#888888',
                              label='Pareto frontier (labelled points)'))
    handles.append(plt.Line2D([], [], marker='o', ls='', color='#aaaaaa', ms=4,
                              label='dominated point (unlabelled)'))
    axs[0].legend(handles=handles, fontsize=5.4, loc='lower left', ncol=2)
    fig.suptitle('F8  Speed against NR cost. Speedup is the matched single-episode '
                 'timing of each method\'s SHIPPED config; the NR delta is the median '
                 'over every comparable clip of this wave -- two different protocols, '
                 'joined on (model, embodiment, method)', y=0.975, fontsize=8.5)
    fig.text(0.008, 0.012, C.wrap(
        'Only the four cells timed under the matched protocol can be placed: '
        + '; '.join(sorted(k for k in KNEE_PANEL_CELL)) + '. '
        + (knee.get('caveat') or ''), 235), fontsize=5.2, color='#666666', va='bottom')
    ctx.csv('f8_pareto', csv_rows)
    ctx.save(fig, 'f8_pareto', tight=False)


# ===========================================================================
# Tier C -- per-frame anatomy
# ===========================================================================

def pick_group(cache, prefer_methods=('itm', 'sito')):
    """An exemplar group whose accel clip is one of `prefer_methods`, else any."""
    groups = cache.exemplar_groups()
    if not groups:
        raise SystemExit('cache has no exemplar groups (run nr_viz_cache without '
                         '--skip-perframe)')
    for want in prefer_methods:
        for g in groups:
            if any(a['method'] == want for a in g['accel']):
                return g, next(a for a in g['accel'] if a['method'] == want)
    g = groups[0]
    return g, g['accel'][0]


@figure('f9', 'Motion anatomy: what Actor Activity is actually measuring, per frame')
def f9_motion_anatomy(ctx):
    """Actor Activity is a median over per-pair flow, so its value hides WHICH pairs

    moved. This traces one dense clip against one accelerated clip of the same episode,
    view and geometry: the foreground p95 flow (the quantity the metric takes), the
    ego-removed variant, the arm/object/background components, and the coverage that
    decides whether a pair counts at all.

    The point of the figure is the failure mode it makes visible: a pruning method does
    not uniformly reduce motion, it flattens the PEAKS -- the metric drops because the
    busy pairs stopped being busy, not because the clip became uniformly still.
    """
    cache, plt, np = ctx.cache, ctx.plt, ctx.np
    group, accel = pick_group(cache)
    dense = cache.perframe(group['dense_id'])
    acc = cache.perframe(accel['id'])
    if not dense or not acc:
        raise SystemExit('exemplar per-frame files missing')
    gates = cache.policy['gates']
    fig, axs = plt.subplots(4, 1, figsize=(12.6, 8.6), sharex=True,
                            gridspec_kw=dict(left=0.068, right=0.86, top=0.885,
                                             bottom=0.075, hspace=0.24))
    series = [
        ('fg_flow_p95', 'foreground p95 flow', True),
        ('fg_flow_ego_p95', 'foreground p95, ego motion removed', True),
        ('arm_flow_med', 'arm median flow', False),
        ('obj_flow_med', 'object median flow', False),
        ('bg_flow_med', 'background median flow (ego proxy)', False),
    ]
    csv_rows = []

    def get(pf, field):
        return [p.get(field) for p in pf['per_frame']]

    # --- panel 0/1: the metric's own quantity, dense vs accel
    for ax, (field, label, _) in zip(axs[:2], series[:2]):
        for pf, name, colour, ls in ((dense, 'dense', '#333333', '-'),
                                     (acc, C.method_label(accel['method']),
                                      C.family_color(accel['method']), '-')):
            vals = get(pf, field)
            xs = list(range(len(vals)))
            ax.plot(xs, vals, ls, color=colour, lw=1.0, alpha=0.9, label=name)
            good = [v for v in vals if v is not None]
            if good:
                ax.axhline(C_median(good), color=colour, lw=0.8, ls=':', alpha=0.7)
            for i, v in enumerate(vals):
                csv_rows.append({'clip': pf['id'], 'method': pf['method'], 'pair': i,
                                 'field': field, 'value': v})
        ax.axhline(gates['active_px'], color='#b8860b', lw=0.9, ls='--')
        ax.annotate(f'active threshold = {gates["active_px"]} px',
                    xy=(0.012, gates['active_px']),
                    xycoords=('axes fraction', 'data'), fontsize=5.6, color='#b8860b',
                    va='bottom', ha='left')
        ax.set_ylabel(C.wrap(label, 26), fontsize=6.5)
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc='upper left', ncol=2, bbox_to_anchor=(0.0, 1.0),
                  framealpha=0.85, frameon=True, edgecolor='none')

    # --- panel 2: the components, dense only (what the fg number is made of)
    ax = axs[2]
    for field, label, _ in series[2:]:
        vals = get(dense, field)
        ax.plot(range(len(vals)), vals, lw=0.9, alpha=0.85, label=label)
        for i, v in enumerate(vals):
            csv_rows.append({'clip': dense['id'], 'method': 'dense', 'pair': i,
                             'field': field, 'value': v})
    ax.set_ylabel('flow components\n(dense clip), px', fontsize=6.5)
    ax.legend(fontsize=5.8, loc='upper left', ncol=3, framealpha=0.85, frameon=True,
              edgecolor='none')
    ax.grid(alpha=0.22)
    ax.tick_params(labelsize=6)

    # --- panel 3: coverage, i.e. which pairs were allowed to count
    ax = axs[3]
    for field, label, colour in (('fg_cov_pct', 'foreground coverage', '#3977a8'),
                                 ('bg_cov_pct', 'background coverage', '#6b6b6b'),
                                 ('obj_cov_pct', 'object coverage', '#5aa68a')):
        vals = get(dense, field)
        ax.plot(range(len(vals)), vals, lw=0.9, color=colour, alpha=0.85, label=label)
    ax.axhline(gates['min_bg_cov_pct'], color='#a32b2b', lw=0.9, ls='--')
    ax.annotate(f'head-mount bg coverage floor = {gates["min_bg_cov_pct"]}%',
                xy=(0.012, gates['min_bg_cov_pct']),
                xycoords=('axes fraction', 'data'), fontsize=5.6, color='#a32b2b',
                va='bottom', ha='left')
    ax.set_ylabel('mask coverage\n(dense clip), %', fontsize=6.5)
    ax.set_xlabel('source frame pair index (within the clip)', fontsize=7)
    ax.legend(fontsize=5.8, loc='upper left', ncol=3, framealpha=0.85, frameon=True,
              edgecolor='none')
    ax.grid(alpha=0.22)
    ax.tick_params(labelsize=6)

    # --- the resulting numbers, beside the plot
    lines = [f"episode {group['pair_key']}", f"model {C.model_label(group['model'])}",
             f"{group['cell'][0]} / mount {group['cell'][1]} / {group['split']}",
             f"view {group['view']}", '']
    for key in ('actor_motion_raw', 'actor_motion_active_rate',
                'active_motion_continuity'):
        d, a = (dense['gated'] or {}).get(key), (acc['gated'] or {}).get(key)
        fmt = lambda v: 'gated out' if v is None else f'{v:.{cache.digits(key)}f}'
        delta = ('' if d is None or a is None else
                 f'   ({a - d:+.{cache.digits(key)}f})')
        lines.append(f'{cache.label(key)}')
        lines.append(f'   dense {fmt(d)}')
        lines.append(f'   {C.method_label(accel["method"])} {fmt(a)}{delta}')
    fig.text(0.868, 0.87, '\n'.join(lines), fontsize=5.8, va='top', family='DejaVu Sans',
             color='#222222',
             bbox=dict(boxstyle='round,pad=0.5', fc='#f7f7f7', ec='#dddddd'))
    if group.get('instruction'):
        fig.text(0.868, 0.35, C.wrap('task: ' + group['instruction'], 34), fontsize=5.4,
                 va='top', color='#555555')
    fig.suptitle(f'F9  Motion anatomy: one clip, per source-frame pair. dense vs '
                 f'{C.method_label(accel["method"])} on the SAME episode, view and '
                 f'geometry.\nThe metric is the median of the top panel -- a method '
                 f'lowers it by flattening the peaks, not by making every pair still',
                 y=0.985, fontsize=9)
    ctx.csv('f9_motion_anatomy', csv_rows)
    ctx.save(fig, 'f9_motion_anatomy', tight=False)


@figure('f10', 'Continuity anatomy: how AMC is built, triple by triple')
def f10_continuity_anatomy(ctx):
    """Active Motion Continuity per frame triple: copy error, VFI error, and the gain.

    AMC asks, for each (t-1, t, t+1): can a video-frame-interpolation model reconstruct
    the middle frame better than simply copying a neighbour? `gain = 1 - vfi_mse/copy_mse`
    -- high when motion is smooth and predictable, low when the middle frame is
    inconsistent with its neighbours.

    The figure exists because a single AMC number hides the TWO ways it can move:
      - `vfi_mse` rises (genuinely less continuous motion), or
      - `copy_mse` falls (less motion at all, so copying already works)
    The second is the over-smoothing artifact the `ACCEL_CAVEATS` entry warns about, and
    the bottom panel is where you can see which one happened. Triples below the copy-error
    floor carry ZERO weight -- they are shaded, and they are why a busy-looking clip can
    still be gated out of the column entirely.
    """
    cache, plt, np = ctx.cache, ctx.plt, ctx.np
    group, accel = pick_group(cache)
    dense, acc = cache.perframe(group['dense_id']), cache.perframe(accel['id'])
    if not dense or not acc:
        raise SystemExit('exemplar per-frame files missing')
    floor = cache.policy['gates']['copy_err_floor']
    fig, axs = plt.subplots(3, 1, figsize=(12.6, 7.4), sharex=True,
                            gridspec_kw=dict(left=0.072, right=0.845, top=0.875,
                                             bottom=0.082, hspace=0.22))
    csv_rows = []
    both = ((dense, 'dense', '#333333'),
            (acc, C.method_label(accel['method']), C.family_color(accel['method'])))

    # --- panel 0: copy error, with the weight floor
    ax = axs[0]
    for pf, name, colour in both:
        vals = (pf['continuity'] or {}).get('copy_mse') or []
        ax.plot(range(len(vals)), vals, color=colour, lw=1.0, alpha=0.9, label=name)
        below = [i for i, v in enumerate(vals) if v is not None and v < floor]
        if below:
            ax.plot(below, [vals[i] for i in below], 'o', ms=2.2, color=colour,
                    alpha=0.5)
    ax.axhline(floor, color='#a32b2b', lw=1.0, ls='--')
    ax.annotate(f'copy-error floor = {floor} -> weight 0 (triple does not count)',
                xy=(0.012, floor), xycoords=('axes fraction', 'data'), fontsize=5.8,
                color='#a32b2b', va='bottom', ha='left')
    ax.set_yscale('log')
    ax.set_ylabel('copy_mse\n(how much the frame moves)', fontsize=6.5)
    ax.legend(fontsize=6, loc='upper left', ncol=2, frameon=True, framealpha=0.85,
              edgecolor='none')
    ax.grid(alpha=0.22)
    ax.tick_params(labelsize=6)

    # --- panel 1: VFI error
    ax = axs[1]
    for pf, name, colour in both:
        vals = (pf['continuity'] or {}).get('vfi_mse') or []
        ax.plot(range(len(vals)), vals, color=colour, lw=1.0, alpha=0.9, label=name)
    ax.set_yscale('log')
    ax.set_ylabel('vfi_mse\n(how unpredictable it is)', fontsize=6.5)
    ax.legend(fontsize=6, loc='upper left', ncol=2, frameon=True, framealpha=0.85,
              edgecolor='none')
    ax.grid(alpha=0.22)
    ax.tick_params(labelsize=6)

    # --- panel 2: the gain, which IS the metric, with zero-weight triples shaded
    ax = axs[2]
    for pf, name, colour in both:
        cont = pf['continuity'] or {}
        gains, copies = cont.get('gain') or [], cont.get('copy_mse') or []
        ax.plot(range(len(gains)), [100 * g if g is not None else None for g in gains],
                color=colour, lw=1.0, alpha=0.9, label=name)
        good = [100 * g for g, c in zip(gains, copies)
                if g is not None and c is not None and c >= floor]
        if good:
            ax.axhline(C_median(good), color=colour, lw=0.9, ls=':', alpha=0.8)
            ax.annotate(f'{name} weighted median {C_median(good):.1f}',
                        xy=(0.999, C_median(good)),
                        xycoords=('axes fraction', 'data'), fontsize=5.6,
                        color=colour, va='bottom', ha='right')
        for i, (g, c) in enumerate(zip(gains, copies)):
            csv_rows.append({'clip': pf['id'], 'method': pf['method'], 'triple': i,
                             'gain_x100': None if g is None else 100 * g,
                             'copy_mse': c,
                             'vfi_mse': (cont.get('vfi_mse') or [None] * len(gains))[i]
                             if i < len(cont.get('vfi_mse') or []) else None,
                             'counts_toward_amc': bool(c is not None and c >= floor)})
    # Shade the zero-weight stretches of the DENSE clip (the reference geometry).
    dcopy = (dense['continuity'] or {}).get('copy_mse') or []
    start = None
    for i, c in enumerate(dcopy + [None]):
        dead = c is not None and c < floor
        if dead and start is None:
            start = i
        elif not dead and start is not None:
            ax.axvspan(start - 0.5, i - 0.5, color='#f0e4e4', zorder=0)
            start = None
    ax.set_ylabel('gain x100\n(= Active Motion Continuity)', fontsize=6.5)
    ax.set_xlabel('frame triple index (within the clip)', fontsize=7)
    ax.legend(fontsize=6, loc='lower left', ncol=2, frameon=True, framealpha=0.85,
              edgecolor='none')
    ax.grid(alpha=0.22)
    ax.tick_params(labelsize=6)

    key = 'active_motion_continuity'
    d, a = (dense['gated'] or {}).get(key), (acc['gated'] or {}).get(key)
    dc = [c for c in dcopy if c is not None]
    ac = [c for c in ((acc['continuity'] or {}).get('copy_mse') or []) if c is not None]
    lines = [f"episode {group['pair_key']}", f"model {C.model_label(group['model'])}",
             f"{group['cell'][0]} / mount {group['cell'][1]}", f"view {group['view']}",
             '', 'Active Motion Continuity',
             f"   dense {'gated out' if d is None else f'{d:.2f}'}",
             f"   {C.method_label(accel['method'])} "
             f"{'gated out' if a is None else f'{a:.2f}'}"
             + ('' if None in (d, a) else f'  ({a - d:+.2f})'), '',
             'zero-weight triples',
             f'   dense {sum(1 for c in dc if c < floor)} / {len(dc)}',
             f"   {C.method_label(accel['method'])} "
             f'{sum(1 for c in ac if c < floor)} / {len(ac)}', '',
             'median copy_mse',
             f'   dense {C_median(dc):.1f}' if dc else '   dense n/a',
             f"   {C.method_label(accel['method'])} {C_median(ac):.1f}" if ac else '']
    fig.text(0.852, 0.86, '\n'.join(lines), fontsize=5.8, va='top', color='#222222',
             bbox=dict(boxstyle='round,pad=0.5', fc='#f7f7f7', ec='#dddddd'))
    caveat = (cache.policy.get('accel_caveats') or {}).get(key)
    if caveat:
        fig.text(0.852, 0.30, C.wrap('Why a HIGHER AMC can be the artifact: ' + caveat,
                                     36), fontsize=4.9, va='top', color='#8a6a14')
    fig.suptitle('F10  Continuity anatomy. AMC = median over triples of '
                 '1 - vfi_mse/copy_mse, weighted, with triples below the copy-error '
                 'floor carrying ZERO weight (shaded).\nA method can raise AMC two ways: '
                 'genuinely smoother motion (vfi_mse falls) or LESS motion '
                 '(copy_mse falls) -- only the first is an improvement',
                 y=0.985, fontsize=9)
    ctx.csv('f10_continuity_anatomy', csv_rows)
    ctx.save(fig, 'f10_continuity_anatomy', tight=False)


@figure('f11', 'Confound panel: which metrics are really reading motion, per mount')
def f11_confound(ctx):
    """Every metric against Actor Activity, on each camera mount.

    A no-reference "quality" score that tracks how much the scene moved is not measuring
    quality -- it is measuring motion, and on a wrist camera almost everything moves. The
    bar is |rho| >= 0.35: above it, a delta on that metric cannot be attributed to
    quality without controlling for motion, which is what `CONFOUNDED_MOUNTS` records and
    what puts the 'x' markers on F6.

    Hexbin rather than a scatter of 38k points: the shape of the cloud is the evidence,
    and overplotting hides it.
    """
    cache, plt, np = ctx.cache, ctx.plt, ctx.np
    red = cache.agg['redundancy']
    mounts = [m for m in C.MOUNT_ORDER if m in (red.get('by_mount') or {})]
    others = [k for k in cache.keys if k != 'actor_motion_raw']
    fig, axs = plt.subplots(len(mounts), len(others),
                            figsize=(2.15 * len(others) + 1.0,
                                     2.05 * len(mounts) + 1.5),
                            gridspec_kw=dict(left=0.072, right=0.995, top=0.865,
                                             bottom=0.075, wspace=0.30, hspace=0.36))
    axs = axs.reshape(len(mounts), len(others))
    BAR = 0.35
    confounded = cache.policy.get('confounded_mounts') or {}
    csv_rows = []
    for ri, mount in enumerate(mounts):
        subset = cache.select(camera_mount=mount)
        xs_all = [c['gated'].get('actor_motion_raw') for c in subset]
        for ci, key in enumerate(others):
            ax = axs[ri, ci]
            ys_all = [c['gated'].get(key) for c in subset]
            pairs = [(x, y) for x, y in zip(xs_all, ys_all)
                     if x is not None and y is not None]
            cell = (red['by_mount'][mount].get(f'actor_motion_raw|{key}')
                    or red['by_mount'][mount].get(f'{key}|actor_motion_raw') or {})
            rho = cell.get('rho')
            flagged = rho is not None and abs(rho) >= BAR
            if len(pairs) >= 30:
                xs = [p[0] for p in pairs]
                ys = [p[1] for p in pairs]
                ax.hexbin(xs, ys, gridsize=26, cmap='Reds' if flagged else 'Blues',
                          mincnt=1, linewidths=0, bins='log')
                # Median trend in motion deciles: the visual form of the rho.
                order = sorted(pairs)
                nb = 8
                step = max(len(order) // nb, 1)
                bx, byy = [], []
                for i in range(0, len(order), step):
                    chunk = order[i:i + step]
                    if len(chunk) >= 8:
                        bx.append(C_median([c[0] for c in chunk]))
                        byy.append(C_median([c[1] for c in chunk]))
                if len(bx) > 1:
                    ax.plot(bx, byy, '-o', ms=2.4, lw=1.1, color='#111111', alpha=0.85)
                ax.set_xlim(0, float(np.percentile(xs, 98)) or 1)
                lo, hi = np.percentile(ys, 1), np.percentile(ys, 99)
                if hi > lo:
                    ax.set_ylim(lo, hi)
            else:
                ax.text(0.5, 0.5, 'too few clips', transform=ax.transAxes,
                        ha='center', va='center', fontsize=6, color='#aaaaaa')
            tag = 'CONFOUNDED' if flagged else 'ok'
            ax.text(0.03, 0.965, (f'rho = {rho:+.2f}  {tag}' if rho is not None
                                  else 'rho n/a'),
                    transform=ax.transAxes, va='top', fontsize=5.8,
                    fontweight='bold' if flagged else 'normal',
                    color='white' if flagged else '#333333',
                    bbox=dict(boxstyle='round,pad=0.22',
                              fc='#a32b2b' if flagged else '#eeeeee', ec='none'))
            ax.text(0.97, 0.965, f'n={len(pairs)}', transform=ax.transAxes, ha='right',
                    va='top', fontsize=5.2, color='#888888')
            if flagged:
                for sp in ax.spines.values():
                    sp.set_color('#a32b2b')
                    sp.set_linewidth(1.3)
            if ri == 0:
                short, _, badge = C.STATUS_STYLE[cache.status(key)]
                ax.set_title(f'{cache.label(key)}', fontsize=7, fontweight='bold',
                             pad=12)
                ax.text(0.5, 1.015, short, transform=ax.transAxes, ha='center',
                        va='bottom', fontsize=4.8, color=badge, fontweight='bold')
            if ci == 0:
                known = mount in str(confounded)
                ax.set_ylabel(f'mount = {mount}'
                              + ('\n(recorded as confounded)' if known else ''),
                              fontsize=7, fontweight='bold')
            if ri == len(mounts) - 1:
                ax.set_xlabel('Actor Activity (px / source frame)', fontsize=5.8)
            ax.tick_params(labelsize=5)
            ax.grid(alpha=0.18)
            csv_rows.append({'mount': mount, 'metric': key, 'rho_vs_actor_motion': rho,
                             'n': len(pairs), 'confounded_at_0.35': bool(flagged)})
    fig.suptitle(f'F11  Every metric against Actor Activity, per camera mount. '
                 f'Red panel = |rho| >= {BAR}: on that mount the column tracks how much '
                 f'the scene moved, so a delta there cannot be read as "quality" '
                 f'without controlling for motion', y=0.975, fontsize=9)
    note = confounded.get('actor_flicker_raw') or next(iter(confounded.values()), '')
    if note:
        fig.text(0.008, 0.012, C.wrap('Recorded: ' + str(note), 250), fontsize=5.2,
                 color='#8a2b2b', va='bottom')
    ctx.csv('f11_confound', csv_rows)
    ctx.save(fig, 'f11_confound', tight=False)


@figure('f12', 'Flicker autopsy: why Actor Flicker is DROPPED, in four measurements')
def f12_flicker_autopsy(ctx):
    """The appendix figure that justifies deleting a metric instead of reporting it.

    Actor Flicker was meant to measure temporal instability (a limb that strobes between
    frames). It cannot, because an RGB residual between consecutive frames rises with ANY
    motion, and no variant escapes that. Four panels:

      A  every flicker variant vs Actor Activity, per mount, against the |rho| < 0.25
         pass criterion the drop decision used. Nothing passes.
      B  the ranking the raw column would have produced vs the ranking Visual Integrity
         gives -- a "quality" order that disagrees with quality.
      C  the distribution per mount, showing the mount effect dwarfs the method effect.
      D  the verbatim recorded verdict.
    """
    cache, plt, np = ctx.cache, ctx.plt, ctx.np
    fv = cache.agg['redundancy'].get('flicker_variants_vs_motion') or {}
    bar = fv.get('bar', 0.25)
    by_mount = fv.get('by_mount') or {}
    variants = fv.get('variants') or []
    fig = plt.figure(figsize=(13.6, 8.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 0.92],
                          left=0.075, right=0.985, top=0.875, bottom=0.075,
                          wspace=0.20, hspace=0.36)
    csv_rows = []

    # --- A: variants vs motion
    ax = fig.add_subplot(gs[0, 0])
    mounts = [m for m in C.MOUNT_ORDER if m in by_mount]
    x = np.arange(len(variants))
    width = 0.8 / max(len(mounts), 1)
    passed_any = []
    for mi, mount in enumerate(mounts):
        rhos = [(by_mount[mount].get(v) or {}).get('rho') for v in variants]
        ax.bar(x + mi * width - 0.4 + width / 2,
               [abs(r) if r is not None else 0 for r in rhos], width=width * 0.92,
               label=f'mount = {mount}', alpha=0.88)
        for v, r in zip(variants, rhos):
            csv_rows.append({'panel': 'A', 'mount': mount, 'variant': v,
                             'rho_vs_motion': r,
                             'n': (by_mount[mount].get(v) or {}).get('n'),
                             'passes_bar': bool(r is not None and abs(r) < bar)})
    for v in variants:
        rr = [(by_mount[m].get(v) or {}).get('rho') for m in mounts]
        rr = [r for r in rr if r is not None]
        if rr and max(abs(r) for r in rr) < bar:
            passed_any.append(v)
    ax.axhline(bar, color='#a32b2b', lw=1.2, ls='--')
    ax.annotate(f'pass criterion: |rho| < {bar} on EVERY mount',
                xy=(0.012, bar), xycoords=('axes fraction', 'data'), fontsize=6.2,
                color='#a32b2b', va='bottom', ha='left', fontweight='bold')
    ax.set_xticks(x, [C.wrap(v.replace('_', ' '), 13) for v in variants], fontsize=4.8,
                  rotation=38, ha='right')
    ax.set_ylabel('|Spearman rho| vs Actor Activity', fontsize=7)
    ax.set_title(f'A. {len(variants) - len(passed_any)} of {len(variants)} flicker '
                 f'variants are motion-confounded on at least one mount', loc='left',
                 fontsize=8)
    if passed_any:
        # Be precise: a few variants DO clear the motion bar. They were still dropped,
        # for the separate reason in panel D (they move the WRONG way on real artifacts),
        # and claiming "none passed" would be false.
        ax.text(0.012, 0.90, 'clears the motion bar but still dropped (see D): '
                + ', '.join(v.replace('_', ' ') for v in passed_any),
                transform=ax.transAxes, fontsize=5.4, color='#8a6a14', va='top')
    ax.legend(fontsize=6, ncol=3, loc='upper right')
    ax.grid(axis='y', alpha=0.25)
    ax.tick_params(labelsize=6)

    # --- B: the ranking it would have produced vs Visual Integrity
    ax2 = fig.add_subplot(gs[0, 1])
    methods = C.sort_methods({c['method'] for c in cache.clips
                              if c.get('method') and c['method'] != 'gt'})
    rank_rows = []
    for m in methods:
        fl = cache.values('actor_flicker_raw', method=m)
        vi = cache.values('visual_integrity', method=m)
        if len(fl) >= 30 and len(vi) >= 30:
            rank_rows.append((m, C_median(fl), C_median(vi)))
    # Actor Flicker: lower = "more stable"; Visual Integrity: higher = better.
    by_fl = [m for m, _, _ in sorted(rank_rows, key=lambda r: r[1])]
    by_vi = [m for m, _, _ in sorted(rank_rows, key=lambda r: -r[2])]
    for i, m in enumerate(by_fl):
        j = by_vi.index(m)
        colour = C.family_color(m)
        ax2.plot([0, 1], [-i, -j], '-', color=colour, lw=1.3, alpha=0.75)
        ax2.plot(0, -i, 'o', ms=4.5, color=colour)
        ax2.plot(1, -j, 'o', ms=4.5, color=colour)
        ax2.text(-0.04, -i, C.method_label(m), ha='right', va='center', fontsize=5.6)
        ax2.text(1.04, -j, C.method_label(m), ha='left', va='center', fontsize=5.6)
        csv_rows.append({'panel': 'B', 'method': m, 'rank_by_flicker': i + 1,
                         'rank_by_visual_integrity': j + 1,
                         'flicker_median': dict((mm, f) for mm, f, _ in rank_rows)[m],
                         'vi_median': dict((mm, v) for mm, _, v in rank_rows)[m]})
    ax2.set_xlim(-0.42, 1.42)
    ax2.set_xticks([0, 1], ['ranked by\nActor Flicker\n(lower = "better")',
                            'ranked by\nVisual Integrity\n(higher = better)'],
                   fontsize=6)
    ax2.set_yticks([])
    for sp in ax2.spines.values():
        sp.set_visible(False)
    # Kendall-style disagreement count: inverted pairs between the two orders.
    inv = sum(1 for i in range(len(by_fl)) for j in range(i + 1, len(by_fl))
              if by_vi.index(by_fl[i]) > by_vi.index(by_fl[j]))
    tot = max(len(by_fl) * (len(by_fl) - 1) // 2, 1)
    ax2.set_title(f'B. The order it would have published disagrees with quality\n'
                  f'{inv} of {tot} method pairs are inverted', loc='left', fontsize=8)

    # --- C: mount effect dwarfs the method effect
    ax3 = fig.add_subplot(gs[1, 0])
    mount_list = [m for m in C.MOUNT_ORDER
                  if any(c.get('camera_mount') == m for c in cache.clips)]
    data, labels, colours = [], [], []
    for mount in mount_list:
        for m in ('dense', 'itm', 'pisa'):
            vals = cache.values('actor_flicker_raw', camera_mount=mount, method=m)
            if len(vals) >= 20:
                data.append(vals)
                labels.append(f'{mount}\n{C.method_label(m)}\nn={len(vals)}')
                colours.append(C.family_color(m))
                csv_rows.append({'panel': 'C', 'mount': mount, 'method': m,
                                 'n': len(vals), 'median': C_median(vals)})
    if data:
        bp = ax3.boxplot(data, widths=0.6, showfliers=False, patch_artist=True,
                         medianprops=dict(color='black', lw=1.1))
        for patch, colour in zip(bp['boxes'], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.55)
        ax3.set_xticks(range(1, len(labels) + 1), labels, fontsize=5)
        # Quantify the two spreads SEPARATELY -- mixing them into one range would be the
        # same confound the figure is about.
        dense_by_mount = [C_median(cache.values('actor_flicker_raw',
                                                camera_mount=mt, method='dense'))
                          for mt in mount_list
                          if len(cache.values('actor_flicker_raw', camera_mount=mt,
                                              method='dense')) >= 20]
        per_mount_method = []
        for mt in mount_list:
            meds = [C_median(cache.values('actor_flicker_raw', camera_mount=mt,
                                          method=m))
                    for m in ('dense', 'itm', 'pisa')
                    if len(cache.values('actor_flicker_raw', camera_mount=mt,
                                        method=m)) >= 20]
            if len(meds) > 1:
                per_mount_method.append(max(meds) - min(meds))
        msg = []
        if len(dense_by_mount) > 1:
            msg.append(f'dense across mounts: {max(dense_by_mount) - min(dense_by_mount):.4f}')
        if per_mount_method:
            msg.append(f'methods within a mount: up to {max(per_mount_method):.4f}')
        ax3.text(0.5, 0.965, '   |   '.join(msg) + '   (RGB residual, 0-1)',
                 transform=ax3.transAxes, ha='center', va='top', fontsize=6,
                 color='#8a2b2b')
        csv_rows.append({'panel': 'C', 'mount': 'ALL', 'method': 'spread_summary',
                         'n': None,
                         'median': None,
                         'dense_mount_spread': (max(dense_by_mount) - min(dense_by_mount)
                                                if len(dense_by_mount) > 1 else None),
                         'max_method_spread_within_mount': (max(per_mount_method)
                                                            if per_mount_method else None)})
    ax3.set_ylabel('Actor Flicker (RGB residual)', fontsize=7)
    ax3.set_title('C. The number moves with the camera mount AND with any method that '
                  'changes motion -- never with quality alone', loc='left', fontsize=8)
    ax3.grid(axis='y', alpha=0.25)
    ax3.tick_params(labelsize=6)

    # --- D: the recorded verdict, verbatim
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    verdict = (cache.policy.get('dropped') or {}).get('actor_flicker_raw', '')
    ax4.text(0.0, 1.0, 'D. Recorded verdict (verbatim)', fontsize=8, va='top',
             fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.0, 0.90, C.wrap(verdict, 62), fontsize=5.6, va='top', color='#7a2020',
             transform=ax4.transAxes,
             bbox=dict(boxstyle='round,pad=0.55', fc='#fdf3f3', ec='#e0bebe'))
    fig.suptitle('F12  Actor Flicker autopsy: it is DROPPED, not reported. An RGB '
                 'residual between consecutive frames rises with any motion, so no '
                 'variant separates instability from activity',
                 y=0.975, fontsize=9.5)
    ctx.csv('f12_flicker_autopsy', csv_rows)
    ctx.save(fig, 'f12_flicker_autopsy', tight=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--cache', type=Path, default=None,
                    help='cache dir (default <run>/figures/cache)')
    ap.add_argument('--out', type=Path, default=None,
                    help='figure dir (default <run>/figures)')
    ap.add_argument('--only', nargs='*', default=None,
                    help='figure ids to build, e.g. --only f1 f5')
    ap.add_argument('--formats', nargs='*', default=['pdf', 'svg', 'png'])
    args = ap.parse_args()
    cache_dir = args.cache or args.run / 'figures' / 'cache'
    if not (cache_dir / 'agg.json').exists():
        raise SystemExit(f'no cache at {cache_dir}; run assets/scripts/nr_viz_cache.py first')
    out = args.out or args.run / 'figures'
    cache = C.Cache(cache_dir)
    print(f'cache: {cache.stamp()}')
    ctx = Ctx(cache, out, tuple(args.formats))
    wanted = args.only or list(FIGURES)
    unknown = [f for f in wanted if f not in FIGURES]
    if unknown:
        raise SystemExit(f'unknown figure ids {unknown}; have {sorted(FIGURES)}')
    failed = []
    for fid in wanted:
        title, fn = FIGURES[fid]
        print(f'[{fid}] {title}')
        try:
            fn(ctx)
        except Exception as exc:                       # keep going: one bad panel must
            failed.append((fid, repr(exc)))            # not cost the other eleven
            print(f'  !! FAILED: {exc!r}')
            import traceback
            traceback.print_exc()
    (out / 'figures_index.json').write_text(json.dumps({
        'generated_by': 'assets/scripts/make_nr_metric_figures.py',
        'cache': str(cache_dir), 'stamp': cache.stamp(),
        'figures': {fid: FIGURES[fid][0] for fid in wanted if fid not in dict(failed)},
        'failed': dict(failed),
    }, indent=1))
    if failed:
        raise SystemExit(f'{len(failed)} figure(s) failed: {[f for f, _ in failed]}')
    print(f'\n{len(wanted)} figures -> {out}')


if __name__ == '__main__':
    main()
