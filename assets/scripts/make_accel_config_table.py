#!/usr/bin/env python3
"""Emit the appendix LaTeX table: rows = block x model x camera-track x embodiment.

This is the per-configuration expansion of the pooled `method x model family` grid
printed by `assets/scripts/table_paper_main.py`, which collapses the camera track and the
embodiment into one line per (method, model).

DEFAULT VIEW = ABSOLUTE, WITH A DENSE BLOCK FIRST
-------------------------------------------------
The table opens with a `Dense` block covering every configuration, then one block per
acceleration method, and the numbers are ABSOLUTE per-configuration medians (the
reference-free column a benchmark normally shows) -- not deltas. That is what makes the
dense block readable: it is the level each method is compared against, in its own
configuration.

Splitting by (model, track, embodiment) is what licenses absolute numbers here. Absolute
Active Motion Continuity is NOT comparable across embodiments (humanoid sits well below
the manipulators) and the pooled absolute view in `table_paper_main.py` is explicitly
flagged `pooled_unpaired_is_confounded`; per-configuration rows do not pool those levels,
so a reader compares a method to the dense row of the SAME configuration only -- never
down a column across configurations.

One column is still withheld by default: absolute `text_condition_alignment` (CLIPScore)
carries a ~7-point per-caption style/length bias, so its absolute median is not a
comparable number no matter how many clips back it (`T.PAIRED_ONLY` in
`assets/scripts/table_dense_nr.py` withholds it for the same reason). `--with-text-align`
forces it in; `--paired` restores it automatically, because in a paired delta the
per-caption bias cancels exactly.

`--paired` emits the delta-vs-dense variant instead (accelerated minus its OWN dense
clip). That view has NO dense block by construction -- a clip's delta against itself is
zero -- and is written to `accel_by_config_paired.tex`.

STRUCTURE ONLY for now: every number is `\\ph` (a grey dot) so the layout can be
reviewed before a scoring wave is joined in. Nothing here is hand-typed -- the
configuration rows are the (model, track, embodiment) cells that actually exist in the
scored records (`metrics/WorldScore/results/dense_nr_full/results/*.json`):

    Ctrl-World  multi-view  only          single-arm, bimanual, humanoid
    DreamGen    multi-view + single-view  single-arm, bimanual, humanoid (each)
    DreamDojo   single-view only          single-arm, bimanual, humanoid

so Ctrl-World / DreamDojo carry ONE track each and DreamGen carries two. Ctrl-World's
two multi-view layouts (3-view stack, and the 4-view grid that exists only for bimanual
and humanoid) are pooled into one Ctrl-World line, matching `FAMILY_OF` in
table_paper_main.py; `--ctrlworld-layouts` breaks them out into their own rows.

SiTo / ITM were never run on DreamDojo (they have no DreamDojo entry in the pooled
grid). Per the acceleration-benchmark policy a missing cell is reported explicitly as
`--` rather than dropped, so the reader can tell "not run" from "ran and lost".

No `\\multirow`: this is a `longtable` and a multirow spanning a page break silently
mis-aligns. Group labels are printed on the first row of their group and the block name
is a full-width section header, which survives any break.

    python3 assets/scripts/make_accel_config_table.py                  # absolute + Dense block
    python3 assets/scripts/make_accel_config_table.py --paired          # delta vs dense
    python3 assets/scripts/make_accel_config_table.py --ctrlworld-layouts
    python3 assets/scripts/make_accel_config_table.py --repeat-labels   # label on every row
"""

from __future__ import annotations

import argparse
import collections
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "assets/paper/appendix/tables/accel_by_config.tex"

# order / display names / family grouping mirror assets/scripts/table_paper_main.py
METHODS = [
    ("dicache", "DiCache", "step caching"),
    ("worldcache", "WorldCache", "step caching"),
    ("fastercache", "FasterCache", "step caching"),
    ("pisa", "PISA", "sparse attention"),
    ("radial", "Radial", "sparse attention"),
    ("svg1", "SVG1", "sparse attention"),
    ("svg2", "SVG2", "sparse attention"),
    ("sito", "SiTo", "token pruning"),
    ("itm", "ITM", "token pruning"),
]

EMB = [("single_arm", "single-arm"), ("bimanual", "bimanual"), ("humanoid", "humanoid")]
EMB_KEY = {d: k for k, d in EMB}      # display label -> the key used in the records

# (method, model family) pairs that were never run -- reported as `--`, not dropped.
NOT_RUN = {("sito", "DreamDojo"), ("itm", "DreamDojo")}

# `paired_only` mirrors T.PAIRED_ONLY in assets/scripts/table_dense_nr.py: an ABSOLUTE
# CLIPScore median mixes a ~7-point per-caption style/length bias across different
# prompts, so it is not a comparable number and is withheld from the absolute view
# (it cancels exactly in the paired view, where pred and its own dense/GT partner are
# scored against the SAME caption).
METRICS = [
    (r"\makecell{Visual\\Integrity}", "visual_integrity", False),
    (r"\makecell{Text\\Align.}", "text_condition_alignment", True),
    (r"\makecell{Actor\\Activity}", "actor_motion_raw", False),
    # Active Rate removed 2026-09-20: rho +0.875 with Actor Activity (+0.946 on wrist)
    # over 2,428 clips, and the "still, then a lurch" pattern it existed to separate
    # occurs in 1 clip of 2,428. See T.NOT_REPORTED in table_dense_nr.py.
    (r"\makecell{Active Motion\\Continuity}", "active_motion_continuity", False),
]

CAPTION_ABS = (
    r"\textbf{Per-configuration results on \ours.} One row per "
    r"(model, camera track, embodiment) configuration that exists in the benchmark, "
    r"for the dense baseline and for every acceleration method. Values are "
    r"\emph{absolute} medians over the clips of that configuration, each with a "
    r"bootstrap CI, exactly as a reference-free benchmark column is normally "
    r"reported; \textbf{Dense} is the reference row of its own configuration and every "
    r"method below it is read against the dense row \emph{in the same configuration}, "
    r"never across configurations. \emph{Speedup} is the measured wall-clock ratio of "
    r"that configuration against its dense run. \texttt{--} marks a configuration the "
    r"method was not run on, or one whose timing was not instrumented; it is never "
    r"back-filled from another cell. $n$ is the largest per-metric clip count in the "
    r"row. Absolute Active Motion Continuity is \emph{not} comparable across "
    r"embodiments (humanoid sits well below the manipulators), which is precisely why "
    r"this table is split by embodiment instead of pooling them. Ctrl-World is "
    r"multi-view only and DreamDojo single-view only, hence one track each; "
    r"Ctrl-World's 3-view and 4-view-grid multi-view layouts are pooled into one line. "
    r"SiTo and ITM were not run on DreamDojo."
)

CAPTION_PAIRED = (
    r"\textbf{Per-configuration acceleration results on \ours.} The same "
    r"configurations, reported as a \emph{paired} median delta against that "
    r"configuration's own dense clip (accelerated $-$ dense, same episode, same matched "
    r"frame grid): negative is worse than dense, and the value carries no absolute "
    r"level, so a per-embodiment level difference cannot leak in. \emph{Speedup} is the "
    r"measured wall-clock ratio of that same configuration. \texttt{--} marks a "
    r"configuration the method was not run on, or one whose timing was not "
    r"instrumented. $n$ is the largest per-metric pair count in the row. "
    r"$^{*}$~resolved (bootstrap CI excludes $0$, Wilcoxon $p<0.05$, $n\ge8$); "
    r"$^{\dagger}$~a solid move in a direction that cannot be an improvement "
    r"(over-smoothing inflating Active Motion Continuity); "
    r"$^{\sim}$~$n<8$, shown but not claimed. Ctrl-World is multi-view only and "
    r"DreamDojo single-view only; SiTo and ITM were not run on DreamDojo."
)

CAPTION_LAYOUTS = (
    r" Here Ctrl-World's two multi-view layouts are kept apart (\emph{3-view} stack "
    r"vs.\ \emph{4-grid} composition); the 4-grid layout exists only for bimanual and "
    r"humanoid."
)


def config_rows(split_layouts: bool):
    """[(model, track, layout, embodiment_display)] in table order."""
    rows = []
    if split_layouts:
        for lay, embs in (("3-view", EMB), ("4-grid", EMB[1:])):
            for _, d in embs:
                rows.append(("Ctrl-World", "multi-view", lay, d))
    else:
        rows += [("Ctrl-World", "multi-view", "", d) for _, d in EMB]
    rows += [("DreamGen", "multi-view", "", d) for _, d in EMB]
    rows += [("DreamGen", "single-view", "", d) for _, d in EMB]
    rows += [("DreamDojo", "single-view", "", d) for _, d in EMB]
    return rows


def load_cells(run: Path):
    """(family, track, embodiment, method) -> {metric_key: [values]} from scored records.

    `track` is derived from the record's path, not from the model name: DreamGen is the
    only family with both a multi-view and a single-view tree, and the only thing that
    distinguishes them in a record is `relative` containing `/multiview/` or
    `/singleview/`.

    Ctrl-World's two multi-view layouts (3-view `ctrlworld`, and the `ctrlworld_4view_grid`
    that exists only for bimanual/humanoid) are POOLED into one Ctrl-World line, matching
    FAMILY_OF in table_paper_main.py. Note the 3-view records only exist for single_arm in
    this wave -- the bimanual/humanoid 3-view renders are deliberately excluded from
    scoring (see excluded_row() in scripts/run_dense_nr.py), so those lines are 4-grid only.
    """
    sys.path.insert(0, str(REPO / 'assets/scripts'))
    import table_dense_nr as T

    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in T.load_records(run):
        method = r.get('method')
        if method == 'gt':
            continue
        model = r.get('model', '')
        rel = r.get('relative', '')
        if model.startswith('ctrlworld'):
            fam = 'Ctrl-World'
        elif model.startswith('dreamdojo'):
            fam = 'DreamDojo'
        elif model.startswith('dreamgen'):
            fam = 'DreamGen'
        else:
            continue
        track = ('multi-view' if '/multiview/' in rel
                 else 'single-view' if '/singleview/' in rel else None)
        if track is None:
            continue
        key = (fam, track, r.get('embodiment'), method)
        sm = T.subject_metrics(r)
        for _, mk, _ in METRICS:
            v = sm[mk] if mk in sm else r.get(mk)
            if v is not None:
                cells[key][mk].append(v)
    return cells


def load_paired(run: Path):
    """Same keying, but each value is (method - its OWN dense clip) for that metric.

    Pairing is on `pair_key` (episode + view, method-free by construction) and gated by
    T.comparable_pair, which drops pairs whose two clips are not the same measurement --
    12.4% of accel pairs on this dataset came from a re-render at a different length/grid
    and would otherwise contribute a difference that is not caused by the method.
    """
    sys.path.insert(0, str(REPO / 'assets/scripts'))
    import table_dense_nr as T

    recs = T.load_records(run)
    dense = {r.get('pair_key'): r for r in recs if r.get('method') == 'dense'}
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        method = r.get('method')
        if method in ('gt', 'dense'):
            continue
        g = dense.get(r.get('pair_key'))
        if g is None or not T.comparable_pair(r, g):
            continue
        model = r.get('model', '')
        rel = r.get('relative', '')
        if model.startswith('ctrlworld'):
            fam = 'Ctrl-World'
        elif model.startswith('dreamdojo'):
            fam = 'DreamDojo'
        elif model.startswith('dreamgen'):
            fam = 'DreamGen'
        else:
            continue
        track = ('multi-view' if '/multiview/' in rel
                 else 'single-view' if '/singleview/' in rel else None)
        if track is None:
            continue
        sa, sb = T.subject_metrics(r), T.subject_metrics(g)
        for _, mk, _ in METRICS:
            a = sa[mk] if mk in sa else r.get(mk)
            b = sb[mk] if mk in sb else g.get(mk)
            if a is not None and b is not None:
                out[(fam, track, r.get('embodiment'), method)][mk].append(a - b)
    return out


def fmt(vals, digits, signed):
    """Median of `vals`, or \\ph when the cell has no data at all."""
    if not vals:
        return r"\ph"
    m = statistics.median(vals)
    return f"{m:+.{digits}f}" if signed else f"{m:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", action="store_true",
                    help="paired delta-vs-dense view (no Dense row) instead of absolute")
    ap.add_argument("--run", type=Path,
                    default=REPO / 'metrics/WorldScore/results/dense_nr_full',
                    help="scored run to read; omit --run only with --structure-only")
    ap.add_argument("--structure-only", action="store_true",
                    help="emit the layout with every number as \\ph (no run needed)")
    ap.add_argument("--with-text-align", action="store_true",
                    help="keep the Text Alignment column in the absolute view "
                         "(withheld by default: absolute CLIPScore is not comparable "
                         "across captions)")
    ap.add_argument("--ctrlworld-layouts", action="store_true",
                    help="break Ctrl-World's 3-view / 4-view-grid layouts into own rows")
    ap.add_argument("--repeat-labels", action="store_true",
                    help="print Model/Track on every row instead of once per group")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: accel_by_config.tex / accel_by_config_paired.tex")
    args = ap.parse_args()

    paired = args.paired
    out = args.out or (OUT.with_name("accel_by_config_paired.tex") if paired else OUT)
    label = ("tab:supp_accel_by_config_paired" if paired
             else "tab:supp_accel_by_config")
    # absolute view drops the paired-only columns unless explicitly asked for
    metrics = [(h, k) for h, k, po in METRICS
               if paired or args.with_text_align or not po]
    nm = len(metrics)

    cfgs = config_rows(args.ctrlworld_layouts)
    data = None if args.structure_only else (
        load_paired(args.run) if paired else load_cells(args.run))
    lay_col = args.ctrlworld_layouts
    # Model | Track | (Layout) | Embodiment | Speedup | metrics... | n
    lead = 3 + (1 if lay_col else 0)
    ncols = lead + 1 + nm + 1
    colspec = "ll" + ("l" if lay_col else "") + "l" + "r" * (nm + 2)
    m_first, m_last = lead + 2, lead + 1 + nm              # metric column span

    # Dense is a configuration-level reference row, not an acceleration method, so it
    # gets its own leading block and is absent from the paired view (a clip's delta
    # against itself is zero by construction).
    blocks = ([] if paired else [("dense", "Dense", "no acceleration")]) + METHODS

    L = []
    A = L.append
    A("% Generated by assets/scripts/make_accel_config_table.py -- do not edit by hand.")
    if data is None:
        A("% STRUCTURE ONLY: every number is a placeholder (--structure-only).")
    else:
        A(f"% Numbers: medians over the scored clips of {args.run.name}.")
        A(r"% Speedup is a wall-clock measurement, not in these records -- still \ph.")
    A(r"% needs: longtable, booktabs, makecell, xcolor, cleveref, xspace")
    A(r"\providecommand{\ph}{\textcolor{black!30}{$\cdot$}}   % unfilled number")
    A(r"\providecommand{\na}{\texttt{--}}                      % not run / not measured")
    A("")
    A(r"{\scriptsize")
    A(r"\setlength{\tabcolsep}{3.5pt}")
    A(r"\begin{longtable}{@{}" + colspec + r"@{}}")
    A(r"\caption{" + (CAPTION_PAIRED if paired else CAPTION_ABS)
      + (CAPTION_LAYOUTS if lay_col else "") + r"}")
    A(rf"\label{{{label}}} \\")

    labels = ["Model", "Track"] + (["Layout"] if lay_col else []) + ["Embodiment"]
    span = (r"Paired $\Delta$ vs.\ dense" if paired
            else r"Absolute median (reference-free)")
    head = [
        r"\toprule",
        (rf"\multicolumn{{{lead}}}{{@{{}}l}}{{Configuration}} & "
         rf"& \multicolumn{{{nm}}}{{c}}{{{span}}} & \\"),
        rf"\cmidrule(lr){{{m_first}-{m_last}}}",
        " & ".join(labels + ["Speedup"] + [h for h, _ in metrics] + [r"$n$"]) + r" \\",
        r"\midrule",
    ]
    L += head
    A(r"\endfirsthead")
    A(rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\emph{{\cref{{{label}}} "
      r"continued from the previous page.}} \\")
    L += head
    A(r"\endhead")
    A(r"\midrule")
    A(rf"\multicolumn{{{ncols}}}{{r@{{}}}}{{\emph{{continued on the next page}}}} \\")
    A(r"\endfoot")
    A(r"\bottomrule")
    A(r"\endlastfoot")

    for bi, (key, disp, family) in enumerate(blocks):
        if bi:
            A(r"\midrule")
        A(rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\textbf{{{disp}}} "
          rf"\textit{{({family})}}}} \\")
        A(r"\addlinespace[1pt]")
        prev = None
        for model, track, layout, emb in cfgs:
            grp = (model, track, layout)
            new_grp = grp != prev
            if new_grp and prev is not None:
                A(rf"\cmidrule(lr){{1-{lead}}}")   # light rule between track groups
            show = args.repeat_labels or new_grp
            row = [model if show else "", track if show else ""]
            if lay_col:
                # only Ctrl-World has more than one multi-view layout; an empty cell
                # here means "single layout", NOT `\na` ("not run")
                row.append(layout if show else "")
            row.append(emb)
            if (key, model) in NOT_RUN:
                body = [r"\na"] * (nm + 2)
            elif data is None:
                # --structure-only: layout review before a wave is joined in
                body = ([r"1.00$\times$"] if key == "dense" else [r"\ph"]) \
                    + [r"\ph"] * (nm + 1)
            else:
                cell = data.get((model, track, EMB_KEY[emb], key), {})
                # Speedup is a wall-clock measurement, not an NR metric, so it is not in
                # these records at all -- left as \ph rather than invented.
                body = [r"1.00$\times$"] if key == "dense" else [r"\ph"]
                for _, mk in metrics:
                    body.append(fmt(cell.get(mk, []), 3 if mk ==
                                    'text_condition_alignment' else 2, paired))
                # n = the largest per-metric count, matching what the caption promises.
                nmax = max((len(cell.get(mk, [])) for _, mk in metrics), default=0)
                body.append(str(nmax) if nmax else r"\ph")
            A(" & ".join(row + body) + r" \\")
            prev = grp

    A(r"\end{longtable}")
    A(r"}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    try:
        shown = out.relative_to(REPO)
    except ValueError:
        shown = out
    print(f"[ok] {shown}   view={'paired' if paired else 'absolute'}   "
          f"{len(blocks)} blocks x {len(cfgs)} configs = "
          f"{len(blocks) * len(cfgs)} data rows, {ncols} columns")
    if not paired and not args.with_text_align:
        print("     Text Alignment column withheld (absolute CLIPScore is not "
              "comparable across captions); --with-text-align to force it in.")


if __name__ == "__main__":
    main()
