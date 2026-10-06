#!/usr/bin/env python3
"""RQ1 table: dense (un-accelerated) baselines, one row per configuration.

This is the RQ1 surface: one row per (embodiment, model, camera track) dense
configuration under both task-dynamics regimes (Markovian / non-Markovian). Real
video is NOT a row here -- see "WHERE THE REAL-VIDEO REFERENCE LIVES" below.

WHY THIS IS NOT THE ACCELERATION TABLE
--------------------------------------
`make_accel_config_table.py` answers "what did method X cost", so it needs a
Speedup column and a delta-vs-dense view. This table answers "how do the audited
world models behave at all", so:

  * there is NO Speedup column -- every row is dense, which is 1.00x by definition;
  * there is no "Absolute median (reference-free)" spanner, because half the
    columns here are the physical audits, which are not reference-free CLIP-style
    metrics at all;
  * Active Rate is absent (near-redundant with Actor Activity, see T.NOT_REPORTED
    in assets/scripts/table_dense_nr.py).

All four reference-free metrics of the benchmark are shown. The header separates TWO
things that must not be conflated, mirroring assets/scripts/table_dense_nr.py (the
source of truth), where COLUMNS declares a polarity and CEILING_FAILED / PAIRED_ONLY are
separate gates -- "a column can have a well-defined direction and still be barred from a
given comparison":

  DIRECTION (the arrow) comes from the DEFINITION, never from the measurements.
      Visual Integrity = P(CLIP prefers "Good photo.") -> more is better -> up.
      Text Alignment = CLIPScore vs the task clause -> more is better -> up.
      Actor Activity = subject flow in px/frame, a magnitude to be MATCHED -> no
      direction, by construction. Motion Continuity (SSIM-based) -> up.
  GATE (the superscript) says which comparisons that direction supports.
      $\S$ Visual Integrity fails the ground-truth-is-ceiling audit, so it is barred
      from model ranking but valid against a method's own dense clip, where the bias
      cancels. $\dagger$ Text Alignment's absolute levels are not comparable between
      cells, so it is compared paired.

Two earlier versions of this table got this wrong in opposite directions: first marking
Text Alignment $\leftrightarrow$ (which asserts higher is not better -- false), then
marking Visual Integrity $\leftrightarrow$ because predictions beat real video. The
second is the subtler error: pred > real video means the METRIC IS BIASED, not that its
direction is undefined. Inferring polarity from measurements cannot distinguish "no
direction" from "weak metric", so polarity is read off the formula and the empirical
finding is recorded as a gate.

The comparability limit is specific about WHERE it bites, and the two tables differ.
CLIPScore's per-caption bias (~7 pts, vs ~2 pts of image-text agreement) is a level
offset attached to the CAPTION SET, not to the model. In the ACCELERATION table a block
is one (embodiment, model, track, split) and every method in it is scored on the same
captions, so the offset cancels and the column can be read down the block; measured on
this run, caption sets are identical across methods in every configuration except
single_arm/ctrlworld/MV (a re-render added 3, giving 128 vs 125, but all 125 dense
captions are still there), and even there the unpaired column difference tracks the
paired delta closely (itm -3.58 vs -3.54, svg2 -0.93 vs -0.77). In THIS table, by
contrast, every row is a DIFFERENT configuration with its own clip pool and therefore
its own caption set, so no two cells share the offset -- hence the dagger, and hence the
ranking claims on this metric being made as paired deltas elsewhere. The size of the
effect is calibrated by real video against itself: 23.28 on humanoid Markovian vs 21.75
on non-Markovian, a 1.5-point spread on clips that are by construction 100% on-task.

WHY ABSOLUTE
------------
Every cell is the measured median, not a delta against real video. Deltas would
make the metric scale invisible -- KIVA is calibrated in absolute units and Actor
Activity is px/frame -- and leave the reader unable to see whether a small
deviation sits on a large or a tiny base.

WHERE THE REAL-VIDEO REFERENCE LIVES (NOT IN THE TABLE)
------------------------------------------------------
The table prints ONLY the twelve generated configurations. Real video is not a row
here: interleaving one reference row per configuration doubled the height and,
because six geometries have no matched reference scored yet, filled half of those
rows with placeholders -- so the grid no longer read as a grid. The references are
still computed and still checked; they live in the JSON sidecar
(`blocks.*.reference`, plus `blocks.*.{rigis,kiva}.reference`), and the
configuration-vs-real-video pairs that carry an argument are quoted in the prose,
where each comparison can be named instead of left for the reader to assemble.

Each reference is real video measured on that configuration's OWN geometry, never a
pooled figure. This matters: Actor Activity is optical flow in px/frame, and on
bimanual Markovian clips the GT of DreamGen's 2x2 grid (0.911 px/frame) and of
Ctrl-World's 4-view grid (2.986) differ by 3x, while the pooled figure the workbook
carries (1.775) is the reference of neither. Read against the pooled number
DreamGen MV's dense 0.961 looks like a 46% loss of motion; read against its own
grid it is level with real video. References come from
`make_rq1_matched_anchors.py`, which also documents why per-view GT pools are
excluded; the RIGIS/KIVA references are per embodiment rather than per geometry.

WHICH AUDIT APPLIES TO WHICH ROW
--------------------------------
RIGIS and KIVA are not two interchangeable columns:

  RIGIS  cross-view coherence -> only the MULTI-VIEW tracks (DreamGen MV, Ctrl-World MV)
  KIVA   embodiment integrity from 3D-lifted body keypoints -> only the SINGLE-VIEW
         body tracks measured so far (DreamGen SV)

so `\rqna` in those columns means "this audit does not apply to this track", while
`\rqns` means "applies, but not scored in this run" (DreamDojo SV under KIVA).
Collapsing the
two would make an un-run cell look like a structural property of the benchmark.

SOURCES
-------
    sampling_dataset/data_for_web/BENCHMARK_NR_main.xlsx
        tabs VisualIntegrity/ActorActivity/AMC-main (A:C Markovian, E:G
        non-Markovian) and RIGIS-main / KIVA-main. This workbook is the curated
        source of record and the ONLY place RIGIS/KIVA live.
    metrics/WorldScore/results/dense_nr_full/tables/table_paper_grid_absolute_by_track.csv
        the workbook's own upstream, used here for the clip counts the workbook
        does not carry -- and to CROSS-VERIFY every reference-free number, so the
        two artifacts cannot silently drift.
    results/reports/rq1_matched_anchors.csv
        the matched real-video anchor per (embodiment, model, track, split), from
        assets/scripts/make_rq1_matched_anchors.py. Run that first; it is a
        separate script because it reads all ~82k clip records (~5 min).

The Ctrl-World row follows the benchmark's reported-layout rule, which the workbook
already applies: the 4-view grid for bimanual and humanoid, the 3-view strip for
single-arm (which has no grid variant). Verified numerically at load time, not
assumed.

Writes:
    assets/paper/tables/dense_baselines.tex
    results/reports/rq1_dense_baselines.json

Usage:
    python3 assets/scripts/make_rq1_dense_table.py
    python3 assets/scripts/make_rq1_dense_table.py --check   # verify, write nothing
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
XLSX = REPO / "sampling_dataset/data_for_web/BENCHMARK_NR_main.xlsx"
GRID_CSV = (REPO / "metrics/WorldScore/results/dense_nr_full/tables"
            / "table_paper_grid_absolute_by_track.csv")
ANCHOR_CSV = REPO / "results/reports/rq1_matched_anchors.csv"
KIVA_CSV = REPO / "results/reports/kiva_audit.csv"
OUT_TEX = REPO / "assets/paper/tables/dense_baselines.tex"
OUT_JSON = REPO / "results/reports/rq1_dense_baselines.json"

EMB_XL = ["BIMANUAL", "HUMANOID", "SINGLE-ARM"]
EMB_CSV = {"BIMANUAL": "bimanual", "HUMANOID": "humanoid",
           "SINGLE-ARM": "single_arm"}
EMB_LABEL = {"BIMANUAL": "Bimanual", "HUMANOID": "Humanoid",
             "SINGLE-ARM": "Single-arm"}

# Row order. Each entry is (workbook row label, display label, anchor key = (model
# as it appears in the run, camera track)). The track is spelled out only for
# DreamGen, the one family rendered on both tracks; Ctrl-World is multi-view only
# and DreamDojo single-view only, so repeating it on their rows says nothing. The
# anchor key ties a row to real video measured on the SAME geometry; for Ctrl-World
# it is resolved per embodiment from CW_ANCHOR below.
TRACKS = [
    ("Dreamgen MV", r"DreamGen \textit{(multi-view)}", ("dreamgen", "multiview")),
    ("Dreamgen SV", r"DreamGen \textit{(single-view)}", ("dreamgen", "singleview")),
    ("DreamDojo SV", "DreamDojo", ("dreamdojo", "singleview")),
    ("Ctrl-World MV", "Ctrl-World", None),
]

# Which Ctrl-World multi-view layout each embodiment reports (assets/scripts/
# benchmark_cells.py). Checked against the workbook at load time. This is a scoping
# fact, not a result: it is recorded in the JSON sidecar and in this docstring, and
# deliberately NOT printed in the table, where it would add a column of provenance
# to every Ctrl-World row without changing how any number is read.
CW_LAYOUT = {"BIMANUAL": "CW-4view MV", "HUMANOID": "CW-4view MV",
             "SINGLE-ARM": "Ctrl-World MV"}
CW_LAYOUT_LABEL = {"BIMANUAL": "4-view grid", "HUMANOID": "4-view grid",
                   "SINGLE-ARM": "3-view strip"}
# The same choice expressed as the run's model name, for the anchor lookup.
CW_ANCHOR = {"BIMANUAL": ("ctrlworld_4view_grid", "multiview"),
             "HUMANOID": ("ctrlworld_4view_grid", "multiview"),
             "SINGLE-ARM": ("ctrlworld", "multiview")}

# tab -> (csv column stem, decimals). RIGIS/KIVA have no csv counterpart.
# Text Alignment is a CLIPScore on a 0-100 scale, so it gets 1 decimal where the
# others get 2: printing 24.92 alongside 0.96 suggests four significant digits of
# agreement that a ~7-point per-caption bias does not support.
NR_TABS = [
    ("VisualIntegrity-main", "visual_integrity", 2),
    ("TextAlign-main", "text_condition_alignment", 1),
    ("ActorActivity-main", "actor_motion_raw", 2),
    ("AMC-main", "active_motion_continuity", 2),
]
AUDIT_TABS = [("RIGIS-main", 3)]

# Which tracks each physical audit can apply to at all.
RIGIS_TRACKS = {"Dreamgen MV", "Ctrl-World MV"}
KIVA_TRACKS = {"Dreamgen SV", "DreamDojo SV"}
# The audit tabs name their rows differently from the reference-free tabs.
AUDIT_ROW = {
    "RIGIS-main": {"Dreamgen": "Dreamgen MV", "Ctrlworld": "Ctrl-World MV"},
}

MISSING = {"---", "--", "", None}


def _num(v):
    if v in MISSING:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_tab(ws, row_map=None):
    """{(embodiment, row_label): {'mak': v, 'non_mak': v}} from one *-main tab.

    Layout is fixed by export_paper_grid_xlsx.py: column A carries the embodiment
    banner, then `Real Video`, then a model label followed by its method rows; C is
    the Markovian median and G the non-Markovian one. Columns J:P are a
    divided-by-Real-Video mirror and are deliberately ignored -- the ratio is
    recomputed here so the table cannot inherit a stale mirror.
    """
    out, emb, model = {}, None, None
    for r in ws.iter_rows(values_only=True):
        a, b, c, g = r[0], r[1], r[2], (r[6] if len(r) > 6 else None)
        if a in EMB_XL:
            emb, model = a, None
            continue
        if a == "Real Video":
            out[(emb, "Real Video")] = {"mak": _num(c), "non_mak": _num(g)}
            continue
        if a and a != "Markovian":
            model = a
        if b == "Dense" and model is not None:
            label = (row_map or {}).get(model, model)
            out[(emb, label)] = {"mak": _num(c), "non_mak": _num(g)}
    return out


def read_counts():
    """{(embodiment_csv, model): {stem: {'mak': n, 'non_mak': n}}} + the values."""
    rows = list(csv.DictReader(GRID_CSV.open()))
    cur, out = "", {}
    for r in rows:
        if r["model"]:
            cur = r["model"]
        if r["method"].strip().lower() not in ("dense", ""):
            continue
        key = (r["embodiment"], "Real Video" if cur == "Real Video" else cur)
        rec = out.setdefault(key, {})
        for _, stem, _ in NR_TABS:
            rec[stem] = {
                "mak": _num(r.get(f"{stem}_makovian")),
                "non_mak": _num(r.get(f"{stem}_non_makovian")),
                "n_mak": _num(r.get(f"{stem}_makovian_n")),
                "n_non_mak": _num(r.get(f"{stem}_non_makovian_n")),
            }
    return out


def read_kiva():
    """{(embodiment, track): {'mak','non_mak'}} + real-video refs, DENSE rows only.

    KIVA does NOT come from the workbook: its `KIVA-main` tab holds an earlier
    detector on a different scale (old real video 0.543 vs new 5.05 on bimanual,
    and the old/new ratio is not even constant across embodiments). The current
    pass lives in results/reports/kiva_audit.csv via make_kiva_audit_data.py.
    Reading both would put two different measurements in one column.
    """
    if not KIVA_CSV.exists():
        raise SystemExit(
            f"missing {KIVA_CSV.relative_to(REPO)} -- run first:\n"
            "  python3 assets/scripts/make_kiva_audit_data.py")
    out, ref = {}, {}
    for r in csv.DictReader(KIVA_CSV.open()):
        vals = {"mak": _num(r["makovian"]), "non_mak": _num(r["non_makovian"])}
        if r["method"] == "real_video":
            ref[r["embodiment"]] = vals
        elif r["method"] == "Dense":
            out[(r["embodiment"], r["track"])] = vals
    return out, ref


def read_anchors():
    """{(embodiment_csv, model, track): {metric: {'mak','non_mak','n_mak',...}}}.

    Matched real-video medians produced by make_rq1_matched_anchors.py. Only the
    `matched_anchor` rows are used; the file also carries the dense rows, which are
    used to CHECK the workbook rather than to print.
    """
    if not ANCHOR_CSV.exists():
        raise SystemExit(
            f"missing {ANCHOR_CSV.relative_to(REPO)} -- run first:\n"
            "  python3 assets/scripts/make_rq1_matched_anchors.py  (~5 min)")
    out = {}
    for r in csv.DictReader(ANCHOR_CSV.open()):
        key = (r["embodiment"], r["model"], r["track"], r["role"])
        rec = out.setdefault(key, {})
        reg = "mak" if r["split"] == "makovian" else "non_mak"
        for _, stem, _ in NR_TABS:
            rec.setdefault(stem, {})[reg] = _num(r.get(stem))
            rec[stem][f"n_{reg}"] = int(r.get(f"{stem}_n") or 0)
    return out


def load():
    import openpyxl
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    nr = {stem: read_tab(wb[tab]) for tab, stem, _ in NR_TABS}
    audits = {tab.split("-")[0]: read_tab(wb[tab], AUDIT_ROW[tab])
              for tab, _ in AUDIT_TABS}
    counts = read_counts()
    anchors = read_anchors()
    kiva, kiva_ref = read_kiva()

    # --- cross-verify every reference-free number against the workbook's upstream,
    # and in doing so RESOLVE which Ctrl-World layout the workbook reported. A
    # mismatch here means the workbook and the run have drifted apart, which would
    # silently put two different measurements in one paper.
    problems, checked = [], 0
    for stem, tab in nr.items():
        for (emb, label), vals in tab.items():
            csv_label = CW_LAYOUT[emb] if label == "Ctrl-World MV" else label
            ref = counts.get((EMB_CSV[emb], csv_label), {}).get(stem)
            if ref is None:
                problems.append(f"{stem} {emb}/{label}: no upstream row")
                continue
            for reg in ("mak", "non_mak"):
                a, b = vals[reg], ref[reg]
                if a is None and b is None:
                    continue
                if a is None or b is None or abs(a - b) > 0.01:
                    problems.append(
                        f"{stem} {emb}/{label}/{reg}: workbook {a} vs run {b}")
                else:
                    checked += 1
    if problems:
        raise SystemExit("cross-verification FAILED:\n  " + "\n  ".join(problems))
    print(f"[ok] {checked} reference-free values match "
          f"{GRID_CSV.relative_to(REPO)}")
    print(f"[ok] Ctrl-World layout resolved per embodiment: "
          + ", ".join(f"{e}={CW_LAYOUT_LABEL[e]}" for e in EMB_XL))

    # --- second cross-check: the dense rows recomputed in the anchor pass must
    # reproduce the workbook too. This is what proves the anchor pass and the
    # workbook were built from the same clip pool, so an anchor cannot be paired
    # with a dense number measured on a different set of clips.
    dense_checked, dense_bad = 0, []
    for emb in EMB_XL:
        for label, _, akey in TRACKS:
            key = CW_ANCHOR[emb] if label == "Ctrl-World MV" else akey
            rec = anchors.get((EMB_CSV[emb], key[0], key[1], "dense"))
            if rec is None:
                continue
            for _, stem, _ in NR_TABS:
                for reg in ("mak", "non_mak"):
                    a = nr[stem].get((emb, label), {}).get(reg)
                    b = rec.get(stem, {}).get(reg)
                    if a is None or b is None:
                        continue
                    if abs(a - b) > 0.01:
                        dense_bad.append(f"{stem} {emb}/{label}/{reg}: "
                                         f"workbook {a} vs anchor-pass {b}")
                    else:
                        dense_checked += 1
    if dense_bad:
        raise SystemExit("dense re-derivation FAILED:\n  " + "\n  ".join(dense_bad))
    print(f"[ok] {dense_checked} dense values reproduced by the anchor pass")

    n_anchor = sum(1 for k in anchors if k[3] == "matched_anchor")
    print(f"[ok] {n_anchor} matched anchors available "
          f"({ANCHOR_CSV.relative_to(REPO)})")
    print(f"[ok] KIVA read from {KIVA_CSV.relative_to(REPO)} (new detector; the "
          f"workbook's KIVA-main tab is a superseded scale and is NOT read)")
    return nr, audits, counts, anchors, kiva, kiva_ref


def caption():
    """Build the caption.

    SCOPE: what a reader needs in order to parse the grid -- what a row is, what M/N
    are, the polarity markers, and the two placeholder symbols. Nothing else.

    An earlier version also carried the CLIPScore bias figures, the ground-truth-is-
    ceiling audit result, why the two audits are disjoint, and the real-video RIGIS /
    KIVA levels, formatted live from the data. Every one of those is a METHOD fact and
    is already stated where it belongs: the polarity restrictions and the audits are
    defined in appendix \ref{supp:metrics_other}, and the real-video levels are quoted
    per configuration in the RQ1 prose, which is where they are actually argued
    against. Repeating them here made a 15-line caption that restated the section.

    Hence no live data in the caption any more: it no longer prints any measured
    number, so it cannot drift from the table and needs no arguments.
    """
    return (
        r"\textbf{Dense world-model baselines.} Absolute medians per dense "
        r"configuration under both task-dynamics regimes (\textbf{M}~=~Markovian, "
        r"\textbf{N}~=~non-Markovian). Rows are read against real video on the same "
        r"clips and camera geometry, quoted per configuration in the text, and not "
        r"across embodiment blocks. Arrows state each metric's direction as its "
        r"definition fixes it; the superscripts are separate gates restricting which "
        r"comparisons that direction supports, defined in "
        r"\cref{supp:metrics_other} together with the audits behind them."
    )


def fmt(v, dec):
    """One numeric cell, in a FIXED-WIDTH right-aligned box.

    Without the box each of the twelve numeric columns is sized to its own widest
    entry, so `24.9` gets a narrower column than `71.12` and the gaps between
    columns come out at 18.8pt, 21.6pt, 22.5pt, 25.3pt... (measured from the PDF).
    The eye reads that as a crooked grid. A common box width makes every column
    identical, so the only horizontal variation left is the deliberate extra space
    between metrics.
    """
    return (r"\rqnc{\rqns}" if v is None
            else r"\rqn{" + f"{v:.{dec}f}" + "}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="cross-verify against the upstream run and exit")
    args = ap.parse_args()

    nr, audits, counts, anchors, kiva, kiva_ref = load()
    if args.check:
        return

    L = []
    A = L.append
    A("% Generated by assets/scripts/make_rq1_dense_table.py -- do not edit by hand.")
    A(f"% Sources: {XLSX.relative_to(REPO)} (tabs *-main, RIGIS-main, KIVA-main)")
    A(f"%          {GRID_CSV.relative_to(REPO)} (clip counts + cross-verification)")
    A(f"%          {ANCHOR_CSV.relative_to(REPO)} (matched real-video anchors)")
    A(f"%          {KIVA_CSV.relative_to(REPO)} (KIVA, new detector -- the")
    A(r"%          workbook's KIVA-main tab is a superseded scale and is NOT read)")
    A(r"% needs: booktabs, makecell, multirow, graphicx, xcolor, adjustbox, xspace")
    A(r"% Table-local macros, NOT \providecommand: \ph/\na are also defined by")
    A(r"% appendix/tables/accel_by_config.tex, and \providecommand silently keeps")
    A(r"% whichever file LaTeX reads first -- which would make this table's legend")
    A(r"% describe symbols it is not actually printing.")
    A(r"\newcommand{\rqna}{\textcolor{black!55}{--}}  % not applicable")
    A(r"\newcommand{\rqns}{\textcolor{black!55}{n/s}}  % not scored in this run")
    # Fixed-width numeric cell: 2.45em holds the widest entry (`71.12`, `1.205`) at
    # \scriptsize, so all twelve numeric columns come out the same width and the
    # column grid is regular. Right-aligned inside the box, which lines the decimal
    # points up as well since the decimal count is constant within each column.
    A(r"\newcommand{\rqn}[1]{\makebox[2.45em][r]{#1}}")
    # Placeholders are CENTRED in the same box: right-aligning a dash or `n/s`
    # against a column of digits makes it look like a value with a missing
    # fractional part.
    A(r"\newcommand{\rqnc}[1]{\makebox[2.45em][c]{#1}}")
    A("")
    A(r"\begin{table}[t]")
    A(r"\centering")
    A(r"\caption{" + caption() + r"}")
    A(r"\label{tab:dense_baselines}")
    A(r"\vspace{2pt}")
    # ONE size for the whole table. Earlier versions mixed \scriptsize cells with
    # \footnotesize header lines and a \scriptsize placeholder; that reads as a
    # typographic accident, not a hierarchy -- and there is no hierarchy here, all
    # six metrics are peers.
    A(r"{\scriptsize")
    A(r"\setlength{\tabcolsep}{3pt}")
    A(r"\adjustbox{max width=\linewidth}{%")
    # Every numeric cell is a fixed-width \rqn box, so the column type only has to
    # centre that box -- `c`, not `r`. Extra space BETWEEN metrics, none inside an
    # M/N pair: with six adjacent pairs under identical sub-headers, uniform
    # \tabcolsep makes the twelve numeric columns read as one block. The wider gaps
    # let a pair be scanned as a unit.
    # Column 1 is a narrow spine carrying the embodiment name ROTATED 90 degrees, and
    # it replaces the old full-width banner row. The banner spent one line per block
    # on a single word and, because it sat between \midrule and the first data row,
    # left three visually empty lines in a twelve-row table. A rotated \multirow spine
    # says the same thing in zero vertical space and additionally BRACKETS each block,
    # so the four configurations of one embodiment read as a group without needing the
    # extra rule.
    A(r"\begin{tabular}{@{}cl" + r" cc@{\hspace{9pt}}" * 5 + r" cc@{}}")
    A(r"\toprule")
    # FLAT header: no "Appearance and motion" / "Physical audits" spanner. The
    # grouping was decorative -- the caption already says which two columns are ours
    # -- and a spanner wider than the columns it spans forces TeX to dump the excess
    # into the span's last column, which is what pushed KIVA's N column ~50pt from
    # its M in two earlier attempts (measured from the PDF, not eyeballed).
    # Polarity is shown as an ARROW in the header, not as a footnote symbol a reader has
    # to decode: `$\uparrow$`/`$\downarrow$` where the direction is usable, and
    # `$\leftrightarrow$` (a horizontal arrow, i.e. "no direction") on the three columns
    # that must not be ranked. Only the horizontal ones carry a marker, and it says what
    # it is rather than pointing at prose.
    #
    # POLARITY COMES FROM THE DEFINITION, NEVER FROM THE MEASUREMENTS. This is the same
    # architecture as assets/scripts/table_dense_nr.py, which is the source of truth:
    # COLUMNS declares `higher` (True/False/None) from what the formula computes, and
    # CEILING_FAILED / PAIRED_ONLY are SEPARATE gates -- "a column can have a well-defined
    # direction and still be barred from a given comparison". The arrow row states the
    # definition; the superscript states the gate.
    #
    #   Visual Integrity  `higher=True`. CLIP-IQA+ is the softmax probability that CLIP
    #     prefers "Good photo." over "Bad photo.", so by construction more is better --
    #     an EARLIER VERSION OF THIS TABLE PRINTED $\leftrightarrow$ HERE, inferring the
    #     direction from the ceiling-audit numbers. That is backwards: pred > real video
    #     means the metric is BIASED, not that its direction is undefined. `$\uparrow$`
    #     with gate $\S$ (barred from model ranking; valid against its own dense clip,
    #     where the bias cancels).
    #   Text Alignment    `higher=True`, CLIPScore against the task clause, and it passes
    #     the ceiling test. Gate $\dagger$: absolute LEVELS are not comparable between
    #     cells (per-caption bias ~7 pts vs ~2 pts of signal; real video itself moves
    #     23.28 -> 21.75 between regimes on humanoid), so it is compared paired.
    #   Actor Activity    `higher=None` -- the ONLY column with no direction, and it is
    #     none BY DEFINITION: the cell is subject flow in px/frame, a magnitude to be
    #     matched to real video. `$\leftrightarrow$`, no gate marker needed.
    #   Motion Contin.    `higher=True` (SSIM-based); comparable within an embodiment.
    #   RIGIS / KIVA      higher / lower is better, both against a real-video level.
    A(r"& Configuration "
      r"& \multicolumn{2}{c}{\makecell{Visual\\Integrity$^{\S}$"
      r"\\[-1pt]{\tiny$\uparrow$}}} "
      r"& \multicolumn{2}{c}{\makecell{Text\\Align.$^{\dagger}$"
      r"\\[-1pt]{\tiny$\uparrow$}}} "
      r"& \multicolumn{2}{c}{\makecell{Actor\\Activity"
      r"\\[-1pt]{\tiny$\leftrightarrow$}}} "
      r"& \multicolumn{2}{c}{\makecell{Motion\\Contin.\\[-1pt]{\tiny$\uparrow$}}} "
      r"& \multicolumn{2}{c}{\makecell{RIGIS\\[-1pt]{\tiny$\uparrow$}}} "
      r"& \multicolumn{2}{c}{\makecell{KIVA \%\\[-1pt]{\tiny$\downarrow$}}} \\")
    A(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} "
      r"\cmidrule(lr){9-10} \cmidrule(lr){11-12} \cmidrule(lr){13-14}")
    A(r"& & " + " & ".join([r"\rqnc{M}", r"\rqnc{N}"] * 6) + r" \\")
    A(r"\midrule")

    bundle = {
        "generated_by": "assets/scripts/make_rq1_dense_table.py",
        "purpose": "RQ1 evidence: dense baselines and their MATCHED real-video "
                   "reference (same clips, same camera geometry), per embodiment / "
                   "model / track / regime. All values absolute.",
        "sources": [str(XLSX.relative_to(REPO)), str(GRID_CSV.relative_to(REPO)),
                    str(ANCHOR_CSV.relative_to(REPO)), str(KIVA_CSV.relative_to(REPO))],
        "kiva_note": (
            "KIVA comes from results/reports/kiva_audit.csv (new detector, % of "
            "invalid 16-frame segments), NOT from the workbook's KIVA-main tab, "
            "which holds an earlier detector on a different scale."),
        "ctrlworld_layout_reported": {e: CW_LAYOUT_LABEL[e] for e in EMB_XL},
        "anchor_policy": (
            "Each configuration is paired with real video scored on its own camera "
            "geometry. A single pooled real-video row is NOT used: Actor Activity is "
            "px/frame optical flow, and on bimanual Markovian clips the GT of "
            "DreamGen's grid (0.911) and of Ctrl-World's 4-view grid (2.986) differ "
            "by 3x, while the pooled figure (1.775) references neither. Per-view GT "
            "pools (input-tree single views) are excluded as unmatched to a grid "
            "prediction. The RIGIS/KIVA references are per embodiment, not per "
            "geometry, so the same figure recurs across a block's reference rows."),
        "restrictions": {
            "visual_integrity": "polarity higher-is-better BY DEFINITION (CLIP-IQA+ = "
                                "P(CLIP prefers 'Good photo.')); gated out of model "
                                "ranking by the GT-is-ceiling audit failure, which "
                                "indicates a domain bias of the metric, NOT an undefined "
                                "direction. Claim-eligible vs a method's own dense clip, "
                                "where the bias cancels.",
            "text_alignment": (
                "higher-is-better (it PASSES the GT-is-ceiling test, unlike Visual "
                "Integrity), but absolute LEVELS are not comparable between the cells of "
                "THIS table. CLIPScore's per-caption style/length bias (~7 pts, vs ~2 "
                "pts of image-text agreement) is a level offset attached to the CAPTION "
                "SET, and every row here is a different configuration with its own clip "
                "pool and captions. In the ACCELERATION table the offset does cancel "
                "along a block, since one block is a single (embodiment, model, track, "
                "split) and every method in it is scored on the same captions -- "
                "verified on this run: caption sets are byte-identical across methods in "
                "every configuration except single_arm/ctrlworld/MV, where a re-render "
                "added 3 captions (128 vs 125) while still containing all 125 dense "
                "ones, and even there the unpaired column difference tracks the paired "
                "delta closely (itm -3.58 vs -3.54, svg2 -0.93 vs -0.77, worldcache "
                "-0.72 vs -0.47). Ranking claims use those paired deltas."),
            "actor_activity": "no polarity BY DEFINITION (subject flow in px/frame is a "
                              "magnitude to be matched to real video, not maximised); "
                              "matched geometry only",
            "motion_continuity": "higher is better, but only WITHIN an embodiment "
                                 "(real video sits near 48 on humanoid vs 65 on the "
                                 "manipulators)",
            "active_rate": "withheld (near-redundant with Actor Activity)",
        },
        "audit_applicability": {
            "rigis": sorted(RIGIS_TRACKS), "kiva": sorted(KIVA_TRACKS),
        },
        "blocks": {},
    }

    for emb in EMB_XL:
        # No banner row. The embodiment is a rotated \multirow in column 1, emitted on
        # the FIRST configuration of the block and spanning all four of them. \rotatebox
        # is wrapped in \multirow[c] so the label is vertically centred on the block
        # rather than sitting on its first line.
        #
        # \smash: rotated, the label's WIDTH becomes its height, and a block is only
        # four \scriptsize rows (~33pt) tall -- so "Humanoid" and "Single-arm" still
        # reported Overfull \vbox by 1.0pt and 2.3pt even at \tiny. \smash zeroes the
        # box's height and depth so it stops inflating the row, while \rotatebox's
        # origin=c keeps it drawn centred on the block. Cosmetic overshoot of ~1pt into
        # the block's rules is invisible; the warning was not.
        block = {}
        spine = (rf"\multirow{{{len(TRACKS)}}}{{*}}"
                 rf"{{\smash{{\rotatebox[origin=c]{{90}}"
                 rf"{{\tiny\textbf{{{EMB_LABEL[emb]}}}}}}}}}")

        for i, (label, disp, akey) in enumerate(TRACKS):
            key = CW_ANCHOR[emb] if label == "Ctrl-World MV" else akey
            anchor = anchors.get((EMB_CSV[emb], key[0], key[1], "matched_anchor"))
            rv_audit = {"RIGIS": audits["RIGIS"].get((emb, "Real Video"), {}),
                        "KIVA": kiva_ref.get(EMB_CSV[emb], {})}

            # ONE row per configuration. The matched real-video reference is carried
            # in `rec` (and so into the JSON sidecar) but not printed: a reference
            # row per configuration doubled the table's height, and with six
            # geometries still unscored half of those rows were placeholders, which
            # is what broke the column grid.
            cells, rec = [], {"reference": {}, "dense": {}}
            ns = []
            for _, stem, dec in NR_TABS:
                v = nr[stem].get((emb, label), {})
                a = (anchor or {}).get(stem, {})
                for reg in ("mak", "non_mak"):
                    cells.append(fmt(v.get(reg), dec))
                rec["dense"][stem] = {"mak": v.get("mak"), "non_mak": v.get("non_mak")}
                rec["reference"][stem] = (
                    {"mak": a.get("mak"), "non_mak": a.get("non_mak")}
                    if anchor else None)
                cnt = counts.get(
                    (EMB_CSV[emb],
                     CW_LAYOUT[emb] if label == "Ctrl-World MV" else label),
                    {}).get(stem, {})
                for k in ("n_mak", "n_non_mak"):
                    if cnt.get(k):
                        ns.append((k, int(cnt[k])))

            # RIGIS is a 0-1 coherence score (3 dp); the new KIVA pass is a
            # PERCENTAGE of invalid segments (1 dp) -- printing it with 3 dp would
            # imply a precision the pass does not have, and would misalign the two
            # audit pairs.
            for audit, dec in (("RIGIS", 3), ("KIVA", 1)):
                ok = (label in RIGIS_TRACKS if audit == "RIGIS"
                      else label in KIVA_TRACKS)
                if audit == "KIVA":
                    v = kiva.get((EMB_CSV[emb], label), {}) if ok else {}
                else:
                    v = audits[audit].get((emb, label), {}) if ok else {}
                for reg in ("mak", "non_mak"):
                    cells.append(fmt(v.get(reg), dec) if ok
                                 else r"\rqnc{\rqna}")
                rec[audit.lower()] = (
                    {"dense": {"mak": v.get("mak"), "non_mak": v.get("non_mak")},
                     "reference": rv_audit[audit]} if ok else "not_applicable")

            # Clip counts are still collected -- they go to the JSON sidecar, where a
            # reader who needs to know that humanoid Markovian has 98 episodes
            # against 150 non-Markovian can find it. They are not a column: the
            # counts are a property of the episode pool (~4 clips per episode in
            # every regime), not a measurement of the model, so in the table they
            # only competed for attention with the six metrics.
            nm = max((n for k, n in ns if k == "n_mak"), default=0)
            nn = max((n for k, n in ns if k == "n_non_mak"), default=0)
            rec["n"] = {"mak": nm or None, "non_mak": nn or None}
            rec["reference_available"] = anchor is not None
            rec["reference_geometry"] = f"{key[0]}/{key[1]}"

            # The spine cell is emitted only on the block's first row; the other three
            # leave column 1 empty so \multirow can span into it. `\quad` before the
            # configuration name is gone: the rotated spine already marks the
            # indentation level, and the extra quad only pushed the names off the
            # header's left edge.
            A(" & ".join([spine if i == 0 else "", disp] + cells) + r" \\")
            block[label] = rec

        bundle["blocks"][EMB_CSV[emb]] = block
        if emb != EMB_XL[-1]:
            # A rule, not \addlinespace: the rotated spine already spans the block, and
            # a full-width rule is what keeps the spine of the next block from reading
            # as a continuation of this one.
            A(r"\midrule")

    A(r"\bottomrule")
    A(r"\end{tabular}}")
    A(r"}")
    # The legend goes in a \parbox, not after a bare \\: inside a centred `table`
    # a trailing \\ line is centred and re-wrapped raggedly, which is what happened
    # to the first version of this note.
    # The legend separates DIRECTION (from the definition) from GATES (what may be
    # compared). Two earlier versions folded a failed audit into the arrow, which says
    # "higher is not better" when the truth is "the metric is biased".
    A(r"\vspace{2pt}")
    A(r"\parbox{\linewidth}{\footnotesize "
      r"Arrows give each metric's direction \emph{as defined}: $\uparrow$/$\downarrow$ "
      r"higher/lower is better, $\leftrightarrow$ no direction (Actor Activity is a "
      r"px/frame magnitude to be matched to real video, not maximised). Superscripts are "
      r"separate gates on what may be compared, not on direction. "
      r"$\S$~not used to rank models: the column fails the ground-truth-is-ceiling "
      r"audit, and is claim-eligible only against a method's own dense clip, where the "
      r"bias cancels. "
      r"$\dagger$~absolute levels are not comparable between cells, so the column is "
      r"compared as a paired delta. "
      r"\rqna{}~audit not applicable to this camera track. "
      r"\rqns{}~applicable but not scored.}")
    A(r"\end{table}")

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(L) + "\n")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(bundle, indent=2) + "\n")
    print(f"[ok] {OUT_TEX.relative_to(REPO)}   "
          f"{len(EMB_XL)} blocks x {len(TRACKS)} configurations, 6 metrics, "
          f"all absolute (real-video references in the JSON sidecar only)")
    print(f"[ok] {OUT_JSON.relative_to(REPO)}")

    missing = [f"{e}/{lab}" for e in EMB_XL for lab, _, ak in TRACKS
               if not bundle["blocks"][EMB_CSV[e]][lab]["reference_available"]]
    if missing:
        print(f"[note] {len(missing)} of {len(EMB_XL) * len(TRACKS)} configurations "
              f"have no matched real-video reference in the sidecar (never "
              f"substituted from another geometry):")
        for m in missing:
            print(f"        {m}")


if __name__ == "__main__":
    main()
