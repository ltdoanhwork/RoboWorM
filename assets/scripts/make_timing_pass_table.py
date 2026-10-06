#!/usr/bin/env python3
"""Turn the uniform timing pass into the appendix speedup table.

Consumes the JSONL emitted by ``scripts/tools/tune/_bench_uniform_timing.sh`` and produces
the matched-protocol speedup numbers for subsection C.2.

What this fixes relative to ``results/reports/speedup_summary_full.csv``: that
file has no timing at all for 69 of its 127 rows, and every row that does have a
speedup inherited it from a per-cell tuning note, i.e. from a sweep with its own
episode, split and scratch layout. Those numbers cannot be compared across
cells. Here every method in a cell is timed on the *same* episode under the same
protocol on an idle GPU, so the ratio is meaningful.

Reported per (cell, method):
  * ``generation_s``   -- the family's authoritative generation time, not wall
                          clock (which includes model load, VAE decode, ffmpeg
                          and metric computation and would dilute every ratio);
  * ``speedup``        -- dense generation time over the method's, same cell;
  * ``steady_state``   -- the same ratio excluding the first segment/chunk,
                          which pays one-off MSE profiling and k-means warm-up
                          for SVG/PISA and otherwise flatters dense.

Cells and methods that never ran are carried through as explicit rows
(``not benchmarked`` / ``no run script``) rather than dropped, so an absent
result is never mistaken for a negative one.

    python3 assets/scripts/make_timing_pass_table.py \
        --results /tmp/uniform_timing/results.jsonl

Writes:
    results/reports/timing_pass.json
    results/reports/timing_pass.csv
    assets/paper/appendix/tables/timing_pass.tex
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_JSON = os.path.join(REPO, "results/reports/timing_pass.json")
OUT_CSV = os.path.join(REPO, "results/reports/timing_pass.csv")
OUT_TEX = os.path.join(REPO, "assets/paper/appendix/tables/timing_pass.tex")

METHOD_DISPLAY = {
    "dense": "dense", "sito": "SiTo", "itm": "ITM",
    "worldcache": "WorldCache", "dicache": "DiCache", "fastercache": "FasterCache",
    "pisa": "PISA", "svg1": "SVG", "svg2": "SVG2", "radial": "Radial",
}
METHOD_ORDER = ["dense", "sito", "itm", "worldcache", "dicache", "fastercache",
                "pisa", "svg1", "svg2", "radial"]
FAMILY = {
    "sito": "token reduction", "itm": "token reduction",
    "worldcache": "caching", "dicache": "caching", "fastercache": "caching",
    "pisa": "sparse attention", "svg1": "sparse attention",
    "svg2": "sparse attention", "radial": "sparse attention",
}
CELL_DISPLAY = {
    # The Ctrl-World 3-view strip is reported only for single-arm; bimanual and
    # humanoid are reported on their 4-view grid, which supersedes the strip
    # (assets/scripts/benchmark_cells.py). The two excluded ids are therefore
    # absent here, so the timing pass never measures a cell the paper drops.
    "ctrlworld_single_arm_mv": ("Ctrl-World", "single-arm MV (3-view)"),
    "ctrlworld4_bimanual_mv": ("Ctrl-World", "bimanual MV (4-view)"),
    "ctrlworld4_humanoid_mv": ("Ctrl-World", "humanoid MV (4-view)"),
    "dreamgen_single_arm_mv": ("DreamGen", "single-arm MV (14B)"),
    "dreamgen_bimanual_mv": ("DreamGen", "bimanual MV"),
    "dreamgen_humanoid_mv": ("DreamGen", "humanoid MV"),
    "dreamgen_single_arm_sv": ("DreamGen", "single-arm SV"),
    "dreamgen_bimanual_sv": ("DreamGen", "bimanual SV"),
    "dreamgen_humanoid_sv": ("DreamGen", "humanoid SV"),
    "dreamdojo_single_arm_sv": ("DreamDojo", "single-arm SV"),
    "dreamdojo_bimanual_sv": ("DreamDojo", "bimanual SV"),
    "dreamdojo_humanoid_sv": ("DreamDojo", "humanoid SV"),
}


def load(path):
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def build(recs):
    """Group by cell, divide by that cell's dense time."""
    by_cell = {}
    for r in recs:
        by_cell.setdefault(r["cell"], {})[r["method"]] = r

    rows = []
    for cell in CELL_DISPLAY:
        got = by_cell.get(cell, {})
        dense = got.get("dense")
        dense_t = dense.get("generation_s") if dense else None
        for method in METHOD_ORDER:
            r = got.get(method)
            if r is None:
                rows.append({"cell": cell, "method": method,
                             "status": "not_benchmarked", "generation_s": None,
                             "speedup": None, "steady_state_speedup": None,
                             "contended": None})
                continue
            t = r.get("generation_s")
            speedup = (dense_t / t) if (dense_t and t) else None
            ss = None
            if dense and dense.get("steady_state_s") and r.get("steady_state_s"):
                ss = dense["steady_state_s"] / r["steady_state_s"]
            rows.append({
                "cell": cell, "method": method, "status": r.get("status"),
                "generation_s": t, "speedup": speedup,
                "steady_state_speedup": ss,
                "first_unit_s": r.get("first_unit_s"),
                "work_unit": r.get("work_unit"),
                "contended": r.get("contended"),
                "script": r.get("script"),
            })
    return rows


def emit_tex(rows, path):
    ran = [r for r in rows if r["speedup"]]
    out = [
        "% Generated by assets/scripts/make_timing_pass_table.py -- do not edit by hand.",
        "% Source: the uniform timing pass (scripts/tools/tune/_bench_uniform_timing.sh).",
        r"\begin{table}[t]\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{\textbf{Matched-protocol speedups.}",
        r"Each method is timed on the \emph{same} episode as its cell's dense"
        r" baseline, sequentially on one idle GPU, and the ratio is taken from the"
        r" family's own generation-time log rather than from wall clock (which"
        r" includes model load, VAE decoding and metric computation and would"
        r" dilute every ratio towards $1$). $S$ is the end-to-end speedup;"
        r" $S_{\text{ss}}$ excludes the first segment or chunk, which pays the"
        r" one-off attention profiling and $k$-means warm-up that several methods"
        r" require, and is therefore the number a long rollout actually sees."
        r" Values below $1$ are genuine results and are kept: on several"
        r" backbones self-attention is too small a share of generation for sparse"
        r" attention to pay for its own overhead. \texttt{--} marks a cell that"
        r" was not benchmarked, which is reported rather than silently omitted.}",
        r"\label{tab:supp_timing_pass}",
        r"\begin{tabular}{@{}llrrr@{}}\toprule",
        r"Cell & Method & Gen.\ (s) & $S$ & $S_{\text{ss}}$ \\",
        r"\midrule",
    ]
    if not ran:
        out.append(r"\multicolumn{5}{@{}l}{\emph{The uniform timing pass has not "
                   r"been executed yet.}} \\")
    else:
        prev = None
        for r in rows:
            if r["speedup"] is None and r["status"] == "not_benchmarked":
                continue
            fam, track = CELL_DISPLAY[r["cell"]]
            label = ("%s, %s" % (fam, track)) if r["cell"] != prev else ""
            prev = r["cell"]
            gen = "$%.1f$" % r["generation_s"] if r["generation_s"] else r"\texttt{--}"
            s = "$%.2f$" % r["speedup"] if r["speedup"] else (
                "$1.00$" if r["method"] == "dense" else r"\texttt{--}")
            ss = "$%.2f$" % r["steady_state_speedup"] if r["steady_state_speedup"] \
                else r"\texttt{--}"
            out.append("%s & %s & %s & %s & %s \\\\" % (
                label, METHOD_DISPLAY[r["method"]], gen, s, ss))
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/tmp/uniform_timing/results.jsonl")
    args = ap.parse_args()

    recs = load(args.results)
    rows = build(recs)
    ran = [r for r in rows if r["speedup"]]
    contended = [r for r in rows if r.get("contended")]

    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    emit_tex(rows, OUT_TEX)

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "cell", "method", "status", "work_unit", "generation_s",
            "first_unit_s", "speedup", "steady_state_speedup", "contended",
            "script"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})

    payload = {
        "generated_by": "assets/scripts/make_timing_pass_table.py",
        "source": args.results,
        "purpose": "Matched-protocol generation times and speedups: one shared "
                   "episode per cell, every method timed sequentially on an idle "
                   "GPU, read from each family's generation-time log.",
        "protocol": {
            "episodes_per_cell": 1,
            "shared_episode": "dense and every method time the same episode",
            "concurrency": "strictly sequential, single GPU, no context parallel",
            "timing_source": {
                "ctrlworld": "per-episode wall_time_s in metrics.json",
                "dreamgen": "Total inference time / LVG Total inference time",
                "dreamdojo": "[Timing] Generated",
            },
            "excluded_from_timing": "model load, VAE decode, ffmpeg, metric "
                                    "computation (wall clock is not used)",
            "first_unit": "reported separately; it pays one-off attention "
                          "profiling and k-means warm-up for SVG and PISA",
            "output": "scratch only; the canonical sampling_dataset tree is never "
                      "written by the timing pass",
        },
        "n_rows": len(rows),
        "n_measured": len(ran),
        "n_contended": len(contended),
        "executed": bool(ran),
        "rows": rows,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2)

    print("read %d records from %s" % (len(recs), args.results))
    print("wrote %s" % os.path.relpath(OUT_TEX, REPO))
    print("wrote %s" % os.path.relpath(OUT_CSV, REPO))
    print("wrote %s" % os.path.relpath(OUT_JSON, REPO))
    print("measured %d of %d (cell, method) pairs" % (len(ran), len(rows)))
    if contended:
        print("WARNING: %d measurements were taken on a contended GPU" % len(contended))
    if not ran:
        print("\nThe pass has not been executed yet; the table renders a "
              "placeholder until it is.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
