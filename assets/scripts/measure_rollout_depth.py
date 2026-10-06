#!/usr/bin/env python3
"""Measure autoregressive rollout depth per (embodiment, model, regime) from the videos.

WHY THIS SCRIPT EXISTS
----------------------
Rollout depth used to be transcribed by hand into make_depth_vs_regime_figure.py as
median chunks/clip. That was wrong for humanoid DreamGen, and the error was invisible
because nothing recomputed it.

THE BUG IT FIXES
----------------
DreamGen humanoid clips take exactly TWO lengths, 185 and 277 frames (= 92*2+1 and
92*3+1), i.e. 2 or 3 chunks, and the split between them is near a coin flip:

    Markovian      60 clips @ 2 chunks, 70 @ 3   -> median 3
    non-Markovian  69 clips @ 2 chunks, 61 @ 3   -> median 2

The median therefore FLIPS 3 -> 2 and reports a 0.67x depth ratio, while the means are
2.54 vs 2.47 (0.97x) and the difference is not significant (Fisher p = 0.32,
Mann-Whitney p = 0.27, common-language effect 0.535). The two regimes carry the SAME
depth here; the median landed on a knife edge of a two-valued distribution.

The same failure mode hides the opposite error on DreamDojo humanoid, whose chunk
distribution has a long tail (1 to 29 chunks): median says 2 vs 2 (1.00x) while the means
are 2.86 vs 3.57 (1.25x). Again not significant (p = 0.71).

CONSEQUENCE FOR REPORTING
-------------------------
We report the MEAN chunks per clip as the depth statistic, because depth is a count of
re-conditioning events and its mean is the quantity the cascade argument is about, and we
gate every "deeper" claim on a two-sided Mann-Whitney test rather than on the ratio alone.
Medians are kept in the JSON for the appendix table, flagged when they disagree in
direction with the mean.

Usage:
    python3 assets/scripts/measure_rollout_depth.py
Writes:
    results/reports/rollout_depth.json
"""

import json
import pathlib
import statistics as st
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from scipy import stats

REPO = pathlib.Path(__file__).resolve().parents[2]
ROOT = REPO / "sampling_dataset" / "dense"
OUT = REPO / "results" / "reports" / "rollout_depth.json"

# Chunk stride in frames, verified by divisibility of the clip-length distribution:
# each clip is stride*C + n_cond frames, so (length - n_cond) % stride == 0 on every clip.
STRIDE = {
    ("humanoid", "dreamgen"): 92,
    ("bimanual", "dreamgen"): 92,
    ("single_arm", "dreamgen"): 92,
    ("humanoid", "dreamdojo"): 49,
    ("bimanual", "dreamdojo"): 12,
    ("single_arm", "dreamdojo"): 12,
}
SPLITS = ("makovian", "non_makovian")


def n_frames(path):
    """Decoded packet count. nb_frames is absent or wrong on some of these mp4s."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return int(r.stdout.strip())
    except ValueError:
        return None


def split_of(path):
    # Exact path-component match: "makovian" in str(path) also matches "non_makovian".
    for part in path.parts:
        if part in SPLITS:
            return part
    return None


def collect(emb, model):
    d = ROOT / emb / "output" / "singleview" / model
    if not d.exists():
        return None
    vids = [p for p in d.rglob("*.mp4")
            if "_gt" not in p.stem and "merged" not in p.stem and split_of(p)]
    with ThreadPoolExecutor(16) as ex:
        counts = list(ex.map(n_frames, vids))
    per_split = {s: [] for s in SPLITS}
    for p, f in zip(vids, counts):
        if f:
            per_split[split_of(p)].append(f)
    return per_split


def summarise(frames, stride):
    chunks = [round(f / stride) for f in frames]
    return dict(
        n_clips=len(frames),
        frames_median=st.median(frames),
        frames_range=[min(frames), max(frames)],
        chunks_mean=round(st.mean(chunks), 3),
        chunks_median=st.median(chunks),
        chunks_range=[min(chunks), max(chunks)],
        chunks_distinct=len(set(chunks)),
        chunks_histogram={str(k): v for k, v in sorted(Counter(chunks).items())},
        # Per-clip lists so downstream figures can plot the DISTRIBUTION, not just the
        # summary: the single-arm regime supports turn out to be disjoint, which no
        # mean/median pair can convey.
        frames_all=sorted(frames),
        chunks_all=sorted(chunks),
    )


TABLE = REPO / "assets" / "paper" / "appendix" / "tables" / "chunk_depth.tex"

ROW_LABEL = {"dreamgen": "DreamGen", "dreamdojo": "DreamDojo"}
EMB_LABEL = {"humanoid": "Humanoid", "bimanual": "Bimanual", "single_arm": "Single-arm"}
# KIVA, percentage of invalid 16-frame segments: results/reports/rq1_dense_baselines.json
# blocks.{emb}.{Dreamgen SV,DreamDojo SV}.kiva.dense.{mak,non_mak}.
KIVA_SRC = REPO / "results" / "reports" / "rq1_dense_baselines.json"
TRACK = {"dreamgen": "Dreamgen SV", "dreamdojo": "DreamDojo SV"}


def write_table(cells):
    kiva_blocks = json.loads(KIVA_SRC.read_text())["blocks"]
    lines = [
        "% GENERATED by assets/scripts/measure_rollout_depth.py -- do not edit by hand.",
        "% Depth measured from every generated clip under",
        "% sampling_dataset/dense/{emb}/output/singleview/{dreamgen,dreamdojo}; KIVA from",
        "% results/reports/rq1_dense_baselines.json (blocks.*.kiva.dense).",
        "\\begin{table}[h]",
        "\\centering\\scriptsize",
        "\\caption{Autoregressive rollout depth against KIVA. $\\Delta$ is the chunk "
        "stride in frames and $\\bar{C}$ the mean number of chunks per clip (range in "
        "brackets); $p$ is a two-sided Mann--Whitney test on the per-clip chunk counts of "
        "the two regimes, and KIVA is the percentage of invalid $16$-frame segments. Rows "
        "are ordered by the non-Markovian/Markovian depth ratio, with the two "
        "configurations whose regimes are statistically indistinguishable in depth listed "
        "first. Among the four configurations with a reliably deeper non-Markovian "
        "rollout, the two deepest are the only two whose KIVA degrades. Depth is reported "
        "as a mean because the median is degenerate on the DreamGen length distribution, "
        "which takes only two values.}",
        "\\label{tab:chunk_depth}",
        "\\begin{tabular}{@{}llccccccc@{}}",
        "\\toprule",
        "& & & \\multicolumn{3}{c}{Rollout depth $\\bar{C}$} & "
        "\\multicolumn{3}{c}{KIVA (\\%)} \\\\",
        "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}",
        "Embodiment & Model & $\\Delta$ & Mak. & Non-mak. & $p$ & Mak. & Non-mak. & "
        "$\\Delta$ \\\\",
        "\\midrule",
    ]
    prev_sig = None
    for c in cells:
        if prev_sig is not None and c["deeper_significant"] != prev_sig:
            lines.append("\\midrule")
        prev_sig = c["deeper_significant"]
        k = kiva_blocks[c["embodiment"]][TRACK[c["model"]]]["kiva"]["dense"]
        rng = lambda s: f"[{s['chunks_range'][0]}--{s['chunks_range'][1]}]"
        p = c["mannwhitney_p"]
        p_txt = "$<\\!10^{-5}$" if p < 1e-5 else f"${p:.2f}$"
        lines.append(
            f"{EMB_LABEL[c['embodiment']]:10s} & {ROW_LABEL[c['model']]:9s} & "
            f"${c['stride']}$ & ${c['makovian']['chunks_mean']:.2f}$ {rng(c['makovian'])} & "
            f"${c['non_makovian']['chunks_mean']:.2f}$ {rng(c['non_makovian'])} & {p_txt} & "
            f"${k['mak']:.2f}$ & ${k['non_mak']:.2f}$ & "
            f"${k['non_mak'] - k['mak']:+.2f}$ \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    TABLE.write_text("\n".join(lines))
    print(f"[ok] {TABLE.relative_to(REPO)}")


def main():
    cells = []
    for (emb, model), stride in STRIDE.items():
        per_split = collect(emb, model)
        if per_split is None:
            continue
        mak = [round(f / stride) for f in per_split["makovian"]]
        non = [round(f / stride) for f in per_split["non_makovian"]]
        u, p = stats.mannwhitneyu(mak, non, alternative="two-sided")
        # Common-language effect size: P(a random Markovian clip is deeper). 0.5 = no
        # difference. Reported because a ratio alone cannot say "distinguishable".
        cl = u / (len(mak) * len(non))
        s_mak, s_non = summarise(per_split["makovian"], stride), summarise(per_split["non_makovian"], stride)
        r_mean = s_non["chunks_mean"] / s_mak["chunks_mean"]
        r_med = s_non["chunks_median"] / s_mak["chunks_median"]
        cells.append(dict(
            embodiment=emb, model=model, stride=stride,
            makovian=s_mak, non_makovian=s_non,
            depth_ratio_mean=round(r_mean, 3),
            depth_ratio_median=round(r_med, 3),
            mannwhitney_p=float(f"{p:.3g}"),
            cl_effect=round(cl, 3),
            deeper_significant=bool(p < 1e-3 and r_mean > 1.0),
            # Flag the cell when the median ratio asserts a substantive depth difference
            # (>15%) that the test does not support, or points the opposite way from the
            # mean. This is what caught humanoid DreamGen's spurious 0.67x.
            median_misleading=bool((abs(r_med - 1.0) > 0.15 and p >= 1e-3)
                                   or (r_med - 1.0) * (r_mean - 1.0) < 0),
        ))

    # Group the indistinguishable cells first, then order by ratio, so the table reads in
    # the same row order as fig_depth_vs_regime.
    cells.sort(key=lambda c: (c["deeper_significant"], c["depth_ratio_mean"]))
    doc = dict(
        what="Autoregressive rollout depth per single-view configuration and regime.",
        depth_statistic="mean chunks per clip",
        why_not_median=(
            "DreamGen humanoid takes only two clip lengths (185/277 frames = 2/3 chunks) "
            "split near 50/50, so the median flips 3->2 and reports 0.67x while the means "
            "are 2.54 vs 2.47 (0.97x, Mann-Whitney p=0.27, not significant). DreamDojo "
            "humanoid has the mirror problem: median 2 vs 2 (1.00x) hides means 2.86 vs "
            "3.57 (1.25x), also not significant. The median is unusable on both."
        ),
        deeper_criterion="two-sided Mann-Whitney p < 1e-3 AND mean ratio > 1",
        stride_verification=(
            "clip length = stride*C + n_cond on every clip; strides 92 (DreamGen), "
            "12 (DreamDojo bimanual/single_arm), 49 (DreamDojo humanoid)"
        ),
        cells=cells,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"[ok] {OUT.relative_to(REPO)}")
    write_table(cells)
    print(f"\n{'cell':24s}{'r_mean':>8s}{'r_med':>8s}{'MWU p':>10s}{'CL':>7s}  verdict")
    for c in cells:
        tag = "deeper" if c["deeper_significant"] else "not distinguishable"
        warn = "  <-- median unusable" if c["median_misleading"] else ""
        print(f"{c['embodiment']+'/'+c['model']:24s}{c['depth_ratio_mean']:8.2f}"
              f"{c['depth_ratio_median']:8.2f}{c['mannwhitney_p']:10.1g}"
              f"{c['cl_effect']:7.3f}  {tag}{warn}")


if __name__ == "__main__":
    main()
