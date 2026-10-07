#!/usr/bin/env python3
"""Emit the RQ3 human-alignment evidence bundle: our metrics vs. the VLM judges.

WHY THIS FILE EXISTS
--------------------
The RQ3 numbers were computed outside this repo (the benchmark workbook's
`RIGIS-main` / `KIVA-main` tabs, same source `assets/scripts/export_paper_grid_xlsx.py`
mirrors for the NR grid) and the paper needs every sentence to cite ONE file. This
script is that file: it holds the transcribed table and emits a CSV + JSON so
`sections/4_experiments.tex` can point at a path instead of at a spreadsheet.

Nothing is recomputed here. The detector code for RIGIS (cross-view point
tracking) and KIVA (3D-lifted body keypoints) is external to this repo -- see
`docs/metrics/README.md` (RQ3 baselines) -- so the numbers are transcribed, and the
transcription is checked: every `mean` row is re-derived from its three
per-embodiment values and the script fails if the workbook mean disagrees by more
than 5e-4.

WHICH RIGIS NUMBER THE PAPER REPORTS (the reconciliation this file settles)
--------------------------------------------------------------------------
Two RIGIS-vs-human correlations exist in this repo and they differ:

    results/ours_vs_vlm/stats_all.csv                 Spearman 0.6414  (N=189, POOLED
                                                      over the three embodiments)
    results/geometry_correlation/
        correlation_summary_consensus.csv             0.656 / 0.588 / 0.775 per embodiment
    this file (workbook)                              0.7063 / 0.7244 / 0.8158,
                                                      mean 0.7488

The reported statistic is the **mean over per-embodiment correlations**, not the
pooled one, for the same reason the NR grid is never pooled across embodiments
(archived `tmp/20261005_refactor/misc/docs_metrics/nr_paper_output_schema.md`, `NOT_CROSS_EMBODIMENT`): the three embodiments
have different human label distributions and different clip counts, so a pooled
rank correlation partly measures the embodiment mix. The pooled value is kept in
the JSON as a diagnostic so the gap is visible rather than quietly dropped.

The per-embodiment values here are also slightly above
`correlation_summary_consensus.csv` because that CSV was computed on an earlier
label resolution (before the geometry arbiter's full 300-clip pass was folded in).
Both are recorded; the workbook pass is the current one.

KIVA IS PENDING ON PURPOSE
--------------------------
The only KIVA-vs-human numbers in this repo are in
`results/hallucinate_final/RESULTS.md` and they are from the OLD pipeline cut
points (segment overlap 0.5599, below Gemini's 0.6434). A revised detector is in
flight, so KIVA rows are emitted with null metrics and
`status: pending_new_detector`. Do not cite the 0.5599 figures as the paper's KIVA
result: they measure a superseded configuration.

Usage:
    python3 assets/scripts/make_rq3_human_alignment.py
    python3 assets/scripts/make_rq3_human_alignment.py --check   # verify only
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results/reports"
OUT_ALIGN = OUT_DIR / "rq3_human_alignment.csv"
OUT_COST = OUT_DIR / "rq3_judge_cost.csv"
OUT_JSON = OUT_DIR / "rq3_human_alignment.json"

EMB = ["bimanual", "humanoid", "single_arm"]
MEAN_TOL = 5e-4

# Display name, access class. `qwen_plus` is the closed 3.6-Plus endpoint and
# `qwen_35b` the open-weight 35B-A3B; both appear in
# scripts/tables/compare_ours_vs_vlm_correlation.py as `qwen_close` / `qwen_open`.
JUDGES = [
    ("gemini_flash", "Gemini 3.6 Flash", "closed"),
    ("qwen_plus", "Qwen 3.6 Plus", "closed"),
    ("gemma_31b", "Gemma 4 31B", "open"),
    ("qwen_35b", "Qwen 3.6 35B", "open"),
]

# ---------------------------------------------------------------------------
# Transcribed workbook values.
#
# correlation: predictor -> strategy -> metric -> [bimanual, humanoid, single_arm,
#                                                 workbook_mean]
# The 4th entry is the workbook's own mean and is CHECKED against the first three,
# never trusted blindly.
# ---------------------------------------------------------------------------
CORRELATION = {
    "gemini_flash": {
        "vanilla": {
            "spearman": [0.3787, 0.1542, 0.2877, 0.2735],
            "kendall": [0.3402, 0.1428, 0.2577, 0.2469],
            "pearson": [0.3760, 0.1879, 0.2875, 0.2838],
            "rank_acc": [0.4790, 0.2034, 0.4236, 0.3687],
        },
        "cot": {
            "spearman": [0.1834, 0.0546, 0.2517, 0.1632],
            "kendall": [0.1636, 0.0478, 0.2275, 0.1463],
            "pearson": [0.1844, 0.0932, 0.2493, 0.1756],
            "rank_acc": [0.3936, 0.2170, 0.4020, 0.3375],
        },
    },
    "qwen_plus": {
        "vanilla": {
            "spearman": [0.0220, 0.2330, 0.2336, 0.1629],
            "kendall": [0.0205, 0.2187, 0.2153, 0.1515],
            "pearson": [0.0087, 0.2162, 0.2323, 0.1524],
            "rank_acc": [0.0977, 0.0749, 0.3594, 0.1773],
        },
        "cot": {
            "spearman": [0.3369, 0.1734, 0.1630, 0.2244],
            "kendall": [0.3181, 0.1591, 0.1537, 0.2103],
            "pearson": [0.3454, 0.1714, 0.1634, 0.2267],
            "rank_acc": [0.2921, 0.2955, 0.3221, 0.3032],
        },
    },
    "gemma_31b": {
        "vanilla": {
            "spearman": [0.3131, 0.3269, 0.3052, 0.3151],
            "kendall": [0.2858, 0.2888, 0.2816, 0.2854],
            "pearson": [0.3073, 0.3430, 0.2974, 0.3159],
            "rank_acc": [0.3236, 0.4685, 0.3966, 0.3962],
        },
        "cot": {
            "spearman": [0.2907, 0.3418, 0.3444, 0.3256],
            "kendall": [0.2720, 0.3129, 0.3192, 0.3014],
            "pearson": [0.2854, 0.3401, 0.3333, 0.3196],
            "rank_acc": [0.3008, 0.4781, 0.4164, 0.3984],
        },
    },
    "qwen_35b": {
        "vanilla": {
            "spearman": [0.3425, 0.1458, 0.0238, 0.1707],
            "kendall": [0.3156, 0.1309, 0.0195, 0.1553],
            "pearson": [0.3024, 0.1539, 0.0373, 0.1645],
            "rank_acc": [0.3735, 0.3589, 0.2882, 0.3402],
        },
        "cot": {
            "spearman": [0.1436, 0.3593, 0.2979, 0.2669],
            "kendall": [0.1267, 0.3156, 0.2694, 0.2372],
            "pearson": [0.1400, 0.3484, 0.2948, 0.2611],
            "rank_acc": [0.3543, 0.5029, 0.4017, 0.4196],
        },
    },
    "rigis": {
        "deterministic": {
            "spearman": [0.7063, 0.7244, 0.8158, 0.7488],
            "kendall": [0.5696, 0.5914, 0.6799, 0.6136],
            "pearson": [0.7283, 0.7228, 0.7623, 0.7379],
            "rank_acc": [0.8474, 0.8615, 0.9144, 0.8744],
        },
    },
}

# Self-consistency across the 3 independent turns (%) and cost in seconds of
# compute per second of video. `self_agreement_pct` for `human` is inter-rater
# agreement -- the honest ceiling this task admits, not a judge property.
COST = {
    ("gemini_flash", "vanilla"): (63.000, 0.7840),
    ("qwen_plus", "vanilla"): (77.000, 3.1210),
    ("gemma_31b", "vanilla"): (57.667, 1.0540),
    ("qwen_35b", "vanilla"): (38.667, 1.1540),
    ("gemini_flash", "cot"): (64.333, 0.8040),
    ("qwen_plus", "cot"): (33.667, 2.8180),
    ("gemma_31b", "cot"): (62.333, 2.7570),
    ("qwen_35b", "cot"): (33.000, 1.1100),
    ("human", "n/a"): (56.00, 1.0),
    ("rigis", "deterministic"): (None, 0.341),
}

DISPLAY = {k: d for k, d, _ in JUDGES}
DISPLAY.update({"rigis": "RIGIS", "kiva": "KIVA", "human": "Human"})
ACCESS = {k: a for k, _, a in JUDGES}
ACCESS.update({"rigis": "deterministic", "kiva": "deterministic", "human": "human"})

TRACK = {"rigis": "multiview_consistency", "kiva": "embodied_physics"}

# The human reference each track is scored against.
REFERENCE = {
    "multiview_consistency": {
        "protocol": "Mode A -- 300 multi-view clips, raw + CoTracker overlay, "
                    "3-level Bad/Medium/Good",
        "resolution": "two agreeing raters, else the geometry arbiter's full "
                      "300-clip pass",
        "n_clips_scored": 189,
        "source_in_repo": "scripts/tables/compare_ours_vs_vlm_correlation.py",
    },
    "embodied_physics": {
        "protocol": "Mode B -- 16-frame segments of 291 videos, blind catalog, "
                    "3-level Bad/Medium/Good",
        "resolution": "4,336 segments by rater agreement, 3,757 adjudicated by a "
                      "checker",
        "n_segments_resolved": 8093,
        "majority_class_baseline": 0.5148,
        "source_in_repo": "results/hallucinate_final/RESULTS.md",
    },
}


def check_means() -> list[str]:
    """Re-derive every workbook mean from its three per-embodiment values."""
    problems = []
    for pred, by_strategy in CORRELATION.items():
        for strategy, metrics in by_strategy.items():
            for metric, vals in metrics.items():
                derived = sum(vals[:3]) / 3.0
                if abs(derived - vals[3]) > MEAN_TOL:
                    problems.append(
                        f"{pred}/{strategy}/{metric}: workbook mean {vals[3]:.4f} "
                        f"!= mean of embodiments {derived:.4f}"
                    )
    return problems


def align_rows() -> list[dict]:
    rows = []
    for pred, by_strategy in CORRELATION.items():
        for strategy, metrics in by_strategy.items():
            for i, emb in enumerate(EMB + ["mean"]):
                rows.append({
                    "track": TRACK.get(pred, "multiview_consistency"),
                    "predictor": pred,
                    "predictor_display": DISPLAY[pred],
                    "access": ACCESS[pred],
                    "strategy": strategy,
                    "embodiment": emb,
                    "spearman": f"{metrics['spearman'][i]:.4f}",
                    "kendall": f"{metrics['kendall'][i]:.4f}",
                    "pearson": f"{metrics['pearson'][i]:.4f}",
                    "rank_acc": f"{metrics['rank_acc'][i]:.4f}",
                    "status": "reported",
                })
    # KIVA placeholders: the track exists, the numbers do not yet.
    for emb in EMB + ["mean"]:
        rows.append({
            "track": "embodied_physics",
            "predictor": "kiva",
            "predictor_display": "KIVA",
            "access": "deterministic",
            "strategy": "deterministic",
            "embodiment": emb,
            "spearman": "",
            "kendall": "",
            "pearson": "",
            "rank_acc": "",
            "status": "pending_new_detector",
        })
    return rows


def cost_rows() -> list[dict]:
    rows = []
    for (pred, strategy), (agree, cost) in COST.items():
        rows.append({
            "predictor": pred,
            "predictor_display": DISPLAY[pred],
            "access": ACCESS[pred],
            "strategy": strategy,
            "self_agreement_pct": "" if agree is None else f"{agree:.3f}",
            "sec_compute_per_sec_video": f"{cost:.4f}",
            "note": ("inter-rater agreement, not self-consistency"
                     if pred == "human" else
                     "deterministic: repeated runs are bit-identical"
                     if agree is None else
                     "exact 3-way agreement over 3 independent turns"),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the transcription and exit without writing")
    args = ap.parse_args()

    problems = check_means()
    if problems:
        raise SystemExit("transcription check FAILED:\n  " + "\n  ".join(problems))
    print(f"[ok] transcription check passed "
          f"({sum(len(s) * 4 for s in CORRELATION.values())} means re-derived)")
    if args.check:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = align_rows()
    with OUT_ALIGN.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    crows = cost_rows()
    with OUT_COST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(crows[0].keys()))
        w.writeheader()
        w.writerows(crows)

    best_vlm = max(
        ((p, s, m["spearman"][3]) for p, bs in CORRELATION.items()
         if p != "rigis" for s, m in bs.items()),
        key=lambda t: t[2],
    )
    bundle = {
        "generated_by": "assets/scripts/make_rq3_human_alignment.py",
        "purpose": "RQ3 evidence: agreement of our deterministic physical metrics "
                   "and of VLM judges with the human reference, plus each "
                   "predictor's stability and cost.",
        "provenance": "transcribed from the benchmark workbook's RIGIS/KIVA tabs; "
                      "detector code for RIGIS and KIVA is external to this repo "
                      "(docs/metrics/README.md, RQ3 baselines).",
        "reported_statistic": "mean over the three per-embodiment correlations, "
                              "never pooled across embodiments (the embodiments "
                              "have different human label distributions and clip "
                              "counts, so a pooled rank correlation partly "
                              "measures the mix).",
        "judges": {
            "models": [d for _, d, _ in JUDGES],
            "prompt_strategies": ["vanilla", "cot"],
            "turns_per_clip": 3,
            "reported_per_turn_and_3turn_mean": True,
        },
        "reference": REFERENCE,
        "headline": {
            "rigis_spearman_mean": CORRELATION["rigis"]["deterministic"]["spearman"][3],
            "rigis_rank_acc_mean": CORRELATION["rigis"]["deterministic"]["rank_acc"][3],
            "best_vlm": {"predictor": DISPLAY[best_vlm[0]],
                         "strategy": best_vlm[1],
                         "spearman_mean": best_vlm[2]},
            "cot_hurts_strongest_judge": {
                "predictor": "Gemini 3.6 Flash",
                "spearman_vanilla": CORRELATION["gemini_flash"]["vanilla"]["spearman"][3],
                "spearman_cot": CORRELATION["gemini_flash"]["cot"]["spearman"][3],
            },
            "self_agreement_collapse_under_cot": {
                "predictor": "Qwen 3.6 Plus",
                "vanilla_pct": COST[("qwen_plus", "vanilla")][0],
                "cot_pct": COST[("qwen_plus", "cot")][0],
            },
            "cost_sec_per_sec_video": {
                "rigis": COST[("rigis", "deterministic")][1],
                "vlm_range": [min(c for (p, _), (_, c) in COST.items()
                                  if p not in ("rigis", "human")),
                              max(c for (p, _), (_, c) in COST.items()
                                  if p not in ("rigis", "human"))],
                "human": COST[("human", "n/a")][1],
            },
            "human_inter_rater_agreement_pct": COST[("human", "n/a")][0],
        },
        "diagnostics": {
            "rigis_spearman_pooled_over_embodiments": {
                "value": 0.6414,
                "n": 189,
                "source": "results/ours_vs_vlm/stats_all.csv",
                "why_not_reported": "pooling mixes the three embodiments' label "
                                    "distributions; kept to make the gap to the "
                                    "reported 0.7488 visible.",
            },
            "rigis_per_embodiment_earlier_resolution": {
                "bimanual": 0.656, "humanoid": 0.5882, "single_arm": 0.7748,
                "source": "results/geometry_correlation/"
                          "correlation_summary_consensus.csv",
                "why_lower": "computed before the geometry arbiter's full 300-clip "
                             "pass was folded into the label resolution.",
            },
        },
        "kiva_status": {
            "status": "pending_new_detector",
            "do_not_cite": "results/hallucinate_final/RESULTS.md",
            "reason": "those numbers (segment overlap 0.5599, below Gemini 0.6434) "
                      "come from the superseded pipeline cut points 1.00/1.50; a "
                      "revised detector is in flight.",
        },
    }
    OUT_JSON.write_text(json.dumps(bundle, indent=2) + "\n")

    for p, n in ((OUT_ALIGN, len(rows)), (OUT_COST, len(crows)), (OUT_JSON, 1)):
        print(f"[ok] {p.relative_to(REPO)}  ({n} rows)" if n > 1
              else f"[ok] {p.relative_to(REPO)}")
    print(f"     RIGIS Spearman mean {bundle['headline']['rigis_spearman_mean']:.4f} "
          f"vs best VLM {best_vlm[2]:.4f} ({DISPLAY[best_vlm[0]]}, {best_vlm[1]})")
    print("     KIVA rows emitted as pending_new_detector (no numbers)")


if __name__ == "__main__":
    main()
