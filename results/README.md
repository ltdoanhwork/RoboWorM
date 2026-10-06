# `results/`: metric outputs read by the paper

Only this README is tracked by git. Everything else here is generated and can be
regenerated (see [`docs/reproduce/README.md`](../docs/reproduce/README.md)).

| dir | written by | read by (paper) |
|---|---|---|
| `metrics/` | `python -m metrics.evaluate --out results/metrics/<run>` (default output root of the metric CLI; not created until you run it) | your own tables |
| `hallucination/` | `metrics/kiva/run.sh` (`HAL_OUT_ROOT` default) | `metrics/kiva/aggregate.py` (KIVA, RQ3) |
| `multiview_consistency/` | `metrics/rigis/run.sh` (`MVC_OUT_ROOT` default) | `metrics/rigis/aggregate.py` (RIGIS) |
| `hallucinate_final/` | `scripts/metrics/export_hallucinate_results.py`, `scripts/tables/measure_after_expert_pass.py` | `assets/scripts/make_rq3_human_alignment.py` (RQ3, App. D) |
| `ours_vs_vlm/` | `scripts/tables/compare_ours_vs_vlm_correlation.py` | `assets/scripts/make_rq3_human_alignment.py` (RIGIS vs VLM judges) |
| `ours_vs_vlm_segment/` | `scripts/tables/compare_ours_vs_vlm_segment_overlap.py` | App. D segment-overlap figures |
| `geometry_correlation/` | `scripts/tables/analyze_geometry_label_correlation.py` | `assets/scripts/make_rq3_human_alignment.py` |

Superseded runs (`dense_nr_v1/`, queue logs, an unreferenced boxplot) were moved to
`tmp/20261005_refactor/misc/results/` on 2026-10-06; restore commands are in
`tmp/20261005_refactor/MANIFEST.md`.

Paper summary tables (`*.json`, `*.tex`, `*.csv`) are written to `metrics/reports/`,
not here.
