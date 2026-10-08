# Reproducing the paper tables and figures

Every number in the paper is produced by a script; nothing is hand-typed. Producers live
in `assets/scripts/` (paper assets, figures, appendix tables) and `scripts/tables/`
(benchmark-wide summaries). They write summary data to `results/reports/` and LaTeX /
PDF assets to `assets/paper/{tables,figures,appendix/tables}/`. The full producer
catalogue, with ordering notes and gates, is [assets/scripts/README.md](../../assets/scripts/README.md).

Prerequisites: `sampling_dataset/` downloaded ([../datasets/sampling_dataset.md](../datasets/sampling_dataset.md))
and the metric outputs in `results/` / `metrics/WorldScore/results/` computed
([../metrics/README.md](../metrics/README.md)). Unless stated otherwise run from the repo
root with `PY=models/dreamgen/.venv/bin/python` (several producers need Python ≥ 3.12).

> Script names carry the RQ numbering of an earlier draft (`make_rq1_*` = Table 2,
> `make_rq3_*` = human alignment). The mapping below is to the numbering of the current PDF.

## Main paper

| paper item | content | producer(s) | output |
|---|---|---|---|
| Fig. 1 | teaser / benchmark overview | `assets/scripts/make_hero_figure.py`, `make_stat_donuts.py` | `assets/paper/figures/fig_hero.*`, `fig_stats_panel.*`, `stat_*.pdf` |
| Fig. 2–3 | dataset composition, metric illustrations | `assets/scripts/plot_benchmark_figures.py`, `make_nr_intuition_figures.py`, `make_ref_metric_figures.py` | `fig_{coverage,composition,diversity}.*`, `fig_nr_intuition_*.pdf`, `fig_ref_intuition_*.pdf` |
| Fig. 4, RQ1–RQ2 | KIVA / RIGIS vs VLM judges: correlation, accuracy, stability, latency | `scripts/tables/compare_ours_vs_vlm_correlation.py`, `compare_ours_vs_vlm_segment_overlap.py`, `analyze_geometry_label_correlation.py`, `scripts/metrics/export_hallucinate_results.py` → `assets/scripts/make_rq3_human_alignment.py` | `results/{ours_vs_vlm,ours_vs_vlm_segment,geometry_correlation,hallucinate_final}/`, `results/reports/rq3_human_alignment.{csv,json}`, `rq3_judge_cost.csv` |
| Table 2, RQ4 | dense results per cell (Visual Integrity, PSNR, Text Alignment, FVD, RIGIS / KIVA; M vs nM) | FVD pools (`fvd_camera_match.py` → `extract_perview_fvd_feats.py` → `pool_perview_fvd.py`, `pool_singleview_fvd.py` → `verify_shared_pool_fairness.py`) → `assets/scripts/export_dense_table_shared_pool.py`, `make_rq1_dense_table.py` | `assets/paper/tables/dense_baselines.tex`, `results/reports/rq1_dense_baselines.json` |
| Fig. 6 (c), RQ4 | degradation by rollout depth | `assets/scripts/measure_rollout_depth.py` → `make_singlearm_depth_figure.py`, `make_depth_vs_regime_figure.py`, `make_error_cascade_figure.py` | `fig_singlearm_depth.*`, `fig_depth_vs_regime.*`, `fig_error_cascade.*`, `appendix/tables/chunk_depth.tex` |
| Fig. 5, Fig. C2–C3, RQ5 | speedup vs RIGIS / KIVA / NR per method and cell | timing pass (below) → `assets/scripts/dump_speedup_figure_data.py` → `make_speedup_audit_figure.py`; statistics `rq2_tradeoff_stats.py` | `results/reports/speedup_figure_data_{bimanual,humanoid,single_arm}.{csv,json}`, `fig_speedup_rigis_kiva_*.pdf`, `results/reports/rq2_tradeoff.json` |

## Appendix

| item | producer(s) | output |
|---|---|---|
| App. A dataset tables, cell matrix | `assets/scripts/make_benchmark_stats.py` → `make_dataset_appendix_tables.py` → `audit_sampling_distribution.py` → `make_cell_matrix.py`; gate `check_dataset_appendix_claims.py` | `appendix/tables/dataset_{cells,taxonomy}.tex`, [../datasets/cell_matrix.md](../datasets/cell_matrix.md) |
| Table C1 / C3: checkpoints, fine-tuning recipes, generation settings | `assets/scripts/make_finetune_config_table.py`, `make_training_data_table.py`, `scripts/tables/make_dense_generation_table.py` | `appendix/tables/{finetune_hyperparams,generation_config,training_data_prep}.tex`, [../inference/dense_generation_config.md](../inference/dense_generation_config.md) |
| Acceleration hyper-parameters | `assets/scripts/make_impl_config_tables.py` | `appendix/tables/accel_hyperparams_{token,cache,sparse}.tex` |
| Uniform timing pass, speedup tables | `scripts/tools/tune/_bench_uniform_timing.sh` → `assets/scripts/merge_timing_shards.py` → `make_timing_pass_table.py`, `scripts/tables/make_speedup_summary.py`, `assets/scripts/speedup_table_full.py` | `appendix/tables/timing_pass.tex`, `results/reports/speedup_summary_*.{csv,tex}`, `speedup_table_full.csv` |
| Time budget, knee curves, fine-tune qualitative | `assets/scripts/make_time_budget_figure.py`, `make_knee_curves_figure.py`, `make_finetune_qualitative_figure.py` | `fig_time_budget.*`, `fig_knee_curves.*`, `fig_finetune_qualitative.*` |
| Per-config results | `assets/scripts/make_accel_config_table.py` | `appendix/tables/accel_by_config{,_paired}.tex` |
| App. C gate | `assets/scripts/check_impl_appendix_claims.py` (run last; reads [main_results_speedup_and_checkpoint.md](main_results_speedup_and_checkpoint.md)) | exit code only |
| App. D human study / VLM judges | same chain as Fig. 4; KIVA audit data `assets/scripts/make_kiva_audit_data.py` | `results/reports/kiva_audit.{csv,json}` |

Notes on the appendix tables (the versions in the paper PDF are the reference; regenerated
numbers can differ slightly):

- `timing_pass.tex` and `results/reports/speedup_summary_*.{csv,tex}` come from the
  uniform timing pass (`scripts/tools/tune/_bench_uniform_timing.sh` → its
  `results.jsonl` → `make_timing_pass_table.py --results .../results.jsonl`). A fresh
  pass on different hardware gives slightly different numbers.
- Regenerating `generation_config.tex` with `scripts/tables/make_dense_generation_table.py`
  now also emits the bimanual and humanoid Ctrl-World 3-view strip rows, which are not in
  the paper table (the paper lists their 4-view grid instead).

## Suggested order

```bash
PY=models/dreamgen/.venv/bin/python

# 1. timing -> speedup
bash scripts/tools/tune/_bench_uniform_timing.sh --all --gpu 0
$PY assets/scripts/merge_timing_shards.py
$PY scripts/tables/make_speedup_summary.py --sampling-root sampling_dataset --out-dir results/reports
$PY assets/scripts/speedup_table_full.py
$PY assets/scripts/dump_speedup_figure_data.py
$PY assets/scripts/rq2_tradeoff_stats.py
$PY assets/scripts/make_speedup_audit_figure.py

# 2. FVD pools -> Table 2
$PY assets/scripts/fvd_camera_match.py
$PY assets/scripts/extract_perview_fvd_feats.py
$PY assets/scripts/pool_perview_fvd.py
$PY assets/scripts/pool_singleview_fvd.py
$PY assets/scripts/verify_shared_pool_fairness.py
$PY assets/scripts/export_dense_table_shared_pool.py
$PY assets/scripts/make_rq1_dense_table.py

# 3. human alignment (Fig. 4)
$PY scripts/metrics/export_hallucinate_results.py
$PY scripts/tables/compare_ours_vs_vlm_correlation.py
$PY scripts/tables/compare_ours_vs_vlm_segment_overlap.py
$PY scripts/tables/analyze_geometry_label_correlation.py
$PY assets/scripts/make_rq3_human_alignment.py

# 4. rollout depth (Fig. 6c)
$PY assets/scripts/measure_rollout_depth.py
$PY assets/scripts/make_singlearm_depth_figure.py
$PY assets/scripts/make_depth_vs_regime_figure.py
$PY assets/scripts/make_error_cascade_figure.py

# 5. appendix C, gate last
for s in make_finetune_config_table make_training_data_table make_impl_config_tables \
         make_time_budget_figure make_knee_curves_figure make_finetune_qualitative_figure \
         make_timing_pass_table check_impl_appendix_claims; do
  $PY assets/scripts/$s.py || break
done
```

Most producers accept `--help`; check it for input-run overrides (e.g. `--run
metrics/WorldScore/results/dense_nr_full` for the NR-table producers). `results/` is not
shipped with the repo: every file under `results/reports/` is created by the producers above.
