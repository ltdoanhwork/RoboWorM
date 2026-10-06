# metrics/reports — what was lost and how to regenerate it

Written 2026-10-05 (Phase 0 of the paper-aligned refactor).

## What happened

At about 16:00 UTC on 2026-10-05, `metrics/reports/` was missing from disk. The cause is unknown. A concurrent
`metrics/` rewrite ran between 15:08 and 15:34 UTC, but nothing proves it did this.

- **21 tracked files:** restored with `git checkout -- metrics/reports`.
  - 10 of these had **uncommitted modifications**, and those are gone. Their content is
    now the last committed version. Regenerate them (see "Tracked but stale" below).
- **85 untracked files** were listed by `git status` earlier in the day:
  - 53 top-level files
  - 31 files under `hallucination_smoke/agg/`
  - `hallucination_smoke/summary.json`
- **Search scope:**
  - `/tmp` and `/tmp/cursor-1479`
  - Inside the repo: `results/`, `tmp/`, `archive/`, `assets/`, `.main-edit/`, `external/`, `logs/`, `compare/` and `docs/`. The rest of the repo was searched to depth 6, and `sampling_dataset/`, `exp_logs/`, `models/`, `metrics/WorldScore/` and `webpage/` to depth 7.
  - Sibling dirs under `VLA/doanh` and `VLA/doanh/video_world`, plus `$HOME` and the workspace `.cursor/`.
  - Both git stashes, including their untracked trees. The two `external/*` worktrees.

## Recovered (31)

- `hallucination_smoke/agg/**`: all 31 files. They were copied from `/tmp/hal_agg/`, which was written by
  `/tmp/hal_probe.sh` and `/tmp/hal_probe2.sh` via `metrics/hallucination/run.sh` (formerly `scripts/run_hallucination.sh`).
  - The file set matches the earlier `git status` listing exactly.
  - The `render/` subdirs were left out because they were never under `metrics/reports`.

## Unrecoverable (54): regenerate

All Python paths are relative to the repo root. The "Producer" column comes from searching `scripts/`, `assets/scripts/`
and `metrics/` for the filename.

| report | producer (writes it) | notes |
|---|---|---|
| active_motion_continuity_per_video.json | assets/scripts/export_activity_scores.py | |
| active_rate_per_video.json | assets/scripts/export_activity_scores.py | |
| dense_psnr_cache.json | assets/scripts/export_dense_table_shared_pool.py | cache |
| dense_table_shared_pool.tex | assets/scripts/export_dense_table_shared_pool.py | needs fvd_*_pooled + shared_pool_fairness first |
| depth_vs_regime.json | assets/scripts/make_depth_vs_regime_figure.py | needs rollout_depth.json, rq1_dense_baselines.json |
| dr_coverage_audit.json | assets/scripts/audit_dr_coverage.py | |
| dr_vs_real_audit.json | assets/scripts/export_dr_grid_xlsx_vs_real.py | |
| error_cascade.json | assets/scripts/make_error_cascade_figure.py | |
| fvd_camera_map.json | assets/scripts/fvd_camera_match.py (`--out` default) | GPU |
| fvd_camera_map_rest.json | assets/scripts/fvd_camera_match.py `--clips 60 --cells humanoid/makovian,single_arm/makovian,single_arm/non_makovian --out metrics/reports/fvd_camera_map_rest.json` | GPU; cmd from agent transcript df731b2f |
| fvd_group_check.json | assets/scripts/verify_fvd_groups.py | |
| fvd_kinetics_domain_audit.json | assets/scripts/fvd_kinetics_domain_audit.py | |
| fvd_perview_pooled.json | assets/scripts/pool_perview_fvd.py | needs extract_perview_fvd_feats.py features + fvd_camera_map*.json |
| fvd_resolution_audit.json | assets/scripts/fvd_resolution_audit.py | |
| fvd_sample_size_probe.json | assets/scripts/fvd_sample_size_probe.py | |
| fvd_singleview_pooled.json | assets/scripts/pool_singleview_fvd.py | |
| fvd_temporal_sensitivity.json | assets/scripts/fvd_temporal_sensitivity_probe.py | |
| gt_align_probe.json | scripts/gt_align_probe.py | |
| gt_render.json | scripts/render_missing_gt.py | |
| hallucinate_human_vs_dr.csv | metrics/hallucination/map_to_dr.py (was scripts/map_hallucinate_to_dr.py) | |
| hallucinate_source_map.json | metrics/hallucination/map_to_dr.py | |
| human_subset_dr_metrics.csv | scripts/make_human_subset_dr_table.py | |
| human_subset_dr_metrics.json | scripts/make_human_subset_dr_table.py | |
| kiva_audit.csv | assets/scripts/make_kiva_audit_data.py | |
| kiva_audit.json | assets/scripts/make_kiva_audit_data.py | |
| multiview_fvd_diagnosis.json | assets/scripts/diagnose_multiview_fvd.py | |
| perview_fvd_probe.json | assets/scripts/perview_fvd_probe.py | |
| rollout_depth.json | assets/scripts/measure_rollout_depth.py | |
| rq1_dense_baselines.json | assets/scripts/make_rq1_dense_table.py | |
| rq1_matched_anchors.csv | assets/scripts/make_rq1_matched_anchors.py | |
| rq2_tradeoff.json | assets/scripts/rq2_tradeoff_stats.py | needs speedup_figure_data_*.csv first |
| rq3_human_alignment.csv | assets/scripts/make_rq3_human_alignment.py | |
| rq3_human_alignment.json | assets/scripts/make_rq3_human_alignment.py | |
| rq3_judge_cost.csv | assets/scripts/make_rq3_human_alignment.py | |
| shared_pool_fairness.json | assets/scripts/verify_shared_pool_fairness.py | |
| shared_pool_feasibility.json | assets/scripts/shared_pool_feasibility.py | |
| shared_pool_humanoid_makovian.json | assets/scripts/shared_pool_feasibility.py `--clips 40 --embodiment humanoid --split makovian --json metrics/reports/shared_pool_humanoid_makovian.json` | cmd from agent transcript df731b2f |
| shared_pool_humanoid_non_makovian.json | same, `--embodiment humanoid --split non_makovian` | |
| shared_pool_single_arm_makovian.json | same, `--embodiment single_arm --split makovian` | |
| singlearm_depth.json | assets/scripts/make_singlearm_depth_figure.py | |
| singlearm_depth_bars.csv | assets/scripts/make_singlearm_depth_figure.py | |
| speedup_figure_data_bimanual.{csv,json} | assets/scripts/dump_speedup_figure_data.py (`--out-prefix metrics/reports/speedup_figure_data`) | 2 files |
| speedup_figure_data_humanoid.{csv,json} | assets/scripts/dump_speedup_figure_data.py | 2 files |
| speedup_figure_data_single_arm.{csv,json} | assets/scripts/dump_speedup_figure_data.py | 2 files |
| speedup_grid_figureB.csv | inline heredoc python (no script in repo) | source in agent transcript a7a5cb74 ("Emit table in figure (b) layout"); should become a script |
| speedup_perchunk.csv | assets/scripts/merge_timing_shards.py | |
| speedup_table_full.csv | assets/scripts/speedup_table_full.py | needs timing_pass.* |
| worldcache_humanoid_mv_tuning.json | scripts/_tune_worldcache_humanoid_mv.sh | |
| worldcache_humanoid_sv_tuning.json | scripts/_tune_worldcache_humanoid_sv.sh (+ scripts/_collect_worldcache_humanoid_sv.py) | scratch outputs may be in tmp/wc_ladder |
| hallucination_smoke/summary.json | scripts/aggregate_hallucination.py, now metrics/hallucination/aggregate.py | exact args unknown; `/tmp/hal_results/summary.json` is a *different* run (single_arm, 5 variants) |

(The table has 50 rows. The three `speedup_figure_data_*` rows each cover 2 files, so it lists 50 + 3 = 53 top-level files plus `hallucination_smoke/summary.json`, for 54 in total.)

Logs from the original runs are still in `logs/`: `cam_match*.log`, `shared_pool_*.log`, `diag_mv_fvd.log`,
`fvd_n_probe.log` and `verify_fvd*.log`.

## Tracked but stale (uncommitted edits lost; restored to HEAD 486f21d)

| report | producer |
|---|---|
| dense_generation_config.json / .tex | scripts/make_dense_generation_table.py |
| finetune_config.json | assets/scripts/make_finetune_config_table.py |
| impl_accel_configs.json | assets/scripts/make_impl_config_tables.py |
| knee_curves.json | assets/scripts/make_knee_curves_figure.py |
| speedup_summary_appendix.tex, speedup_summary_by_method.csv, speedup_summary_full.csv | scripts/make_speedup_summary.py |
| timing_pass.csv / timing_pass.json | assets/scripts/merge_timing_shards.py |

## Running jobs at snapshot time (2026-10-05 ~16:10 UTC, host f750d639c356)

Nothing was killed.

- **`nvidia-smi`:** 1 GPU (index 0) at 0 MiB and 0% utilization, with no compute apps.
- **`tmux ls`:** `hal2`, `halprobe` and `halprobe2`. All three are finished and sitting at an idle shell.
  - `hal2` printed `DONE`.
  - `halprobe` printed `HAL_PROBE_DONE`.
  - `halprobe2` printed `HAL_PROBE2_DONE`.
- **`ps -u $USER`:** no live python or `bash scripts/*` jobs. There are hundreds of `<defunct>` python zombies left over from earlier
  sessions, plus idle Cursor terminal shells.
- **`/tmp/bsc_inflight_note.txt`:** says the DreamDojo `ddojo_bl_sc` run was "DONE 5/5 rc=0, committed ff0c96b; nothing of mine in flight".
- **`det task list`:** lists cluster-wide tasks, and none of them are on this shell. Another Determined shell owned by this user could
  still write to the shared filesystem; this was not checked from here.

## Suggested regeneration order

1. Timing: `merge_timing_shards.py`, then `make_speedup_summary.py`, then `speedup_table_full.py` and `dump_speedup_figure_data.py`, then `rq2_tradeoff_stats.py` and the figure-B grid.
2. FVD: `fvd_camera_match.py` (both maps), then `extract_perview_fvd_feats.py`, then `pool_perview_fvd.py` and `pool_singleview_fvd.py`, then `verify_shared_pool_fairness.py`, then `export_dense_table_shared_pool.py`. The FVD probes and audits can run in any order.
3. RQ1/RQ3/KIVA: `make_rq1_dense_table.py`, `make_rq1_matched_anchors.py`, `make_kiva_audit_data.py`, `make_rq3_human_alignment.py`.
4. Depth: `measure_rollout_depth.py`, then `make_singlearm_depth_figure.py`, `make_depth_vs_regime_figure.py` and `make_error_cascade_figure.py`.

> 2026-10-06 (Phase 9): `metrics/reports/` moved to `results/reports/` (still git-tracked via a
> `.gitignore` exception). Every producer path above now writes `results/reports/<file>`.
