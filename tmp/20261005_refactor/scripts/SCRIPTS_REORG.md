# Phase 6 — scripts/ reorganisation (2026-10-05 refactor)

Branch `refactor/paper-cleanup`. Every move is listed in `path_map.tsv` (old_path, new_path, kind ∈ move | archived | merged:<target>).
Archived files are in this directory; each has a row (with restore command) in `../MANIFEST.md`.

## Final layout (file counts, excluding `__init__.py`)

| dir | files |
|---|---|
| `scripts/prepare` | 11 |
| `scripts/infer` | 7 |
| `scripts/run (root: ablation base + augment-all)` | 2 |
| `scripts/run/ctrlworld` | 50 |
| `scripts/run/dreamgen` | 70 |
| `scripts/run/dreamdojo` | 33 |
| `scripts/train` | 5 |
| `scripts/metrics` | 56 |
| `scripts/tables` | 10 |
| `scripts/tools/tune` | 12 |
| `scripts/tools/diag` | 3 |
| `scripts/tools/ops` | 4 |
| `scripts/tools/hf` | 7 |
| `scripts/tools/data` | 12 |
| `scripts/augment (unchanged)` | 15 |
| `data/labels` | 8 |

Totals: 289 `git mv` moves (281 scripts + 8 data files to `data/labels/`), 77 archived, 8 merged-away sources (archived too), 1 new script (`scripts/metrics/run_dense_nr_full.sh`), 10 new `__init__.py` (Python subpackages only; run/ and train/ hold shell launchers). The `scripts/` root now only holds `__init__.py`.

## Placement rules

- `prepare/` — `prepare_*.py` + `repair_humanoid_singleview_split_meta.py` (dataset → model input).
- `infer/` — the 7 live `infer_*.py` entrypoints called by launchers.
- `run/{ctrlworld,dreamgen,dreamdojo}/` — every per-cell dense + method launcher, logic unchanged (only ROOT derivation and inter-script paths updated). Canonical dense entrypoints unchanged: DreamDojo single_arm `_dense.sh`, humanoid/bimanual `_full_episode.sh`; DreamGen multiview `_full.sh`, singleview `_static16fps.sh`; Ctrl-World `run_{emb}_multiview_ctrlworld[_4view_grid].sh`.
- `train/` — `run_*_ctrlworld_train*.sh`. `run_bimanual_multiview_ctrlworld_train_5fps.sh` keeps its `# BROKEN (2026-10-05 refactor)` marker (warm-start ckpt path no longer exists) — not fixed here.
- `metrics/` — scoring (PhyWorld NR/DR, compute_dr_metrics shim, dense-NR runners/slots, `_exp_*` experiments whose outputs feed docs/paper).
- `tables/` — paper table/figure builders that live under scripts/.
- `tools/tune` (sweeps/timing), `tools/diag` (kernel/attn diagnostics), `tools/ops` (env bootstrap, shard-summary merge, DreamDojo SV queue), `tools/hf` (HF upload/download), `tools/data` (dataset download/split/caption + makovian classification: `classify_markovian_*.py`, `join_labels_to_episodes.py` — KEPT as documented dataset-curation, see `classify_makovian.md`).
- `data/labels/` — `task_labels*.csv/jsonl`, `gr1_*` labels, `gr1_episode_sampled.csv`, `clasify_makovian_task.ipynb`. `gr1_episode_sampled copy.csv` was NOT identical (120 vs 115 rows, old paths) → archived, not deleted.

## Merged families

- `run_dense_nr_full_m0..m4.sh` + `run_dense_nr_full_extra.sh` → `scripts/metrics/run_dense_nr_full.sh <machine_idx|extra>` (sets `MACHINE=<idx>` or `REVERSE=1 MACHINE=0` and execs `_run_dense_nr_full.sh`, identical to the old wrappers).
- `_run_dreamdojo_{bimanual,humanoid}_sv_queue.sh` → archived; `_run_dreamdojo_sv_queue_multi.sh` (already `emb:method:split` parameterised) renamed to `scripts/tools/ops/run_dreamdojo_sv_queue.sh`.

## Path / import fixes

- Shell: every launcher derives `ROOT` with `$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)` (replaces the 43 hardcoded `/pfss/...` ROOTs and the `..` traversals). Inter-script calls (4-view-grid wrappers → base launcher, tmux shard self-relaunch, queue/dispatch/timing harness, `merge_ctrlworld_shard_summaries.py`, `repair_humanoid_singleview_split_meta.py`) point at new paths. Dynamic paths in `_bench_uniform_timing.sh` resolve `scripts/run/<family>/…` (`--estimate` finds 117/120 cell×method scripts, same 3 gaps as before).
- Python: repo-root derivations use `parents[2]` / `parents[3]` by depth; sibling imports (`run_dense_nr`, `compute_phyworld`, `phyworld_index`, `gt_cells`) put their own dir on `sys.path`; package imports updated to `scripts.metrics.compute_phyworld`, `scripts.infer.infer_single_arm_multiview_ctrlworld`.
- `assets/scripts`: `sys.path` inserts → `scripts/metrics` (for `compute_dr_metrics`) / `scripts/tables`; `make_impl_config_tables.py` indexes launchers in `scripts/run/*/` (verified: same rows as at HEAD) and records `script` as the new relative path.
- Docs/rules/skills/README/finetuning/methods/metrics path strings rewritten mechanically from `path_map.tsv`. References to archived scripts are rewritten as `archived:<name>` (see this dir).

## Unresolved / notes

- Some Python modules still hold absolute `/pfss/...` constants (e.g. `prepare_ctrlworld_*.py` DATASET_BASE/SVD_PATH, `split_humanoid_singleview.py --workspace` default). They are location-independent of the reorg and still resolve; left as-is.
- `assets/scripts/table_main_results.py` needs Python ≥3.12 (pre-existing f-string syntax); compiles in the project venvs, not in system python3.10.
- `run_bimanual_multiview_ctrlworld_train_5fps.sh`: BROKEN warm-start (marker kept).

## ARCHIVED (77 + 8 merged sources)

- `scripts/.bench_dist.sh` — Phase 6: generated copy of _bench_uniform_timing.sh (dispatcher re-creates .bench_dist.sh)
- `scripts/.bench_running.sh` — Phase 6: generated copy of _bench_uniform_timing.sh (dispatcher re-creates .bench_dist.sh)
- `scripts/_check_method_coverage.py` — Phase 6: one-off coverage check, superseded by nr_table_coverage.py; zero refs
- `scripts/_demo_motion_detector.py` — Phase 6: one-off demo of Actor Activity / Active Rate; zero refs
- `scripts/_demo_motion_downstream.py` — Phase 6: one-off demo of Actor Activity / Active Rate; zero refs
- `scripts/_demo_motion_patterns.py` — Phase 6: one-off demo of Actor Activity / Active Rate; zero refs
- `scripts/_exp_e3_build_manifest.py` — Phase 6: E3 flicker experiment concluded (column dropped); doc-only refs
- `scripts/_exp_e3_posthoc.py` — Phase 6: E3 flicker experiment concluded (column dropped); doc-only refs
- `scripts/_exp_e3_verdict.py` — Phase 6: E3 flicker experiment concluded (column dropped); doc-only refs
- `scripts/_rerender_single_arm_mv_dreamgen_dense.sh` — Phase 6: one-off supervisor of a finished rerender
- `scripts/_rerender_single_arm_mv_dreamgen_supervise.sh` — Phase 6: one-off supervisor of a finished rerender
- `scripts/_run_dreamdojo_bimanual_sv_queue.sh` — Phase 6: merged into scripts/tools/ops/run_dreamdojo_sv_queue.sh (behaviour kept)
- `scripts/_run_dreamdojo_humanoid_sv_queue.sh` — Phase 6: merged into scripts/tools/ops/run_dreamdojo_sv_queue.sh (behaviour kept)
- `scripts/_tune_dreamdojo_bimanual_sv_report.sh` — Phase 6: superseded by generic _tune_dreamdojo_sv_{trial,report}.sh
- `scripts/_tune_dreamdojo_bimanual_sv_trial.sh` — Phase 6: superseded by generic _tune_dreamdojo_sv_{trial,report}.sh
- `scripts/_tune_dreamdojo_single_arm_sv_report.sh` — Phase 6: superseded by generic _tune_dreamdojo_sv_{trial,report}.sh
- `scripts/_tune_dreamdojo_single_arm_sv_trial.sh` — Phase 6: superseded by generic _tune_dreamdojo_sv_{trial,report}.sh
- `scripts/_watchdog_dreamdojo_bimanual_sv.sh` — Phase 6: one-off watchdog with hardcoded Aug-2026 job lists
- `scripts/_watchdog_dreamdojo_sv.sh` — Phase 6: one-off watchdog with hardcoded Aug-2026 job lists
- `scripts/analyze_dense_nr_smoke.py` — Phase 6: smoke analysis of a finished NR run; zero refs
- `scripts/bench_itm_single_episode.sh` — Phase 6: ad-hoc bench; BROKEN marker / zero refs
- `scripts/bench_worldcache_thresh.sh` — Phase 6: ad-hoc bench; BROKEN marker / zero refs
- `scripts/bootstrap_mjcf_assets.sh` — Phase 6: fetches MJCFs for metrics/embodied_exec, which no longer exists
- `scripts/compare_grid_results.py` — Phase 6: only used by archived ITM grid searches
- `scripts/gr1_episode_sampled copy.csv` — stray copy of gr1_episode_sampled.csv; NOT identical (120 vs 115 rows, old sampling paths) so archived, not deleted
- `scripts/hf_prune_stale.py` — Phase 6: one-off HF cleanup; zero refs
- `scripts/infer_bimanual_singleview_full_episode.py` — Phase 6: no caller
- `scripts/infer_dreamgen_singleview_native_fps.py` — Phase 6: BROKEN (all_singleview ckpt gone); only caller archived
- `scripts/infer_humanoid_singleview_long_autoregressive.py` — Phase 6: only caller (long_autoregressive launcher) archived
- `scripts/infer_single_arm_singleview_full_episode.py` — Phase 6: only caller (single_arm _full_episode launcher) archived; canonical is _dense.sh
- `scripts/infer_single_arm_singleview_single_segment.py` — Phase 6: only caller (single_segment_test) archived
- `scripts/jobs_fill_missing.sh` — Phase 6: one-off fill-missing job list (hand-off doc)
- `scripts/materialize_humanoid_symlinks.py` — Phase 6: one-off symlink materialisation, already applied; zero refs
- `scripts/materialize_single_arm_symlinks.py` — Phase 6: one-off symlink materialisation, already applied; zero refs
- `scripts/migrate_ctrlworld_multiview_layout.py` — Phase 6: one-shot layout migration, already applied; zero refs
- `scripts/prepare_batch_input_json_csv.py` — Phase 6: legacy prepare, zero refs
- `scripts/prepare_droid_batch.py` — Phase 6: only used by archived run_droid_batch.sh
- `scripts/prepare_droid_for_ctrl_world.py` — Phase 6: only used by archived run_ctrl_world_droid_batch.sh
- `scripts/prepare_multiview_humanoid_batch.py` — Phase 6: only used by archived humanoid batch/airbot launchers (pre-grid flow)
- `scripts/resolve_prompts.py` — Phase 6: web manifest helper; web stack archived in Phase 5
- `scripts/run_all_singleview_dreamgen_70k.sh` — Phase 6: *70k* test run
- `scripts/run_all_singleview_dreamgen_dense.sh` — Phase 6: BROKEN (all_singleview ckpt gone)
- `scripts/run_bimanual_multiview_dreamgen_all.sh` — Phase 6: legacy per-task flow writing pre-layout sampling_dataset/bimanual/...; superseded by _full.sh
- `scripts/run_bimanual_multiview_dreamgen_batch.sh` — Phase 6: legacy per-task flow writing pre-layout sampling_dataset/bimanual/...; superseded by _full.sh
- `scripts/run_bimanual_multiview_dreamgen_seg_rollout.sh` — Phase 6: BROKEN (bimanual_multiview_seg iter10000 ckpt gone) test/rollout
- `scripts/run_bimanual_multiview_dreamgen_seg_test.sh` — Phase 6: BROKEN (bimanual_multiview_seg iter10000 ckpt gone) test/rollout
- `scripts/run_bimanual_singleview_dreamdojo_single_segment_test.sh` — Phase 6: *_test.sh
- `scripts/run_ctrl_world_droid_batch.sh` — Phase 6: DROID batch legacy (*_batch.sh)
- `scripts/run_ctrlworld_replay_original.sh` — Phase 6: one-off replay of original Ctrl-World
- `scripts/run_dense_nr_fit.sh` — Phase 6: superseded by run_dense_nr_complement / grid; zero refs
- `scripts/run_dense_nr_full_extra.sh` — Phase 6: merged into scripts/metrics/run_dense_nr_full.sh (behaviour kept)
- `scripts/run_dense_nr_full_m0.sh` — Phase 6: merged into scripts/metrics/run_dense_nr_full.sh (behaviour kept)
- `scripts/run_dense_nr_full_m1.sh` — Phase 6: merged into scripts/metrics/run_dense_nr_full.sh (behaviour kept)
- `scripts/run_dense_nr_full_m2.sh` — Phase 6: merged into scripts/metrics/run_dense_nr_full.sh (behaviour kept)
- `scripts/run_dense_nr_full_m3.sh` — Phase 6: merged into scripts/metrics/run_dense_nr_full.sh (behaviour kept)
- `scripts/run_dense_nr_full_m4.sh` — Phase 6: merged into scripts/metrics/run_dense_nr_full.sh (behaviour kept)
- `scripts/run_dreamgen_humanoid_singleview.sh` — Phase 6: legacy GR00T-Teleop captioning + gr00t flow (old sampling layout)
- `scripts/run_dreamgen_singleview_fps_test.sh` — Phase 6: BROKEN fps test / native-fps experiment
- `scripts/run_dreamgen_singleview_native_fps.sh` — Phase 6: BROKEN fps test / native-fps experiment
- `scripts/run_droid_batch.sh` — Phase 6: DROID batch legacy (*_batch.sh)
- `scripts/run_humanoid_gr1_video2world_720p16fps.sh` — Phase 6: 14B 720p video2world experiment, not a paper cell
- `scripts/run_humanoid_multiview_dreamgen_70k_test.sh` — Phase 6: *70k* test run
- `scripts/run_humanoid_multiview_dreamgen_seg16.sh` — Phase 6: alternate seg16 output tree (dreamgen_seg16, not in layout); canonical is _full.sh
- `scripts/run_humanoid_singleview_dreamdojo_70k_test.sh` — Phase 6: *70k* test run
- `scripts/run_humanoid_singleview_dreamdojo_long_autoregressive.sh` — Phase 6: long-autoregressive experiment, not a paper cell
- `scripts/run_humanoid_singleview_dreamgen_base.sh` — Phase 6: superseded singleview DreamGen variants (canonical is _static16fps.sh)
- `scripts/run_humanoid_singleview_dreamgen_prompt_refine.sh` — Phase 6: superseded singleview DreamGen variants (canonical is _static16fps.sh)
- `scripts/run_humanoid_singleview_dreamgen_short.sh` — Phase 6: superseded singleview DreamGen variants (canonical is _static16fps.sh)
- `scripts/run_itm_grid_search.sh` — Phase 6: ITM/SiTo grid searches (tuning done, numbers in rules)
- `scripts/run_itm_grid_search_v2.sh` — Phase 6: ITM/SiTo grid searches (tuning done, numbers in rules)
- `scripts/run_itm_grid_search_v2b.sh` — Phase 6: ITM/SiTo grid searches (tuning done, numbers in rules)
- `scripts/run_itm_grid_search_v3_light.sh` — Phase 6: ITM/SiTo grid searches (tuning done, numbers in rules)
- `scripts/run_multiview_airbot_all.sh` — Phase 6: legacy Airbot all-datasets flow (pre-grid)
- `scripts/run_multiview_humanoid_batch.sh` — Phase 6: DROID batch legacy (*_batch.sh)
- `scripts/run_observe_dense_vs_itm.sh` — Phase 6: ITM/SiTo grid searches (tuning done, numbers in rules)
- `scripts/run_phyworld_wave_supervise.sh` — Phase 6: one-off wave supervisor
- `scripts/run_pruning_benchmark.sh` — Phase 6: ITM/SiTo grid searches (tuning done, numbers in rules)
- `scripts/run_single_arm_multiview_video2world_2b_tmp.sh` — Phase 6: *_tmp.sh
- `scripts/run_single_arm_singleview_dreamdojo_full_episode.sh` — Phase 6: GT-anchored segment variant, zero refs; canonical single_arm dense is _dense.sh
- `scripts/run_single_arm_singleview_dreamdojo_single_segment_test.sh` — Phase 6: *_test.sh
- `scripts/run_single_arm_singleview_dreamgen_base.sh` — Phase 6: superseded singleview DreamGen variants (canonical is _static16fps.sh)
- `scripts/run_sito_grid_search.sh` — Phase 6: ITM/SiTo grid searches (tuning done, numbers in rules)
- `scripts/run_video2prompt_humanoid.sh` — Phase 6: legacy GR00T-Teleop captioning + gr00t flow (old sampling layout)
- `scripts/test_actor_flicker.py` — Phase 6: E3 flicker smoke test (Phase 3 plan: archive with actor_flicker)
- `scripts/video2prompt.py` — Phase 6: legacy GR00T-Teleop captioning + gr00t flow (old sampling layout)

## KEPT

All other files: see `path_map.tsv` rows with kind `move` (289).
