# Extending RoboWorM

RoboWorM is built to be extended under a fixed protocol (paper App. E, "Using and
Extending RoboWorM"). A new entrant is directly comparable to every number in the paper
because the episodes, the conditioning, the analysis grid and each metric's reference
are already fixed. Scoring is **unpaired when computed and paired when tabulated**:
every metric is stored per view-clip, so adding something never requires re-scoring
what is already in the benchmark.

| path | you add | needs training? | reuses | guide |
|---|---|---|---|---|
| **A** new world model (App. E.1) | a row: one model on the existing grid | yes, fine-tune on the released corpus | episodes, conditioning, metrics | [new_world_model.md](new_world_model.md) |
| **B** new embodiment / dataset (App. E.2) | a column: a new cell of the grid | yes, every backbone you cover | the pipeline, the metrics' code | [new_embodiment_dataset.md](new_embodiment_dataset.md) |
| **C** new acceleration method (App. E.3) | a point on the speed–quality plane | no (training-free) | episodes, conditioning, checkpoints, dense rollouts | [new_acceleration_method.md](new_acceleration_method.md) |
| new metric | a column of every table | no | every rollout already in `sampling_dataset/` | [new_metric.md](new_metric.md) |

Path B is the expensive one: it *supplies* the invariants the physical audits rely on
(which links are rigid, how much motion is normal, where each camera is mounted), so a
KIVA keypoint reader, KIVA real-segment thresholds, a RIGIS camera-set declaration and an
FVD real pool must all exist before the first generated clip can be scored.

## The pieces every path touches

| piece | where | convention |
|---|---|---|
| source data | `datasets/{single_arm,bimanual,humanoid}/{singleview,multiview}/{makovian,non_makovian}/[<task>/]` | LeRobot v2: `meta/`, `data/chunk-XXX/*.parquet`, `videos/chunk-XXX/<camera>/*.mp4` ([../datasets/README.md](../datasets/README.md)) |
| pipeline | `scripts/prepare/` → `scripts/infer/` (or `models/*/examples/`) → `scripts/run/<family>/` | three stages per cell ([../../scripts/README.md](../../scripts/README.md)) |
| launcher name | `scripts/run/<family>/run_<embodiment>_<view>_<model>[_4view_grid]_<method>.sh` | method launchers mirror the cell's dense launcher |
| rollouts | `sampling_dataset/{method}/{embodiment}/{input,output}/{view}/{model}/{split}/...` | `input/` only under `dense/`; the method is the top-level dir ([../datasets/sampling_dataset.md](../datasets/sampling_dataset.md)) |
| checkpoints | `checkpoints/{base,finetuned,metrics}/` | one row per cell in [../../checkpoints/CHECKPOINTS.md](../../checkpoints/CHECKPOINTS.md) |
| metrics | `metrics/registry.py`, `metrics/base.py`, `python -m metrics.evaluate` | [../../metrics/README.md](../../metrics/README.md), [../metrics/README.md](../metrics/README.md) |
| physical audits | `metrics/kiva/`, `metrics/rigis/` | [../metrics/kiva.md](../metrics/kiva.md), [../metrics/rigis.md](../metrics/rigis.md) |
| acceleration | `methods/{cache_strategy,sparse_attention,prunning}/` | `add_<method>_args` + a builder / `enable_*` hook ([../acceleration/](../acceleration/README.md)) |
| cell list for tables | `assets/scripts/benchmark_cells.py`, `scripts/tools/tune/_bench_uniform_timing.sh` (`CELLS=`) | one place decides which cells are reported / timed |
| paper producers | `assets/scripts/`, `scripts/tables/` | [../../assets/scripts/README.md](../../assets/scripts/README.md), [../reproduce/README.md](../reproduce/README.md) |

## Rules that hold on every path (App. E.4, shared checklist)

1. **Cover a cell completely**: both regimes (`makovian`, `non_makovian`), every camera
   the source provides, every curated episode. Partial coverage of the grid is fine;
   partial cells are not.
2. **Keep the analysis grid**: drop the conditioning frame, stride to ≈5 fps, crop the
   requested view, resample to 256×192 (`metrics/video_io.py::nr_analysis_frames`);
   clips with fewer than 3 surviving frames are not scored.
3. **Aggregate inside one stratum** (model, embodiment, camera mount, regime): median,
   95% bootstrap CI, valid `n`, exclusion rate. A difference is a claim only if the CI
   excludes 0 **and** Wilcoxon p < 0.05 **and** n ≥ 8; otherwise "untested".
4. **Paired deltas need the same measurement**: same source and effective fps, length
   within 10%.
5. **Unmeasurable is `None`, never `0`.**
6. Respect per-column restrictions: no model ranking on Visual Integrity (CLIP-IQA+), no
   unpaired Text Alignment (CLIPScore) median, no per-cell claim on FVD (effective n = 1).
7. **Never write trials or sweeps into the canonical tree**: send them to a scratch
   `OUTPUT_ROOT` / `SAVE_ROOT` and delete it afterwards.
8. **Negative results stay**: a slow or lossy method is a data point, not something to
   delete or swap. "Not benchmarked" (`--`) stays distinct from "measured and lost".
