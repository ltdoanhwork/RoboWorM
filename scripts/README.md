# `scripts/`

Launchers, inference entrypoints and tooling for the RoboWorM benchmark
(3 world models x 9 acceleration methods x 12 paper cells, Table C1).
Every script resolves the repo root on its own
(`git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel` in bash,
`Path(__file__).resolve().parents[N]` in Python), so you can call it from any
working directory. The examples below run from the repo root.

## Layout

| dir | what lives there |
|---|---|
| `prepare/` | dataset -> model-input converters (`prepare_*.py`, `repair_humanoid_singleview_split_meta.py`) |
| `infer/` | Python inference entrypoints the launchers call (`infer_*.py`) |
| `run/ctrlworld/`, `run/dreamgen/`, `run/dreamdojo/` | one shell launcher per (cell, method): dense plus every accelerator |
| `run/` (root) | `_run_checkpoint_ablation_base.sh`, `run_augment_all.sh` |
| `train/` | Ctrl-World fine-tuning launchers (`run_*_ctrlworld_train*.sh`). `..._train_5fps.sh` is marked BROKEN because its warm-start checkpoint no longer exists |
| `metrics/` | PhyWorld NR/DR scoring, `compute_dr_metrics.py`, dense-NR wave runners (`run_dense_nr*.sh`), `_exp_*` experiments that feed paper numbers |
| `tables/` | paper table and figure builders that live under `scripts/` (most builders are in `assets/scripts/`) |
| `tools/tune/` | tuning sweeps and the uniform timing harness (`_bench_uniform_timing.sh`, `_dispatch_timing_cells.sh`, `_tune_*`) |
| `tools/diag/` | kernel and attention diagnostics (`_diag_dreamdojo_svg.py`, ...) |
| `tools/ops/` | env bootstrap (radial/svg), `merge_ctrlworld_shard_summaries.py`, `run_dreamdojo_sv_queue.sh` |
| `tools/hf/` | Hugging Face upload/download |
| `tools/data/` | dataset download, split, captioning, makovian classification |
| `augment/` | augment track (core runners + per-model infer); see the `augment-track-layout` rule |

The Markovian label CSV/JSONL files are not shipped (the splits are baked into the HF
dataset `doanh25032004/vgp_datasets`); the `tools/data/` classifiers write them to a local,
untracked `tmp/20261005_refactor/misc/data_labels/` by default.

## Launcher naming

```
scripts/run/<model>/run_<embodiment>_<view>_<model>[_4view_grid]_<method>.sh
  <model>      ctrlworld | dreamgen | dreamdojo
  <embodiment> single_arm | bimanual | humanoid
  <view>       multiview | singleview
  <method>     dicache | worldcache | fastercache | pisa | radial | svg1 | svg2 | sito | itm
               (+ augment, and the DreamGen bimanual-MV extras bsc / bwcache / scalingcache)
```

A method launcher mirrors its cell's dense launcher: same checkpoint, dataset,
entrypoint and paths. It only adds the method flags and writes to
`sampling_dataset/<method>/...` instead of `sampling_dataset/dense/...`. The
dense launcher name varies by family (see the table below), so look it up
rather than guessing it.

## Canonical dense launcher per cell (Table C1)

| # | cell (timing id) | dense launcher | entrypoint |
|---|---|---|---|
| 1 | Ctrl-World single_arm MV (`ctrlworld_single_arm_mv`) | `run/ctrlworld/run_single_arm_multiview_ctrlworld.sh` | `infer/infer_single_arm_multiview_ctrlworld.py` |
| 2 | Ctrl-World bimanual MV, 3-view strip | `run/ctrlworld/run_bimanual_multiview_ctrlworld.sh` | `infer/infer_bimanual_multiview_ctrlworld.py` |
| 3 | Ctrl-World bimanual MV, 4-view grid (`ctrlworld4_bimanual_mv`) | `run/ctrlworld/run_bimanual_multiview_ctrlworld_4view_grid.sh` | wraps #2 with the grid ckpt/config |
| 4 | Ctrl-World humanoid MV, 3-view strip | `run/ctrlworld/run_humanoid_multiview_ctrlworld.sh` | `infer/infer_bimanual_multiview_ctrlworld.py --config humanoid` |
| 5 | Ctrl-World humanoid MV, 4-view grid (`ctrlworld4_humanoid_mv`) | `run/ctrlworld/run_humanoid_multiview_ctrlworld_4view_grid.sh` | wraps #4 with the grid ckpt/config |
| 6 | DreamGen single_arm MV, 14B (`dreamgen_single_arm_mv`) | `run/dreamgen/run_single_arm_multiview_dreamgen_all.sh` (per split: `..._batch.sh`) | `video2world_gr00t --gr00t_variant droid` |
| 7 | DreamGen bimanual MV (`dreamgen_bimanual_mv`) | `run/dreamgen/run_bimanual_multiview_dreamgen_full.sh` | `infer/infer_bimanual_multiview_rollout.py` |
| 8 | DreamGen humanoid MV (`dreamgen_humanoid_mv`) | `run/dreamgen/run_humanoid_multiview_dreamgen_full.sh` | `infer/infer_bimanual_multiview_rollout.py` |
| 9 | DreamGen single_arm SV (`dreamgen_single_arm_sv`) | `run/dreamgen/run_single_arm_singleview_dreamgen_static16fps.sh` | `video2world_gr00t_lvg` |
| 10 | DreamGen bimanual SV (`dreamgen_bimanual_sv`) | `run/dreamgen/run_bimanual_singleview_dreamgen_16fps.sh` | `video2world_gr00t_lvg --gr00t_variant droid` |
| 11 | DreamGen humanoid SV (`dreamgen_humanoid_sv`) | `run/dreamgen/run_humanoid_singleview_dreamgen_static16fps.sh` | `video2world_lvg` |
| 12 | DreamDojo single_arm SV (`dreamdojo_single_arm_sv`) | `run/dreamdojo/run_single_arm_singleview_dreamdojo_dense.sh` | `infer/infer_single_arm_singleview_autoregressive.py` |
| 13 | DreamDojo bimanual SV (`dreamdojo_bimanual_sv`) | `run/dreamdojo/run_bimanual_singleview_dreamdojo_full_episode.sh` | `infer/infer_bimanual_singleview_single_segment.py` |
| 14 | DreamDojo humanoid SV (`dreamdojo_humanoid_sv`) | `run/dreamdojo/run_humanoid_singleview_dreamdojo_full_episode.sh` | `infer/infer_humanoid_singleview_full_episode.py` |

The paper reports Ctrl-World bimanual and humanoid on the 4-view grid (rows 3 and 5).
The 3-view strips (rows 2 and 4) are kept runnable. Checkpoints for every row are
listed in `checkpoints/CHECKPOINTS.md`.

**Non-canonical dense launchers.** These write into the same `sampling_dataset/dense/` tree but use a different entrypoint. Do not use them as the speedup reference:

- `run/dreamgen/run_{bimanual,humanoid}_multiview_dreamgen_dense.sh`: a single-shot `examples.video2world` batch. The canonical launcher is the `_full.sh` roll-out.
- `run/dreamdojo/run_{bimanual,humanoid}_singleview_dreamdojo_dense.sh`: `python -m models.DreamDojo.examples.action_conditioned`. The canonical launcher is `_full_episode.sh`. Only the DreamDojo `sito`/`itm` launchers still use this entrypoint, and those methods are not reported for DreamDojo.

## Coverage (cell x method launcher)

`Y` = launcher exists. `-` = no launcher (not applicable or never implemented).

| cell | dense | dicache | worldcache | fastercache | pisa | radial | svg1 | svg2 | sito | itm |
|---|---|---|---|---|---|---|---|---|---|---|
| CW single_arm MV | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| CW bimanual MV 3-view | Y | Y | Y | Y | - | - | - | - | - | - |
| CW bimanual MV 4-view grid | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| CW humanoid MV 3-view | Y | Y | Y | - | Y | Y | Y | Y | Y | Y |
| CW humanoid MV 4-view grid | Y | Y | Y | - | Y | Y | Y | Y | Y | Y |
| DG single_arm MV (14B) | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| DG bimanual MV | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| DG humanoid MV | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| DG single_arm SV | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| DG bimanual SV | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| DG humanoid SV | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| DD single_arm SV | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| DD bimanual SV | Y | Y | Y | Y | Y | Y | Y | Y | - | - |
| DD humanoid SV | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |

The paper grid counts 117 of 120 (cell, method) pairs:
`bash scripts/tools/tune/_bench_uniform_timing.sh --estimate`.

There are three known gaps:

- FasterCache on `ctrlworld4_humanoid_mv`.
- SiTo and ITM on `dreamdojo_bimanual_sv`.

The CW bimanual 3-view sparse/pruning launchers never existed. That cell is
superseded by the 4-view grid. None of these was archived by the refactor.

Other per-cell variants:

- `run/ctrlworld/run_single_arm_multiview_ctrlworld_pi.sh`: pi0.5 policy-in-the-loop.
- `*_augment.sh`: augment track.
- `run/ctrlworld/run_bimanual_4view_all_methods_full.sh` and `..._4view_grid_all_caches.sh`: method loops.
- `run/dreamdojo/run_{bimanual,humanoid}_singleview_dreamdojo_all_methods.sh`: method loops.
- `run/dreamgen/run_bimanual_multiview_dreamgen_{bsc,bwcache,scalingcache}.sh`: AM_DiT cache methods.

## Environments

| family | venv | notes |
|---|---|---|
| Ctrl-World (`run/ctrlworld`, `train/`, `infer/*ctrlworld*`) | `models/Ctrl-World/.venv` | launchers `cd` to the repo root and set `PYTHONPATH=models/Ctrl-World:.` (the infer scripts import `methods.*`; the import fails without it) |
| DreamGen (`run/dreamgen`) | `models/dreamgen/.venv` | **SVG2 uses `models/dreamgen/.venv_svg`** (flashinfer 0.3.1 block-sparse); the svg2 launchers default to it |
| DreamDojo (`run/dreamdojo`, `infer/infer_*singleview*`) | `models/DreamDojo/.venv` | sparse attention needs flashinfer there; no context parallelism with pisa/radial/svg |
| metrics / tables / tools | `models/dreamgen/.venv` | RIGIS uses `metrics/rigis/.venv`. `scripts/tables/{analyze_geometry_label_correlation,compare_ours_vs_vlm_*,measure_after_expert_pass}.py` also need `seaborn`/`sklearn`, which `models/dreamgen/.venv` does not ship |

Cache, pruning and sparse-attention launchers are single-GPU. Unset `NUM_SHARDS` and
`SHARD_ID` before launching, because they leak through the tmux global env.

## Scratch output for trials and sweeps

Never write sweep or trial output into the canonical `sampling_dataset/` tree.
Point the launcher at a scratch root and delete it afterwards:

| family | output env | input / data env |
|---|---|---|
| Ctrl-World | `OUTPUT_ROOT` | `INPUT_ROOT`, `NUM_EPISODES`, `SPLIT`, `SKIP_PREPARE` |
| DreamGen multiview (`_full.sh` + methods) | `SAVE_ROOT` | `MULTIVIEW_ROOT`, `MAX_EPISODES`, `MAX_SEGMENTS` |
| DreamGen singleview | `OUTPUT_ROOT` | `INPUT_ROOT`, `SPLITS`, `NUM_CHUNKS`, `EPISODE_IDS` (humanoid SV worldcache) |
| DreamGen single_arm MV (14B) | `OUTPUT_ROOT` | `INPUT_DIR`, `CATEGORY`, `MAX_EPISODES` |
| DreamDojo | `SAVE_ROOT` | `SPLITS`, `NUM_EPISODES` |

```bash
CUDA_VISIBLE_DEVICES=0 NUM_EPISODES=2 SPLIT=val \
  OUTPUT_ROOT=/tmp/cw_tune/wc/output INPUT_ROOT=/tmp/cw_tune/wc/input \
  bash scripts/run/ctrlworld/run_humanoid_multiview_ctrlworld_worldcache.sh
```

All launchers are resumable: they skip episodes whose output already exists. To
measure speed, read the per-episode, per-segment or per-chunk generation time from
the log (`wall_time_s`, `Total inference time`, `LVG Total inference time`,
`[Timing] Generated`), not wall clock. To run a matched dense-vs-method timing
pass, use `tools/tune/_bench_uniform_timing.sh --cell <id>`.
