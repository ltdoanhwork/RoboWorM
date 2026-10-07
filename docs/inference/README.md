# Inference: dense and accelerated rollouts

Every rollout is produced by a shell launcher in `scripts/run/{dreamgen,dreamdojo,ctrlworld}/`.
Launchers are **resumable** (episodes whose prediction already exists are skipped),
discover episodes from `datasets/` / `sampling_dataset/dense/*/input/` (no hard-coded
lists), and write to the [sampling_dataset layout](../datasets/sampling_dataset.md):
`sampling_dataset/{method}/{embodiment}/output/{view}/{model}/{split}/...`.

Naming: `scripts/run/<family>/run_<embodiment>_<view>_<model>[_4view_grid]_<method>.sh`.
An accelerated launcher is an exact mirror of the dense launcher of its cell (same
checkpoint, inputs, sampler, steps, guidance, rollout); it only adds the method flags,
swaps `sampling_dataset/dense/` → `sampling_dataset/<method>/` for the output, and forces
one GPU. Python entrypoints live in `scripts/infer/` and `models/*/examples/`.

## The 12 dense cells (Table C1)

| cell | dense launcher | env | entrypoint |
|---|---|---|---|
| DreamGen single_arm MV (14B) | `scripts/run/dreamgen/run_single_arm_multiview_dreamgen_all.sh` (per split: `..._batch.sh`) | `models/dreamgen/.venv` | `models.dreamgen.examples.video2world_gr00t --gr00t_variant droid` |
| DreamGen bimanual MV (2B grid) | `scripts/run/dreamgen/run_bimanual_multiview_dreamgen_full.sh` | `models/dreamgen/.venv` | `scripts/infer/infer_bimanual_multiview_rollout.py` |
| DreamGen humanoid MV (2B grid) | `scripts/run/dreamgen/run_humanoid_multiview_dreamgen_full.sh` | `models/dreamgen/.venv` | `scripts/infer/infer_bimanual_multiview_rollout.py` (embodiment auto-detected) |
| DreamGen single_arm SV (2B) | `scripts/run/dreamgen/run_single_arm_singleview_dreamgen_static16fps.sh` | `models/dreamgen/.venv` | `models.dreamgen.examples.video2world_gr00t_lvg` |
| DreamGen bimanual SV (2B) | `scripts/run/dreamgen/run_bimanual_singleview_dreamgen_16fps.sh` | `models/dreamgen/.venv` | `models.dreamgen.examples.video2world_gr00t_lvg` |
| DreamGen humanoid SV (2B) | `scripts/run/dreamgen/run_humanoid_singleview_dreamgen_static16fps.sh` | `models/dreamgen/.venv` | `models.dreamgen.examples.video2world_lvg` |
| DreamDojo single_arm SV | `scripts/run/dreamdojo/run_single_arm_singleview_dreamdojo_dense.sh` | `models/DreamDojo/.venv` | `scripts/infer/infer_single_arm_singleview_autoregressive.py` |
| DreamDojo bimanual SV | `scripts/run/dreamdojo/run_bimanual_singleview_dreamdojo_full_episode.sh` | `models/DreamDojo/.venv` | `scripts/infer/infer_bimanual_singleview_single_segment.py` |
| DreamDojo humanoid SV | `scripts/run/dreamdojo/run_humanoid_singleview_dreamdojo_full_episode.sh` | `models/DreamDojo/.venv` | `scripts/infer/infer_humanoid_singleview_full_episode.py` |
| Ctrl-World single_arm MV (3 views) | `scripts/run/ctrlworld/run_single_arm_multiview_ctrlworld.sh` | `models/Ctrl-World/.venv` | `scripts/infer/infer_single_arm_multiview_ctrlworld.py` |
| Ctrl-World bimanual MV (4-view grid) | `scripts/run/ctrlworld/run_bimanual_multiview_ctrlworld_4view_grid.sh` | `models/Ctrl-World/.venv` | `scripts/infer/infer_bimanual_multiview_ctrlworld.py --config bimanual_grid` |
| Ctrl-World humanoid MV (4-view grid) | `scripts/run/ctrlworld/run_humanoid_multiview_ctrlworld_4view_grid.sh` | `models/Ctrl-World/.venv` | `scripts/infer/infer_bimanual_multiview_ctrlworld.py` |

3-view Ctrl-World variants: `run_bimanual_multiview_ctrlworld.sh` (pass
`CKPT=$PWD/checkpoints/finetuned/ctrlworld/bimanual_multiview_5fps/checkpoint-280000.pt`)
and `run_humanoid_multiview_ctrlworld.sh` (`CKPT=.../humanoid_multiview/checkpoint-200000.pt
DATASET_NAME=humanoid_multiview DOWN_SAMPLE=6`). The checkpoint each launcher loads is
listed in [checkpoints/CHECKPOINTS.md](../../checkpoints/CHECKPOINTS.md); the generation
settings per cell (steps, guidance, chunking, fps, resolution) are in
[dense_generation_config.md](dense_generation_config.md).

## Dense examples

```bash
# DreamGen bimanual multiview, all episodes, 1 GPU
CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamgen/run_bimanual_multiview_dreamgen_full.sh

# DreamGen humanoid singleview, makovian split only
CUDA_VISIBLE_DEVICES=0 SPLITS=makovian bash scripts/run/dreamgen/run_humanoid_singleview_dreamgen_static16fps.sh

# DreamDojo bimanual singleview
CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamdojo/run_bimanual_singleview_dreamdojo_full_episode.sh

# Ctrl-World humanoid 4-view grid; 4-GPU sharding then merge summaries
NUM_SHARDS=4 bash scripts/run/ctrlworld/run_humanoid_multiview_ctrlworld_4view_grid.sh --launch-shards
```

Common env overrides (see each launcher's header): `CUDA_VISIBLE_DEVICES`, `CKPT`,
`OUTPUT_ROOT` / `INPUT_ROOT` / `SAVE_ROOT` (redirect to a scratch tree for trials),
`SPLITS` / `SPLIT` / `CATEGORY`, `MAX_EPISODES` / `NUM_EPISODES`, `NUM_SHARDS` / `SHARD_ID`,
`FORCE_REGENERATE`. Run `unset NUM_SHARDS SHARD_ID` first if your shell might inherit them.

## Acceleration methods

| family | method | code | flag prefix / launcher env (defaults differ per cell) |
|---|---|---|---|
| Feature caching | **WorldCache** | `methods/cache_strategy/WorldCache/` | `WORLDCACHE_REL_L1_THRESH`, `WORLDCACHE_RET_RATIO`, `WORLDCACHE_PROBE_DEPTH` |
| | **DiCache** | `methods/cache_strategy/DiCache/` | `DICACHE_REL_L1_THRESH`, `DICACHE_RET_RATIO`, `DICACHE_PROBE_DEPTH` |
| | **FasterCache** | `methods/cache_strategy/FasterCache/` | `FASTERCACHE_MODEL_INTERVAL`, `FASTERCACHE_BLOCK_INTERVAL`, `FASTERCACHE_START_STEP` |
| Sparse attention | **PISA** | `methods/sparse_attention/pisa/` | `PISA_DENSITY`, `PISA_BLOCK_SIZE`, `PISA_START_LAYER_IDX` |
| | **Radial** | `methods/sparse_attention/radial/` | `RADIAL_DECAY_FACTOR`, `RADIAL_START_LAYER_IDX`, `RADIAL_MODEL_TYPE` |
| | **SVG1** (Sparse-VideoGen, flex_attention) | `methods/sparse_attention/svg/` | `SVG_SPARSITY`, `SVG_START_LAYER_IDX`, `SVG_NUM_SAMPLED_ROWS`, `SVG_DENSE_STEP_FRAC` |
| | **SVG2** (k-means + FlashInfer block-sparse; `.venv_svg`) | `methods/sparse_attention/svg/` (+ vendored `Sparse-VideoGen/`) | `SVG_NUM_Q_CENTROIDS`, `SVG_NUM_K_CENTROIDS`, `SVG_TOP_P_KMEANS`, `SVG_START_LAYER_IDX` |
| Token reduction | **SiTo** | `methods/prunning/SiTo/` | `SITO_PRUNE_RATIO`, `SITO_START_LAYER_IDX`, `SITO_GROUP_MODE` (Ctrl-World: `--sito_spatiotemporal_hold`) |
| | **ITM** | `methods/prunning/importance_token_merge/` | `ITM_COMPRESS_RATIO`, `ITM_START_LAYER_IDX`, `ITM_KEEP_LAST_N_DENSE` (Ctrl-World: `ITM_ST_KEEP_RATIO`) |

The defaults baked into each launcher are the tuned per-cell operating points reported
in App. C (`assets/paper/appendix/tables/accel_hyperparams_{cache,sparse,token}.tex`, regenerated
by `assets/scripts/make_impl_config_tables.py`). Not every method exists on every cell
(e.g. no FasterCache on Ctrl-World humanoid 4-view); `docs/datasets/cell_matrix.md` lists
the rollout count per method.

```bash
# one method on one cell (same GPU, same episodes as dense)
CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamgen/run_humanoid_multiview_dreamgen_dicache.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamgen/run_humanoid_multiview_dreamgen_svg2.sh   # uses .venv_svg
CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamdojo/run_humanoid_singleview_dreamdojo_radial.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run/ctrlworld/run_bimanual_multiview_ctrlworld_4view_grid_worldcache.sh

# override an operating point into a scratch tree (never into the canonical tree)
CUDA_VISIBLE_DEVICES=0 MAX_EPISODES=1 SAVE_ROOT=/tmp/trial/pisa_d10 PISA_DENSITY=0.1 \
    bash scripts/run/dreamgen/run_humanoid_multiview_dreamgen_pisa.sh

# all methods of a DreamDojo cell back to back
CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamdojo/run_humanoid_singleview_dreamdojo_all_methods.sh
```

## Measuring speedup

Speedup is **dense time / method time on the same episodes, same GPU, run sequentially**,
using each family's **internal per-segment timer** (never wall clock, which includes
model load, ffmpeg and metric computation):

| family | timer |
|---|---|
| DreamGen | `Total inference time` (multiview) / `LVG Total inference time`, `GR00T LVG Total inference time` (singleview) |
| DreamDojo | `[Timing] Generated` |
| Ctrl-World | per-episode `wall_time_s` in `metrics.json` |

The uniform timing pass used for the paper:

```bash
bash scripts/tools/tune/_bench_uniform_timing.sh --estimate            # GPU-hour estimate only
bash scripts/tools/tune/_bench_uniform_timing.sh --cell dreamdojo_humanoid_sv --gpu 0
bash scripts/tools/tune/_bench_uniform_timing.sh --all --gpu 0         # every cell x method
python3 assets/scripts/merge_timing_shards.py                          # -> results/reports/timing_pass.{json,csv}
models/dreamgen/.venv/bin/python scripts/tables/make_speedup_summary.py \
    --sampling-root sampling_dataset --out-dir results/reports
```

It reports the median per-chunk time with the first chunk dropped as warm-up (k-means /
profiling one-offs), on an idle GPU (it refuses to start if another process is resident).
Cell ids: `{dreamgen,dreamdojo,ctrlworld,ctrlworld4}_{humanoid,bimanual,single_arm}_{mv,sv}`.

Quality against the paired dense rollout (PSNR / SSIM / LPIPS, PhyFid-DR) is computed by
the metrics pipeline, see [../metrics/README.md](../metrics/README.md).
