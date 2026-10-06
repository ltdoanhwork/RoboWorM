# Training-free acceleration methods

RoboWorM audits **9 training-free acceleration methods** against a **dense** baseline on
every (world model, embodiment, view) cell. Each method is applied to the same checkpoint
as dense, with the same inputs, sampler, steps and rollout schedule. Only the method flags
and the output root (`sampling_dataset/<method>/...`) change.

| page | contents |
|---|---|
| [caching.md](caching.md) | WorldCache, DiCache, FasterCache: algorithm, code, flags, knobs, pitfalls |
| [sparse_attention.md](sparse_attention.md) | PISA, Radial, SVG1, SVG2 |
| [pruning.md](pruning.md) | SiTo, ITM (token pruning / merging) |
| [configs.md](configs.md) | full (cell × method) matrix: launcher, effective config, env, output dir, gaps, known discrepancies |
| [../extending/new_acceleration_method.md](../extending/new_acceleration_method.md) | how to add a new method (written separately) |

## Taxonomy

| family | method | what it removes | original paper | code (adapter in this repo) |
|---|---|---|---|---|
| feature caching | **FasterCache** | the CFG-unconditional forward on non-anchor steps (rebuilt from the conditional output with FFT low/high-band deltas) + per-block self-attention outputs (linear extrapolation) | Lv et al., *FasterCache: Training-Free Video Diffusion Model Acceleration with High Quality*, ICLR 2025 | `methods/cache_strategy/FasterCache/` |
| | **DiCache** | whole denoiser forwards: a shallow probe decides when to skip, the skipped output is extrapolated from cached residuals | Bu et al., *DiCache: Let Diffusion Model Determine Its Own Cache*, arXiv 2508.17356 | `methods/cache_strategy/DiCache/` |
| | **WorldCache** | as DiCache, with a motion-adaptive threshold and optional HF / saliency / OSI / flow guards | Nawaz et al., *WorldCache: Content-Aware Caching for Accelerated Video World Models*, ECCV 2026 | `methods/cache_strategy/WorldCache/` |
| sparse attention | **SVG1** (Sparse VideoGen) | self-attention outside a profiled spatial/temporal pattern | Xi et al., *Sparse VideoGen: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity*, arXiv 2502.01776 | `methods/sparse_attention/svg/` (upstream `Sparse-VideoGen/`) |
| | **SVG2** (Sparse VideoGen2) | self-attention outside top-p k-means centroid blocks (semantic-aware permutation) | Yang et al., *Sparse VideoGen2: Accelerate Video Generation with Sparse Attention via Semantic-Aware Permutation*, arXiv 2505.18875 | `methods/sparse_attention/svg/` (upstream `Sparse-VideoGen/`) |
| | **Radial** | self-attention outside a log-decaying radial band mask | Li et al., *Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation*, arXiv 2506.19852 | `methods/sparse_attention/radial/` (upstream `radial-attention/`) |
| | **PISA** | exact attention only on a `density` fraction of blocks, block-wise Taylor approximation elsewhere | Li et al., *PISA: Piecewise Sparse Attention Is Wiser for Efficient Diffusion Transformers*, arXiv 2602.01077 | `methods/sparse_attention/pisa/` (upstream `piecewise-sparse-attention/`) |
| token reduction | **SiTo** | tokens that are similar to an anchor (pruned, then recovered) | Zhang et al., *Training-free and Hardware-friendly Acceleration for Diffusion Models via Similarity-based Token Pruning*, AAAI 2025 | `methods/prunning/SiTo/` |
| | **ITM** | low-importance tokens, merged using a CFG-derived importance map | Wu et al., *Importance-Based Token Merging for Efficient Image and Video Generation*, arXiv 2411.16720 | `methods/prunning/importance_token_merge/` (upstream `importance-token-merge/`) |

BibTeX keys are in `assets/paper/iclr2027_conference.bib`: `lv2025fastercache`,
`bu2026dicache`, `nawaz2026worldcache`, `xi2025sparse`, `yang2025sparse`, `li2025radial`,
`li2026pisa`, `sito`, `itm`.

`methods/cache_strategy/config.py` also registers ScalingCache, BWCache and BSC (AM_DiT
backends, DreamGen bimanual MV only, `run_bimanual_multiview_dreamgen_{scalingcache,bwcache,bsc}.sh`).
They are not among the 9 audited methods and are not covered here.

## How each method hooks into each model family

| | DreamGen (Cosmos-Predict2 DiT, 2B / 14B) | DreamDojo (action-conditioned DiT, 2B) | Ctrl-World (SVD U-Net) |
|---|---|---|---|
| caching | `build_cache_runtime_config()` → `Video2WorldPipeline.from_config(cache_config=…)` → `apply_{worldcache,dicache,fastercache}(pipe.dit)` replaces `dit.forward` (`models/dreamgen/cosmos_predict2/pipelines/video2world.py`); state reset per chunk by `reset_cache_runtime` | infer script builds `{WorldCache,DiCache,FasterCache}Config` → same `apply_*` on the DiT (`models/DreamDojo/cosmos_predict2/_src/predict2/inference/video2world.py`) | `enable_worldcache/enable_dicache` attach a runtime to `unet._ctrl_cache_runtime`; the U-Net forward calls `begin_step / observe_probe / finish_step` (`models/Ctrl-World/models/unet_spatio_temporal_condition.py`). FasterCache installs `SVDFasterCacheAttnProcessor` via `set_attn_processor` |
| sparse attention | per-block `config.net.{pisa,radial,svg}_parameters` (`models/dreamgen/cosmos_predict2/configs/base/config_{pisa,radial,svg}.py`), dispatched to attention ops in `cosmos_predict2/module/a2a_cp.py` | `model.config.net.self_attention_backend={pisa,radial,svg}` overrides → `build_dreamdojo_*_parameters` in `networks/minimal_v4_dit.py`, ops in `networks/a2a_cp.py` | diffusers attention processors (`methods/sparse_attention/*/processor*.py`) installed by `enable_pisa / enable_radial_attention / enable_svg`; only `attn1` self-attention with T ≥ 128 is sparsified |
| token reduction | per-block `config.net.{sito,itm}_parameters` → `SiToTokenPruner` / `ITMTokenMerger` in `cosmos_predict2/models/text2image_dit.py`; ITM heat map from `cond − uncond` in the pipeline | `model.config.net.use_{sito,itm}` overrides via `examples/action_conditioned.py` only | `methods/prunning/SiTo/spatiotemporal_hold.py` (ST-hold, shipped), `SiTo/adapter.py` (per-attention), `importance_token_merge/{adapter,block_hold}.py` |

All backends within a family are **mutually exclusive** (enforced in
`methods/cache_strategy/config.py::build_cache_runtime_config`,
`config_video2world.py::get_cosmos_predict2_video2world_pipeline`,
`models/DreamDojo/cosmos_predict2/cache_runtime.py::validate_runtime_backend_args`,
`methods/cache_strategy/ctrl_world_utils.py::validate_ctrl_world_backend_args`), and all are
**single-GPU only** (no context parallelism). Every method launcher therefore forces one GPU,
even where the dense launcher of the same cell runs 2–4 GPUs.

## Environments

| family | env | notes |
|---|---|---|
| DreamGen | `models/dreamgen/.venv` (torch 2.6, flashinfer 0.2.5) | all methods except SVG2 |
| DreamGen SVG2 | `models/dreamgen/.venv_svg` (flashinfer 0.3.1) | the `*_svg2.sh` launchers set `VENV` to it by default. In `.venv` SVG2 falls back to a dense-mask SDPA path that OOMs on multiview |
| DreamDojo | `models/DreamDojo/.venv` (torch 2.7, flashinfer 0.3.1) | all methods, including Radial and SVG2 |
| Ctrl-World | `models/Ctrl-World/.venv` (torch 2.7.1, flashinfer 0.3.1) | all methods; SVG2 launchers accept a `VENV` override |

Setup: [../setup/README.md](../setup/README.md).

## Run one method on one cell

Launchers are named `scripts/run/<family>/run_<embodiment>_<view>_<model>[_4view_grid]_<method>.sh`.
They are resumable, discover episodes themselves, and default to the tuned per-cell
configuration listed in [configs.md](configs.md).

```bash
unset NUM_SHARDS SHARD_ID                       # these leak via the tmux global env
CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamgen/run_humanoid_multiview_dreamgen_dicache.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamgen/run_humanoid_multiview_dreamgen_svg2.sh      # .venv_svg
CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamdojo/run_humanoid_singleview_dreamdojo_radial.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run/ctrlworld/run_bimanual_multiview_ctrlworld_4view_grid_worldcache.sh
```

Override any knob through its env var and send trials to a scratch tree, never to
`sampling_dataset/`:

```bash
CUDA_VISIBLE_DEVICES=0 MAX_EPISODES=1 SAVE_ROOT=/tmp/trial/pisa_d10 PISA_DENSITY=0.1 \
    bash scripts/run/dreamgen/run_humanoid_multiview_dreamgen_pisa.sh
CUDA_VISIBLE_DEVICES=0 NUM_EPISODES=2 SPLIT=val \
    OUTPUT_ROOT=/tmp/cw/out INPUT_ROOT=/tmp/cw/in WORLDCACHE_REL_L1_THRESH=0.03 \
    bash scripts/run/ctrlworld/run_humanoid_multiview_ctrlworld_4view_grid_worldcache.sh
```

Scratch/episode-cap variables per family are listed in [../../scripts/README.md](../../scripts/README.md)
("Scratch output for trials and sweeps").

## Speedup protocol

Speedup = dense time / method time, **same episodes, same GPU, run sequentially on an idle
device**, using each family's internal timer (never wall clock, which includes model load,
ffmpeg and metric computation):

| family | timer read by the harness |
|---|---|
| Ctrl-World, DreamDojo | `[ChunkTiming] chunk N: X s` (per-chunk median, every `chunk 1` dropped as warm-up); fallbacks: `wall_time_s` in `metrics.json` (Ctrl-World), `[Timing] Generated` (DreamDojo) |
| DreamGen | per-chunk `Total inference time` lines (the aggregate `LVG Total inference time` is excluded), median with the first chunk dropped |

Harness and producers:

```bash
bash scripts/tools/tune/_bench_uniform_timing.sh --estimate             # GPU-hour estimate only
bash scripts/tools/tune/_bench_uniform_timing.sh --cell dreamdojo_humanoid_sv --gpu 0
bash scripts/tools/tune/_bench_uniform_timing.sh --all --gpu 0
bash scripts/tools/tune/_dispatch_timing_cells.sh                       # spread cells over Determined shells
python3 assets/scripts/merge_timing_shards.py                           # -> results/reports/speedup_perchunk.csv
python3 assets/scripts/make_timing_pass_table.py --results /tmp/uniform_timing/results.jsonl
#   -> results/reports/timing_pass.{json,csv}, assets/paper/appendix/tables/timing_pass.tex
models/dreamgen/.venv/bin/python scripts/tables/make_speedup_summary.py \
    --sampling-root sampling_dataset --out-dir results/reports           # speedup_summary_*.csv/.tex
python3 assets/scripts/speedup_table_full.py --only chunk               # per-chunk subset only
python3 assets/scripts/make_impl_config_tables.py                       # App. C accel_hyperparams_*.tex
```

What `_bench_uniform_timing.sh` guarantees:

- One dense control per cell, re-run in the same batch with the same `EPS_PER_CATEGORY`
  episodes (default 1, category `makovian`) and `MAX_CHUNKS` (default 3) as every method.
- `NUM_GPUS=1`, `SEQUENTIAL=1`, output to `SCRATCH` (default `/tmp/uniform_timing`).
- Refuses to run on a busy GPU (≥ 15 % util or ≥ 8 GB used) unless `FORCE_BUSY_GPU=1`; such
  records are flagged `contended`.
- Pins the configuration the released rollouts were produced with: `ITER_NAME` for DreamGen
  singleview, `NUM_INFERENCE_STEPS=50` for Ctrl-World single_arm (its SiTo/ITM launchers
  default to 25 steps), and the DreamDojo checkpoint (`verify_dd_ckpt`).
- Flags `suspect_unit_mismatch` / `suspect_cap_ignored` records; `merge_timing_shards.py` drops them.

Fidelity against the paired dense rollout (PSNR / SSIM / LPIPS) is computed by the metrics
pipeline: [../metrics/README.md](../metrics/README.md). Paper-item → producer mapping:
[../reproduce/README.md](../reproduce/README.md), [../reproduce/main_results_speedup_and_checkpoint.md](../reproduce/main_results_speedup_and_checkpoint.md).

## Related

- Dense launchers and generation settings: [../inference/README.md](../inference/README.md),
  [../inference/dense_generation_config.md](../inference/dense_generation_config.md)
- Rollout counts per (cell, method): [../datasets/cell_matrix.md](../datasets/cell_matrix.md)
- Output layout: [../datasets/sampling_dataset.md](../datasets/sampling_dataset.md)
- Script tree and coverage table: [../../scripts/README.md](../../scripts/README.md)
