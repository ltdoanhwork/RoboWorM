# Evaluating a new acceleration method (path C)

Paper App. E.3. The cheapest path: acceleration is training-free, so a new method reuses
the released episodes, conditioning **and** checkpoints, and only touches inference. It
is benchmarked per cell against that cell's dense rollouts, at its quality–speed knee,
and placed on the speed–quality plane beside the nine existing methods.

The nine existing methods, their hooks per model family, flags and the full
(cell × method) config matrix are documented in [../acceleration/](../acceleration/README.md);
this page is the procedure for adding a tenth.

## 1. Declare the family and the environment

| family | code | existing methods | speedup bounded by | family doc |
|---|---|---|---|---|
| step / feature caching | `methods/cache_strategy/` | `WorldCache/`, `DiCache/`, `FasterCache/` | number of skipped network evaluations (not Amdahl-bounded per step) | [caching.md](../acceleration/caching.md) |
| sparse attention | `methods/sparse_attention/` | `pisa/`, `radial/`, `svg/` (SVG1 + SVG2; upstream `Sparse-VideoGen/`) | self-attention share of runtime | [sparse_attention.md](../acceleration/sparse_attention.md) |
| token reduction | `methods/prunning/` | `SiTo/`, `importance_token_merge/` (ITM) | share of the pruned blocks | [pruning.md](../acceleration/pruning.md) |

Profile before sweeping a sparse-attention or token-reduction method: self-attention is
≈27% of generation on the DreamGen multi-view DiT, 15–18% on the DreamDojo single-view
DiTs and ≈7% (spatial) on the Ctrl-World U-Net, i.e. Amdahl ceilings of about 1.36×,
1.18–1.22× and 1.08× even with a free kernel (App. E.3). Existing profilers:
`DD_ATTN_PROFILE=1` (DreamDojo), `_install_attn_profiler` in
`scripts/infer/infer_single_arm_multiview_ctrlworld.py`, and
`scripts/tools/diag/_diag_dreamdojo_svg.py` for kernel-level timing.

Declare the execution environment. Methods run in the model's own venv; SVG2 on DreamGen
needs `models/dreamgen/.venv_svg` (flashinfer 0.3.1 variable-block-sparse wrapper),
otherwise it falls back to a dense mask and runs out of memory on long multi-view
sequences.

## 2. Package the method (the pattern every existing method follows)

```
methods/<family>/<YourMethod>/
  config.py        add_<yourmethod>_args(parser)   # one flag group
  adapter.py       enable_<yourmethod>(net_or_unet, **params)   # patch modules in place
  runtime.py       per-step state / apply_<yourmethod>(net, config)   # optional
```

Real examples to copy:

| hook | example |
|---|---|
| flag group | `methods/cache_strategy/WorldCache/config.py::add_worldcache_args` (`--use_worldcache`, `--worldcache_rel_l1_thresh`, ...); `methods/sparse_attention/pisa/config.py::add_pisa_args` (`--use_pisa`, `--pisa_density`, ...); `methods/prunning/SiTo/config.py::add_sito_args` / `add_dreamgen_sito_args` |
| enable on a U-Net | `methods/cache_strategy/DiCache/adapter.py::enable_dicache(unet, num_steps=..., rel_l1_thresh=..., ret_ratio=..., probe_depth=...)`, `methods/sparse_attention/pisa/adapter.py::enable_pisa`, `methods/prunning/SiTo/adapter.py::enable_sito` |
| frozen runtime config (DreamGen caches) | dataclasses in `methods/cache_strategy/common.py` (`WorldCacheConfig`, `DiCacheConfig`, `FasterCacheConfig`, ...), built by `methods/cache_strategy/config.py::build_cache_runtime_config(args)` |
| apply to a DiT | `methods/cache_strategy/WorldCache/runtime.py::apply_worldcache(net, config)` called from `Video2WorldPipeline._apply_cache_backend` (`models/dreamgen/cosmos_predict2/pipelines/video2world.py`) |
| config-driven DiT backend (DreamDojo) | per-block parameter lists such as `methods/sparse_attention/pisa/config.py::build_dreamdojo_pisa_parameters`; `scripts/infer/infer_humanoid_singleview_full_episode.py::build_experiment_opts` turns flags into `model.config.net.self_attention_backend=<backend>` overrides |

Minimal flag group + adapter skeleton (names are yours; the shape matches the existing
modules):

```python
# methods/cache_strategy/MyCache/config.py
def add_mycache_args(parser) -> None:
    parser.add_argument("--use_mycache", action="store_true",
                        help="Enable MyCache (training-free, inference-only).")
    parser.add_argument("--mycache_thresh", type=float, default=0.05,
                        help="Primary quality/speed knob.")
    parser.add_argument("--mycache_warmup_ratio", type=float, default=0.3,
                        help="Fraction of initial steps always computed (warm-up).")

# methods/cache_strategy/MyCache/adapter.py
def enable_mycache(unet, *, num_steps: int, thresh: float, warmup_ratio: float):
    """Patch unet.forward in place; keep all per-step state on the module."""
    ...
    return unet
```

## 3. Wire it into the inference scripts (flags only, no model edits)

Each inference entry point registers every method's flag group and enables at most one:

| family | entry point | registration | enable |
|---|---|---|---|
| Ctrl-World | `scripts/infer/infer_single_arm_multiview_ctrlworld.py`, `infer_bimanual_multiview_ctrlworld.py` | `_import_cache_module("WorldCache", "config").add_worldcache_args(parser)`, ... `add_pisa_args`, `add_svg_args`, `add_radial_args`, `add_sito_args`, `add_itm_args` | `if getattr(args, "use_<m>", False): ...enable_<m>(agent.model.unet, ...)` after `CtrlWorldAgent(...)` |
| DreamGen | `models/dreamgen/examples/video2world_lvg.py`, `video2world_gr00t_lvg.py`, `scripts/infer/infer_bimanual_multiview_rollout.py` | `add_dreamgen_cache_args`, `add_pisa_args`, `add_dreamgen_svg_args`, `add_dreamgen_radial_args`, `add_dreamgen_sito_args`, `add_dreamgen_itm_args` | caches: `build_cache_runtime_config(args)` → `Video2WorldPipeline.from_config(..., cache_config=...)`; sparse / pruning: kwargs of `get_cosmos_predict2_video2world_pipeline(...)` |
| DreamDojo | `scripts/infer/infer_humanoid_singleview_full_episode.py` and siblings | inline `--<m>-enabled` / `--use_<m>` flags in `parse_args()` | `build_cache_config_from_args(args)`; sparse via `build_experiment_opts(args)` |

Then extend the single mutual-exclusion check: `CTRL_WORLD_BACKEND_FLAGS` in
`methods/cache_strategy/ctrl_world_utils.py` (Ctrl-World) or `CACHE_SELECTOR_FLAGS` in
`methods/cache_strategy/config.py` (DreamGen caches). Two methods patching the same
network at once give a number attributable to neither.

Implementation constraints that already exist: token pruning is DiT-oriented (on the
Ctrl-World U-Net it needs the spatio-temporal hold variant to exceed ~1.1×); caches and
sparse attention have no context parallelism (launchers are single-GPU).

## 4. One launcher per cell, mirroring dense

For every cell, copy the cell's **dense** launcher (table "Canonical dense launcher per
cell" in [../../scripts/README.md](../../scripts/README.md)) to
`scripts/run/<family>/run_<embodiment>_<view>_<model>[_4view_grid]_<yourmethod>.sh` and
change only:

1. the method flags (with env-overridable defaults, e.g. `MYCACHE_THRESH="${MYCACHE_THRESH:-0.05}"`);
2. the leading `sampling_dataset/dense/` of the **output** root → `sampling_dataset/<yourmethod>/`
   (inputs stay under `dense/`).

Same checkpoint, dataset, entrypoint, step count and paths as dense. `diff` the two files
to check (`diff scripts/run/ctrlworld/run_single_arm_multiview_ctrlworld.sh scripts/run/ctrlworld/run_single_arm_multiview_ctrlworld_dicache.sh`
is the reference shape). Keep the launcher resumable and env-overridable for
`OUTPUT_ROOT` / `SAVE_ROOT` and the episode cap. Add a row per cell to the
(cell × method) launcher / config / env / output matrix in
[../acceleration/configs.md](../acceleration/configs.md).

Add the method name to the `{method}` set of the layout
([../datasets/sampling_dataset.md](../datasets/sampling_dataset.md)) and to method lists
of the producers (grep for an existing name, e.g. `"dicache"` in
`scripts/tables/make_speedup_summary.py::METHODS`, `scripts/metrics/compute_dr_metrics.py`,
`assets/scripts/dump_speedup_figure_data.py`).

## 5. Tune to the knee, per cell

- Vary the primary knob with warm-up parameters fixed; score each point with PSNR, SSIM,
  LPIPS against the dense rollout of the **same** episode
  (`scripts/metrics/_score_vs_dense.py`, or `python -m metrics.evaluate --metrics fidelity --ref-root <dense tree>`),
  and inspect the final frame (per-view smear and block artefacts hide in aggregate PSNR;
  past the collapse point reference metrics stop ranking).
- Ship the most aggressive setting that is still indistinguishable from dense; publish the
  curve including the degraded points.
- **Per cell**, never transfer: median backbone disagreement in retention is 0.102 and
  20 / 45 pairs land on opposite sides of Q = 0.95.
- Sweeps go to a scratch tree (`OUTPUT_ROOT=/tmp/...` / `SAVE_ROOT=/tmp/...`) and are
  deleted; never into `sampling_dataset/<method>/`.

## 6. Time it against a matched dense control

- Same cell, same episodes, same pinned config; ideally the **identical command line with
  the method flags dropped** (the DreamGen humanoid SV WorldCache launcher has
  `WORLDCACHE_ENABLED=0` for this).
- One idle A100-80GB per run, strictly sequential, no context parallelism; read the
  family's log line, not wall clock: Ctrl-World `wall_time_s`, DreamGen
  `Total inference time` / `LVG Total inference time`, DreamDojo `[Timing] Generated`.
- Sample both regimes; report the first segment / chunk separately from the
  steady-state ratio (warm-up costs such as attention profiling or k-means).
- Harness: `scripts/tools/tune/_bench_uniform_timing.sh` (`--estimate`, `--cell <id>`,
  `--all --gpu N`); it pins per-cell env (`ITER_NAME`, `NUM_INFERENCE_STEPS`) so the
  control and candidate load the same checkpoint and step count. Add your method to its
  method loop (`for method in dense sito itm worldcache ...`) and to its per-method cost
  multipliers used by `--estimate`.

## 7. Report quality as paired deltas

Per cell: speedup, `ΔKIVA` (percentage points of invalid segments, single-view cells),
`ΔRIGIS` (multi-view cells), paired deltas of the reference-free columns (Visual
Integrity = CLIP-IQA+, Text Alignment = CLIPScore), PSNR vs dense, FVD; each with valid
`n`, exclusion rate and the claim gate. The aggregate retention `Q` (App. C.2) may order
methods but is not an effect size. Keep slower-than-dense results; mark runnable but
unreleased cells as not benchmarked (`--`).

## Checklist (paper App. E.4, items xx–xxv)

- [ ] Family declared; execution env declared (own venv vs `.venv_svg` etc.).
- [ ] `methods/<family>/<YourMethod>/{config.py,adapter.py[,runtime.py]}` with `add_<m>_args` + enable/apply hook.
- [ ] Registered in every inference entry point you support; mutual-exclusion list updated.
- [ ] One launcher per cell, `diff` vs dense = method flags + output root only.
- [ ] Per-cell sweep in scratch, knee shipped, full curve (incl. degraded points) recorded.
- [ ] Matched dense control (same command, flags dropped), idle single GPU, family timing line, both regimes, first-unit vs steady-state.
- [ ] Paired deltas with `ΔKIVA` / `ΔRIGIS` / reference-free columns, `n`, exclusion rate, claim gate.
- [ ] Negative speedups kept; "not benchmarked" distinct from "measured and lost".
