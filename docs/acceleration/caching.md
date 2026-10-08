# Feature caching: WorldCache, DiCache, FasterCache

Caching methods skip work *across denoising steps*. They are the only family that can remove
whole denoiser forwards, so they are not bounded by the attention share of the model (see
[sparse_attention.md](sparse_attention.md#amdahl-ceiling)).

Shared code: `methods/cache_strategy/common.py` (config dataclasses, per-chunk reset),
`methods/cache_strategy/config.py` (DreamGen CLI + `build_cache_runtime_config` + mutual
exclusion), `methods/cache_strategy/stats.py` (block-work accounting),
`methods/cache_strategy/ctrl_world_utils.py` (Ctrl-World exclusivity check).

## Step-count convention on the DiTs

DreamGen and DreamDojo run classifier-free guidance **sequentially**: the DiT forward is
called twice per denoising step (cond, then uncond). The DiT runtimes therefore count *net
forwards*: `reset_worldcache_state` / `reset_dicache_state` set the internal step budget to
`2 × num_steps`, keep one cache slot per branch (`cnt % 2`), and are reset at every chunk by
`reset_cache_runtime` (DreamGen `pipelines/video2world.py`; DreamDojo
`inference/video2world.py`, where `_get_effective_num_steps` uses the cache config's
`num_steps`). `--*-num-steps 35` in the launchers must equal the sampler's step count.

Ctrl-World runs CFG off (`guidance_scale = 1.0` in every `models/Ctrl-World/config_*.py`,
50 steps), so its runtimes see one U-Net call per step.

## DiCache

**What it does.** From step `⌊ret_ratio · steps⌋` on, run only the first `probe_depth`
blocks (the probe) and measure the relative L1 change of the probe output against the last
fully computed step. While the accumulated change stays below `rel_l1_thresh`, skip the rest
of the network and reconstruct the output from the cached residuals: `x + r[-2] + γ (r[-1] − r[-2])`,
with γ ∈ [1, 2] estimated from the probe residuals. Otherwise reset the accumulator and finish
the forward (the probe output is reused, not recomputed).

| family | code | entry |
|---|---|---|
| DreamGen, DreamDojo (DiT) | `DiCache/runtime.py::apply_dicache` → `dicache_mini_train_dit_forward` replaces `dit.forward` | `DiCacheConfig` |
| Ctrl-World (U-Net) | `DiCache/adapter.py::enable_dicache` → `DiCache/ctrl_world_runtime.py::CtrlWorldDiCacheRuntime` | `unet._ctrl_cache_runtime` |

On Ctrl-World the probe is the output of the `probe_depth`-th attention-bearing U-Net stage
(down blocks with cross-attention, then the mid block). The skip decision is per step (not
accumulated), and the skipped step returns early from `UNetSpatioTemporalConditionModel.forward`,
which is why `probe_depth = 1` is the fastest setting there.

| knob | DreamGen flag | DreamDojo flag | Ctrl-World flag | launcher env | effect |
|---|---|---|---|---|---|
| enable | `--dicache-enabled` | `--dicache-enabled` | `--use_dicache` | — | |
| threshold | `--dicache-rel-l1-thresh` (0.08) | same (0.08) | `--dicache_rel_l1_thresh` (0.08) | `DICACHE_REL_L1_THRESH` | higher → more skips, faster, lossier |
| warm-up | `--dicache-ret-ratio` (0.2) | same | `--dicache_ret_ratio` (0.2) | `DICACHE_RET_RATIO` | leading fraction of steps always computed |
| probe | `--dicache-probe-depth` (2) | same | `--dicache_probe_depth` (2) | `DICACHE_PROBE_DEPTH` | blocks (DiT) / stages (U-Net) run before the gate; paid on every gated step |
| steps | `--dicache-num-steps` (35) | same | from `--num_inference_steps` | `DICACHE_NUM_STEPS` | must match the sampler |

Values in parentheses are the CLI defaults. Shipped values are in [configs.md](configs.md).

## WorldCache

**What it does.** DiCache's probe-and-extrapolate scheme with a content-aware gate. The
threshold is scaled by input motion, `thresh / (1 + α · Δx)`, where `Δx` is the relative L1
change of the denoiser input and `α = motion_sensitivity`, so fast-moving scenes are cached
less. Optional guards (all off in the shipped launchers unless listed in [configs.md](configs.md)):

| guard | flag (DreamGen / Ctrl-World) | effect |
|---|---|---|
| dynamic decay | `--worldcache-dynamic-decay` / `--worldcache_dynamic_decay` | relaxes the threshold toward late steps; **on** for DreamGen single_arm MV/SV and humanoid SV |
| HF drift | `--worldcache-hf-enabled`, `--worldcache-hf-thresh` | forces a full step when Laplacian high-frequency input drift exceeds the threshold |
| saliency | `--worldcache-saliency-enabled`, `--worldcache-saliency-weight` | weights the probe drift by a channel-variance saliency map |
| OSI | `--worldcache-osi-enabled` | least-squares γ instead of the clipped ratio |
| flow warp | `--worldcache-flow-enabled`, `--worldcache-flow-scale` | warps the cached residual with estimated optical flow (DiT only) |
| AdUC | `--worldcache-aduc-enabled`, `--worldcache-aduc-start` | reuses the previous uncond output late in sampling (DiT only) |
| parallel CFG | `--worldcache-parallel-cfg` | cond+uncond batched (B = 2); not used |

| family | code |
|---|---|
| DreamGen, DreamDojo | `WorldCache/runtime.py::apply_worldcache` → `worldcache_mini_train_dit_forward` |
| Ctrl-World | `WorldCache/adapter.py::enable_worldcache` → `WorldCache/ctrl_world_runtime.py::CtrlWorldWorldCacheRuntime` (subclass of the DiCache runtime) |

Core knobs mirror DiCache: `--worldcache-{rel-l1-thresh,ret-ratio,probe-depth,motion-sensitivity,num-steps}`
(DreamGen/DreamDojo, CLI defaults 0.03 / 0.4 / 4 / 5.0 / 35) and
`--worldcache_{rel_l1_thresh,ret_ratio,probe_depth,motion_sensitivity}` (Ctrl-World,
defaults 0.03 / 0.4 / 3 / 5.0); env `WORLDCACHE_*`. The DreamDojo infer scripts expose only
the five core knobs, not the guards.

## FasterCache

**What it does.** Two independent reuse mechanisms, both starting at `start_step`
(`0` means `⌈0.3 · steps⌉`):

1. **Model level (CFG).** On steps that are not multiples of `model_interval` after
   `start_step`, the unconditional forward is skipped and rebuilt from the current
   conditional output plus the cached low/high-frequency (FFT) delta between the two
   branches, measured on the last anchor step.
2. **Block level.** On steps that are not multiples of `block_interval`, the self-attention
   output of an eligible block is extrapolated from its last two values, `y[-1] + 0.3 (y[-1] − y[-2])`.

| family | code | eligible blocks |
|---|---|---|
| DreamGen, DreamDojo | `FasterCache/runtime.py::apply_fastercache(cfg_mode="sequential")` wraps `dit.forward` and each `block.self_attn.forward`; history kept on CPU | blocks `[4, N−2)` for 28-block DiTs (`[6, N−2)` for ≥ 36), sequence length ≥ 2048 |
| Ctrl-World | `FasterCache/adapter.py::enable_fastercache` installs `FasterCache/processor.py::SVDFasterCacheAttnProcessor`; the CFG shortcut lives in `models/Ctrl-World/models/pipeline_ctrl_world.py` | self-attention layers after the first 2, **only at sequence lengths {180, 720, 2880}** |

| knob | DreamGen / DreamDojo flag | Ctrl-World flag | env | effect |
|---|---|---|---|---|
| enable | `--fastercache-enabled` | `--use_fastercache` | — | |
| start | `--fastercache-start-step` (0) | `--fastercache_start_step` (0) | `FASTERCACHE_START_STEP` | first step eligible for reuse |
| model interval | `--fastercache-model-interval` (5) | `--fastercache_model_interval` (5) | `FASTERCACHE_MODEL_INTERVAL` | larger → more uncond skips |
| block interval | `--fastercache-block-interval` (3) | `--fastercache_block_interval` (3) | `FASTERCACHE_BLOCK_INTERVAL` | larger → more attention reuse |
| debug | `--fastercache-debug` | `--fastercache_debug` | `FASTERCACHE_DEBUG` | per-step logs (on in some DreamGen launchers; no effect on output) |

DreamGen also exposes a legacy v1 path (`--use_fastercache` + underscore flags, applied to
`config.net.fastercache_config`). The launchers use the v2 `--fastercache-enabled` path; the
14B single_arm MV launcher passes the underscore spellings of the interval flags, which set
the same argparse destinations.

## Pitfalls

- **FasterCache on Ctrl-World is close to a no-op.** CFG is off, so the model-level shortcut
  never fires; only block-level attention reuse remains. On the 4-view grid the self-attention
  lengths are 3840 / 960 / 240, none of which is in `ELIGIBLE_SEQ_LENS`, so nothing is reused
  at all. Ctrl-World FasterCache is not in the released coverage.
- **FasterCache can be slower than dense** when the FFT delta is recomputed too often
  (DreamGen bimanual MV/SV ship `model_interval = block_interval = 2`, ≈ 0.94× / 0.76×), and it
  corrupts quickly when the intervals are pushed (DreamGen humanoid MV: `block_interval ≤ 3`
  or `model_interval ≥ 8`).
- **The probe is not free.** A gated step that is not skipped costs `num_blocks + probe_depth`
  block evaluations, i.e. more than dense. Low thresholds can therefore be net-negative
  (DreamGen humanoid MV: DiCache `rel_l1_thresh = 0.10` is ≈ 0.8×).
- **`ret_ratio` is the cliff.** Lowering warm-up without tightening the threshold degrades
  quality first (DreamGen humanoid SV WorldCache collapses between `ret_ratio` 0.15 and 0.10).
  On the Ctrl-World 4-view grid, by contrast, `ret_ratio = 0` and `probe_depth = 1` are what
  make caching fast; the shared base-script defaults (`0.4 / 2–3`, tuned on the removed 3-view
  checkpoints) are both slower and lossier there.
- **Never stack backends.** `build_cache_runtime_config` raises if two cache selectors are set;
  sparse/token backends are rejected together with caches on DreamDojo and Ctrl-World.
- Cache backends are single-GPU only on every family (`--num_gpus 1` / `context_parallel_size 1`).
