# Cache Strategy Guide for DreamDojo

`video_gen_physics/README_cache.md` is the authoritative document for DreamDojo cache backends in this workspace.

The legacy snapshot under `world_cache/VLA-Humanoid/Models/DreamDojo` is kept only as historical reference. The primary code path now lives in:

- `video_gen_physics/methods/cache_strategy/WorldCache`
- `video_gen_physics/methods/cache_strategy/DiCache`
- `video_gen_physics/models/DreamDojo`

## Scope

This integration targets the DreamDojo action-conditioned inference path in `video_gen_physics/models/DreamDojo`.

It does not change the generic `cosmos_predict2/inference.py` flow.

Cache backends are treated as inference-only and v1 single-GPU only.

## Backend Rules

The DreamDojo action-conditioned path treats the following acceleration families as mutually exclusive:

- `WorldCache`
- `DiCache`
- `FasterCache`
- `SiTo`
- `SVG`
- `ITM`
- dense-vs-sparse backend overrides such as `NATTEN`

If a cache backend is enabled, DreamDojo will require dense `minimal_a2a` attention at runtime.

## Model Geometry

The recommended cache thresholds should be reasoned from DreamDojo latent token geometry, not from raw pixel frames alone.

For the current DreamDojo GR1 action-conditioned path:

- tokenizer spatial compression = `8`
- latent patch spatial = `2`
- effective spatial token factor = `16`
- temporal compression = `4`

Sequence length is:

```text
S = state_t * (H / 16) * (W / 16)
```

For the common GR1 evaluation setting:

- pixel resolution = `480 x 640`
- pixel frames per chunk = `13`
- latent frames = `state_t = 1 + 12 // 4 = 4`

So:

```text
S = 4 * (480 / 16) * (640 / 16)
  = 4 * 30 * 40
  = 4800 tokens
```

This is the reference geometry used for the starting values below.

## Tuning Defaults

These are recommended starting points derived from:

- DreamDojo latent geometry in this repo
- WorldCache, arXiv `2603.22286`, published March 23, 2026
- DiCache, arXiv `2508.17356`, published August 24, 2025

They should be treated as starting points, not as verified benchmark claims for this workspace. Local checkpoints were not available during this integration pass, so no local speed or quality claim is asserted here.

### WorldCache

Recommended starting values:

- 2B / 7B, `28` blocks:
  - `--worldcache-probe-depth 4`
  - `--worldcache-ret-ratio 0.4`
  - `--worldcache-rel-l1-thresh 0.03`
- 14B, `36` blocks:
  - `--worldcache-probe-depth 6`
  - `--worldcache-ret-ratio 0.45`
  - `--worldcache-rel-l1-thresh 0.026`

Interpretation:

- `probe_depth` should cover enough early blocks to reflect deep-feature drift without paying most of the block stack.
- `ret_ratio` is the unconditional warm-up portion before reuse starts.
- `rel_l1_thresh` should be lower for deeper or more sensitive models.

Optional quality guards:

- `--worldcache-hf-enabled` when high-frequency detail is unstable
- `--worldcache-saliency-enabled` when object-centric motion matters
- `--worldcache-dynamic-decay` when late denoising steps become easier to cache
- `--worldcache-aduc-enabled` only after baseline cache behavior is stable

### DiCache

Recommended starting values:

- 2B / 7B, `28` blocks:
  - `--dicache-probe-depth 2`
  - `--dicache-ret-ratio 0.2`
  - `--dicache-rel-l1-thresh 0.08`
- 14B, `36` blocks:
  - `--dicache-probe-depth 3`
  - `--dicache-ret-ratio 0.22`
  - `--dicache-rel-l1-thresh 0.07`

Interpretation:

- DiCache uses a shallower probe and simpler reuse schedule than WorldCache.
- It is usually the safer first cache baseline if you want minimal tuning.

## Threshold Scaling Rule

For other sequence lengths, start from the `S = 4800` reference and rescale the threshold as:

```text
thresh_scaled = thresh_ref * sqrt(4800 / S)
```

Practical guidance:

- If `S` increases, decrease the threshold.
- If the dataset has stronger motion, decrease the threshold further.
- If `num_latent_conditional_frames = 2`, decrease the threshold further because temporal change is less stationary.
- If quality drifts too much, lower threshold before increasing probe depth.
- If speedup is too weak but quality is stable, raise `ret_ratio` carefully only after threshold is already reasonable.

## Runtime Notes

- Cache `num_steps` must match the denoising step count used by the sampler.
- In the current DreamDojo action-conditioned path, the cache config drives the effective denoising step count for cache-enabled runs.
- `worldcache_parallel_cfg` means batched cond/uncond classifier-free guidance in one forward pass. It is not multi-GPU context parallelism.
- Cache backends are currently blocked when `context_parallel_size > 1`.

## How To Run

Run from the `video_gen_physics` root, consistent with the other method READMEs:

```bash
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/DreamDojo/.venv/bin/activate
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics
export PYTHONPATH="./models/DreamDojo:${PYTHONPATH}"
```

### Baseline

```bash
python -m models.DreamDojo.examples.action_conditioned \
  -o ./output/action_conditioned/cache_baseline \
  --checkpoints-dir ./checkpoints/dreamdojo/2B_GR1_post-train \
  --experiment dreamdojo_2b_480_640_gr1 \
  --save-dir ./output/dreamdojo_cache/baseline \
  --num-frames 49 \
  --num-samples 65 \
  --dataset-path ./checkpoints/dreamdojo/datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot \
  --data-split test \
  --deterministic-uniform-sampling
```

### DiCache

```bash
python -m models.DreamDojo.examples.action_conditioned \
  -o ./output/action_conditioned/cache_dicache \
  --checkpoints-dir ./checkpoints/dreamdojo/2B_GR1_post-train \
  --experiment dreamdojo_2b_480_640_gr1 \
  --save-dir ./output/dreamdojo_cache/dicache \
  --num-frames 49 \
  --num-samples 65 \
  --dataset-path ./checkpoints/dreamdojo/datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot \
  --data-split test \
  --deterministic-uniform-sampling \
  --dicache-enabled \
  --dicache-num-steps 35 \
  --dicache-probe-depth 2 \
  --dicache-ret-ratio 0.2 \
  --dicache-rel-l1-thresh 0.08
```

### WorldCache

```bash
python -m models.DreamDojo.examples.action_conditioned \
  -o ./output/action_conditioned/cache_worldcache \
  --checkpoints-dir ./checkpoints/dreamdojo/2B_GR1_post-train \
  --experiment dreamdojo_2b_480_640_gr1 \
  --save-dir ./output/dreamdojo_cache/worldcache \
  --num-frames 49 \
  --num-samples 65 \
  --dataset-path ./checkpoints/dreamdojo/datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot \
  --data-split test \
  --deterministic-uniform-sampling \
  --worldcache-enabled \
  --worldcache-num-steps 35 \
  --worldcache-probe-depth 4 \
  --worldcache-ret-ratio 0.4 \
  --worldcache-rel-l1-thresh 0.03
```

## Suggested Tuning Order

For `WorldCache`:

1. Fix `num_steps`
2. Start with the recommended `probe_depth`
3. Tune `rel_l1_thresh`
4. Then tune `ret_ratio`
5. Only then try `hf`, `saliency`, `dynamic_decay`, or `aduc`

For `DiCache`:

1. Fix `num_steps`
2. Start with the recommended `probe_depth`
3. Tune `rel_l1_thresh`
4. Tune `ret_ratio` last

## Verification Checklist

When checkpoints become available locally, compare:

- baseline wall time
- cache wall time
- skip ratio from runtime logs
- output PSNR / SSIM / LPIPS artifacts already produced by the DreamDojo eval path

Keep prompt, chunking, checkpoint, dataset split, and seed policy fixed across comparisons.
