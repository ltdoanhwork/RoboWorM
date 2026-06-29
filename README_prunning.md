# Pruning Guide

## SiTo Overview

SiTo is integrated as an inference-only pruning backend for self-attention.

- DreamGen: supported through transformer block-level prune/recover on flattened video tokens.
- DreamDojo: supported through the same block-level path used by DreamGen.
- Ctrl-World: supported through a diffusers `AttnProcessor` on large spatial self-attention layers.

## Consistency Rules

- SiTo is self-attention only in v1. Cross-attention and MLP stay dense.
- SiTo is single-GPU only in v1 for DreamGen and DreamDojo. Do not combine it with context parallelism.
- SiTo is mutually exclusive with `NATTEN`, `PISA`, `Radial`, and `SVG`.
- DreamGen and DreamDojo use `per_frame` spatial grouping.
- Ctrl-World uses `full_2d` spatial grouping.

## Recommended Defaults

- DreamGen 28-block models: `--sito_start_layer_idx 4 --sito_keep_last_n_dense 2`
- DreamGen 36-block models: `--sito_start_layer_idx 6 --sito_keep_last_n_dense 2`
- DreamDojo 2B/7B: `--sito_start_layer_idx 4 --sito_keep_last_n_dense 2`
- DreamDojo 14B: `--sito_start_layer_idx 6 --sito_keep_last_n_dense 2`
- Ctrl-World: `--sito_max_downsample_ratio 4`

Auto prune ratio is chosen from the per-group token count:

- `>= 2048`: `0.45`
- `>= 512`: `0.35`
- `>= 128`: `0.20`
- `< 128`: disabled

## Geometry Notes

For DreamGen and DreamDojo, SiTo should be reasoned on post-patchify tokens, not raw latent cells.

- Example latent grid `(T=4, H=60, W=80)`
- With `patch_spatial=2`, the DiT token grid becomes `(4, 30, 40)`
- Runtime attention sequence length is `4 * 30 * 40 = 4800`, not `19200`

For Ctrl-World, SiTo is applied only on supported spatial sequence lengths:

- `2880 -> 72 x 40`
- `720 -> 36 x 20`
- `180 -> 18 x 10`
- `45` and temporal `11` are skipped

## DreamGen

Base run:

```bash
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1

torchrun -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS}
```

With SiTo:

```bash
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1

torchrun -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS} \
  --use_sito \
  --sito_start_layer_idx 6 \
  --sito_prune_ratio 0.15 \
  --sito_keep_last_n_dense 2 \
  --sito_patch_h 2 \
  --sito_patch_w 2




torchrun -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS} \
  --use_sito \
  --sito_start_layer_idx 18 \
  --sito_keep_last_n_dense 2 \
  --sito_patch_h 2 \
  --sito_patch_w 2


torchrun -m models.dreamgen.examples.video2world \
  --model_size 2B \
  --batch_input_json ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS} \
  --use_sito \
  --sito_start_layer_idx 12 \
  --sito_keep_last_n_dense 2 \
  --sito_prune_ratio 0.4 \
  --sito_patch_h 2 \
  --sito_patch_w 2



torchrun -m models.dreamgen.examples.video2world \
  --model_size 2B \
  --batch_input_json ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS} \
  --use_sito \
  --use_sito \
  --sito_start_layer_idx 16 \
  --sito_keep_last_n_dense 4 \
  --sito_prune_ratio 0.25 \
  --sito_patch_h 2 \
  --sito_patch_w 2 \
  --sito_sim_beta 1.0 \
  --sito_noise_alpha 0.1

torchrun --master_port=29501 -m models.dreamgen.examples.video2world \
  --model_size 2B \
  --batch_input_json ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS} \
  --use_sito \
  --sito_start_layer_idx 20 \
  --sito_keep_last_n_dense 4 \
  --sito_prune_ratio 0.25 \
  --sito_patch_h 2 \
  --sito_patch_w 2 \
  --sito_sim_beta 1.0 \
  --sito_noise_alpha 0.05



```

Text-to-image with SiTo:

```bash
export PYTHONPATH="./models/dreamgen"
python -m models.dreamgen.examples.text2image \
  --model_size 2B \
  --use_sito \
  --sito_start_layer_idx 4 \
  --sito_keep_last_n_dense 2
```

## DreamDojo

Base run:

```bash
python -m models.DreamDojo.examples.action_conditioned \
  -o ./output/dreamdojo_eval_2b \
  --checkpoints-dir ./checkpoints/dreamdojo/2B_GR1_post-train/iter_000050000 \
  --experiment dreamdojo_2b_480_640_gr1 \
  --save-dir ./output/dreamdojo_results/base \
  --num-frames 49 \
  --num-samples 65 \
  --dataset-path "./checkpoints/dreamdojo/datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot" \
  --data-split test \
  --deterministic-uniform-sampling
```

With SiTo:

```bash
python -m models.DreamDojo.examples.action_conditioned \
  -o ./output/dreamdojo_sito_eval \
  --checkpoints-dir ./checkpoints/dreamdojo/2B_GR1_post-train/iter_000050000 \
  --experiment dreamdojo_2b_480_640_gr1 \
  --save-dir ./output/dreamdojo_results/sito \
  --num-frames 49 \
  --num-samples 65 \
  --dataset-path "./checkpoints/dreamdojo/datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot" \
  --data-split test \
  --deterministic-uniform-sampling \
  --use_sito \
  --sito_start_layer_idx 4 \
  --sito_keep_last_n_dense 2
```

## Ctrl-World

Replay trajectory base run:

```bash
export PYTHONPATH="./models/Ctrl-World"
python models/Ctrl-World/scripts/rollout_replay_traj.py \
  --dataset_root_path ./models/Ctrl-World/dataset_example \
  --dataset_meta_info_path ./models/Ctrl-World/dataset_meta_info \
  --dataset_names ./models/Ctrl-World/droid_subset \
  --svd_model_path ./checkpoints/stabilityai/stable-video-diffusion-img2vid \
  --clip_model_path ./checkpoints/openai/clip-vit-base-patch32 \
  --ckpt_path ./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt
```

Interactive rollout with SiTo:

```bash
export PYTHONPATH="./models/Ctrl-World"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python models/Ctrl-World/scripts/rollout_interact_pi.py \
  --dataset_root_path ./models/Ctrl-World/dataset_example \
  --dataset_meta_info_path ./models/Ctrl-World/dataset_meta_info \
  --dataset_names ./models/Ctrl-World/droid_subset \
  --svd_model_path ./checkpoints/stabilityai/stable-video-diffusion-img2vid \
  --clip_model_path ./checkpoints/openai/clip-vit-base-patch32 \
  --ckpt_path ./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt \
  --task_type pickplace \
  --use_sito \
  --sito_max_downsample_ratio 4 \
  --sito_patch_h 2 \
  --sito_patch_w 2
```

Replay trajectory with SiTo:

```bash
export PYTHONPATH="./models/Ctrl-World"
python models/Ctrl-World/scripts/rollout_replay_traj.py \
  --dataset_root_path ./models/Ctrl-World/dataset_example \
  --dataset_meta_info_path ./models/Ctrl-World/dataset_meta_info \
  --dataset_names ./models/Ctrl-World/droid_subset \
  --svd_model_path ./checkpoints/stabilityai/stable-video-diffusion-img2vid \
  --clip_model_path ./checkpoints/openai/clip-vit-base-patch32 \
  --ckpt_path ./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt \
  --use_sito \
  --sito_max_downsample_ratio 4
```

## ITM Overview

Importance Token Merge (ITM) is integrated as an inference-only token-merging backend driven by CFG error heat maps.

- DreamGen: supported through block-level token merge/unmerge on post-patchify video tokens.
- DreamDojo: supported through the same block-level path and per-frame token heat maps.
- Ctrl-World: supported through a diffusers `AttnProcessor` on large spatial self-attention layers.

## ITM Consistency Rules

- ITM is distinct from SiTo and cannot be combined with `SiTo`, `NATTEN`, `PISA`, `Radial`, or `SVG`.
- DreamGen and DreamDojo are single-GPU only for ITM v1. Do not combine ITM with context parallelism.
- ITM needs classifier-free guidance. Use `guidance > 0` for DreamGen and DreamDojo, and `guidance_scale > 1` for Ctrl-World.
- DreamGen and DreamDojo use post-patchify `per_frame` heat maps.
- Ctrl-World uses framewise spatial heat maps on supported spatial self-attention layers only.

## ITM Recommended Defaults

- DreamGen 28-block models: `--itm_start_layer_idx 4 --itm_keep_last_n_dense 2`
- DreamGen 36-block models: `--itm_start_layer_idx 6 --itm_keep_last_n_dense 2`
- DreamDojo 2B/7B: `--itm_start_layer_idx 4 --itm_keep_last_n_dense 2`
- DreamDojo 14B: `--itm_start_layer_idx 6 --itm_keep_last_n_dense 2`
- Ctrl-World: `--itm_max_downsample_ratio 4`

Auto `compress_ratio` is chosen from tokens per group:

- `>= 2048`: `0.35`
- `>= 512`: `0.25`
- `>= 128`: `0.15`
- `< 128`: disabled

## DreamGen ITM Examples

Video-to-world with ITM:

```bash
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1

torchrun -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS} \
  --use_itm \
  --itm_start_layer_idx 18 \
  --itm_keep_last_n_dense 2 \
  --itm_prune_from_step 1 \
  --itm_merge_from_step 2
```

Text-to-image with ITM:

```bash
export PYTHONPATH="./models/dreamgen"
python -m models.dreamgen.examples.text2image \
  --model_size 2B \
  --guidance 4 \
  --use_itm \
  --itm_start_layer_idx 4 \
  --itm_keep_last_n_dense 2
```

## DreamDojo ITM Example

```bash
python -m models.DreamDojo.examples.action_conditioned \
  -o ./output/dreamdojo_itm_eval \
  --checkpoints-dir ./checkpoints/dreamdojo/2B_GR1_post-train/iter_000050000 \
  --experiment dreamdojo_2b_480_640_gr1 \
  --save-dir ./output/dreamdojo_results/itm \
  --num-frames 49 \
  --num-samples 65 \
  --dataset-path "./checkpoints/dreamdojo/datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot" \
  --data-split test \
  --deterministic-uniform-sampling \
  --use_itm \
  --itm_start_layer_idx 4 \
  --itm_keep_last_n_dense 2 \
  --itm_prune_from_step 1 \
  --itm_merge_from_step 2
```
CUDA_VISIBLE_DEVICES=0 python -m models.DreamDojo.examples.action_conditioned \
    -o ./output/grid_v3_itm_4layers_10pct_late_makovian \
    --checkpoints-dir ./checkpoints/dreamdojo/2B_GR1_post-train/iter_000050000 \
    --experiment dreamdojo_2b_480_640_gr1 \
    --save-dir ./tmp_2/grid_search_v2/itm_4layers_10pct_late/makovian \
    --num-frames 49 \
    --num-samples 2 \
    --dataset-path ./datasets/humanoid/singleview/makovian \
    --data-split full \
    --single-base-index \
    --use_itm \
    --itm_start_layer_idx 4 \
    --itm_keep_last_n_dense 2 \
    --itm_prune_from_step 1 \
    --itm_merge_from_step 2

## Ctrl-World ITM Examples

Interactive rollout with ITM:

```bash
export PYTHONPATH="./models/Ctrl-World"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python models/Ctrl-World/scripts/rollout_interact_pi.py \
  --dataset_root_path ./models/Ctrl-World/dataset_example \
  --dataset_meta_info_path ./models/Ctrl-World/dataset_meta_info \
  --dataset_names ./models/Ctrl-World/droid_subset \
  --svd_model_path ./checkpoints/stabilityai/stable-video-diffusion-img2vid \
  --clip_model_path ./checkpoints/openai/clip-vit-base-patch32 \
  --ckpt_path ./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt \
  --task_type pickplace \
  --use_itm \
  --itm_max_downsample_ratio 4 \
  --itm_prune_from_step 1 \
  --itm_merge_from_step 2
```

Replay trajectory with ITM:

```bash
export PYTHONPATH="./models/Ctrl-World"
python models/Ctrl-World/scripts/rollout_replay_traj.py \
  --dataset_root_path ./models/Ctrl-World/dataset_example \
  --dataset_meta_info_path ./models/Ctrl-World/dataset_meta_info \
  --dataset_names ./models/Ctrl-World/droid_subset \
  --svd_model_path ./checkpoints/stabilityai/stable-video-diffusion-img2vid \
  --clip_model_path ./checkpoints/openai/clip-vit-base-patch32 \
  --ckpt_path ./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt \
  --use_itm \
  --itm_max_downsample_ratio 4
```

## ITM Tuning Guide

### Parameter Reference

| Flag | Default | Range | Effect |
|---|---|---|---|
| `--itm_compress_ratio` | auto | `[0, 0.75)` | Fraction of tokens to merge/prune per group. Leave unset to use auto schedule. |
| `--itm_start_layer_idx` | auto (6 for 36-block, 4 for 28-block) | `[0, num_blocks)` | First layer ITM is active. Higher = more dense compute = better quality. |
| `--itm_keep_last_n_dense` | `2` | `[0, ...]` | Always keep this many tail layers dense. Increase for sharper final-step outputs. |
| `--itm_prune_from_step` | `1` | `[0, T)` | Diffusion step index where ITM starts pruning (no merge). Step 0 is the noisiest. |
| `--itm_merge_from_step` | `2` | `[prune, T)` | Step where ITM switches from prune-only to full merge (mean-pool into dst tokens). |
| `--itm_merge_attn` | `true` | flag | Apply merging around self-attention. Always keep on. |
| `--itm_merge_crossattn` | `false` | flag | Apply merging around cross-attention. May hurt text-following. Experimental. |
| `--itm_merge_mlp` | `false` | flag | Apply merging around MLP blocks. Largest VRAM savings, highest quality risk. |

### Token Geometry for DreamGen 14B / 480p / 16fps

Video tokens after patchification (patch_spatial=2):

- Latent grid: `(T=24, H=60, W=80)` → DiT grid: `(T=24, H=30, W=40)`
- Tokens per frame (group): `30 × 40 = 1200`
- Total sequence length: `24 × 1200 = 28800`

Auto compress schedule for `tokens_per_group = 1200` → **0.25** (merge 25% of tokens per frame).

Effective tokens kept per frame with `compress_ratio=0.25`: `1200 × 0.75 = 900`

### Prune vs Merge Phase Explained

ITM operates in two phases within a single diffusion trajectory:

- **Prune phase** (`step < itm_merge_from_step`): tokens are removed without aggregation. Fast but reconstructed positions copy from their nearest dst anchor.
- **Merge phase** (`step >= itm_merge_from_step`): removed tokens are mean-pooled into their anchor dst token before attention. More accurate, slightly slower due to scatter_reduce.

Setting `--itm_prune_from_step 0 --itm_merge_from_step 0` applies full merge from the very first step (max aggression). Setting `--itm_prune_from_step 3 --itm_merge_from_step 5` preserves early denoising steps fully dense (max quality, lower speedup).

### DreamGen 14B Presets

#### Conservative — high quality, moderate speedup

Merging starts late, only 20% tokens reduced, last 4 layers dense:

```bash
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1

torchrun --master_port=29501 -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS} \
  --use_itm \
  --itm_start_layer_idx 20 \
  --itm_keep_last_n_dense 4 \
  --itm_compress_ratio 0.20 \
  --itm_prune_from_step 2 \
  --itm_merge_from_step 4
```

#### Balanced — good quality, solid speedup

```bash
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1

torchrun --master_port=29501 -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS} \
  --use_itm \
  --itm_start_layer_idx 16 \
  --itm_keep_last_n_dense 2 \
  --itm_compress_ratio 0.25 \
  --itm_prune_from_step 1 \
  --itm_merge_from_step 2
```

#### Aggressive — max speedup, acceptable quality for evals

```bash
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1

torchrun --master_port=29501 -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS} \
  --use_itm \
  --itm_start_layer_idx 10 \
  --itm_keep_last_n_dense 2 \
  --itm_compress_ratio 0.30 \
  --itm_prune_from_step 0 \
  --itm_merge_from_step 0
```

### ITM vs SiTo Decision Matrix

| Scenario | Recommendation |
|---|---|
| Benchmark quality reporting | No pruning (base run) |
| Best quality with speedup | ITM Conservative |
| Fast iteration / ablation | ITM Balanced |
| Max throughput, quality acceptable | ITM Aggressive or SiTo Balanced |
| VRAM reduction priority | Add `--itm_merge_mlp` to any ITM preset |
| Multi-GPU inference needed | Neither (SiTo/ITM are single-GPU only) |

## Performance Insight — SiTo can be SLOWER than dense on DreamGen DiT

Measured on `single_arm` multiview DreamGen 14B (`run_single_arm_multiview_dreamgen_sito.sh`),
SiTo did **not** speed up inference and was in fact slightly slower than the dense
baseline. More aggressive pruning made it worse, not better.

### Observed numbers (per DiT forward, single A100-80GB)

| Run | Total Self-Attention | DiT Blocks | Denoise / step | it/s |
|---|---|---|---|---|
| Dense (`--sito` off, auto defaults) | ~1.892s | ~5.30s | ~10.6s | ~10.65 s/it |
| SiTo `prune_ratio=0.5 start_layer=4 keep_last=1` | ~1.893s | ~5.29s | ~10.7s | ~10.75 s/it |

Self-Attention time is essentially identical (1.893 ≈ 1.892s) — token pruning
bought **zero** attention speedup, and the extra bookkeeping made the total slower.

### Root causes

1. **Prune is undone every block.** In `text2image_dit.py` the block does
   `prepare → prune → self_attn → recover` inside a single block, so the
   sequence length returns to full length immediately. The shortened sequence
   only feeds one attention call, never propagating across blocks.

2. **The `transformer_engine` self-attention backend does not benefit from this
   token reduction at this scale.** `DotProductAttention(qkv_format="bshd",
   attn_mask_type="no_mask")` is overhead/launch-bound for the per-frame token
   counts here, so cutting ~50% of tokens per frame does not reduce wall-clock
   attention time.

3. **The timing log hides SiTo cost.** Only `recover()` is inside the measured
   `self_attn` window; `prepare()` + `prune()` (a Python per-frame loop doing
   `F.normalize`, `matmul`, `randn_like`, `argsort`, `nonzero`, `unique`,
   `scatter_`) run *before* the timer. That cost is attributed to "DiT Blocks
   time", not "Total Self-Attention", which is why Self-Attention looks unchanged
   while the total grows.

4. **"Stronger" pruning is slower.** `start_layer_idx=4 keep_last_n_dense=1`
   prunes more blocks than the defaults (`start_layer_idx=6 keep_last_n_dense=2`),
   so it adds more `prepare`/`recover` overhead while attention still does not
   speed up → net slower than even the default SiTo config.

### Takeaways

- SiTo's prune-then-recover-per-block design is a poor fit for DreamGen's
  full-3D self-attention on the `transformer_engine` backend. Do not expect a
  speedup here; expect a small slowdown.
- To actually reduce wall-clock time on DreamGen DiT, prefer:
  - **Cache methods** (DiCache / WorldCache / FasterCache) — cut redundant
    recompute across denoise steps, attacking "Denoise time" directly.
  - **Kernel-level sparse attention** (PISA / Radial / SVG / NATTEN) — change the
    attention kernel itself instead of pruning around it.
- If SiTo must be used, a real fix requires (a) pruning **once and keeping it
  across multiple blocks** (no per-block recover), and (b) a backend that
  actually scales with shorter sequence length. This is a non-trivial code change.
- When benchmarking any acceleration method, compare **total wall-clock** (e.g.
  Denoise/step or it/s), not the "Total Self-Attention" line — that line excludes
  `prepare`/`prune` overhead and can be misleading.
