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
  --sito_keep_last_n_dense 2 \
  --sito_patch_h 2 \
  --sito_patch_w 2
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
  --itm_start_layer_idx 6 \
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
  --guidance 1.5 \
  --use_itm \
  --itm_start_layer_idx 4 \
  --itm_keep_last_n_dense 2 \
  --itm_prune_from_step 1 \
  --itm_merge_from_step 2
```

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
