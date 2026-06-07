# Video World - Video Generation Physics Benchmark

## Project Overview
Benchmark framework for evaluating video world models on robotics datasets. Compares video generation quality across different models (Ctrl-World, DreamDojo/DreamGen, Video2World) on various robot embodiments (single-arm, bimanual, humanoid) with different camera setups (singleview, multiview).

## Directory Structure
```
datasets/                  # Source datasets in LeRoBot format (parquet + mp4 videos)
  {embodiment}/{view_type}/{makovian,non_makovian}/
    data/chunk-XXX/episode_XXXXXX.parquet
    videos/chunk-XXX/{view_name}/episode_XXXXXX.mp4
    meta/{info.json, episodes.jsonl, ...}

sampling_dataset/dense/    # Benchmark I/O
  {embodiment}/input/      # GT inputs (video, actions, metadata)
  {embodiment}/output/     # Model predictions + metrics

models/                    # Model implementations
  Ctrl-World/              # SVD-based action-conditioned world model
  DreamDojo/               # Cosmos-based video2world model

scripts/                   # Inference & preparation scripts
checkpoints/               # Model weights
```

## Dataset Conventions
- **makovian**: Episodes where next state depends primarily on current state + action
- **non_makovian**: Episodes with longer temporal dependencies
- All datasets use LeRoBot format: parquet for states/actions, mp4 for videos
- Original fps: 15 (single_arm DROID), 10 (humanoid GR1)
- Video resolutions: 320x180 (DROID), 640x480 (GR1)

## Ctrl-World Specifics
- Operates at **5fps** (downsample factor 3 from 15fps DROID data)
- Video resolution: **320x192** (resized from 180→192 height)
- Input format: annotation JSON with `states` (7D: cartesian 6D + gripper), `joints` (8D: joint 7D + gripper), `videos` (3 views), `texts`
- Generation: autoregressive, pred_step=5 frames per iteration, overlap 1 frame
- Output coverage: ~95-100% of input episode length
- Stat normalization: uses `dataset_meta_info/droid_subset/stat.json`

## DreamDojo/DreamGen Specifics
- Operates at native dataset fps
- Uses `MultiVideoActionDataset` — always access data via `dataset[data_id]`, never access `lerobot_ds` directly
- Generation: chunked autoregressive, chunk_size=12, GT-anchored segments
- Resolution: matches training config (480x640 for humanoid, varies for arms)

## Inference Script Pattern
Each model+dataset combo has:
1. `prepare_*.py` — Convert LeRoBot dataset → model-specific format
2. `infer_*.py` — Run generation, save input/output/metrics
3. `run_*.sh` — Shell wrapper with env setup + paths

### Output Structure Per Episode
```
input/{model}/{subset}/episode_{id}/
  full_gt.mp4              # GT video (primary view)
  view_0.mp4, view_1.mp4   # All GT views
  metadata.json            # instruction, states, joints, fps, num_frames

output/{model}/{subset}/episode_{id}/
  full_pred.mp4            # Generated video
  full_merged.mp4          # GT | Pred side-by-side
  metrics.json             # PSNR, SSIM, LPIPS per episode
```

## Key Rules
- When adding a new model, follow the existing `prepare → infer → run` pattern
- Always compute `interact_num` or segment count per-episode to match GT length
- Save both input and output for reproducible benchmarking
- Use GPU 1 by default (`CUDA_VISIBLE_DEVICES=1`)
- Metrics: PSNR, SSIM, LPIPS (computed per-episode, summarized across dataset)
- Never hardcode episode lists — discover from annotation/video directories
