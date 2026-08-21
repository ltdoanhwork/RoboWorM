# Multiview Consistency Metrics using VGGT

## Metrics

- **Camera pose stability** (`pose_stability`, \([0, 1]\), higher is better)
  - Measures how stable **extrinsics + intrinsics** are across time for each view, then aggregates across views.
- **Geometric consistency** (`geometric_consistency`, \([0, 1]\), higher is better)
  - For each frame: predict depth per view, unproject to unified 3D pointcloud, re-render depth into each camera within the covisible region, then score the agreement.
  - Aggregated across steps.

## Input format

An episode/video is a folder with one subfolder per view, containing frames named like `frame_0000.png`, `frame_0001.png`, ...

Example:

```text
/path/to/episode_000003_generated/
  view1/
    frame_0000.png
    frame_0001.png
    ...
  view2/
    frame_0000.png
    frame_0001.png
    ...
```

## Install

[Follow the original VGGT-Omega repository to download 1B the checkpoint](https://github.com/facebookresearch/vggt-omega)
## Run (model service + client)

### 1) Start the model service

```bash
python -m multiview_consistency.model_service --host 127.0.0.1 --port 6111 --dtype float16 --omega_checkpoint "path/to/checkpoint/vggt_omega_1b_512.pt" --omega_image_resolution 512 --omega_patch_size 16
```
* `--compile` *(default)*: Enable `torch.compile` on the aggregator for lower steady-state latency after the first few runs. Best for batch evaluation of many episodes on a long-lived service.
* `--no-compile`: Run the model eagerly (no `torch.compile`). Avoids first-run compile overhead; better for single-episode execution.
### 2) Compute metrics for one episode/video or a folder containing multiple episodes

An episode (a folder)
```bash
python -m multiview_consistency.run_video_metrics \
  --episode_dir "/path/to/episode" \
  --views "view1,view2" \
  --host 127.0.0.1 \
  --port 6111
```

A list of episodes (a folder contains multiple episode folders inside)
```bash
python -m multiview_consistency.run_video_metrics \
  --episodes_root /path/to/folder \
  --out_csv /path/result.csv \
  --views "view1,view2" \
  --host 127.0.0.1 \
  --port 6111

```


