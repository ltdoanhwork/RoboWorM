# Multiview Consistency Metrics using VGGT

## Metrics

- **Camera pose stability** (`pose_stability`, \([0, 1]\), higher is better)
  - Measures how stable **extrinsics + intrinsics** are across time for each view, then aggregates across views.
- **Geometric consistency** (`geometric_consistency`, \([0, 1]\), higher is better)
  - For each frame: predict depth per view, unproject to 3D, re-render depth into each camera, then score the agreement (correlation + residual).
  - Aggregated across frames (default: mean).

## Input format

An episode/video is a folder with one subfolder per view, containing aligned frames named like `frame_0000.png`, `frame_0001.png`, ...

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

The code uses the intersection of frame indices across views.

## Install

[Follow the original VGGT repository to setup the model](https://github.com/facebookresearch/vggt)

## Run (model service + client)

### 1) Start the model service (loads VGGT once)

```bash
python -m multiview_consistency.model_service --host 127.0.0.1 --port 6111 --dtype float16
```

### 2) Compute metrics for one episode/video or a folder containing multiple episodes

An episode
```bash
python -m multiview_consistency.run_video_metrics \
  --episode_dir "/path/to/episode" \
  --views "view1,view2" \
  --stride 1 \
  --host 127.0.0.1 \
  --port 6111
```

A folder
```bash
python -m multiview_consistency.run_video_metrics \
  --episodes_root /path/to/folder \
  --out_csv /path/result.csv \
  --views "view1,view2" \
  --stride 1 \
  --host 127.0.0.1 \
  --port 6111
  --frame_batch_size 4

```


