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

The code uses the intersection of frame indices across views.

## Fixed-camera merge (calibrate once from real)

For a fixed 2-view rig, estimate cameras **once** from the real reference video, then merge any pair of single-view depths with that frozen setup:

```bash
# 1) Calibrate from real frames (run once)
python -m multiview_consistency.fixed_camera_merge calibrate \
  --real_dir /path/to/frames/real \
  --out /path/to/cameras.npz \
  --checkpoint /path/to/vggt_omega_1b_512.pt \
  --image_resolution 512 --stride 2

# 2) Merge a generated (or real) frame pair
python -m multiview_consistency.fixed_camera_merge merge \
  --episode_dir /path/to/frames/bad \
  --cameras /path/to/cameras.npz \
  --frame_id 0 \
  --out_dir /path/to/output \
  --checkpoint /path/to/vggt_omega_1b_512.pt
```

Outputs: `cameras.npz`, `merged_frame_XXXX.ply` / `.npz`. Each single depth is scaled onto the joint depth of the same view (`s_i = median(d_joint_i / d_single_i)` on high-confidence pixels), then unprojected with frozen cameras.

## Install

[Follow the original VGGT  repository to setup the model](https://github.com/facebookresearch/vggt)
[or VGGT-Omega](https://github.com/facebookresearch/vggt-omega)

## Run (model service + client)

### 1) Start the model service (loads VGGT once)

Using VGGT-Omega
```bash
python -m multiview_consistency.model_service --host 127.0.0.1 --port 6111 --backend vggt_omega --dtype float16 --omega_checkpoint "path/to/checkpoint/vggt_omega_1b_512.pt" --omega_image_resolution 512 --omega_patch_size 16
```
or using VGGT
```bash
python -m multiview_consistency.model_service --host 127.0.0.1 --port 6111 --dtype float16
```

### 2) Compute metrics for one episode/video or a folder containing multiple episodes

An episode (a folder)
```bash
python -m multiview_consistency.run_video_metrics \
  --episode_dir "/path/to/episode" \
  --views "view1,view2" \
  --stride 1 \
  --host 127.0.0.1 \
  --port 6111
```

A list of episodes (a folder contains multiple episode folders inside)
```bash
python -m multiview_consistency.run_video_metrics \
  --episodes_root /path/to/folder \
  --out_csv /path/result.csv \
  --views "view1,view2" \
  --stride 1 \
  --host 127.0.0.1 \
  --port 6111

```


