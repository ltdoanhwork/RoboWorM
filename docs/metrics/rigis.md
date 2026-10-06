# RIGIS — Robotic Inter-view Geometry & Intrinsic-extrinsic Stability (↑)

**Ours** (paper Sec. 3.4.2, App. B.2). Multi-view consistency of a rollout from a fixed
camera rig, reference-free, in [0, 1]. Code: `metrics/rigis/` — the VGGT-Ω metric code
vendored from branch `thanhnv/multiview-consistency-indep` @8015d97
(`metrics/rigis/VENDORED.md`). Backbone: a modified VGGT-Ω (Wang et al., CVPR 2026).

## What it computes

Per timestep, VGGT-Ω predicts per-view depth (independent and joint heads, aligned by a
per-view median log-ratio on the centre crop) and a camera per view.

* **S_geo (geometric consistency).** Unproject each view's depth to a world point map,
  drop flying pixels, rasterise the fused cloud into every camera (z-buffer) and compare on
  the co-visible region: RGB RMSE and relative depth residual give a per-view score
  `s = ½(s_X + s_D)`; the timestep score is the product over views. A timestep with < 2 %
  mutual overlap is *disjoint* and scores 0.
* **S_pose (camera pose stability).** The rig is fixed, so intrinsics and the relative
  extrinsics must not drift: `S_int = mean_v exp(−σ_K)`, `S_ext` from the std of the
  relative rotation angle and translation of the non-primary views; `S_pose = ½(S_int + S_ext)`
  (0 if fewer than two non-disjoint frames).
* **RIGIS = ½(S_geo + S_pose)** per rollout. The benchmark score of a morphology is the mean
  divided by the mean of its real-video subset (raw real-video level ≈ 0.89).

## Setup

```bash
(cd metrics/rigis && uv sync)          # builds metrics/rigis/.venv (run.sh does it if missing)
mkdir -p checkpoints/metrics/vggt_omega
ln -sfn <path-to>/vggt_omega_1b_512.pt checkpoints/metrics/vggt_omega/vggt_omega_1b_512.pt
```

## Checkpoints

One file: `vggt_omega_1b_512.pt` (VGGT-Ω 1B, 512 px, patch 16; 4.26 GiB), released by
<https://github.com/facebookresearch/vggt-omega>. Default
`checkpoints/metrics/vggt_omega/vggt_omega_1b_512.pt`, override `MVC_CKPT`. It is loaded
strictly, so `--omega_image_resolution 512 --omega_patch_size 16` must match.

## Running

The metric reads **extracted frames**: `<episode>/view1/frame_0000.png`, `<episode>/view2/...`.

```bash
# mosaic mp4 -> per-view frames (once per variant)
python metrics/rigis/extract_frames.py \
  --videos-root sampling_dataset/pisa/bimanual/output/multiview/ctrlworld_4view_grid \
  --out-root    sampling_dataset/multiview_consistency/frames/pisa/bimanual/ctrlworld \
  --video-name pred_all_views.mp4 --layout 2x2 --fps 2 --skip-existing
# real-video reference (score it as one more variant, "real")
python metrics/rigis/extract_frames.py \
  --videos-root sampling_dataset/dense/bimanual/input/multiview/ctrlworld_4view_grid \
  --out-root    sampling_dataset/multiview_consistency/frames/real/bimanual/ctrlworld \
  --layout per_view --view-globs view_0.mp4,view_1.mp4 --fps 2 --skip-existing

# one cell through the registry
models/dreamgen/.venv/bin/python -m metrics.evaluate --pred-root <frames>/<task> --pred-glob '*' \
    --ref-template '' --text-template '' --metrics rigis --out results/metrics/<run>

# full sweep (one model-service load) + leaderboard vs real
VARIANTS="real dense pisa radial svg1 svg2" bash metrics/rigis/run.sh --tmux
python metrics/rigis/aggregate.py --results-root results/multiview_consistency \
  --variants real dense pisa radial svg1 svg2 --baseline real
```

Env of `run.sh`: `MVC_ROOT MVC_CKPT MVC_PYTHON MVC_DATA_ROOT MVC_OUT_ROOT VARIANTS SUBTREE
VIEWS FRAME_BATCH_SIZE MAX_EPISODES MVC_PORT MVC_DTYPE MVC_COMPILE EPISODE_DIR`. Cost:
~10 s per 65-frame episode on an A100 plus ~25 s model load.

## Known gotchas

* `numba` is an undeclared dependency of the vendored code; `run.sh` adds `--with numba`.
* Only the **first two views** are scored, and extrinsic stability uses view 2 only.
* Disjoint timesteps score 0, not NaN — read `num_disjoint_frames` in the CSV.
* fp16 autocast + a geometry thread pool: small run-to-run variation is possible.
* Compare variants only at the same extraction fps.
* The model service is single-tenant: concurrent sweeps need different `MVC_PORT`.

The pre-refactor process notes and smoke-test numbers are archived at
`tmp/20261005_refactor/misc/docs_metrics/multiview_consistency_metric.md`.
