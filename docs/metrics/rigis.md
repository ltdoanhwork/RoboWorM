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

> **Check the mosaic layout before extracting.** Ctrl-World `pred_all_views.mp4` is a
> 1×4 **row** (1280×192 on the 4-view cells, 960×192 = 1×3 on single-arm), not a 2×2
> grid. `--layout 2x2` on it, as in the command above, produces 640×96 "views" that each
> hold the top halves of two cameras. The frames already under
> `sampling_dataset/multiview_consistency/frames/*/bimanual/ctrlworld/` have that shape.
> For per-camera crops use `--layout 1x4` (`1x3` on single-arm), and never mix the two in
> one comparison. See [Input format](#input-format) below.

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

---

# Detailed setup reference

The sections below expand on the summary above, with names checked against the code.

## Score definitions, mapped to code

Paper App. B.2. `tau` runs over timesteps `>= 1`; frame 0 is the conditioning frame and
only anchors the pose.

| paper | formula | code |
|---|---|---|
| scale alignment (B.3) | `D = s * D_single`, `s = exp(median(log D_joint - log D_single))` on the central H/3 × W/3 crop | `geometric_consistency.py::align_depth_scale` (scale stays 1.0 if < 10 valid pixels) |
| lift + reproject (B.4–B.5) | unproject, drop flying pixels (relative neighbour-depth jump > 0.1), z-buffer into every camera, covisible ROI | `render_depth_from_point_map`, `_render_depth`, `_fov_overlap` |
| per-view agreement (B.6–B.8) | `s_X = 1 - RMSE_RGB`, `s_D = 1 - (mean abs depth residual / median depth)^2`, `s_v = (s_X + s_D)/2` | `geometric_calculation`, `geometric_consistency_metric` |
| timestep score (B.9) | `S_geo(tau) = prod_v s_v` | `geometric_consistency_metric` (views 0 and 1) |
| disjoint frame | ROI < 2% in either view → `S_geo(tau) = 0`, excluded from pose stability | `video_metrics.py::_process_geometric_frame` |
| `S_geo` (B.10) | mean of `S_geo(tau)` over all `tau >= 1`, disjoint included as 0 | `video_metrics.py::_aggregate_geometric_results` |
| `S_int` (B.11) | `mean_v exp(-sigma_K)`, `k = (fx, fy, cx, cy)` normalised by the anchor | `camera_pose_stability.py::_intrinsic_stability_score` |
| `S_ext` (B.12–B.13) | `(exp(-sigma_R/5) + exp(-sigma_t))/2` on the second view | `_extrinsic_stability_score(E[:, 1])` |
| `S_pose` (B.14) | `(S_ext + S_int)/2` | `compute_pose_stability_metric` |
| `S`, `S_target` (B.2.3) | `(S_geo + S_pose)/2`; `mean(S) / mean(S_real)` per morphology | **not in `metrics/rigis/`** — table time |

Sec. 3.4.2 describes the `s_X`, `s_D` mapping as an "exponential decay"; App. B.2 and the
code use `1 - x` and `1 - x^2`.

## Files

| path | role | upstream? |
|---|---|---|
| `metrics/rigis/run.sh` | launcher: uv interpreter, model service, variant sweep, `EPISODE_DIR` mode | ours |
| `metrics/rigis/extract_frames.py` | mosaic / per-view mp4 → `<episode>/view1, view2` PNGs | ours |
| `metrics/rigis/aggregate.py` | leaderboard from the CSVs (`--intersect` by default) | ours |
| `metrics/rigis/metric.py` | `MultiviewConsistency(ExternalMetric)`, registry name `rigis` (aliases `multiview_consistency`, `mvc`, `multiview`) | ours |
| `metrics/rigis/multiview_consistency/model_service.py` | loads VGGT-Ω once, serves over `multiprocessing.connection` | upstream |
| `metrics/rigis/multiview_consistency/run_video_metrics.py` | client: `--episode_dir` or `--episodes_root` → CSV | upstream |
| `metrics/rigis/multiview_consistency/{video_metrics,geometric_consistency,camera_pose_stability,utils}.py` | the metric | upstream |
| `metrics/rigis/vggt_omega/` | VGGT-Ω model code | upstream |
| `metrics/rigis/pyproject.toml`, `uv.lock` | the uv project (package `vggt`) | upstream / lock ours |

## Environment details

- Own uv project: Python 3.13, torch 2.14.1+cu126, torchvision 0.29.1+cu126, numpy
  1.26.4 in `metrics/rigis/.venv`. Do not run it from a model venv.
- Install uv if needed (`curl -LsSf https://astral.sh/uv/install.sh | sh`), then
  `cd metrics/rigis && uv sync` (about 7 GB). If `~/.cache/uv` is quota-limited, set
  `UV_CACHE_DIR` to a large disk first.
- The pyproject pins the cu126 PyTorch index; re-lock with another index for a different
  CUDA build.
- Check: `(cd metrics/rigis && uv run --project . --with numba python -m multiview_consistency.run_video_metrics --help)`.
- `MVC_PYTHON="<interpreter>"` replaces uv (split on spaces); `MVC_UV_EXTRA_DEPS`
  replaces the injected `numba`.
- `extract_frames.py` and `aggregate.py` need only the standard library plus `ffmpeg`;
  `metric.py` runs from any Python with numpy (e.g. `models/dreamgen/.venv`) and shells
  out to `run.sh`.
- Weights: one file, 4,576,706,117 bytes. Default path
  `checkpoints/metrics/vggt_omega/vggt_omega_1b_512.pt`; legacy fallback
  `checkpoints/vggt_omega/vggt_omega_1b_512.pt`; overrides `MVC_CKPT` or
  `--opt rigis.ckpt=...`. On this cluster it is a symlink to
  `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/thanhnv/VideoWorldModelBenchmark/checkpoint/vggt_omega_1b_512.pt`.
  No DINO / MASt3R / FeatUp weights are used.

## Input format

```
<episodes_root>/<episode_id>/view1/frame_0000.png ...
<episodes_root>/<episode_id>/view2/frame_0000.png ...
```

- Names must match `frame_(\d+)\.png`; only indices present in every view are used
  (`utils.py::intersect_frames`). Frame 0 = pose anchor, excluded from `S_geo`
  (`num_frames` = frames − 1).
- View folder names are free (`VIEWS` / `--views`, default `view1,view2`); only the first
  two are scored. Resolution is free (resized to 512 internally). Cameras are predicted,
  not read.
- `run.sh` finds every directory whose children are episode folders, recursively.

Extraction per cell type (`metrics/rigis/extract_frames.py`):

| rollout file | geometry | `--layout` |
|---|---|---|
| Ctrl-World 4-view `pred_all_views.mp4` (bimanual, humanoid) | 1280×192, 1×4 row of 320×192 (`per_view` order in `metrics.json`, e.g. `cam_high, cam_low, cam_left_wrist, cam_right_wrist`) | `1x4` |
| Ctrl-World single-arm `pred_all_views.mp4` | 960×192, 1×3 row (`exterior_1_left, exterior_2_left, wrist_left`) | `1x3` |
| DreamGen multiview `full_pred.mp4` | 768×432, 2×2 grid, top row = the two static cameras | `2x2` |
| real anchor `sampling_dataset/dense/<emb>/input/multiview/ctrlworld_4view_grid/<split>/episode_*/view_{0,1}.mp4` | one file per view | `per_view --view-globs view_0.mp4,view_1.mp4` |

The scored pair should be two static, overlapping cameras (paper App. E.2). Other flags:
`--view-names` (must equal `VIEWS`), `--flat`, `--limit N`, `--skip-existing`. Keep `--fps`
identical across compared variants (2 fps everywhere so far).

## One episode, end to end

```bash
export PATH=/path/to/static-ffmpeg/bin:$PATH        # if system ffmpeg is missing / broken
python3 metrics/rigis/extract_frames.py \
  --video-path sampling_dataset/dense/bimanual/output/multiview/ctrlworld_4view_grid/non_makovian/episode_fold_blue_towel__000070/pred_all_views.mp4 \
  --out-dir /tmp/rigis/frames/episode_fold_blue_towel__000070 --layout 1x4 --fps 2
CUDA_VISIBLE_DEVICES=0 MVC_PORT=6177 \
EPISODE_DIR=/tmp/rigis/frames/episode_fold_blue_towel__000070 OUT_CSV=/tmp/rigis/one.csv \
  bash metrics/rigis/run.sh
```

Smoke-tested on 2026-10-06 (idle A100-80GB, 66 frames/view at 2 fps): 110 s wall
including the model load; the client printed the JSON result and wrote the CSV row
(`num_frames` 65). Re-run on 2026-10-07 (model cache warm, KIVA sharing the GPU): 52 s
wall, `geometric_consistency` 0.8407, `pose_stability` 0.6806, 1 disjoint frame (ratio
0.0154).

## Whole tree

`run.sh` scores `MVC_DATA_ROOT/<variant>[/SUBTREE]` and writes
`MVC_OUT_ROOT/<relative episodes root>.csv`:

```bash
MAX_EPISODES=1 VARIANTS="dense pisa" SUBTREE=bimanual/ctrlworld_4view_grid/non_makovian \
  bash metrics/rigis/run.sh                                         # smoke, 1 ep per root
CUDA_VISIBLE_DEVICES=0 VARIANTS="real dense pisa radial svg1 svg2" \
  bash metrics/rigis/run.sh --tmux                                  # session mvc_<HHMMSS>
python3 metrics/rigis/aggregate.py --results-root results/multiview_consistency \
  --variants real dense pisa radial svg1 svg2 --baseline real \
  --filter bimanual --json results/multiview_consistency/summary.json
```

| env | default |
|---|---|
| `MVC_ROOT` | `metrics/rigis` |
| `MVC_CKPT` | `checkpoints/metrics/vggt_omega/vggt_omega_1b_512.pt` |
| `MVC_PYTHON` / `MVC_UV_EXTRA_DEPS` | `uv run --project $MVC_ROOT --with numba python` / `numba` |
| `MVC_DATA_ROOT` / `MVC_OUT_ROOT` | `sampling_dataset/multiview_consistency/frames` / `results/multiview_consistency` |
| `VARIANTS`, `SUBTREE`, `VIEWS` | `dense`, —, `view1,view2` |
| `FRAME_BATCH_SIZE`, `MAX_EPISODES` | `4`, `0` (all) |
| `MVC_HOST`, `MVC_PORT`, `MVC_AUTHKEY` | `127.0.0.1`, `6111`, `vggt_omega` |
| `MVC_DTYPE`, `MVC_COMPILE` | `float16`, `0` |
| `MVC_IMAGE_RESOLUTION`, `MVC_PATCH_SIZE` | `512`, `16` (must match the checkpoint) |
| `EPISODE_DIR`, `OUT_CSV` | single-episode mode |

Shard a large tree by `SUBTREE`, one GPU and one `MVC_PORT` per shard.

Through `python -m metrics.evaluate` (stages symlinks for frame dirs and runs
`extract_frames.py` for mp4s):

```bash
models/dreamgen/.venv/bin/python -m metrics.evaluate \
  --pred-root sampling_dataset/dense/bimanual/output/multiview/ctrlworld_4view_grid/non_makovian \
  --pred-glob '*/pred_all_views.mp4' --ref-template '' --text-template '' \
  --metrics rigis --opt rigis.layout=1x4 --opt rigis.fps=2 --out /tmp/rigis/eval
```

`--opt rigis.*` keys: `views`, `frame_batch_size`, `dtype`, `layout`
(`2x2|1x2|1x3|1x4|per_view`, default `2x2`), `fps`, `view_globs`, `ckpt`, `work_dir`,
`log_path`.

## Outputs

CSV columns per episode: `name`, `geometric_consistency` (`S_geo`, ↑), `pose_stability`
(`S_pose`, ↑), `num_disjoint_frames` (↓), `disjoint_frame_ratio` (↓).
`metrics.evaluate` writes `rigis.json`, `per_sample.csv`, `summary.json` with the same
keys. `aggregate.py` prints `n`, mean ± sem and `Δ` vs the baseline per term, plus the
disjoint ratio.

Reading them: report `S = (S_geo + S_pose)/2` normalised by the real anchor of the same
morphology, with `disjoint_frame_ratio` beside it; compare only on the same episodes,
views, fps and extraction layout; `pose_stability` moves little across methods. For an
accelerator, report the per-episode paired `ΔRIGIS` against its own dense rollout with
the claim gate (CI excludes 0, Wilcoxon p < 0.05, n ≥ 8).

## Common errors

| symptom | cause / fix |
|---|---|
| `ERROR: VGGT-Omega checkpoint not found` | symlink the weights or set `MVC_CKPT` |
| `ERROR: uv not found and MVC_PYTHON unset` | install uv or set `MVC_PYTHON` |
| `No module named 'numba'` | modules called without `--with numba`; use `run.sh` |
| `Disk quota exceeded (os error 122)` during `uv sync` | set `UV_CACHE_DIR` to a large disk |
| `model service died before becoming ready` | port in use (change `MVC_PORT`), GPU full, or wrong checkpoint |
| `No valid episode folders found` / `WARN no view1+view2 episodes` | extractor `--view-names` ≠ `VIEWS`, or wrong `SUBTREE` |
| ffmpeg segfault / `no frames written` in a process that imported torch | system ffmpeg vs torch BLAS clash; put a standalone ffmpeg build (`bin/` and `lib/`) first on `PATH` / `LD_LIBRARY_PATH` |
| very low score, two cameras inside one view image | wrong `--layout` for the mosaic (see the warning above) |

Also: scale alignment can silently stay at 1.0 on featureless centre crops; the depth
residual is a mean (an older p90 variant is commented out, old p90 CSVs are not
comparable); the upstream `metrics/rigis/README.md` has paths from the author's machine —
use `run.sh`.

Related: [kiva.md](kiva.md), [README.md](README.md), [../extending/new_embodiment_dataset.md](../extending/new_embodiment_dataset.md).
