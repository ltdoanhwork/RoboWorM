# Multi-view consistency metric (RIGIS; VGGT-Omega rigid-geometry variant)

> **2026-10-05 refactor:** the metric code is now **vendored** into `metrics/rigis` (`git archive 8015d97`, see `metrics/rigis/VENDORED.md`); the old `external/multiview-consistency` git worktree was removed after a parity test. Registry name `rigis` (old name `multiview_consistency` kept as an alias). Historical sections below that mention a worktree describe the pre-refactor setup; the code is byte-identical.

How to measure whether our generated multi-view robot videos are *geometrically
consistent across views*, and how to turn that into a one-command SOTA
comparison across our method variants.

**TL;DR**

```bash
# 1. frames (once per variant)
python metrics/rigis/extract_frames.py \
  --videos-root sampling_dataset/pisa/bimanual/output/multiview/ctrlworld_4view_grid \
  --out-root    sampling_dataset/multiview_consistency/frames/pisa/bimanual/ctrlworld \
  --video-name pred_all_views.mp4 --layout 2x2 --fps 2

# 2. score every variant in one invocation
VARIANTS="dense pisa radial svg1 svg2" bash metrics/rigis/run.sh --tmux

# 3. leaderboard
python metrics/rigis/aggregate.py \
  --results-root results/multiview_consistency \
  --variants dense pisa radial svg1 svg2
```

Both reported numbers are in `[0, 1]` and **higher is better**.

---

## 1. Which metric is this, and why it is the right one

### 1.1 It is NOT MEt3R

The starting point for this work was
`/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/thanhnv/VideoWorldModelBenchmark/test_met3r.py`,
which calls the pip package `met3r` (MEt3R: *Measuring Multi-View Consistency in
Generated Images*, CVPR 2025) with `backbone="mast3r"`, `feature_backbone="dino16"`,
`upsampler="featup"`, `distance="cosine"`.

That file is a **standalone 2-image probe only**:

- it hard-codes two PNG paths (`test_met3r.py:9-11`),
- it is not wired into any video/episode loop, any CSV writer, or any launcher,
- `met3r` is **not installed** anywhere on this cluster
  (`python -c "import met3r"` → `ModuleNotFoundError`; `pip show met3r` → not
  found; `find / -maxdepth 8 -name met3r -type d` → **zero hits**),
- the only checkpoint on disk (`checkpoint/vggt_omega_1b_512.pt`) is **not** a
  MASt3R/DUSt3R/FeatUp checkpoint — MEt3R would have to download MASt3R + FeatUp
  from the hub at first use, which has never happened here.

So MEt3R is a dead-end exploratory script. The metric that is actually
**implemented, batched, CSV-emitting, and already used to score 654 result
files** is the VGGT-Omega multi-view consistency metric below.

### 1.2 What we actually use

Branch `origin/thanhnv/multiview-consistency-indep` of this repo contains a
self-contained package `multiview_consistency/` + a vendored `vggt_omega/`
backbone. It reports **two** scores per episode:

| Key | Range | Direction | What it measures |
|---|---|---|---|
| `geometric_consistency` | [0, 1] | ↑ better | Do the two views describe **the same rigid 3D scene**? Lift each view's depth to a point cloud, re-project (rigid-body + pinhole) into the *other* camera, and score depth + RGB agreement in the covisible region. |
| `pose_stability` | [0, 1] | ↑ better | Does the **camera rig stay rigid over time**? Extrinsic rotation/translation and intrinsics of each view should not drift or jitter across frames of a generated video. |

Why this is the right metric for our question: a video world model that
hallucinates geometry will produce two views that cannot be explained by one
rigid 3D scene seen from two fixed cameras. Both failure modes (scene disagreement
and camera drift) are exactly what a 3D-reconstruction foundation model exposes,
and neither is captured by per-view metrics (PSNR/FVD/CLIP) or by single-view
temporal metrics.

### 1.3 Which variant is the "rigid" one — and our choice

> The request was "lấy **rigis** method cho multiview" (typo). There is **no
> literal `rigid` / `--method rigid` flag anywhere in the code**
> (`git grep -i "rigid\|procrustes\|umeyama\|kabsch\|icp"` over both metric
> branches returns only unrelated `num_register_tokens` hits).

The real, documented option set is:

| # | Option | Where | Nature |
|---|---|---|---|
| A | **MEt3R** (MASt3R + DINO16/FeatUp features, cosine distance) | `VideoWorldModelBenchmark/test_met3r.py` | **Feature-space**, soft. Warps *features* and compares with cosine similarity. Tolerant to texture/appearance change; does not test rigid geometry directly. |
| B | **VGGT-Omega `geometric_consistency`** | `multiview_consistency/geometric_consistency.py` | **Explicit rigid 3D geometry.** Unproject depth → world points, apply the rigid extrinsic `[R\|t]` + pinhole `K`, z-buffer render, compare depth and RGB. |
| C | **VGGT-Omega `pose_stability`** | `multiview_consistency/camera_pose_stability.py` | **Rigid-body camera pose** stability over time (rotation angle σ, translation σ, intrinsics σ). |
| D | *Fixed-camera merge* (`fixed_camera_merge calibrate\|merge`) | documented in the README of `origin/thanhnv/multiview-consistency` | A **frozen/rigid camera rig** utility: calibrate cameras once from the real video, reuse for all generations. **Not present as a module in either branch's tree** — the README documents it but `git ls-tree` shows no `fixed_camera_merge.py`. Unusable today. |
| E | Backbone choice `--backend vggt \| vggt_omega` | only on `origin/thanhnv/multiview-consistency` (`model_service.py:28`); removed on `-indep` (vggt_omega only) | Model选择, not a metric variant. |
| F | Depth source: **joint** (`preds["depth"]`) vs **independent** (`preds["depth_single"]`, scale-aligned) | `-indep` branch uses independent; see §3.1 | The `-indep` suffix of the branch name. |

**Our choice: (B) + (C), i.e. the VGGT-Omega metric on branch
`origin/thanhnv/multiview-consistency-indep`, with `geometric_consistency` as the
headline number.** This is the "rigid" method in the only meaningful sense
available: it is the variant that enforces a *rigid 3D scene + rigid camera
transform*, as opposed to MEt3R's feature-space warping. It is also the only
option that is runnable, batched, and already has 654 result CSVs on disk.

Trade-offs of this choice:

- **Pro**: measures geometry, not appearance; two outputs that separate "scene
  disagrees" from "camera drifts"; no camera calibration needed (inferred);
  already validated on our data.
- **Pro**: `-indep` (independent per-view depth) is the *harder* test — see §3.1.
- **Con**: not a published metric, so there are **no external SOTA numbers to
  compare against** (see §7). MEt3R would give published baselines but is not set up.
- **Con**: scores only the **first two views** (`view0`/`view1` are hard-coded in
  `geometric_consistency.py:439-441` and `camera_pose_stability.py:63-66`), even
  if you pass more.

---

## 2. Exact metric definition, with citations

All citations are to the metric checkout, i.e.
`metrics/rigis/...` == `origin/thanhnv/multiview-consistency-indep`.

### 2.1 Per-frame pipeline

For each frame index `t > 0` of an episode, the two views at that timestep are
fed to VGGT-Omega as a 2-frame set (`video_metrics.py:168-194`):

```
preds = model(images)            # images: (B, S=2, 3, H, W), H=W=512
pose_enc    -> extrinsic (B,2,3,4), intrinsic (B,2,3,3)   # encoding_to_camera
depth       -> joint depth       (B,2,H,W,1)
depth_single-> independent depth (B,2,H,W,1)              # each view alone
```

**Step 1 — scale-align the independent depths** (`geometric_consistency.py:6-62`).
For each `(b, s)` a single global scale is fitted on a center crop (H/3 × W/3)
using the median log-ratio against the joint depth, over pixels where both depths
are in `(1e-4, max_depth=1.0)` and finite:

$$ s_{b,s} = \exp\!\big(\mathrm{median}\,[\log d^{\text{joint}} - \log d^{\text{indep}}]\big),\qquad
   \tilde d^{\text{indep}}_{b,s} = s_{b,s}\, d^{\text{indep}}_{b,s} $$

(`geometric_consistency.py:56-60`; called from `video_metrics.py:377`.) Only the
*scale* is fitted — no shift, no per-pixel warp. This is what makes the
comparison rigid: a model is not allowed to pass by locally deforming depth.

**Step 2 — unproject to one world point cloud** (`video_metrics.py:379-381`):
`unproject_depth_map_to_point_map(d̃, E, K)` → `point_map` of shape `(S, H, W, 3)`
in world coordinates, using the rigid extrinsic `E = [R|t]` and pinhole `K`.

**Step 3 — rigid re-render into every camera** (`geometric_consistency.py:223-307`,
entry `render_depth_from_point_map` at `:310-341`). Flying/edge pixels are first
discarded (`_detect_flying_pixels`, relative depth discontinuity ≥ 10%,
`:78-89`). Then all surviving points from *both* views are projected into each
camera `s`:

$$ P_c = E_s \begin{bmatrix}P_w\\1\end{bmatrix},\qquad
   (u, v) = \Big(\tfrac{(K_s P_c)_0}{(K_s P_c)_2},\ \tfrac{(K_s P_c)_1}{(K_s P_c)_2}\Big),\qquad z = (P_c)_2 $$

Points with `z <= 0.01` or out of frame are dropped; the rest go through a numba
z-buffer where nearest-z wins and carries its RGB (`_zbuffer_scatter_numba`,
`:92-123`). Output: `depth_rendered (B,S,H,W)` and `rgb_rendered (B,S,H,W,3)`.

**Step 4 — covisibility mask** (`_fov_overlap`, `:166-221`). A pixel of view `s`
is covisible if *either* direction holds: view-`s`'s own point projects inside
another camera's frustum (`own → other`), or some other view's point lands on
that pixel (`other → own`, scattered with `_scatter_hits_numba`).

**Step 5 — per-view residuals** (`geometric_calculation`, `:343-417`). Inside the
ROI (covisible **and** both depths finite and `> eps = 1e-6`):

$$ \text{residual}_s = \mathrm{mean}\big|\,\tilde d_s - d^{\text{rendered}}_s\,\big|,\qquad
   \text{scale\_ref}_s = \mathrm{median}(\tilde d_s),\qquad
   \text{rgb\_rmse}_s = \sqrt{\mathrm{mean}\big(I_s - I^{\text{rendered}}_s\big)^2} $$

(`:388-394`; note `residual` is a **mean**, the p90 variant is commented out at
`:389`.) `valid_ratio` = fraction of pixels in the ROI (`:399`).

**Step 6 — per-view score and the frame score**
(`geometric_consistency_metric`, `:419-441`):

$$ s^{\text{resid}}_s = \max\!\Big(0,\ 1 - \big(\tfrac{\text{residual}_s}{\text{scale\_ref}_s}\big)^2\Big),
\qquad s^{\text{rgb}}_s = \max(0,\ 1 - \text{rgb\_rmse}_s) $$

$$ \text{score}_s = \tfrac{1}{2}\big(s^{\text{resid}}_s + s^{\text{rgb}}_s\big),
\qquad \boxed{\;\text{GC}(t) = \mathrm{clip}_{[0,1]}\big(\text{score}_0 \cdot \text{score}_1\big)\;} $$

The **product** over the two views (`:441`) is deliberate: it is a logical AND —
one bad view sinks the frame.

**Step 7 — disjoint-frame gate** (`video_metrics.py:398-404`). If either view's
`valid_ratio < 0.02` (less than 2% covisible), the frame is marked `disjoint` and
**scored 0.0**, not skipped.

### 2.2 Reduction over frames

`_aggregate_geometric_results` (`video_metrics.py:413-433`):

$$ \text{geometric\_consistency} = \frac{1}{|T|}\sum_{t \in T} \text{GC}(t),\qquad
   T = \{t : t > 0,\ t \in \text{frames common to all views}\} $$

Plain mean over **all** scored frames, disjoint zeros included (`:415` selects
every frame; the non-disjoint-only variant is commented out at `:414`). Frame `0`
is excluded from geometry (`video_metrics.py:449`) but still contributes its pose
to pose-stability.

Companion counters in the CSV: `num_disjoint_frames`, `disjoint_frame_ratio`,
`num_frames`.

### 2.3 Pose stability

`camera_pose_stability.py:46-67`. Over the T frames that were **not** disjoint
(`video_metrics.py:323-324` only buffers non-disjoint poses):

Extrinsics of view 1 (`:66`) — rotation is referenced to frame 0 and reduced to
an angle:

$$ \sigma_R = \mathrm{std}_t\,\angle\!\big(R_t R_0^{\top}\big)\ [\text{deg}],\qquad
   \sigma_t = \mathrm{mean}_{\text{xyz}}\ \mathrm{std}_t\, t_t $$
$$ S_E = \tfrac{1}{2}\big(e^{-\sigma_R/5} + e^{-\sigma_t/1}\big) \quad (\text{\tt :15-29}) $$

Intrinsics, averaged over both views (`:31-44`, `:63-65`), on
`(f_x, f_y, c_x, c_y)` normalised by their frame-0 values:

$$ S_K^{(v)} = \exp\!\Big(-\mathrm{mean}_{k}\ \mathrm{std}_t \tfrac{k_t}{k_0}\Big),
   \qquad S_K = \tfrac{1}{2}\big(S_K^{(0)} + S_K^{(1)}\big) $$

$$ \boxed{\;\text{pose\_stability} = \tfrac{1}{2}\big(S_E + S_K\big)\;} \quad (\text{\tt :67}) $$

### 2.4 Episode output

`compute_video_metrics` returns (`video_metrics.py:516-525`) and
`run_video_metrics.py:88-101` writes these CSV columns:

```
name,geometric_consistency,pose_stability,num_disjoint_frames,disjoint_frame_ratio
```

> Note: `thanhnv`'s *uncommitted* working copy at
> `VideoWorldModelBenchmark/video_gen_physics/` adds two more columns
> (`num_frames,time_seconds`), so the 654 CSVs already on disk under
> `VideoWorldModelBenchmark/output/` have 7 columns. Our
> `metrics/rigis/aggregate.py` accepts both schemas.

---

## 3. Inputs

### 3.1 `-indep` vs the non-indep branch

| | `origin/thanhnv/multiview-consistency` | `origin/thanhnv/multiview-consistency-indep` ← **we use this** |
|---|---|---|
| head | `f7f99b1 Refactor` | `8015d97 Update` (5 commits ahead of `f7f99b1`) |
| point cloud from | joint depth `preds["depth"]` | **independent** depth `preds["depth_single"]`, scale-aligned to joint |
| backbones | `--backend vggt \| vggt_omega` | vggt_omega only |
| `torch.compile` | no | `--compile` / `--no-compile` |
| extra tree | also vendors plain `vggt/` (15 files) | `vggt_omega/` only |

Why `-indep` is the right pick: `preds["depth"]` is produced by VGGT's *joint*
cross-view attention, so the two views' depths are already reconciled by the
reconstruction model — scoring them would partly measure VGGT, not our generator.
`preds["depth_single"]` is each view estimated **alone**; only a global scale is
then matched (§2.1 Step 1). If the two views are mutually inconsistent, nothing
hides it.

### 3.2 On-disk input format

The metric reads **PNG frames, never mp4**:

```
<episodes_root>/
  <episode_id>/
    view1/frame_0000.png frame_0001.png ...
    view2/frame_0000.png frame_0001.png ...
```

- View folder names are free-form, passed via `--views "view1,view2"`
  (real-data runs use e.g. `observation.images.exterior_1_left,...`).
  Exactly the **first two** are scored (§1.3).
- Filenames must match `frame_(\d+)\.png` (`utils.py:60-64`). The run uses the
  **intersection** of frame indices across views (`utils.py:67-76`).
- Resolution is free: `load_and_preprocess_images(..., image_resolution=512,
  patch_size=16)` resizes/pads internally. Our bimanual grid crops are
  `640 × 96` and work fine.
- Batch discovery: `discover_episode_dirs` (`utils.py:79-103`) accepts any
  directory whose immediate children are episode folders containing all views.
- **Camera poses and intrinsics are NOT required.** They are *inferred* per frame
  by VGGT-Omega's camera head (`video_metrics.py:193`). No calibration file,
  no GT poses.

### 3.3 Getting from mp4 to that layout

Use `metrics/rigis/extract_frames.py` (adapted from
`VideoWorldModelBenchmark/frame_extractor_one.py`). It supports mosaic videos
(`--layout 2x2|1x2|1x3|1x4`, crops view1/view2 out of the grid) and
per-view mp4s (`--layout per_view --view-globs view_0.mp4,view_1.mp4`).

Verified equivalence: running it on
`sampling_dataset/svg1/bimanual/output/multiview/ctrlworld_4view_grid/non_makovian/episode_fold_blue_towel__000070/pred_all_views.mp4`
with `--layout 2x2 --fps 2` produced **66 frames/view at 640 × 96**, byte-for-byte
the same shape as `thanhnv`'s pre-extracted
`VideoWorldModelBenchmark/data/preprocessing/svg1/.../view1/`.

> **fps matters for comparability.** `--fps 2` is the setting used for all
> already-extracted data. Scoring one variant at 2 fps and another at 5 fps is
> not a fair comparison (different inter-frame baselines → different covisibility).

---

## 4. Checkpoints

Exactly **one** checkpoint is needed.

| File | Size | Model | Source |
|---|---|---|---|
| `vggt_omega_1b_512.pt` | 4 576 706 117 B (4.26 GiB) | VGGT-Omega 1B, 512 px input, patch 16 | Released by the upstream VGGT-Omega repo. The branch README points to `https://github.com/facebookresearch/vggt-omega`. |

On-cluster copy (the only one that exists today):

```
/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/thanhnv/VideoWorldModelBenchmark/checkpoint/vggt_omega_1b_512.pt
```

Our scripts default to `checkpoints/metrics/vggt_omega/vggt_omega_1b_512.pt` (which is
gitignored via `/checkpoints`). Set it up once:

```bash
mkdir -p checkpoints/metrics/vggt_omega
ln -sfn /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/thanhnv/VideoWorldModelBenchmark/checkpoint/vggt_omega_1b_512.pt \
        checkpoints/metrics/vggt_omega/vggt_omega_1b_512.pt
```

Override anywhere with `MVC_CKPT=/path/to/vggt_omega_1b_512.pt`. **No path under
another user's tree is hard-coded in any committed script.**

It is loaded with a strict `load_state_dict` into `VGGTOmega(patch_size=16)`
(`video_metrics.py:96-98`), so `--omega_patch_size` / `--omega_image_resolution`
must match the checkpoint (16 / 512).

No DUSt3R, MASt3R, FeatUp, or DINO checkpoint is used or needed.

---

## 5. Running it

### 5.1 One-time setup

```bash
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics

# code is vendored in metrics/rigis/ (VENDORED.md); only the venv may need building
(cd metrics/rigis && uv sync)        # ~4 min, builds numpy 1.26.4 -> metrics/rigis/.venv

mkdir -p checkpoints/metrics/vggt_omega && ln -sfn <path-to>/vggt_omega_1b_512.pt checkpoints/metrics/vggt_omega/
```

`metrics/rigis/run.sh` runs `uv run --project metrics/rigis`, which creates the venv if missing.

### 5.2 Extract frames per variant

```bash
for V in dense pisa radial svg1 svg2; do
  python metrics/rigis/extract_frames.py \
    --videos-root sampling_dataset/$V/bimanual/output/multiview/ctrlworld_4view_grid \
    --out-root    sampling_dataset/multiview_consistency/frames/$V/bimanual/ctrlworld \
    --video-name pred_all_views.mp4 --layout 2x2 --fps 2 --skip-existing
done
```

Already-extracted frames exist for 8 variants under
`/pfss/.../thanhnv/VideoWorldModelBenchmark/data/preprocessing/<variant>/` — you
can symlink those into `sampling_dataset/multiview_consistency/frames/` instead
of re-extracting (that is what the smoke test below did).

### 5.3 Score — the single-command SOTA comparison

```bash
# all variants, one model-service load, backgrounded in tmux
CUDA_VISIBLE_DEVICES=0 VARIANTS="dense pisa radial svg1 svg2" \
  bash metrics/rigis/run.sh --tmux
# tmux attach -t mvc_<HHMMSS>   |   log: logs/exp_logs/tmux_logs/mvc_<HHMMSS>.log

python metrics/rigis/aggregate.py \
  --results-root results/multiview_consistency \
  --variants dense pisa radial svg1 svg2 \
  --baseline dense \
  --json results/multiview_consistency/summary.json
```

Other modes:

```bash
# smoke: 1 episode per episodes-root, restricted subtree
MAX_EPISODES=1 VARIANTS="dense pisa" SUBTREE=bimanual/ctrlworld/non_makovian \
  bash metrics/rigis/run.sh

# one explicit episode folder
EPISODE_DIR=/tmp/mvc_frames/episode_000070 OUT_CSV=/tmp/one.csv \
  bash metrics/rigis/run.sh
```

### 5.3b Score the real-GT reference (do this first)

The scores have no meaning without a ceiling (§7). Extract the **real** input
views and score them as if they were another variant:

```bash
python metrics/rigis/extract_frames.py \
  --videos-root sampling_dataset/dense/bimanual/input/multiview/ctrlworld_4view_grid \
  --out-root    sampling_dataset/multiview_consistency/frames/real/bimanual/ctrlworld \
  --layout per_view --view-globs view_0.mp4,view_1.mp4 --fps 2 --skip-existing

VARIANTS="real dense pisa radial svg1 svg2" bash metrics/rigis/run.sh --tmux
python metrics/rigis/aggregate.py \
  --results-root results/multiview_consistency \
  --variants real dense pisa radial svg1 svg2 --baseline real
```

With `--baseline real`, the `Δ` column directly reads as "how far below real".

Full env-var list is in the header of `metrics/rigis/run.sh`
(`MVC_ROOT MVC_CKPT MVC_PYTHON MVC_DATA_ROOT MVC_OUT_ROOT VARIANTS SUBTREE VIEWS
FRAME_BATCH_SIZE MAX_EPISODES MVC_PORT MVC_DTYPE MVC_COMPILE EPISODE_DIR`).

### 5.4 Cost

Measured on one A100-80GB that was already ~98% busy with another job:
**~10-11 s per 65-frame episode** at `FRAME_BATCH_SIZE=4`, plus ~25 s one-time
model load. 1 610 episodes per variant ⇒ **≈ 5 h per variant**, ≈ 25 h for five
variants on one GPU. Shard by `SUBTREE` across GPUs/shells to parallelise (see
the multi-GPU shells guide). Use `MVC_COMPILE=1` only for long sweeps.

---

## 6. Smoke-test results (actually measured)

Date: 2026-10-01. GPU: one A100-SXM4-80GB (shared, ~98% util from another job).
Metric checkout: `metrics/rigis` @ `8015d97`.
Interpreter: `uv run --project metrics/rigis --with numba python`
(torch 2.14.1+cu126, numpy 1.26.4, numba 0.68.0).

### 6.1 Variant comparison, 4 variants × 2 episodes (pre-extracted frames, 2 fps)

Subtree `bimanual/ctrlworld/non_makovian/handover_clear_zip_bag_upright`,
episodes `000004` + `000025`, 65 scored frames each, `view1,view2` @ 640×96,
`FRAME_BATCH_SIZE=4`. Produced by a single invocation
(`MAX_EPISODES=2 VARIANTS='dense pisa radial svg1' SUBTREE=... bash metrics/rigis/run.sh`)
then aggregated with `--baseline dense`:

```
variant          n  geom_cons    ±sem        Δ    pose_stab    ±sem        Δ   disjoint
---------------------------------------------------------------------------------------
svg1             2     0.8153  0.0393  +0.0617       0.7194  0.0017  +0.0027     0.0462
dense *          2     0.7536  0.0970  +0.0000       0.7166  0.0003  +0.0000     0.1077
radial           2     0.6425  0.1820  -0.1111       0.7157  0.0040  -0.0009     0.2308
pisa             2     0.5931  0.1492  -0.1606       0.7158  0.0034  -0.0009     0.2846
```

Total wall time for all 4 variants: **2 min 4 s** (one model load, ~10 s/episode).
CSVs under `results/multiview_consistency/<variant>/bimanual/ctrlworld/non_makovian/handover_clear_zip_bag_upright.csv`;
JSON at `results/multiview_consistency/smoke_summary.json`.

**These are n=2 numbers with ±sem up to 0.18. They are a plumbing proof, NOT a
result — do not quote them.** Note `svg1` above `dense` is well inside the noise.
What they *do* establish: the metric is discriminative, it spreads variants over a
useful range, and `geometric_consistency` tracks `disjoint_frame_ratio` inversely
(0.046 → 0.285 as the score falls 0.82 → 0.59), exactly as §2.2 Step 7 predicts.
`pose_stability` barely moves (0.7157-0.7194), confirming it is a rig property
rather than a method discriminator.

### 6.2 Full mp4 → number end-to-end

`sampling_dataset/svg1/bimanual/output/multiview/ctrlworld_4view_grid/non_makovian/episode_fold_blue_towel__000070/pred_all_views.mp4`
→ `metrics/rigis/extract_frames.py --layout 2x2 --fps 2` (66 frames/view,
640×96) → `metrics/rigis/run.sh EPISODE_DIR=...`:

```json
{
  "views": ["view1", "view2"],
  "geometric_consistency": 0.7021434144255196,
  "pose_stability": 0.7095510999002339,
  "num_frames": 65,
  "num_disjoint_frames": 12,
  "disjoint_frame_ratio": 0.18461538461538463
}
```

Wall time 34 s including model load. This proves the committed extractor +
launcher chain works from a raw generated mp4, with no dependency on
`thanhnv`'s pre-extracted frames.

### 6.3 Real-GT reference ceiling (`--layout per_view`)

Scoring **real** (non-generated) videos gives the practical upper bound for the
scale. Two GT episodes from
`sampling_dataset/svg1/bimanual/input/multiview/ctrlworld_4view_grid/non_makovian/`,
extracted with `--layout per_view --view-globs view_0.mp4,view_1.mp4 --fps 2`
(34 frames/view), 33 scored frames each:

| episode | `geometric_consistency` | `pose_stability` | `disjoint_frame_ratio` |
|---|---|---|---|
| `episode_brush_screws_into_dustpan_human_brush_left_hold__000001` | 0.8733 | 0.7069 | **0.0000** |
| `..._000006` | 0.8941 | 0.7715 | **0.0000** |

This also exercises the `--layout per_view` path. Two things to take from it:

1. **Real data scores ≈ 0.87-0.89, not 1.0.** The residual gap is VGGT-Omega's own
   reconstruction error, so ~0.89 is a realistic ceiling. The best generated
   variant in §6.1 (0.82) is therefore ~92% of the achievable score, not 82%.
2. **Real data has `disjoint_frame_ratio = 0.0`.** The 0.05-0.28 seen on generated
   videos in §6.1 is a genuine generation failure mode, not a quirk of the camera
   geometry — and it is the single most informative companion number.

Caveat: these are different episodes from §6.1 and are the Ctrl-World *input*
views, so the two tables are not a paired comparison — treat 0.87-0.89 as a
scale anchor only, and re-measure the real reference on the exact episode set
before quoting a "% of real" figure (recipe in §5.3b).

---

## 7. How to interpret the numbers

- **Both metrics: [0, 1], higher is better.** Not a distance, not a loss.
- `geometric_consistency` is the headline multi-view-consistency number.
  `pose_stability` is a companion diagnostic — it is largely a property of the
  *camera rig in the video*, so within one dataset it varies much less across
  methods (see §6.1: 0.7164 vs 0.7124) and should not be used as the headline.
- `disjoint_frame_ratio` is a **validity flag, not a score**. Those frames are
  scored `0.0` and dragged into the mean (§2.2 Step 7), so a variant can lose
  mostly by producing more frames with <2% covisibility. Always report it
  alongside; a big gap in `disjoint_frame_ratio` means the two numbers are not
  comparing the same thing.
- Absolute scale is **not** calibrated to anything. `exp(-σ_R/5)`,
  `1 - rel_resid²`, `1 - rgb_rmse` are hand-chosen shapings
  (`camera_pose_stability.py:27-28`, `geometric_consistency.py:429-437`), so a
  "0.70" has no external meaning. **Only differences between variants scored on
  the identical episode set at the identical fps are meaningful.** This is why
  `metrics/rigis/aggregate.py` defaults to `--intersect`.
  The one useful anchor we measured: **real video scores ≈ 0.87-0.89 with
  `disjoint_frame_ratio = 0.0`** (§6.3). Treat ~0.89, not 1.0, as the ceiling.
- The product over views (§2.1 Step 6) makes the metric *conjunctive*: scores
  compress toward 0 fast, and a 0.05 absolute gap near 0.6 is substantial.
- **Published numbers to compare against: none exist.** This metric is internal
  and unpublished; no paper reports it. Searching the metric branches finds no
  reference table. MEt3R *does* have published numbers in its paper, but MEt3R is
  not what we run (§1.1), and MEt3R's scale is **lower-is-better** feature
  distance — the two are not interconvertible. Concretely: **we cannot claim
  "SOTA on multi-view consistency" on an external leaderboard with this metric.**
  What we *can* claim is a controlled internal statement like "variant X
  preserves N% of the dense baseline's geometric consistency at Y× speedup".
  See §9 for what would be needed for an external claim.

### Reading the aggregator output

```
variant          n  geom_cons    ±sem        Δ    pose_stab    ±sem        Δ   disjoint
```

`n` = episodes in the comparison set, `±sem` = standard error of the mean,
`Δ` = this variant's mean minus the baseline's (positive = better). Rows are
sorted by `geom_cons` descending; `*` marks the baseline. With `--intersect`
(default) only episodes present in **every** requested variant are used, so `n`
is identical across rows.

---

## 8. Known limitations and gotchas

1. **`numba` is an undeclared dependency.** `geometric_consistency.py:5` does
   `from numba import njit`, but `numba` is absent from the branch's
   `pyproject.toml` and `uv.lock`. A plain `uv sync && uv run` therefore crashes
   with `ModuleNotFoundError: No module named 'numba'`. Our launcher injects
   `--with numba` (override via `MVC_UV_EXTRA_DEPS`). Fix upstream if you own
   that branch.
2. **Only the first 2 views count**, hard-coded (`geometric_consistency.py:439-441`,
   `camera_pose_stability.py:63-66`). Passing `--views view1,view2,view3` silently
   scores only the first two pairwise. A 4-view grid is only partially evaluated.
3. **Extrinsic stability uses view 1 only** (`camera_pose_stability.py:66` indexes
   `E[:, 1]`); view 0's extrinsic drift is not scored. Intrinsics use both.
4. **Disjoint frames score 0, not NaN** — see §7. The `num_disjoint_frames`
   counter in the CSV is the only way to tell "bad geometry" from "no overlap".
5. **`frame 0` is excluded from geometry** (`video_metrics.py:449`) but included
   in pose stability, so `num_frames` in the CSV is `len(frames) - 1`.
6. **Scale alignment is a single global scalar per view** fitted on a center crop
   only (`geometric_consistency.py:32-37`). For images with extreme aspect ratio
   (our 640×96 crops) the crop is 213×32 px — if the center is featureless, the
   `n_valid < 10` guard leaves `scale = 1.0` silently (`:51-52`).
7. **`residual` is a mean, not p90.** The p90 line is commented out
   (`geometric_consistency.py:389`); older CSVs in `VideoWorldModelBenchmark/output/`
   produced under `output_p90_tol03_no-alpha/` used a different reduction and are
   **not** comparable to anything produced today.
8. **No determinism guarantee.** fp16 autocast + a thread pool for the geometry
   stage (`video_metrics.py:455`) mean small run-to-run variation is possible.
9. **Service is not multi-tenant.** `model_service` holds one CUDA context; two
   concurrent sweeps on one GPU must use different `MVC_PORT` **and** fit in
   memory. The 1B model at 512 px uses well under 80 GB, but the GPU here was
   already 27 GB busy.
10. **fps mismatch silently invalidates comparisons** (§3.3).
11. **The metric measures VGGT-Omega's *perception* of our video.** A generation
    that is self-consistent but out-of-distribution for VGGT can be penalised.
    There is no human-agreement study for this metric in the repo.
12. **The upstream `run.sh` and `run_one_end2end.sh` hard-code
    `thanhnv`-owned absolute paths.** Do not run those; use our scripts.

---

## 9. Status — what is verified, what is not

### 9.1 Branch-name discrepancy (resolved)

The task referenced branch **`thanhnv/multiview-consistency-indep`**. A prior
`git branch -a` had shown only `origin/thanhnv/multiview-consistency`, suggesting
the `-indep` branch did not exist. **That was stale remote state.** After
`git fetch --all`, `origin/thanhnv/multiview-consistency-indep` **does exist**
(head `8015d97 Update`, 5 commits ahead of the non-indep head `f7f99b1`).

We use `origin/thanhnv/multiview-consistency-indep`, as originally requested.
It is vendored into `metrics/rigis/` at `8015d97`; to re-pin, re-run the `git archive`
command in `metrics/rigis/VENDORED.md` with the new SHA.

### 9.2 "rigid" option (resolved with a caveat)

No literal `rigid` flag exists. We chose the VGGT-Omega rigid-3D-geometry metric
(`geometric_consistency` + `pose_stability`) over MEt3R's feature-space warping.
Full option inventory and reasoning: §1.3. If the ask really meant *MEt3R*, that
is a separate setup job (install `met3r`, fetch MASt3R + FeatUp weights, write a
video loop) — none of which exists today.

### 9.3 Which variants have generations ready to score

Mosaic `pred_all_views.mp4` counts in this repo under `sampling_dataset/<variant>/`
(measured 2026-10-01), and pre-extracted `view1/` episode counts under
`/pfss/.../thanhnv/VideoWorldModelBenchmark/data/preprocessing/<variant>/`:

| Variant | `pred_all_views.mp4` in repo | pre-extracted episodes (bimanual / humanoid / single_arm) | Ready to score? |
|---|---|---|---|
| `dense` (baseline) | 1353 | 600 / 496 / 514 = **1610** | **yes** |
| `pisa` | 805 | 600 / 496 / 514 = **1610** | **yes** |
| `radial` | 805 | 600 / 496 / 514 = **1610** | **yes** |
| `svg1` | 805 | 600 / 496 / 514 = **1610** | **yes** |
| `svg2` | 805 | 600 / 496 / 514 = **1610** | **yes** |
| `itm` | 1053 | **1610** | yes (not requested) |
| `dicache` | 1353 | **1610** | yes (not requested) |
| `sito` | 1053 | **1610** | yes (not requested) |
| `worldcache` | 1353 | 0 | mp4s only — needs extraction |
| `fastercache` | 0 | 300 / 248 / 261 = **809** | frames only, partial |
| `augment` | — | 53 / 20 / 54 = **127** | partial |

Mapping to the git branch names in the question:

| Branch | Variant dir | Status |
|---|---|---|
| `doanh/multiview_base` | `dense` | generations ready |
| `doanh/multiview_pisa` | `pisa` | generations ready |
| `doanh/multiview_radial` | `radial` | generations ready |
| `doanh/multiview_svg` | `svg1` **and** `svg2` | both ready |

So **nothing needs a new generation run** for the five requested variants.
`svg` maps to two variant dirs (SVG1/SVG2) — score both.

`thanhnv` also has **654 already-scored CSVs** under
`VideoWorldModelBenchmark/output/{pisa,radial,svg1,svg2,fastercache,data_for_web}/`
(118 each for pisa/radial/svg1/svg2, 60 fastercache, 120 data_for_web). **There is
no `dense` baseline among them**, so they cannot answer the comparison question on
their own, and their 7-column schema implies an uncommitted code state (§2.4) —
treat them as unreproducible until the baseline is scored with the pinned commit.

### 9.4 Verified

- `origin/thanhnv/multiview-consistency-indep` exists; vendored at `8015d97` (was a worktree).
- Python env builds and imports (torch 2.14.1+cu126 / numpy 1.26.4 / numba 0.68.0).
- Checkpoint loads (strict `load_state_dict`, 4.26 GiB).
- Model service + client IPC round-trip works.
- Multi-variant sweep, `MAX_EPISODES` staging, single-episode mode, CSV writing:
  all exercised (§6.1).
- mp4 → frames → score end-to-end on a real generated video (§6.2).
- Aggregator leaderboard + `--json`, including graceful handling of variants with
  no data and of the 5- vs 7-column CSV schemas.
- Frame extractor output matches `thanhnv`'s pre-extracted frames in count and size.
- **Both** extractor layouts: `--layout 2x2` (§6.2) and `--layout per_view` (§6.3).
- Real-GT reference anchor measured: ≈0.87-0.89, `disjoint_frame_ratio = 0.0` (§6.3).

### 9.5 NOT verified / open

- **No full-scale run.** Everything above is n≤2 episodes, with ±sem up to 0.18.
  No variant has a complete 1610-episode score from this harness.
- **No SOTA answer.** Blocked on (a) running the full sweep (~25 GPU-hours for
  five variants on one GPU, §5.4) and (b) the absence of any external published
  number for this metric (§7). An external "we are SOTA" claim is **not
  achievable with this metric as-is**; it supports internal variant ranking only.
- **Humanoid / single_arm not exercised** — only bimanual Ctrl-World subtrees.
  Resolution there differs; re-check `disjoint_frame_ratio` before trusting it.
- **Real reference not scored on the matched episode set** — §6.3 used different
  episodes from §6.1, so the "% of real" framing is not yet valid. Do this first;
  it is cheap and it is what makes the numbers interpretable.
- **MEt3R is not installed** and was not benchmarked.
- **`fixed_camera_merge`** (option D, §1.3) is documented upstream but absent from
  both branch trees.
- **No human-agreement validation** of this metric in this repo.

---

## 10. Files

Created in this repo:

| Path | Role |
|---|---|
| `metrics/rigis/run.sh` | Launcher: uv interpreter, model service, multi-variant sweep, tmux mode, single-episode mode. |
| `metrics/rigis/extract_frames.py` | mp4 (mosaic or per-view) → `view1/frame_XXXX.png` layout. |
| `metrics/rigis/aggregate.py` | Per-episode CSVs → variant leaderboard (+ JSON). |
| `docs/multiview_consistency_metric.md` | This document. |

Created outside git tracking:

| Path | Role |
|---|---|
| `metrics/rigis/` | vendored code @ `8015d97` + our wrappers (`VENDORED.md`), plus its git-ignored `.venv`. |
| `checkpoints/metrics/vggt_omega/vggt_omega_1b_512.pt` | symlink to the shared checkpoint (gitignored). |
| `sampling_dataset/multiview_consistency/frames/` | staged/extracted frames. |
| `results/multiview_consistency/` | output CSVs + `smoke_summary.json`. |

Upstream, read-only references:

| Path | Role |
|---|---|
| `/pfss/.../thanhnv/VideoWorldModelBenchmark/test_met3r.py` | MEt3R 2-image probe (not used, §1.1). |
| `/pfss/.../thanhnv/VideoWorldModelBenchmark/frame_extractor_one.py` | original extractor our script is adapted from. |
| `/pfss/.../thanhnv/VideoWorldModelBenchmark/checkpoint/vggt_omega_1b_512.pt` | the checkpoint. |
| `/pfss/.../thanhnv/VideoWorldModelBenchmark/data/preprocessing/<variant>/` | pre-extracted frames (8 variants). |
| `/pfss/.../thanhnv/VideoWorldModelBenchmark/output/` | 654 prior CSVs, no dense baseline (§9.3). |
