# `metrics/dynamics` — motion / dynamics metrics

WorldScore's three **dynamics** aspects plus our paired-GT (reference-based) variants.
All of these need the dreamgen venv (`models/dreamgen/.venv`) because they load SAM2,
SEA-RAFT, VFIMamba and/or GroundingDINO from `metrics/WorldScore/.../third_party`.

## Files

| file | metric | models used |
|---|---|---|
| `motion_accuracy_gdino.py` | `motion_accuracy` (**our version**) | GroundingDINO (arm + object) → SAM2 → SEA-RAFT |
| `motion_mag_smooth_probe.py` | `motion_magnitude` + `motion_smoothness` | SEA-RAFT (median flow), VFIMamba (interpolation) |
| `physics_metrics.py` | paired-GT flow EPE / magnitude ratio vs GT | (backend-agnostic, numpy only) |
| `flow_backend.py` | SEA-RAFT wrapper for `physics_metrics.py` | SEA-RAFT |

Why not WorldScore's original `motion_accuracy` probe (archived to
`tmp/20261005_refactor/misc/metrics/dynamics/motion_accuracy_probe.py`): it only tracks the *object* and
treats everything else as background, but on robot data the **moving robot arm sits in
that "background"**, which poisons the `max(bg_flow)` term. `motion_accuracy_gdino.py`
detects the arm *and* the object from the task prompt and merges them into one
**foreground**, so the background is genuinely static.

## How each metric works

### `motion_accuracy`

WorldScore's original (`motion_accuracy_metrics.py`), per consecutive frame pair:

```python
flow            = SEA_RAFT(image1, image2)                  # dense (H,W,2)
flow_magnitude  = sqrt(flow[...,0]**2 + flow[...,1]**2)     # px/frame
object_flow     = flow_magnitude[mask]                      # SAM2-propagated object mask
background_flow = flow_magnitude[~mask]
score          += object_flow.max() - background_flow.max()  # <-- MAX minus MAX
```
final score = mean over frame pairs.

Pipeline: seed a mask on frame 0 (GroundingDINO box → SAM2) → **SAM2 video predictor
propagates** that mask through all frames → SEA-RAFT flow per pair → sample flow inside vs
outside the mask.

**Two structural problems on robot data:**
1. It's a **difference of two maxima**, so it's decided by *two pixels* out of ~300k. A
   single high-flow outlier anywhere in "background" collapses the score.
2. The **robot arm is in "background"** by construction. The arm is usually the fastest
   thing in frame, so `max(bg_flow)` ≈ arm speed and the score measures
   "object faster than arm" — nearly always negative, and unrelated to physical correctness.

Our version (`motion_accuracy_gdino.py`) fixes both: GroundingDINO detects **arm *and*
object** from the task prompt, both are seeded into SAM2 and merged into one **foreground**
mask, so background is genuinely static. It reports several statistics instead of one
fragile number — `ws_fg_vs_bg` (the original max-minus-max formula, for comparability),
`med_fg_vs_bg` (**median**-based, outlier-robust), plus raw `mean_{obj,arm,bg}_flow_med`
and foreground coverage `mean_fg_cov_pct` so you can see *why* a score moved.

### `motion_magnitude`

`median(flow_magnitude)` over all pixels of each frame pair, averaged over pairs
(`motion_mag_smooth_probe.py`, mirroring WorldScore's `flow_metrics.py`). Median (not mean)
so it reports typical scene motion rather than the fastest object. **Reference-free and
speed-only**: it says how fast, never whether the motion was *correct*. In our paired-GT
setting we additionally report the ratio `pred/GT` and `|log ratio|` so under-moving and
over-moving are penalized symmetrically.

### `motion_smoothness`

Hide every odd frame and ask a video-interpolation model to redraw it:

1. Split frames into **even anchors** (0,2,4,…) and **odd targets** (1,3,5,…).
2. For each triple `(2k, 2k+1, 2k+2)`, give **VFIMamba** only the two even anchors and let
   it interpolate the middle frame.
3. Compare the interpolation against the withheld real odd frame → **SSIM ↑ / MSE ↓**
   (WorldScore also computes LPIPS).

Smooth, temporally continuous motion is easy to interpolate → high SSIM / low MSE. Jerky or
flickering motion breaks the interpolation. **Caveat: it rewards slowness** — a near-static
clip is trivially interpolatable — so it must always be paired with a magnitude term.
(The per-triple `anchor | interpolated | real | |error|` viewer `motion_smoothness_viz.py`
was archived to `tmp/20261005_refactor/misc/metrics/dynamics/`.)

### `physics_metrics.py` (our paired-GT addition, not in WorldScore)

Because we have aligned GT, we can compare **flow fields directly** instead of scoring the
prediction alone: `flow_epe` = mean `‖flow_pred − flow_gt‖` (0 = identical motion), plus
`motion_mag_ratio = |flow_pred| / |flow_gt|` (1.0 ideal, <1 frozen, >1 hallucinated). This
is strictly more informative than the reference-free metrics — it checks motion is *correct*,
not merely present.

## Run

```bash
# motion_accuracy — our GroundingDINO+SAM2 version (raw grid)
CUDA_VISIBLE_DEVICES=0 bash scripts/metrics/run_motion_accuracy_gdino.sh

# motion_accuracy on the MATCHED grid (GT resampled to each model's res+fps) <- correct protocol
CUDA_VISIBLE_DEVICES=0 bash scripts/metrics/run_motion_accuracy_matched.sh

# motion_magnitude + motion_smoothness on the matched grid
CUDA_VISIBLE_DEVICES=0 SMOOTH_STRIDE=8 bash scripts/metrics/run_motion_mag_smooth_matched.sh
```

## ⚠️ Protocol that matters: always match the grid

Optical flow is measured in **px/frame**, so it scales with resolution *and* fps. GT
(800 frames), DreamGen (768×432@16) and DreamDojo (640×480@10) are **not commensurable**
on the raw grid — comparing them directly inverts the ranking. Resample GT to each
prediction's own (resolution, fps) and score each model against *its own matched GT*
(`run_motion_accuracy_matched.sh`). See `docs/metrics/WORLDSCORE_PHYSICS_REPORT.md` §8.

## Known findings

- **`motion_magnitude` is speed-only** and rewards the over-mover; `motion_smoothness`
  captures the temporal coherence the eye actually judges. Neither alone is sufficient —
  the physics score must combine magnitude-distance + smoothness.
- **`motion_smoothness` rewards slowness**: a near-static clip is trivially interpolatable,
  so it must be paired with a magnitude term.
- VFIMamba needs `mamba_ssm`. To keep the venv on torch 2.6 (flashinfer/cosmos), a
  **pure-PyTorch `mamba_ssm` shim** (`selective_scan_ref`) is vendored into the venv
  instead of the CUDA kernel (which requires torch 2.9 / triton 3.5). It is numerically
  equivalent but slower — hence the frame-triple subsampling (`SMOOTH_STRIDE`).
