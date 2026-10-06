# Metrics

The metric set of the paper (Sec. 3.4, Sec. 4.1, App. B). Registry, CLI flags,
Python API and checkpoint overrides are in [metrics/README.md](../../metrics/README.md).

Every registered metric runs through one CLI:

```bash
PY=models/dreamgen/.venv/bin/python
export HF_HUB_OFFLINE=1
CELL=sampling_dataset/dense/single_arm/output/singleview/dreamgen/makovian
$PY -m metrics.evaluate --pred-root $CELL --metrics <name> --out results/metrics/<run>
```

`--out` gets `<metric>.json`, `per_sample.csv` and `summary.json`.

> `metrics/` is being refactored (`clip_score.py` → `text_alignment.py`,
> `fvd.py` → `metrics/fvd/`, new `visual_integrity.py` / `motion_smoothness.py` /
> `motion_amplitude.py`). The registry names below are the post-refactor ones;
> `clip_score` stays an alias. If a name is not registered yet, check
> `metrics/registry.py` and [metrics/README.md](../../metrics/README.md).

## Proposed metrics

### KIVA — Kinematic Invariant Violation Assessment (↓)

Embodiment hallucination in single-view rollouts. A diagnostic diffusion model,
trained only on real robot videos (DINOv3 features → forward-kinematics keypoints),
lifts each rollout to a 3D metric keypoint trajectory (mean of 5 DDIM runs). Per
segment it measures **rigidity** error (bone-length deformation, median) and
**jerk** (third-order finite difference, max), each divided by the 95th percentile
on real videos. A segment is a violation when `s = max(s_rig, s_jerk)` exceeds a
per-embodiment threshold calibrated against human labels (App. B.1); KIVA is the
percentage of violating segments, averaged over the rollouts of a cell.

- Ours. Code: `metrics/kiva/` (registry `kiva`; own `hdmi` env, launched for you).
- Setup and full guide: [kiva.md](kiva.md).

```bash
$PY -m metrics.evaluate --pred-root $CELL --metrics kiva \
    --opt kiva.embodiment=single_arm --opt kiva.score_ref=true --out results/metrics/<run>
```

### RIGIS — Robotic Inter-view Geometry & Intrinsic-Extrinsic Stability (↑)

Multi-view consistency. A modified VGGT-Ω backbone (Wang et al., 2026) predicts
per-view depth and per-frame cameras; views are scale-aligned, fused and
reprojected. **S_geo** averages exponential-decay scores of the RGB RMSE and depth
residual over views and time; **S_pose** scores the temporal std of intrinsics and
extrinsics relative to the first frame (the rig is fixed). RIGIS = ½(S_geo + S_pose) ∈ [0, 1].

- Ours. Code: `metrics/rigis/` (registry `rigis`; own uv env `metrics/rigis/.venv`).
- Setup and full guide: [rigis.md](rigis.md).

```bash
$PY -m metrics.evaluate --pred-root <frames>/<task> --pred-glob '*' \
    --ref-template '' --text-template '' --metrics rigis --out results/metrics/<run>
```

## Reference metrics (App. B.3)

| metric | what it measures | adapted from | code | registry |
|---|---|---|---|---|
| **Visual Integrity** ↑ | frame-level visual quality: CLIP-IQA+ per frame, **10th percentile** over frames ×100 (instead of the mean, so brief autoregressive collapses count) | CLIP-IQA+ (Wang et al., 2023) | `metrics/visual_integrity.py` | `visual_integrity` |
| **Text Alignment** ↑ | whether the rollout still depicts the commanded task: CLIPScore `max(100·cos, 0)` vs the **task clause only**, mean over frames; text-conditioned models only | CLIPScore (Hessel et al., 2021) | `metrics/text_alignment.py` | `text_alignment` |
| **FVD** ↓ | set-level distance between generated and real videos in Kinetics-400 I3D feature space; 16-frame windows at 224×224, all videos at 5 fps, one shared real pool per embodiment × camera × regime (multi-view: per camera, averaged) | FVD (Unterthiner et al., 2018) | `metrics/fvd/` | `fvd` |
| **PSNR / SSIM / LPIPS** | fidelity of an accelerated rollout to the **dense rollout** of the same episode (not to real video); PSNR is the one used in the retention index | standard full-reference metrics | `metrics/fidelity.py` | `fidelity` (`psnr` `ssim` `lpips`) |

```bash
$PY -m metrics.evaluate --pred-root $CELL --metrics visual_integrity text_alignment --out results/metrics/<run>
$PY -m metrics.evaluate --pred-root $CELL --metrics fvd      --out results/metrics/<run>
$PY -m metrics.evaluate --pred-root <method cell> --ref-root <dense cell> \
    --ref-template '{stem}{suffix}' --metrics fidelity --out results/metrics/<run>
```

FVD is only compared within the same reference pool and sampling setup. Text
Alignment rankings use paired deltas under the same prompt. Visual Integrity is
used for the acceleration comparison, not to rank models.

## RQ3 baselines (Sec. 4.2)

Compared against KIVA / RIGIS; not headline columns.

| baseline | compared with | adapted from | code | registry |
|---|---|---|---|---|
| **Motion Smoothness (MS)** — VFIMamba re-synthesises each odd frame from its neighbours | KIVA | WorldScore (Duan et al., 2025) | `metrics/motion_smoothness.py` | `motion_smoothness` |
| **Motion Amplitude Score (MAS)** — optical-flow magnitude of the robot + manipulated object | KIVA | RBench (Deng et al., 2026) | `metrics/motion_amplitude.py` | `motion_amplitude` |
| **MVC** — feature-matching multi-view consistency | RIGIS | VBench-2.0 (Zheng et al., 2025) | not in this repo; numbers in `assets/scripts/make_rq3_human_alignment.py` | — |
| **SLAM RPE** — relative pose error of a SLAM reconstruction | RIGIS | MultiWorld / Geometry Forcing (Wu et al., 2026b;a) | not in this repo; as above | — |

```bash
$PY -m metrics.evaluate --pred-root $CELL --metrics motion_smoothness motion_amplitude --out results/metrics/<run>
```

## Speedup

Dense / accelerated generation time per segment from the internal timer (excludes
model load and VAE decode), paired episodes on the same GPU. See
[../inference/README.md](../inference/README.md#measuring-speedup).

Full-benchmark sweeps and the paper table producers: [../reproduce/README.md](../reproduce/README.md).
