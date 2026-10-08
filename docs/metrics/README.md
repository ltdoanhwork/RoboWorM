# Metrics

The metric set of the paper (Sec. 3.4, Sec. 4.1, App. B). Code-level reference — layout,
registry, interpreters, checkpoints, Python API — is [metrics/README.md](../../metrics/README.md).

Every registered metric runs through one CLI:

```bash
PY=models/dreamgen/.venv/bin/python
export HF_HUB_OFFLINE=1
CELL=sampling_dataset/dense/single_arm/output/singleview/dreamgen/makovian
$PY -m metrics.evaluate --pred-root $CELL --metrics <name> --out results/metrics/<run>
```

`--out` gets `<metric>.json`, `per_sample.csv` and `summary.json`.

Setup guides for the two proposed metrics (environment, weights, input format, commands,
outputs, common errors): [kiva.md](kiva.md), [rigis.md](rigis.md). Adding a metric:
[../extending/new_metric.md](../extending/new_metric.md).

## Where each metric comes from

Names are the paper's. Except KIVA and RIGIS (ours), each one adapts a published metric;
the citation is the entry in the paper's bibliography.

| paper metric | registry | source metric | cited source |
|---|---|---|---|
| KIVA ↓ | `kiva` | — (ours) | this paper, App. B.1 |
| RIGIS ↑ | `rigis` | — (ours) | this paper, App. B.2 |
| Visual Integrity ↑ | `visual_integrity` | CLIP-IQA+ | J. Wang, K. C. K. Chan, C. C. Loy. *Exploring CLIP for Assessing the Look and Feel of Images.* AAAI 2023 |
| Text Alignment ↑ | `text_alignment` | CLIPScore | J. Hessel, A. Holtzman, M. Forbes, R. Le Bras, Y. Choi. *CLIPScore: A Reference-free Evaluation Metric for Image Captioning.* EMNLP 2021 |
| FVD ↓ | `fvd` | Fréchet Video Distance | T. Unterthiner, S. van Steenkiste, K. Kurach, R. Marinier, M. Michalski, S. Gelly. *Towards Accurate Generative Models of Video: A New Metric & Challenges.* arXiv:1812.01717, 2018 |
| PSNR ↑ (SSIM ↑, LPIPS ↓) | `psnr` `ssim` `lpips` `fidelity` | full-reference fidelity | standard; not cited in the paper ("following prior acceleration benchmarks") |
| Motion Smoothness (MS) — RQ3 | `motion_smoothness` | WorldScore motion smoothness | H. Duan, H.-X. Yu, S. Chen, L. Fei-Fei, J. Wu. *WorldScore: A Unified Evaluation Benchmark for World Generation.* ICCV 2025 |
| Motion Amplitude Score (MAS) — RQ3 | `motion_amplitude` | RBench motion amplitude | Y. Deng, Z. Pan, H. Zhang, X. Li, R. Hu, Y. Ding, Y. Zou, Y. Zeng, D. Zhou. *Rethinking Video Generation Model for the Embodied World.* ICML 2026 |
| MVC — RQ3 | — | feature-matching multi-view consistency | D. Zheng et al. *VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness.* arXiv:2503.21755, 2025 |
| SLAM RPE — RQ3 | — | SLAM relative pose error | H. Wu et al. *MultiWorld* (arXiv:2604.18564, 2026b); H. Wu et al. *Geometry Forcing* (ICLR 2026a) |

CLIP-IQA+ and CLIPScore both run on CLIP (A. Radford et al., ICML 2021). MVC and SLAM RPE
were computed outside this repository; there is no code for them here.

## Proposed metrics

### KIVA — Kinematic Invariant Violation Assessment (↓)

Embodiment hallucination in single-view rollouts. A diffusion reader trained only on real
robot video (DINOv3 features → forward-kinematics keypoints) lifts each rollout to a 3-D
keypoint trajectory (mean of 5 DDIM runs). Per 16-frame segment it measures **rigidity**
(bone-length deformation, median) and **jerk** (third finite difference, max), each divided
by the 95th percentile on real video. Segments whose score exceeds a human-calibrated
per-embodiment cut are violations; KIVA is the percentage of violating segments, averaged
over the rollouts of a cell. Guide: [kiva.md](kiva.md).

```bash
$PY -m metrics.evaluate --pred-root $CELL --metrics kiva \
    --opt kiva.embodiment=single_arm --opt kiva.score_ref=true --out results/metrics/<run>
```

### RIGIS — Robotic Inter-view Geometry & Intrinsic-extrinsic Stability (↑)

Multi-view consistency. A modified VGGT-Ω predicts per-view depth and cameras; depths are
scale-aligned, fused and re-rendered into every view. **S_geo** scores RGB and depth
agreement on the co-visible region (product over views), **S_pose** the temporal stability of
intrinsics and relative extrinsics (the rig is fixed). RIGIS = ½(S_geo + S_pose) ∈ [0, 1],
reported relative to the real-video mean. Guide: [rigis.md](rigis.md).

```bash
$PY -m metrics.evaluate --pred-root <frames>/<task> --pred-glob '*' \
    --ref-template '' --text-template '' --metrics rigis --out results/metrics/<run>
```

## Reference metrics (App. B.3)

| metric | what it measures, as we compute it | code |
|---|---|---|
| **Visual Integrity** ↑ | CLIP-IQA+ ("Good photo." vs "Bad photo.") per frame; **10th percentile** over frames ×100 instead of the source's mean, so a brief autoregressive collapse is not averaged away (mean kept as `visual_integrity_mean`) | `metrics/visual_integrity.py` |
| **Text Alignment** ↑ | CLIPScore `max(100·cos(E_I, E_C), 0)` against the **task clause only** (DreamGen's grid-layout text is stripped), mean over frames; None for image-only models (DreamDojo) | `metrics/text_alignment.py` |
| **FVD** ↓ | Kinetics-400 I3D features of non-overlapping 16-frame windows at 224×224, all clips at 5 fps; one shared real pool per embodiment × camera × regime, multi-view scored per camera and averaged over common cameras | `metrics/fvd/` — setup in [metrics/fvd/README.md](../../metrics/fvd/README.md) |
| **PSNR** ↑ (SSIM, LPIPS) | fidelity of an accelerated rollout to the **dense rollout** of the same episode (never real video); PSNR is the one in the retention index. Backend: `piq` PSNR/LPIPS (VGG), torchmetrics SSIM (`data_range=1`, no downsampling) — SSIM is the exact call of `scripts/metrics/compute_dr_metrics.py`, and all three reproduce the stored DR records | `metrics/fidelity.py` |

Visual Integrity and Text Alignment are scored on the paper's analysis frames: 5 fps,
conditioning frame dropped, one camera tile, 256×192, every 5th frame (`--meta layout=grid
--meta view_index=<i> --meta n_views=4` selects a tile of a mosaic).

```bash
$PY -m metrics.evaluate --pred-root $CELL --ref-template '' \
    --metrics visual_integrity text_alignment --out results/metrics/<run>
$PY -m metrics.evaluate --pred-root $CELL --metrics fvd --out results/metrics/<run>
$PY -m metrics.evaluate --pred-root sampling_dataset/<method>/<cell> \
    --ref-root sampling_dataset/dense/<cell> --ref-template '{name}' --metrics fidelity --out results/metrics/<run>
```

Use them as the paper does: FVD only within the same reference pool and sampling setup
(it is biased by sample size, Fig. B3); Text Alignment rankings with paired deltas under
the same prompt; Visual Integrity for the acceleration comparison, not to rank models.

## RQ3 baselines (Sec. 4.2)

Compared against KIVA; not headline columns.

| baseline | as we compute it | code |
|---|---|---|
| **MS** | VFIMamba re-synthesises every odd frame from its even neighbours; global SSIM ×100 between interpolated and real frame (higher = smoother) | `metrics/motion_smoothness.py` |
| **MAS** | SEA-RAFT flow inside a GroundingDINO + SAM 2 robot ∪ object mask, p95 per frame pair (ego-compensated on head cameras, whole-frame median on wrist cameras), px per source frame | `metrics/motion_amplitude.py` |

```bash
$PY -m metrics.evaluate --pred-root $CELL --ref-template '' --metrics motion_smoothness motion_amplitude \
    --meta embodiment=single_arm --meta view=global --out results/metrics/<run>
```

## Full-benchmark waves (what the paper tables read)

```bash
bash scripts/metrics/run_dense_nr.sh              # Visual Integrity, Text Alignment, MS, MAS on every rollout
MACHINES=1 bash scripts/metrics/run_dr_metrics.sh 0   # PSNR / SSIM / LPIPS + I3D features vs dense
bash metrics/kiva/run.sh --tmux                   # KIVA   (see kiva.md)
bash metrics/rigis/run.sh --tmux                  # RIGIS  (see rigis.md)
# FVD pooling: metrics/fvd/README.md
```

Per-clip records land in `metrics/WorldScore/results/{dense_nr_full,dr_full,fvd_perview}/`;
the table producers are listed in [../reproduce/README.md](../reproduce/README.md) and write
their summaries to `results/reports/`.

## Speedup

Dense / accelerated generation time per segment from the internal timer (excludes model load
and VAE decode), paired episodes on the same GPU. See
[../inference/README.md](../inference/README.md#measuring-speedup).
