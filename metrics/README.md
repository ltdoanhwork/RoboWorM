# `metrics/` — the RoboWorM paper's metrics

Only the metrics the paper reports (Sec. 3.4, Sec. 4.1, App. B) plus the two RQ3
baselines, behind one interface (`metrics.base.BaseMetric`), runnable from Python or one
CLI (`python -m metrics.evaluate`). Paper PDF: `docs/paper/RoboWM_ICLR2027_check.pdf`.
User guide: [docs/metrics/README.md](../docs/metrics/README.md).

## Metrics and where they come from

Every metric except KIVA and RIGIS is **adapted** from a published metric; the name we use
is the paper's, the column "source" is the paper it comes from (exact entries of the
paper's bibliography).

| paper name | registry | ↑/↓ | source metric | source paper | our adaptation | code |
|---|---|---|---|---|---|---|
| **KIVA** | `kiva` | ↓ | — (ours) | this paper, App. B.1 | — | `kiva/` |
| **RIGIS** | `rigis` | ↑ | — (ours) | this paper, App. B.2 | — | `rigis/` |
| **Visual Integrity** | `visual_integrity` | ↑ | CLIP-IQA+ | Wang, Chan & Loy, *Exploring CLIP for Assessing the Look and Feel of Images*, AAAI 2023 | 10th percentile over frames ×100 instead of the mean | `visual_integrity.py` |
| **Text Alignment** | `text_alignment` | ↑ | CLIPScore | Hessel, Holtzman, Forbes, Le Bras & Choi, *CLIPScore: A Reference-free Evaluation Metric for Image Captioning*, EMNLP 2021 | scored against the task clause only; text-conditioned models only | `text_alignment.py` |
| **FVD** | `fvd` | ↓ | Fréchet Video Distance | Unterthiner, van Steenkiste, Kurach, Marinier, Michalski & Gelly, *Towards Accurate Generative Models of Video: A New Metric & Challenges*, arXiv:1812.01717, 2018 | 5 fps, shared real pool per embodiment × camera × regime, per camera for multi-view | `fvd/` |
| **PSNR** (SSIM, LPIPS) | `psnr` `ssim` `lpips` `fidelity` | ↑ ↑ ↓ | full-reference fidelity | standard (no citation in the paper; SSIM = Wang et al., TIP 2004, LPIPS = Zhang et al., CVPR 2018) | reference = the **dense rollout** of the same episode, not real video | `fidelity.py` |
| **Motion Smoothness (MS)** — RQ3 baseline | `motion_smoothness` | ↑ | WorldScore motion smoothness (VFIMamba) | Duan, Yu, Chen, Fei-Fei & Wu, *WorldScore: A Unified Evaluation Benchmark for World Generation*, ICCV 2025 | global SSIM, no TTA, on the analysis frames | `motion_smoothness.py` |
| **Motion Amplitude Score (MAS)** — RQ3 baseline | `motion_amplitude` | — | RBench motion amplitude | Deng, Pan, Zhang, Li, Hu, Ding, Zou, Zeng & Zhou, *Rethinking Video Generation Model for the Embodied World*, ICML 2026 | flow inside a GroundingDINO + SAM 2 subject mask, per camera mount | `motion_amplitude.py` |

Backbone models cited in the paper: CLIP (Radford et al., ICML 2021) for CLIP-IQA+ and
CLIPScore, VGGT-Ω (Wang et al., CVPR 2026) for RIGIS, DINOv3 (Siméoni et al., TMLR 2026)
for KIVA. Not cited in the paper but used by the code: I3D Kinetics-400 (FVD), VFIMamba
(MS), GroundingDINO + SAM 2 + SEA-RAFT (MAS).

The other two RQ3 baselines — **MVC** feature matching (Zheng et al., *VBench-2.0*, 2025)
and **SLAM RPE** (Wu et al., *MultiWorld* 2026b; *Geometry Forcing* 2026a) — were computed
outside this repo; there is no code for them here.

## Layout

```
metrics/
├── README.md            this file
├── base.py              BaseMetric -> FrameMetric (-> ReferenceFreeFrameMetric) / ExternalMetric; Sample, MetricResult
├── registry.py          name -> class (lazy), deprecated aliases, register(), build_metric()
├── evaluate.py          CLI: discover samples, run metrics, write json/csv
├── video_io.py          decoding, prompt sidecars, nr_analysis_frames() (the reference-free frame protocol)
├── fidelity.py          PSNR, SSIM, LPIPS, Fidelity (piq; SSIM via torchmetrics)
├── visual_integrity.py  Visual Integrity (CLIP-IQA+ p10)
├── text_alignment.py    Text Alignment (CLIPScore), task_clause()
├── fvd/                 FVD: core.py (I3D, Fréchet), pool.py (shared real pool), README.md (setup)
├── motion_smoothness.py MS (VFIMamba)
├── motion_amplitude.py  MAS (GroundingDINO -> SAM 2 -> SEA-RAFT)
├── third_party.py       paths + checkpoints of the WorldScore third_party backends, SEA-RAFT wrapper
├── kiva/                KIVA, vendored kinescore (VENDORED.md) + run.sh / aggregate.py / metric.py
├── rigis/               RIGIS, vendored VGGT-Ω code (VENDORED.md) + run.sh / extract_frames.py / aggregate.py / metric.py
├── WorldScore/          third-party dependency (see below)
└── test_framework.py    unit + parity tests
```

`metrics/WorldScore/` is the vendored WorldScore checkout (Duan et al., 2025; its own git
repo, ~43 GB). It is kept only as a dependency: `third_party.py` puts its
`worldscore/benchmark/metrics/third_party/{SEA-RAFT,VFIMamba,groundingdino,sam2}` on
`sys.path` without importing the `worldscore` package. Its `results/` folder holds the
stored per-clip records the paper tables read (`dense_nr_full/` reference-free wave,
`dr_full/` PSNR/SSIM/LPIPS + I3D, `fvd_perview/` per-camera I3D features).

Removed on 2026-10-06 (moved to `tmp/20261005_refactor/misc/metrics/`, restore commands in
`tmp/20261005_refactor/MANIFEST.md`): the PhyWorld schema/scoring (`phyworld/`), the
WorldScore quality probe (`quality/`), the non-paper dynamics code (`dynamics/`: paired-GT
flow EPE, actor flicker, motion magnitude) and the legacy `video_metrics.py`. The paper
summary tables that used to live in `metrics/reports/` are now in `results/reports/`.

## Interpreters and checkpoints

| registry | needs | interpreter |
|---|---|---|
| `visual_integrity` `text_alignment` `fidelity` `fvd` | pred (+ prompt / ref) | `models/dreamgen/.venv` (pyiqa, torchmetrics, piq) |
| `motion_smoothness` `motion_amplitude` | pred (+ prompt, embodiment, view) | `models/dreamgen/.venv` + WorldScore third_party (added for you) |
| `rigis` | multiview episode dir (`view1/ view2/`) | `metrics/rigis/.venv` (uv), launched for you |
| `kiva` | pred mp4 at the reader resolution (+ GT anchor) | `hdmi` conda env + `PYTHONPATH=metrics/kiva/src`, launched for you |

| metric | default path | override |
|---|---|---|
| FVD | `checkpoints/metrics/fvd/i3d_torchscript.pt` | `FVD_I3D_CKPT` |
| MS / MAS | `checkpoints/metrics/worldscore/{VFIMamba.pkl, Tartan-C-T-TSKH-spring540x960-M.pth, groundingdino_swint_ogc.pth, sam2.1_hiera_base_plus.pt}` | `WORLDSCORE_CKPT_DIR` |
| RIGIS | `checkpoints/metrics/vggt_omega/vggt_omega_1b_512.pt` | `MVC_CKPT` |
| KIVA readers | `checkpoints/metrics/kiva/` (else `$HAL_STORE/ckpt`) | `HAL_CKPT_DIR`, `HAL_CKPT_{SINGLE_ARM,BIMANUAL,HUMANOID}` |
| CLIP-IQA+ / CLIPScore | HF / pyiqa caches (`openai/clip-vit-base-patch16`) | `HF_HOME` |

## Quick start

```bash
PY=models/dreamgen/.venv/bin/python
export HF_HUB_OFFLINE=1
CELL=sampling_dataset/dense/single_arm/output/singleview/dreamgen/makovian

# reference-free (prompt from <episode>.txt)
$PY -m metrics.evaluate --pred-root $CELL --ref-template '' \
    --metrics visual_integrity text_alignment --out results/metrics/dense_sa_mk

# one camera of a multiview mosaic: tell the protocol which tile
$PY -m metrics.evaluate --pred-root <dreamgen mv cell> --pred-glob '**/full_pred.mp4' --ref-template '' \
    --meta layout=grid --meta view_index=0 --meta n_views=4 --metrics visual_integrity --out <out>

# PSNR/SSIM/LPIPS of an accelerated cell vs the dense rollout of the same episode
$PY -m metrics.evaluate --pred-root sampling_dataset/pisa/<cell> --ref-root sampling_dataset/dense/<cell> \
    --ref-template '{name}' --metrics fidelity --out <out>

# FVD of one cell vs its own GT (setup + shared pool: metrics/fvd/README.md)
$PY -m metrics.evaluate --pred-root $CELL --metrics fvd --out <out>

# KIVA, with the real-video anchor scored in the same launcher call
$PY -m metrics.evaluate --pred-root $CELL --metrics kiva \
    --opt kiva.embodiment=single_arm --opt kiva.score_ref=true --out <out>

# RIGIS over extracted episode dirs (<ep>/view1, <ep>/view2)
$PY -m metrics.evaluate --pred-root <frames>/<task> --pred-glob '*' --ref-template '' \
    --text-template '' --metrics rigis --out <out>

# RQ3 baselines
$PY -m metrics.evaluate --pred-root $CELL --ref-template '' --metrics motion_smoothness motion_amplitude \
    --meta embodiment=single_arm --meta view=global --out <out>
```

`--out` gets `<metric>.json` (full result incl. config and per-sample rows),
`per_sample.csv` and `summary.json`, rewritten after every metric. Per-metric constructor
arguments go through `--opt METRIC.KEY=VALUE` (JSON-parsed); per-sample metadata through
`--meta KEY=VALUE`.

The full-benchmark waves that produced the paper's stored records:
`scripts/metrics/run_dense_nr.sh` (Visual Integrity, Text Alignment, MS, MAS on every
rollout), `scripts/metrics/run_dr_metrics.sh` (PSNR/SSIM/LPIPS + I3D features vs dense),
`metrics/kiva/run.sh`, `metrics/rigis/run.sh`; FVD pooling in `metrics/fvd/README.md`.

## Python API

```python
from metrics import Sample, build_metric

vi = build_metric("visual_integrity", device="cuda")
res = vi.evaluate([Sample("ep0", pred="ep0.mp4", text="put the cup on the coaster")])
res.summary["visual_integrity"]   # {'mean', 'std', 'sem', 'n', 'n_missing'}
res.per_sample["ep0"]             # {'visual_integrity', 'visual_integrity_mean', ...}
res.save("out/visual_integrity.json")

build_metric("fidelity")(Sample("ep1", pred="pisa/ep1.mp4", ref="dense/ep1.mp4"))  # one sample -> dict
```

## Conventions that decide whether numbers are comparable

* **Reference-free frame protocol** (`ReferenceFreeFrameMetric`, `protocol="paper"`):
  `video_io.nr_analysis_frames` decimates to 5 fps by integer stride, drops the
  conditioning frame, cuts the camera tile (`--meta layout/view_index/n_views`), resizes to
  256×192 (`INTER_AREA`); Visual Integrity / Text Alignment then score every 5th analysis
  frame after a cv2 JPEG round trip, with Resize 512 → CLIP. This reproduces the stored
  `dense_nr_full` values exactly. `protocol="raw"` scores any clip as decoded.
* **Text Alignment** is defined only for text-conditioned models (`text_conditioned()`;
  DreamDojo is image-only → None). Rank with paired deltas under the same prompt.
* **Visual Integrity** fails the real-video-is-ceiling audit: use it for the acceleration
  comparison against dense, not to rank models.
* **Fidelity**: reference = the dense rollout of the same episode. Float [0,1] NCHW,
  frames truncated to `min(T_pred, T_ref)`, no implicit resize. PSNR and LPIPS use `piq`
  defaults; SSIM uses `torchmetrics.functional.image.structural_similarity_index_measure`
  (`data_range=1.0`, no downsampling), the same call as the DR wave
  (`scripts/metrics/compute_dr_metrics.py`) that produced the stored numbers, so all three
  reproduce the stored records. Note `piq.ssim` (used in the inference-time `metrics.json`)
  downsamples first and gives different SSIM at ≥ 384 px.
* **FVD** is a set statistic (one number per cell, effective n = 1), `None` below 64 windows
  per side, only comparable at the same reference and sample size — see `fvd/README.md`.
* **KIVA** rates are meaningful only next to the real-video anchor (`score_ref=true`);
  kinescore re-calibrates per call and samples unseeded noise, so score all compared
  variants and the anchor in **one** call. **RIGIS** is reported normalised by the
  real-video mean of the morphology (≈ 0.89 raw ceiling). Details: `docs/metrics/kiva.md`,
  `docs/metrics/rigis.md`.
* **CLIP weights offline**: `metrics/kiva/run.sh` exports `HF_HOME` into the kinescore
  store; if that leaks into a tmux server's env, `text_alignment` falls back to
  `~/.cache/huggingface` and prints which cache it used.

## Writing a new metric

```python
from metrics.base import FrameMetric
from metrics.registry import register

@register("mean_abs_diff")
class MeanAbsDiff(FrameMetric):
    name = "mean_abs_diff"
    higher_is_better = {"mad": False}
    requires_ref = True

    def score_frames(self, pred, ref, text):          # aligned uint8 (T,H,W,3)
        return {"mad": float(abs(pred.astype(int) - ref).mean())}
```

In-process metrics subclass `FrameMetric` (decoding, alignment, `n_frames_*` bookkeeping
inherited; load models in `setup()`). Out-of-process metrics subclass `ExternalMetric`,
set `launcher` and implement `run_batch(samples)` (one model load per batch). Return floats
keyed like `higher_is_better`; NaN = not scorable, raise for real failures.

## Tests

```bash
models/dreamgen/.venv/bin/python -m pytest metrics/test_framework.py -q    # ~2 min, CPU
```
