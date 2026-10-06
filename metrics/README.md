# `metrics/` — the RoboWorM paper's evaluation metrics

This directory holds exactly the metric set of the paper (App. B), behind one
abstract interface (`metrics.base.BaseMetric`), so every metric can be run,
aggregated and saved the same way from Python or from one CLI
(`python -m metrics.evaluate`). Paper PDF: `docs/paper/RoboWM_ICLR2027_check.pdf`.

## Paper metric → code

| paper metric | what it measures | code | registry name | ↑/↓ |
|---|---|---|---|---|
| **KIVA** | embodiment hallucination (kinescore rigidity / jerk violation rate, excess over real-video anchor) | `kiva/` (vendored `quang/hallucination-official` @5225ce7) | `kiva` (alias `hallucination`) | ↓ |
| **RIGIS** | cross-view rigid-geometry consistency (VGGT-Omega) | `rigis/` (vendored `thanhnv/multiview-consistency-indep` @8015d97) | `rigis` (alias `multiview_consistency`) | ↑ |
| **Visual Integrity** | CLIP-IQA+ (pyiqa `clipiqa+`), reported as the per-clip **p10** over frames | `quality/quality_control_probe.py` (`IQA`), p10 in `scripts/metrics/compute_phyworld.py::score_qual` | — (probe) | ↑ |
| **Text Alignment** | CLIPScore, torchmetrics `clip-vit-base-patch16`, 0–100 | `clip_score.py` | `clip_score` | ↑ |
| **FVD** | I3D (StyleGAN-V torchscript), non-overlapping 16-frame windows @ 5 fps, shared per-camera real pool (multi-view) | `fvd.py` | `fvd` | ↓ |
| **PSNR / SSIM / LPIPS vs dense** | fidelity of an accelerated rollout to the **dense** rollout of the same episode (PhyFid-DR) | `fidelity.py` (piq), DR pass `scripts/metrics/compute_dr_metrics.py` | `psnr` `ssim` `lpips` `fidelity` | ↑ ↑ ↓ |
| **MS** (RQ3 baseline) | motion smoothness, VFIMamba odd-from-even interpolation | `dynamics/motion_mag_smooth_probe.py` | — (probe) | ↑ |
| **MAS-style actor activity** | ego-compensated arm∪object flow (GroundingDINO → SAM2 → SEA-RAFT), RBench MAS analogue | `dynamics/motion_accuracy_gdino.py` | — (probe) | ↑ |
| **PhyWorld NR** | PhyQual-NR / PhyDyn-NR no-reference scoring of every method incl. dense | `phyworld/` (+ `scripts/metrics/run_dense_nr.py`) | — (pipeline) | ↑ |

Registered metrics run through `metrics.evaluate`; the probes/pipelines marked
"—" keep their own launchers (`scripts/run_*`), documented in
`quality/README.md`, `dynamics/README.md` and `docs/metrics/phyworld_metrics.md`.

| registry name | needs | backend | interpreter |
|---|---|---|---|
| `psnr` `ssim` `lpips` `fidelity` | pred + ref | `piq` (LPIPS = VGG16) | dreamgen venv |
| `clip_score` | pred + prompt | torchmetrics CLIP ViT-B/16 | dreamgen venv |
| `fvd` | pred set + ref set (or `ref_pool=<.npy>`) | I3D torchscript | dreamgen venv |
| `rigis` | multiview episode (`view1/ view2/` or mosaic mp4) | VGGT-Omega, `rigis/run.sh` | `uv` project `metrics/rigis/.venv` (launched for you) |
| `kiva` | pred mp4 (+ gt anchor) | kinescore, `kiva/run.sh` | `hdmi` conda env + `PYTHONPATH=metrics/kiva/src` (launched for you) |

```
metrics/
├── base.py              BaseMetric (ABC) -> FrameMetric / ExternalMetric, Sample, MetricResult
├── registry.py          name -> class (lazy), aliases, register(), build_metric()
├── evaluate.py          CLI: discover samples, run metrics, write json/csv
├── video_io.py          mp4 / frame dir / array -> uint8 (T,H,W,3); prompt sidecars
├── fidelity.py          PSNR, SSIM, LPIPS, Fidelity
├── clip_score.py        CLIPScore
├── fvd.py               I3D features, fps decimation, mosaic split, Frechet, FVD metric
├── rigis/               vendored VGGT-Omega code + run.sh, extract_frames.py, aggregate.py, metric.py, VENDORED.md
├── kiva/                vendored kinescore (src/, configs/, docs/) + run.sh, aggregate.py, probe_clips.py,
│                        map_to_dr.py, metric.py, VENDORED.md
├── quality/             Visual Integrity + Text Alignment probe (WorldScore lineage)
├── dynamics/            MS, MAS-style actor activity, paired-GT flow, actor flicker
├── phyworld/            PhyWorld NR schema + scoring
├── WorldScore/          vendored WorldScore + third_party models (43 GB; results/ = run outputs)
├── video_metrics.py     legacy function API (calculate_psnr ...)
└── test_framework.py    unit + parity tests
```

Full write-ups: `docs/metrics/hallucination_metric.md` (KIVA),
`docs/metrics/multiview_consistency_metric.md` (RIGIS),
`tmp/20261005_refactor/docs/WORLDSCORE_PHYSICS_REPORT.md` (WorldScore lineage). The two
external launchers still work standalone for large sweeps
(`VARIANTS="dense pisa ..." bash metrics/kiva/run.sh`).

### Checkpoints (env overrides)

| metric | default path | override |
|---|---|---|
| FVD | `checkpoints/metrics/fvd/i3d_torchscript.pt` | `FVD_I3D_CKPT` |
| RIGIS | `checkpoints/metrics/vggt_omega/vggt_omega_1b_512.pt` | `MVC_CKPT` |
| KIVA readers | `checkpoints/metrics/kiva/` (else `$HAL_STORE/ckpt`) | `HAL_CKPT_DIR`, `HAL_CKPT_{SINGLE_ARM,BIMANUAL,HUMANOID}` |
| WorldScore models (GDINO, SAM2, SEA-RAFT, VFIMamba) | `checkpoints/metrics/worldscore/` | per-probe `--*_ckpt` flags |

Each default falls back to the pre-refactor location (`checkpoints/{fvd,vggt_omega}/`,
the kinescore store) while the compat symlinks exist.

## Quick start

```bash
PY=models/dreamgen/.venv/bin/python        # has piq + torchmetrics + CLIP weights
export HF_HUB_OFFLINE=1

# fidelity + CLIP over one cell (pred X.mp4, ref X_gt.mp4, prompt X.txt)
$PY -m metrics.evaluate \
    --pred-root sampling_dataset/dense/single_arm/output/singleview/dreamgen/makovian \
    --metrics fidelity clip_score --out results/metrics/dense_sa_mk

# FVD of one cell vs its own GT (per-configuration); target_fps=5 is the default
$PY -m metrics.evaluate --pred-root <same dir> --metrics fvd --out results/metrics/dense_sa_mk_fvd

# KIVA, with the real-video anchor scored in the same launcher call
$PY -m metrics.evaluate --pred-root <same dir> --metrics kiva \
    --opt kiva.embodiment=single_arm --opt kiva.score_ref=true \
    --out results/metrics/dense_sa_mk_kiva

# RIGIS over extracted episode dirs (<ep>/view1, <ep>/view2)
$PY -m metrics.evaluate \
    --pred-root sampling_dataset/multiview_consistency/frames/dense/bimanual/ctrlworld/non_makovian/<task> \
    --pred-glob '*' --ref-template '' --text-template '' \
    --metrics rigis --out results/metrics/dense_bi_rigis
```

`--out` gets `<metric>.json` (full result incl. config and per-sample rows),
`per_sample.csv` (all metrics joined on sample id) and `summary.json`. Outputs
are rewritten after every metric, so a crash in the last one keeps the rest.

Per-metric constructor arguments go through `--opt METRIC.KEY=VALUE` (values
parsed as JSON: `true`, `16`, `"x"`).

### Sample discovery

| flag | default | meaning |
|---|---|---|
| `--pred-glob` | `**/*.mp4` | what counts as a prediction (files or dirs) |
| `--exclude-suffix` | `_gt` | stems ending in this are never predictions |
| `--ref-template` | `{stem}_gt{suffix}` | ref next to pred; `''` disables; `--ref-root` looks elsewhere |
| `--text-template` | `{stem}.txt` | prompt sidecar, `[Prompt]` section; `--prompt` overrides |

Sample id = path relative to `--pred-root` without extension. A missing ref or
prompt does not crash discovery; only metrics that need it fail for that sample
(reported under `errors`).

## Python API

```python
from metrics import Sample, build_metric

fid = build_metric("fidelity", device="cuda")
res = fid.evaluate([Sample("ep0", pred="ep0.mp4", ref="ep0_gt.mp4")])
res.summary["psnr"]          # {'mean', 'std', 'sem', 'n', 'n_missing'}
res.per_sample["ep0"]        # {'psnr', 'ssim', 'lpips', 'n_frames_pred', 'n_frames_ref', 'n_frames'}
res.save("out/fidelity.json")

fid(Sample("ep1", pred=np_uint8_THWC, ref=np_uint8_THWC))   # one sample -> dict
```

## Writing a new metric

```python
from metrics.base import FrameMetric
from metrics.registry import register

@register("mean_abs_diff")
class MeanAbsDiff(FrameMetric):
    higher_is_better = {"mad": False}
    requires_ref = True
    name = "mean_abs_diff"

    def score_frames(self, pred, ref, text):          # aligned uint8 (T,H,W,3)
        return {"mad": float(abs(pred.astype(int) - ref).mean())}
```

* **In-process** metrics subclass `FrameMetric`: decoding, frame alignment,
  shape checks and the `n_frames_*` bookkeeping are inherited. Load models in
  `setup()` (called lazily once). `self._chunks(pred, ref)` yields float [0,1]
  NCHW tensors on `self.device`.
* **Out-of-process** metrics (own venv/model service) subclass
  `ExternalMetric`, set `launcher`, and implement `run_batch(samples)`: stage
  inputs under `self.new_run_dir()`, `self.run_launcher(env, run_dir)`, parse
  the output into `{sample.id: {key: float}}`. The model is loaded once per
  batch; never call it per sample in a loop.
* Return floats keyed like `higher_is_better`; extra keys are kept per-sample
  but not aggregated. Use NaN for "not scorable", raise for real failures.

## Conventions that decide whether numbers are comparable

* **Fidelity = the inference scripts' numbers.** `piq` with defaults on float
  [0,1] NCHW, frames truncated to `min(T_pred, T_gt)` — the exact recipe in
  every `scripts/infer_*.py`. `test_parity_with_inference_scripts` checks this
  on a real episode (rel. tol 1e-4). Generated and GT clips often differ in
  length (e.g. 185 vs 164 frames); the truncation is reported per sample via
  `n_frames_pred / n_frames_ref / n_frames`. Use `--opt fidelity.frame_align="error"`
  to refuse mismatches instead.
* piq vs skimage (used by `scripts/metrics/_score_vs_dense.py`): identical frames give
  ~80 dB in piq vs `inf` in skimage; piq SSIM downsamples by
  `round(min(H,W)/256)` first. Do not mix the two in one table.
* **No implicit resizing.** pred/ref resolution mismatch raises.
* **CLIP score** reproduces `metrics/quality/quality_control_probe.py`
  (WorldScore `content_alignment`): Resize 512 → uint8 → CLIPScore, frame by
  frame, mean. Same scale as the PhyWorld `task_lock` input. (The probe
  round-trips frames through JPEG on disk, so expect ~0.1-level differences,
  not bit-identity.) It is reference-free, so it scores **all** pred frames even
  when the GT is shorter; only reference-based metrics truncate.
* **CLIP weights offline**: `metrics/kiva/run.sh` exports `HF_HOME`
  into the kinescore store. If that leaks into a tmux server's global env
  (`tmux show-environment -g`), CLIP is looked up in the wrong cache;
  `clip_score` falls back to `~/.cache/huggingface` and prints which it used.
* **RIGIS**: reference-free; the real-video ceiling is ≈ 0.89,
  so report the GT anchor alongside every variant. Deterministic: the wrapper
  reproduces the standalone launcher's CSV to the last digit.
* **KIVA**: rates are only meaningful as excess over the real-video
  anchor (`score_ref=true` adds `<det>_rate_ref` and `<det>_rate_excess`).
  Only the single-arm A1X reader is validated to fire; bimanual/humanoid
  readers currently read ~0 everywhere (see the doc). Inputs must be at the
  reader's resolution (640×480 single-arm, 768×432 bimanual/humanoid).
  **Not bit-reproducible**: kinescore re-calibrates on every call and its
  diffusion keypoint head samples unseeded `torch.randn`
  (`heads/diffusion.py:305`), so the same clip can move by one segment between
  runs (measured: A1X segment threshold 8.34 vs 8.18 mm; one clip's
  rigidity_rate 0.083 vs 0.167; jerk_rate identical). Score every variant
  being compared — and its anchor — in **one** call so they share thresholds,
  and treat sub-segment differences as noise. With torch seeded (`manual_seed(0)`) the
  vendored copy reproduces the old `external/hallucination` worktree bit-for-bit
  (2026-10-05 parity test, 12/12 segments identical).
* **FVD is a set statistic** (one number per cell, effective n = 1): no CI, no
  per-cell claim, < 64 windows per side -> NaN. `fvd.py` is the single implementation
  behind `scripts/metrics/compute_dr_metrics.py` (dr-v1, native fps, own GT) and the
  shared-pool scripts in `assets/scripts/` (via the `fvd_pool_common.py` shim);
  the refactor reproduced their I3D features exactly (max abs diff 0.0).

## Tests

```bash
models/dreamgen/.venv/bin/python -m pytest metrics/test_framework.py -q    # ~4 min, CPU
```

The legacy function API (`calculate_psnr` & co. in `video_metrics.py`) is kept
importable but uses a box-filter SSIM and the `lpips` package — its numbers are
**not** the published ones; prefer `fidelity`. (Its old smoke test
`test_metrics.py` was archived to `tmp/20261005_refactor/misc/metrics/`.)
