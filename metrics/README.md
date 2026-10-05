# `metrics/` — unified evaluation metrics

Every metric implements one abstract interface, `metrics.base.BaseMetric`, so
any of them can be run, aggregated and saved the same way, from Python or from
one CLI.

| name | family | needs | ↑/↓ | backend | interpreter |
|---|---|---|---|---|---|
| `psnr` | fidelity | pred + ref | ↑ | `piq.psnr` | dreamgen venv |
| `ssim` | fidelity | pred + ref | ↑ | `piq.ssim` | dreamgen venv |
| `lpips` | fidelity | pred + ref | ↓ | `piq.LPIPS` (VGG16) | dreamgen venv |
| `fidelity` | all three in one decode pass | pred + ref | | `piq` | dreamgen venv |
| `clip_score` | text alignment | pred + prompt | ↑ (0–100) | torchmetrics CLIP ViT-B/16 | dreamgen venv |
| `multiview_consistency` | 3D geometry | multiview episode | ↑ | VGGT-Omega, `multiview_consistency/run.sh` | own `uv` project (launched for you) |
| `hallucination` | embodiment physics | pred mp4 (+ gt anchor) | ↓ | kinescore, `hallucination/run.sh` | `hdmi` conda env (launched for you) |

```
metrics/
├── base.py                  BaseMetric (ABC) -> FrameMetric / ExternalMetric, Sample, MetricResult
├── registry.py              name -> class (lazy), register(), build_metric()
├── evaluate.py              CLI: discover samples, run metrics, write json/csv
├── video_io.py              mp4 / frame dir / array -> uint8 (T,H,W,3); prompt sidecars
├── fidelity.py              PSNR, SSIM, LPIPS, Fidelity
├── clip_score.py            CLIPScore
├── multiview_consistency/   run.sh, extract_frames.py, aggregate.py, metric.py
├── hallucination/           run.sh, aggregate.py, probe_clips.py, map_to_dr.py, metric.py
├── test_framework.py        unit + parity tests
└── (pre-existing) dynamics/ quality/ phyworld/ WorldScore/ video_metrics.py
```

The two external metrics' scripts used to live in `scripts/`; their full
write-ups are `docs/multiview_consistency_metric.md` and
`docs/hallucination_metric.md`. Those launchers still work standalone for
large sweeps (`VARIANTS="dense pisa ..." bash metrics/hallucination/run.sh`).

## Quick start

```bash
PY=models/dreamgen/.venv/bin/python        # has piq + torchmetrics + CLIP weights
export HF_HUB_OFFLINE=1

# fidelity + CLIP over one cell (pred X.mp4, ref X_gt.mp4, prompt X.txt)
$PY -m metrics.evaluate \
    --pred-root sampling_dataset/dense/single_arm/output/singleview/dreamgen/makovian \
    --metrics fidelity clip_score --out results/metrics/dense_sa_mk

# hallucination, with the real-video anchor scored in the same launcher call
$PY -m metrics.evaluate --pred-root <same dir> --metrics hallucination \
    --opt hallucination.embodiment=single_arm --opt hallucination.score_ref=true \
    --out results/metrics/dense_sa_mk_hal

# multiview consistency over extracted episode dirs (<ep>/view1, <ep>/view2)
$PY -m metrics.evaluate \
    --pred-root sampling_dataset/multiview_consistency/frames/dense/bimanual/ctrlworld/non_makovian/<task> \
    --pred-glob '*' --ref-template '' --text-template '' \
    --metrics multiview_consistency --out results/metrics/dense_bi_mvc
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
* piq vs skimage (used by `scripts/_score_vs_dense.py`): identical frames give
  ~80 dB in piq vs `inf` in skimage; piq SSIM downsamples by
  `round(min(H,W)/256)` first. Do not mix the two in one table.
* **No implicit resizing.** pred/ref resolution mismatch raises.
* **CLIP score** reproduces `metrics/quality/quality_control_probe.py`
  (WorldScore `content_alignment`): Resize 512 → uint8 → CLIPScore, frame by
  frame, mean. Same scale as the PhyWorld `task_lock` input. (The probe
  round-trips frames through JPEG on disk, so expect ~0.1-level differences,
  not bit-identity.) It is reference-free, so it scores **all** pred frames even
  when the GT is shorter; only reference-based metrics truncate.
* **CLIP weights offline**: `metrics/hallucination/run.sh` exports `HF_HOME`
  into the kinescore store. If that leaks into a tmux server's global env
  (`tmux show-environment -g`), CLIP is looked up in the wrong cache;
  `clip_score` falls back to `~/.cache/huggingface` and prints which it used.
* **Multiview consistency**: reference-free; the real-video ceiling is ≈ 0.89,
  so report the GT anchor alongside every variant. Deterministic: the wrapper
  reproduces the standalone launcher's CSV to the last digit.
* **Hallucination**: rates are only meaningful as excess over the real-video
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
  and treat sub-segment differences as noise.

## Tests

```bash
models/dreamgen/.venv/bin/python -m pytest metrics/test_framework.py -q    # ~4 min, CPU
```

The pre-existing `metrics/test_metrics.py` covers the legacy function API
(`calculate_psnr` & co. in `video_metrics.py`), which is kept importable but
uses a box-filter SSIM and the `lpips` package — its numbers are **not** the
published ones; prefer `fidelity`.
