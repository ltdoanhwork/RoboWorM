# Adding a new metric

Every metric in RoboWorM is a subclass of `metrics.base.BaseMetric`, registered by name
in `metrics/registry.py` and run by `python -m metrics.evaluate`. A new metric that
follows this interface is immediately runnable on any `sampling_dataset/` tree and
writes the same per-sample / summary schema as the existing ones.

Read first: [`metrics/base.py`](../../metrics/base.py) (the contract, in its module
docstring), [`metrics/registry.py`](../../metrics/registry.py),
[`metrics/evaluate.py`](../../metrics/evaluate.py), and
[`metrics/README.md`](../../metrics/README.md).

## 1. Pick the base class

| base class (`metrics/base.py`) | use when | you implement | existing examples |
|---|---|---|---|
| `FrameMetric` | in-process, per clip, on decoded RGB frames (optionally vs a reference) | `score_frames(pred, ref, text) -> dict` | `PSNR`, `SSIM`, `LPIPS` (`metrics/fidelity.py`) |
| `ReferenceFreeFrameMetric(FrameMetric)` | no reference; must use the paper analysis grid (5 fps, conditioning frame dropped, view crop, 256×192, every 5th frame, JPEG round trip) | `score_frames`, using `self.pil_frames(pred)` | `VisualIntegrity` (CLIP-IQA+, `metrics/visual_integrity.py`), `TextAlignment` (CLIPScore, `metrics/text_alignment.py`) |
| `ExternalMetric` | needs its own environment / a long-lived model; batch-only | `run_batch(samples) -> {id: {key: float}}` + `launcher` | `MultiviewConsistency` = RIGIS (`metrics/rigis/metric.py`), `Hallucination` = KIVA (`metrics/kiva/metric.py`) |
| `BaseMetric` directly | a set-level statistic over the whole batch | `compute`, override `evaluate` | `FVD` (`metrics/fvd/core.py`) |

The contract (from `metrics/base.py`):

- `compute` returns plain floats keyed by the names in `higher_is_better`. Extra
  bookkeeping keys are allowed; only keys listed in `higher_is_better` are aggregated.
- `NaN` = "not scorable for this sample": excluded from the mean, counted in
  `n_missing`. Raise for hard failures; `evaluate` records them in
  `MetricResult.errors` (or re-raises with `--fail-fast`).
- No silent resampling. Shape mismatches raise; frame-count mismatches follow
  `FrameMetric(frame_align="truncate" | "error")`.
- Load heavy models in `setup()`; it runs lazily once, so constructing a metric is cheap.
- Class attributes: `name`, `higher_is_better`, `requires_ref`, `requires_text`.

## 2. Minimal skeletons

### 2.1 Frame metric (verified against the current interfaces)

```python
# metrics/mean_brightness.py
import numpy as np

from .base import FrameMetric


class MeanBrightness(FrameMetric):
    """Toy example: mean pixel intensity in [0, 1]."""

    name = "mean_brightness"
    higher_is_better = {"mean_brightness": True}

    def score_frames(self, pred: np.ndarray, ref, text) -> dict[str, float]:
        # pred: aligned uint8 (T, H, W, 3); ref is None unless requires_ref = True
        return {"mean_brightness": float(pred.astype(np.float32).mean() / 255.0)}
```

For GPU work inside `score_frames`, `self._chunks(pred, ref)` yields float `[0, 1]`
NCHW tensor chunks of size `self.chunk` on `self.device` (see `_PiqFrameMetric` in
`metrics/fidelity.py`).

### 2.2 Reference-free metric on the paper grid

```python
import numpy as np

from .base import ReferenceFreeFrameMetric


class MyNR(ReferenceFreeFrameMetric):
    name = "my_nr"
    higher_is_better = {"my_nr": True}

    def setup(self) -> None:
        self._model = ...                      # load once

    def score_frames(self, pred, ref, text) -> dict[str, float]:
        frames = self.pil_frames(pred)         # every frame_stride-th frame, JPEG round trip
        if not frames:
            raise ValueError("no frames left to score")
        scores = [self._score(im) for im in frames]
        return {"my_nr": float(np.mean(scores)), "n_frames_scored": float(len(scores))}
```

With `protocol="paper"` (default) and an mp4 `Sample.pred`, `compute` calls
`metrics.video_io.nr_analysis_frames(path, layout, view_index, n_views)` with
`layout` / `view_index` / `n_views` taken from `Sample.meta` (`"single"`, `"strip"` for
Ctrl-World 1×N, `"grid"` for DreamGen 2×2). `metrics.evaluate` does **not** fill
`Sample.meta`, so from the CLI every clip is treated as `layout="single"`. For mosaic
rollouts, build `Sample(..., meta={"layout": "strip", "view_index": i, "n_views": 4})`
in Python, as `scripts/metrics/run_dense_nr.py` does per view.

### 2.3 External metric (own environment)

```python
from pathlib import Path
from typing import Sequence

from ..base import ExternalMetric, Sample, staged_names


class MyExternal(ExternalMetric):
    name = "my_external"
    launcher = "metrics/my_external/run.sh"    # relative to the repo root
    higher_is_better = {"score": True}

    def run_batch(self, samples: Sequence[Sample]) -> dict[str, dict[str, float]]:
        run_dir = self.new_run_dir()           # under work_dir or $TMPDIR, kept after the run
        names = staged_names(samples)          # filesystem-safe, collision-checked ids
        inp = run_dir / "in"
        inp.mkdir()
        for name, s in names.items():
            (inp / f"{name}.mp4").symlink_to(Path(s.pred).resolve())
        self.run_launcher({"IN_DIR": inp, "OUT_CSV": run_dir / "out.csv"}, run_dir)
        rows = {}
        ...                                    # parse out.csv, map staged name -> s.id
        return rows
```

`run_launcher` runs `bash <launcher>` from the repo root with your env overlaid, sets
`CUDA_VISIBLE_DEVICES` from `device="cuda:N"` if unset, tees output to
`<run_dir>/launcher.log`, and raises with the log tail on a non-zero exit. The launcher
owns its interpreter (RIGIS: `uv run --project metrics/rigis`; KIVA: `HAL_PYTHON`), so
the wrapper runs from any Python with numpy.

## 3. Register it

Add one **string** entry to `_REGISTRY` in `metrics/registry.py`. Entries are imported
lazily, so listing metrics never imports torch:

```python
_REGISTRY = {
    ...
    "mean_brightness": "metrics.mean_brightness:MeanBrightness",
}
ALIASES = {..., "brightness": "mean_brightness"}   # optional, old / short names
```

`@register("name")` from `metrics.registry` also works, but only once the defining module
has been imported, which `python -m metrics.evaluate` does not do for you. Use the
`_REGISTRY` string for anything that should be reachable from the CLI.

Check: `python -c "from metrics.registry import available; print(available())"`.

## 4. Run it

```bash
PY=models/dreamgen/.venv/bin/python
CELL=sampling_dataset/dense/single_arm/output/singleview/dreamgen/makovian
$PY -m metrics.evaluate --pred-root $CELL --metrics mean_brightness --limit 5 \
    --opt mean_brightness.chunk=16 --out /tmp/metric_trial
```

Sample discovery (`metrics.evaluate.discover_samples`): `--pred-glob` (default
`**/*.mp4`) under `--pred-root`; reference `--ref-template` (default `{stem}_gt{suffix}`,
`''` = none, `--ref-root` to look elsewhere); prompt `--text-template` (default
`{stem}.txt`, `[Prompt]` section) or `--prompt`; files ending in `--exclude-suffix`
(default `_gt`) are never predictions. A glob that matches directories yields directory
samples (RIGIS episode dirs).

`--opt METRIC.KEY=VALUE` becomes a constructor kwarg (value parsed as JSON when
possible). The `METRIC` part is matched against the name **as typed in `--metrics`**,
not the alias-resolved name: `--metrics kiva --opt kiva.embodiment=bimanual`.

Each `--out` gets `<metric>.json` (full `MetricResult`: per-sample, summary with
`mean/std/sem/n/n_missing`, `higher_is_better`, `config`, `errors`, `seconds`),
`per_sample.csv` (all metrics joined on sample id) and `summary.json`. Outputs are
rewritten after each metric, so a later crash keeps earlier results.

Python API:

```python
from metrics import Sample, build_metric
m = build_metric("mean_brightness", device="cpu")
res = m.evaluate([Sample(id="ep0", pred="ep0.mp4")])
res.summary["mean_brightness"]["mean"]; res.save("out/mean_brightness.json")
```

Unit tests for the framework: `models/dreamgen/.venv/bin/python -m pytest metrics/test_framework.py -q`.
Add a CPU-only synthetic test for your metric there (see `test_reference_free_metric_ignores_ref`).

## 5. Getting it into the benchmark tables

The registry gives a runnable metric; the paper tables are fed by two batch "waves"
that store per-clip records and pair them at table time:

| family | wave | stores | notes |
|---|---|---|---|
| reference-free (like Visual Integrity / Text Alignment) | `scripts/metrics/run_dense_nr.py` (via `run_dense_nr.sh`) | one record per view-clip, `KEYS`, `PROTOCOL` | inventory walks every `sampling_dataset/<method>` tree (`_pred_rows`) plus co-located GT; `pair_key` joins accel ↔ dense ↔ GT |
| reference-based vs dense (PSNR / SSIM / LPIPS, FVD) | `scripts/metrics/compute_dr_metrics.py` (via `run_dr_metrics.sh`) | per-clip `vs_dense` and `vs_gt` | frames truncated to `min(len)`, size mismatch skipped |
| physical audits | `metrics/kiva/run.sh`, `metrics/rigis/run.sh` | per-segment / per-episode CSVs | see [../metrics/kiva.md](../metrics/kiva.md), [../metrics/rigis.md](../metrics/rigis.md) |

Adding a column to a wave is a code change in that script (append to its key list, bump
its `PROTOCOL` only if the change is not reconstructible from stored records). Then add
the column to the producer that builds the table (see
[../reproduce/README.md](../reproduce/README.md)).

## 6. Validity protocol for a new metric (paper Sec. 3, App. B.3, App. E.4)

- **Polarity is not permission to rank.** Audit the column against real video: on paired
  clips, real video (the plausibility ceiling) must outscore the prediction. Visual
  Integrity fails this audit and is therefore restricted to acceleration comparisons.
- **Return `NaN` (stored as `None`) when the quantity is not measurable** — e.g. no
  robot detected, too few valid pairs. `0` means "measured, nothing there".
- **Store the per-frame / per-pair inputs** with the score, so validity gates and
  thresholds can be re-applied at table time without re-scoring.
- **Use the analysis grid** for anything reference-free (`ReferenceFreeFrameMetric`), and
  aggregate only within one (model, embodiment, camera mount, regime) stratum.
- **Fixed constants, not fitted to human labels.** If thresholds come from real video,
  compute them per source, as KIVA does.
- If the metric needs weights, put them in `checkpoints/metrics/<name>/` and add a row to
  "Metric checkpoints" in [`checkpoints/CHECKPOINTS.md`](../../checkpoints/CHECKPOINTS.md).
  Document the setup in `docs/metrics/<name>.md` and link it from
  [../metrics/README.md](../metrics/README.md).

## Checklist

- [ ] Subclass of the right base (`FrameMetric`, `ReferenceFreeFrameMetric`, `ExternalMetric`, or `BaseMetric`).
- [ ] `name`, `higher_is_better`, `requires_ref` / `requires_text` set; heavy loading in `setup()`.
- [ ] `NaN` for not-scorable, exceptions for hard failures, no implicit resize.
- [ ] String entry in `metrics/registry.py::_REGISTRY` (+ optional `ALIASES`); appears in `available()`.
- [ ] `python -m metrics.evaluate --metrics <name> --limit 5 --out /tmp/...` works on a real cell.
- [ ] CPU test in `metrics/test_framework.py`.
- [ ] Real-video ceiling audit run; ranking restrictions written down.
- [ ] Weights under `checkpoints/metrics/<name>/` + `CHECKPOINTS.md` row; env documented (own env → `ExternalMetric`).
- [ ] Wave / producer integration if it is to appear in the paper tables.
- [ ] `docs/metrics/<name>.md` + a row with the source-paper citation in `metrics/README.md` and `docs/metrics/README.md`.
