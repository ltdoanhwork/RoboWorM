"""Abstract metric interface shared by every metric in ``metrics/``.

Three layers::

    BaseMetric      (ABC)  sample -> {key: float};  evaluate() -> MetricResult
    ├── FrameMetric        in-process, decodes frames, scores (pred[, ref][, text])
    │   ├── PSNR / SSIM / LPIPS              metrics/fidelity.py
    │   ├── MotionSmoothness / MotionAmplitude  metrics/motion_{smoothness,amplitude}.py
    │   └── ReferenceFreeFrameMetric   paper analysis frames (5 fps, 256x192, JPEG)
    │       ├── VisualIntegrity (CLIP-IQA+)   metrics/visual_integrity.py
    │       └── TextAlignment   (CLIPScore)   metrics/text_alignment.py
    ├── FVD                set-level I3D Frechet distance  metrics/fvd/
    └── ExternalMetric     batch-only, shells out to a launcher in its own env
        ├── RIGIS                            metrics/rigis/metric.py
        └── KIVA                             metrics/kiva/metric.py

Why two subclasses: the reference metrics need ``piq``/``torchmetrics`` (the
``models/dreamgen/.venv``), while VGGT-Omega runs in a ``uv`` project and kinescore
in the ``hdmi`` conda env. They cannot share one interpreter, so ExternalMetric
stages the samples on disk, runs the existing launcher once for the whole batch
(one model load), and parses its output back into the same per-sample schema.

Contract every subclass must honour:

* ``compute`` returns plain floats keyed by names listed in ``higher_is_better``
  (extra bookkeeping keys such as ``n_frames`` are allowed and not aggregated
  unless also listed there).
* NaN means "not scorable for this sample"; it is excluded from the mean and
  counted in ``n_missing``. Hard failures go to ``MetricResult.errors``.
* No silent resampling: shape mismatches raise, frame-count mismatches follow an
  explicit policy (see ``FrameMetric.frame_align``).
"""
from __future__ import annotations

import json
import math
import statistics
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Iterable, Sequence

import numpy as np

@dataclass
class Sample:
    """One unit of evaluation.

    ``pred`` / ``ref`` are video sources: an mp4 path, a directory of frame
    images, or a uint8 (T, H, W, 3) array. Multiview consistency instead takes an
    episode dir holding ``view1/`` and ``view2/`` frame folders (or a mosaic mp4).

    ``id`` must be unique within a batch; it is the join key between per-sample
    rows of different metrics.
    """

    id: str
    pred: Any
    ref: Any = None
    text: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    metric: str
    per_sample: dict[str, dict[str, float]]
    summary: dict[str, dict[str, float]]
    higher_is_better: dict[str, bool]
    config: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def headline(self) -> str:
        parts = []
        for k, hib in self.higher_is_better.items():
            s = self.summary.get(k)
            if s and s["n"]:
                arrow = "↑" if hib else "↓"
                parts.append(f"{k}{arrow}={s['mean']:.4f}±{s['sem']:.4f} (n={s['n']})")
        tail = f"  [{len(self.errors)} failed]" if self.errors else ""
        return f"{self.metric}: " + ("  ".join(parts) or "no scored samples") + tail


class BaseMetric(ABC):
    """Abstract metric.

    Subclasses set the class attributes and implement :meth:`compute`.
    Heavy models are loaded in :meth:`setup`, which runs lazily on first use so
    constructing a metric (e.g. to inspect its config) is always cheap.
    """

    #: registry name, e.g. ``"psnr"``
    name: ClassVar[str] = ""
    #: aggregated output keys -> True if higher is better
    higher_is_better: ClassVar[dict[str, bool]] = {}
    #: sample.ref must be set
    requires_ref: ClassVar[bool] = False
    #: sample.text must be set
    requires_text: ClassVar[bool] = False

    def __init__(self, device: str = "cuda", **config: Any) -> None:
        self.device = device
        self.config: dict[str, Any] = {"device": device, **config}
        self._ready = False

    # -- lifecycle --------------------------------------------------------
    def setup(self) -> None:
        """Load models / resolve checkpoints. Called once, lazily."""

    def ensure_setup(self) -> None:
        if not self._ready:
            self.setup()
            self._ready = True

    # -- per-sample -------------------------------------------------------
    @abstractmethod
    def compute(self, sample: Sample) -> dict[str, float]:
        """Score one sample. Must return floats keyed like ``higher_is_better``."""

    def validate(self, sample: Sample) -> None:
        if self.requires_ref and sample.ref is None:
            raise ValueError(f"{self.name} needs a reference video (sample {sample.id!r})")
        if self.requires_text and not sample.text:
            raise ValueError(f"{self.name} needs a text prompt (sample {sample.id!r})")

    # -- batch ------------------------------------------------------------
    def evaluate(self, samples: Sequence[Sample], *, fail_fast: bool = False,
                 verbose: bool = True) -> MetricResult:
        """Score every sample, collect failures, aggregate."""
        _check_unique_ids(samples)
        t0 = time.time()
        self.ensure_setup()
        per_sample: dict[str, dict[str, float]] = {}
        errors: dict[str, str] = {}
        for i, s in enumerate(samples):
            try:
                self.validate(s)
                per_sample[s.id] = {k: float(v) for k, v in self.compute(s).items()}
            except Exception as e:  # noqa: BLE001 -- recorded, not swallowed
                if fail_fast:
                    raise
                errors[s.id] = f"{type(e).__name__}: {e}"
                if verbose:
                    print(f"[{self.name}] FAIL {s.id}: {errors[s.id]}")
                    traceback.print_exc()
                continue
            if verbose:
                row = "  ".join(f"{k}={v:.4f}" for k, v in per_sample[s.id].items()
                                if k in self.higher_is_better)
                print(f"[{self.name}] {i + 1}/{len(samples)} {s.id}: {row}")
        return self._result(per_sample, errors, time.time() - t0)

    def aggregate(self, per_sample: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        """Mean / std / sem over samples per key; NaN is excluded and counted."""
        out: dict[str, dict[str, float]] = {}
        for key in self.higher_is_better:
            vals = [r[key] for r in per_sample.values() if key in r]
            finite = [v for v in vals if v is not None and math.isfinite(v)]
            out[key] = _stats(finite)
            out[key]["n_missing"] = len(per_sample) - len(finite)
        return out

    def _result(self, per_sample, errors, seconds) -> MetricResult:
        return MetricResult(
            metric=self.name,
            per_sample=per_sample,
            summary=self.aggregate(per_sample),
            higher_is_better=dict(self.higher_is_better),
            config=self.describe(),
            errors=errors,
            seconds=round(seconds, 2),
        )

    def describe(self) -> dict[str, Any]:
        """Everything needed to reproduce a number; stored in every result."""
        return {"class": f"{type(self).__module__}.{type(self).__qualname__}", **self.config}

    def __call__(self, sample: Sample) -> dict[str, float]:
        self.ensure_setup()
        self.validate(sample)
        return self.compute(sample)

    def __repr__(self) -> str:
        cfg = ", ".join(f"{k}={v!r}" for k, v in self.config.items())
        return f"{type(self).__name__}({cfg})"


class FrameMetric(BaseMetric):
    """Metric over decoded RGB frames, computed in this process.

    Subclasses implement :meth:`score_frames`; this class handles decoding,
    frame alignment and chunking so every frame metric aligns identically.

    ``frame_align``:
      * ``"truncate"`` (default) -- keep the first ``min(T_pred, T_ref)`` frames
        of both. This is what every ``scripts/infer_*.py`` does before its
        ``piq`` call, so numbers stay comparable with the published ones.
      * ``"error"`` -- raise on any length mismatch.
    ``n_frames_pred`` / ``n_frames_ref`` / ``n_frames`` are always reported, so a
    truncation is visible in the per-sample rows.
    """

    def __init__(self, device: str = "cuda", frame_align: str = "truncate",
                 chunk: int = 32, **config: Any) -> None:
        if frame_align not in ("truncate", "error"):
            raise ValueError(f"frame_align must be 'truncate' or 'error', got {frame_align!r}")
        super().__init__(device=device, frame_align=frame_align, chunk=chunk, **config)
        self.frame_align = frame_align
        self.chunk = int(chunk)

    def compute(self, sample: Sample) -> dict[str, float]:
        from .video_io import load_frames

        pred = load_frames(sample.pred)
        # Only reference-based metrics may look at (or be truncated by) the ref.
        ref = load_frames(sample.ref) if (self.requires_ref and sample.ref is not None) else None
        bookkeeping = {"n_frames_pred": float(len(pred))}
        if ref is not None:
            bookkeeping["n_frames_ref"] = float(len(ref))
            if pred.shape[1:] != ref.shape[1:]:
                raise ValueError(f"resolution mismatch pred {pred.shape[1:]} vs ref {ref.shape[1:]}"
                                 " (no implicit resize; resample upstream)")
            if len(pred) != len(ref):
                if self.frame_align == "error":
                    raise ValueError(f"frame count mismatch pred={len(pred)} ref={len(ref)}")
                n = min(len(pred), len(ref))
                pred, ref = pred[:n], ref[:n]
        if len(pred) == 0:
            raise ValueError("no frames decoded")
        bookkeeping["n_frames"] = float(len(pred))
        return {**self.score_frames(pred, ref, sample.text), **bookkeeping}

    @abstractmethod
    def score_frames(self, pred: np.ndarray, ref: np.ndarray | None,
                     text: str | None) -> dict[str, float]:
        """``pred``/``ref`` are aligned uint8 (T, H, W, 3)."""

    # helper for subclasses: uint8 THWC numpy -> float [0,1] TCHW tensor chunks
    def _chunks(self, *arrays: np.ndarray | None) -> Iterable[tuple]:
        import torch

        T = len(arrays[0])
        for s in range(0, T, self.chunk):
            yield tuple(
                None if a is None else
                torch.from_numpy(np.ascontiguousarray(a[s:s + self.chunk]))
                .to(self.device).permute(0, 3, 1, 2).float().div_(255.0).clamp_(0, 1)
                for a in arrays
            )


class ReferenceFreeFrameMetric(FrameMetric):
    """Per-frame reference-free metric scored on the paper's analysis frames.

    ``protocol="paper"`` (default) reproduces the published numbers: a video file is
    decoded with :func:`metrics.video_io.nr_analysis_frames` (5 fps, conditioning frame
    dropped, camera tile from ``sample.meta`` ``layout`` / ``view_index`` / ``n_views``,
    256x192), every ``frame_stride``-th frame (default 5) is kept, and each kept frame
    goes through the same JPEG round trip the original pipeline had. An in-memory
    array is taken to be analysis frames already (stride + JPEG still apply).

    ``protocol="raw"`` scores the frames exactly as decoded (stride default 1, no JPEG):
    use it for generic clips that are not benchmark rollouts.
    """

    def __init__(self, device: str = "cuda", protocol: str = "paper",
                 frame_stride: int | None = None, jpeg: bool | None = None,
                 **config: Any) -> None:
        if protocol not in ("paper", "raw"):
            raise ValueError(f"protocol must be 'paper' or 'raw', got {protocol!r}")
        paper = protocol == "paper"
        frame_stride = int(frame_stride or (5 if paper else 1))
        jpeg = paper if jpeg is None else bool(jpeg)
        super().__init__(device=device, protocol=protocol, frame_stride=frame_stride,
                         jpeg=jpeg, **config)
        self.protocol, self.frame_stride, self.jpeg = protocol, max(1, frame_stride), jpeg

    def compute(self, sample: Sample) -> dict[str, float]:
        if self.protocol == "paper" and not isinstance(sample.pred, np.ndarray) \
                and Path(sample.pred).is_file():
            from .video_io import nr_analysis_frames

            m = sample.meta or {}
            frames, _ = nr_analysis_frames(sample.pred, m.get("layout", "single"),
                                           int(m.get("view_index", 0)), int(m.get("n_views", 1)))
            out = self.score_frames(frames, None, sample.text)
            return {**out, "n_frames_pred": float(len(frames)), "n_frames": float(len(frames))}
        return super().compute(sample)

    def pil_frames(self, frames: np.ndarray) -> list:
        """The frames that are actually scored, as PIL RGB images."""
        from PIL import Image

        from .video_io import jpeg_roundtrip

        sel = frames[::self.frame_stride]
        return [jpeg_roundtrip(f) if self.jpeg else Image.fromarray(f) for f in sel]


class ExternalMetric(BaseMetric):
    """Batch metric that runs an out-of-process launcher.

    Subclasses implement :meth:`run_batch`, which receives every valid sample
    at once and returns ``{sample_id: {key: float}}``. ``compute`` on a single
    sample is supported but pays one full model load per call -- prefer
    :meth:`evaluate`.

    ``work_dir``: where staged inputs + raw launcher outputs go (default: a fresh
    dir under ``$TMPDIR``; the pfss workspace quota is tight). Kept after the run
    so the raw CSVs can be re-aggregated with the launcher's own aggregate.py.
    """

    #: launcher path relative to the repo root
    launcher: ClassVar[str] = ""

    def __init__(self, device: str = "cuda", work_dir: str | None = None,
                 log_path: str | None = None, **config: Any) -> None:
        super().__init__(device=device, work_dir=work_dir, log_path=log_path, **config)
        self.work_dir = work_dir
        self.log_path = log_path

    def setup(self) -> None:
        if not self.launcher_path().is_file():
            raise FileNotFoundError(self.launcher_path())

    def launcher_path(self) -> Path:
        return REPO_ROOT / self.launcher

    def new_run_dir(self) -> Path:
        import tempfile

        if self.work_dir:
            base = Path(self.work_dir)
            base.mkdir(parents=True, exist_ok=True)
            return Path(tempfile.mkdtemp(prefix=f"{self.name}_", dir=base))
        return Path(tempfile.mkdtemp(prefix=f"{self.name}_"))

    def run_launcher(self, env: dict[str, str], run_dir: Path) -> None:
        """``bash <launcher>`` with ``env`` overlaid; output tee'd to a log."""
        import os
        import subprocess

        log = Path(self.log_path) if self.log_path else run_dir / "launcher.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        full_env = {**os.environ, **{k: str(v) for k, v in env.items()}}
        full_env.setdefault("CUDA_VISIBLE_DEVICES", _cuda_index(self.device))
        print(f"[{self.name}] launcher {self.launcher}  (log {log})")
        with log.open("w") as fh:
            proc = subprocess.run(["bash", str(self.launcher_path())], cwd=REPO_ROOT,
                                  env=full_env, stdout=fh, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"{self.launcher} exited {proc.returncode}; tail of {log}:\n"
                               + _tail(log))

    def compute(self, sample: Sample) -> dict[str, float]:
        out = self.run_batch([sample])
        if sample.id not in out:
            raise RuntimeError(f"{self.name}: launcher produced no row for {sample.id!r}")
        return out[sample.id]

    @abstractmethod
    def run_batch(self, samples: Sequence[Sample]) -> dict[str, dict[str, float]]:
        """Stage ``samples``, run the launcher once, parse per-sample rows."""

    def evaluate(self, samples: Sequence[Sample], *, fail_fast: bool = False,
                 verbose: bool = True) -> MetricResult:
        _check_unique_ids(samples)
        t0 = time.time()
        self.ensure_setup()
        errors: dict[str, str] = {}
        valid: list[Sample] = []
        for s in samples:
            try:
                self.validate(s)
                valid.append(s)
            except Exception as e:  # noqa: BLE001
                if fail_fast:
                    raise
                errors[s.id] = f"{type(e).__name__}: {e}"
        per_sample = self.run_batch(valid) if valid else {}
        for s in valid:
            if s.id not in per_sample:
                errors.setdefault(s.id, "launcher produced no row (see launcher log)")
        if fail_fast and errors:
            raise RuntimeError(f"{self.name}: {len(errors)} samples failed: {errors}")
        result = self._result(per_sample, errors, time.time() - t0)
        if verbose:
            print(result.headline())
        return result


# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]


def safe_id(sample_id: str) -> str:
    """Filesystem-safe, collision-free name for staging (``a/b c`` -> ``a__b_c``)."""
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id.replace("/", "__")).strip("._") or "sample"


def staged_names(samples: Sequence[Sample]) -> dict[str, Sample]:
    out: dict[str, Sample] = {}
    for s in samples:
        n = safe_id(s.id)
        if n in out:
            raise ValueError(f"sample ids {out[n].id!r} and {s.id!r} collide after sanitising")
        out[n] = s
    return out


def free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _cuda_index(device: str) -> str:
    return device.split(":", 1)[1] if device.startswith("cuda:") else "0"


def _tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "<no log>"


def _stats(values: list[float]) -> dict[str, float]:
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "sem": float("nan"), "n": 0}
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    return {"mean": mean, "std": std, "sem": std / math.sqrt(n), "n": n}


def _check_unique_ids(samples: Sequence[Sample]) -> None:
    seen: set[str] = set()
    for s in samples:
        if s.id in seen:
            raise ValueError(f"duplicate sample id {s.id!r}")
        seen.add(s.id)


def _json_safe(x: Any) -> Any:
    """Strict JSON: NaN/inf -> None, numpy scalars -> python."""
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, Path):
        return str(x)
    return x
