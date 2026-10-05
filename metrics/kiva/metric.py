"""KIVA: kinescore embodiment hallucination as a :class:`~metrics.base.BaseMetric`.

Wraps ``metrics/kiva/run.sh`` (kinescore vendored from branch
``quang/hallucination-official`` @5225ce7 into ``metrics/kiva/src``, see
``metrics/kiva/VENDORED.md``;
full write-up in ``docs/hallucination_metric.md``). The launcher owns the
interpreter (``hdmi`` conda env), the kinescore store (reader checkpoints,
URDFs, calibration clips, DINOv3 cache) and the ffmpeg workaround; this class
stages mp4 symlinks, runs it once, and reduces ``segments.csv`` with the SAME
functions ``aggregate.py`` uses -- so there is exactly one definition of
"violation rate" in the repo.

Inputs: ``Sample.pred`` = one mp4 per clip, at the reader's exact input size
(single_arm 640x480, bimanual 768x432, humanoid 768x432); kinescore rejects any
other size. ``embodiment`` picks the reader and is required.

Outputs, per detector (all LOWER is better):
  <det>_rate        fraction of the clip's 16-frame segments flagged as violating
  <det>_severity    median value / real-motion p95 threshold (>1 = worse than real)

Read a rate as EXCESS over the real-video anchor, never as an absolute: score
the paired ``*_gt.mp4`` the same way (``score_ref=True`` does it in the same
launcher call and adds ``<det>_rate_excess = pred - ref``). Only the
single_arm (A1X) reader is validated to fire; the bimanual/humanoid readers
currently read ~0 on everything (doc §9).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

from ..base import ExternalMetric, Sample, staged_names
from .aggregate import ALL_DETECTORS, HEADLINE, _load_segments, _per_clip

EMBODIMENTS = ("single_arm", "bimanual", "humanoid")
REF_SUFFIX = "__ref"


class Hallucination(ExternalMetric):
    name = "kiva"
    launcher = "metrics/kiva/run.sh"
    higher_is_better = {f"{d}_rate": False for d in HEADLINE}

    def __init__(self, device: str = "cuda", embodiment: str = "single_arm",
                 detectors: Sequence[str] = HEADLINE, score_ref: bool = False,
                 frame_chunk: int = 16, max_frames: int = 0,
                 calib_clips: int = 24, percentile: float = 95.0, **config: Any) -> None:
        if embodiment not in EMBODIMENTS:
            raise ValueError(f"embodiment must be one of {EMBODIMENTS}, got {embodiment!r}")
        detectors = list(detectors)
        bad = [d for d in detectors if d not in ALL_DETECTORS]
        if bad:
            raise ValueError(f"unknown detectors {bad}; available {ALL_DETECTORS}")
        super().__init__(device=device, embodiment=embodiment, detectors=detectors,
                         score_ref=score_ref, frame_chunk=frame_chunk, max_frames=max_frames,
                         calib_clips=calib_clips, percentile=percentile, **config)
        self.detectors = detectors
        self.score_ref = bool(score_ref)
        hib = {}
        for d in detectors:
            hib[f"{d}_rate"] = False
            hib[f"{d}_severity"] = False
            if self.score_ref:
                hib[f"{d}_rate_ref"] = False
                hib[f"{d}_rate_excess"] = False
        self.higher_is_better = hib  # instance-level: depends on detectors/score_ref

    @property
    def requires_ref(self) -> bool:  # type: ignore[override]
        return self.score_ref

    def validate(self, sample: Sample) -> None:
        super().validate(sample)
        for src in (sample.pred, sample.ref if self.score_ref else None):
            if src is None:
                continue
            p = Path(src)
            if not p.is_file() or p.suffix.lower() != ".mp4":
                raise ValueError(f"hallucination scores mp4 files, got {p}")

    # ------------------------------------------------------------------ #
    def run_batch(self, samples: Sequence[Sample]) -> dict[str, dict[str, float]]:
        run_dir = self.new_run_dir()
        videos = run_dir / "videos"
        videos.mkdir(parents=True)
        names = staged_names(samples)
        for name, s in names.items():
            (videos / f"{name}.mp4").symlink_to(Path(s.pred).resolve())
            if self.score_ref:
                (videos / f"{name}{REF_SUFFIX}.mp4").symlink_to(Path(s.ref).resolve())

        out_dir = run_dir / "out"
        env = {
            "VIDEOS_DIR": videos,
            "OUT_DIR": out_dir,
            "EMBODIMENTS": self.config["embodiment"],
            "HAL_PATTERN": "*.mp4",
            "FRAME_CHUNK": self.config["frame_chunk"],
            "MAX_FRAMES": self.config["max_frames"],
            "CALIB_CLIPS": self.config["calib_clips"],
            "PERCENTILE": self.config["percentile"],
            "MAX_CLIPS": 0,
            "HAL_RENDER": 0,
        }
        self.run_launcher(env, run_dir)
        if not (out_dir / "segments.csv").is_file():
            raise RuntimeError(f"launcher wrote no segments.csv in {out_dir} (log in {run_dir})")

        per_clip = _per_clip(_load_segments(out_dir, self.detectors), self.detectors)
        clip_errors = _clip_errors(out_dir / "metrics.csv")
        rows: dict[str, dict[str, float]] = {}
        for name, s in names.items():
            pred = per_clip.get(name)
            if pred is None:
                if name in clip_errors:
                    print(f"[{self.name}] {s.id}: kinescore error: {clip_errors[name]}")
                continue
            row = {k: v for k, v in pred.items()}
            if self.score_ref:
                ref = per_clip.get(f"{name}{REF_SUFFIX}") or {}
                for d in self.detectors:
                    r = ref.get(f"{d}_rate", float("nan"))
                    row[f"{d}_rate_ref"] = r
                    row[f"{d}_rate_excess"] = row.get(f"{d}_rate", float("nan")) - r
            rows[s.id] = row
        print(f"[{self.name}] raw outputs kept in {run_dir}")
        return rows


def _clip_errors(metrics_csv: Path) -> dict[str, str]:
    if not metrics_csv.is_file():
        return {}
    with metrics_csv.open(newline="") as f:
        return {(r.get("id") or "").strip(): r["error"] for r in csv.DictReader(f)
                if (r.get("error") or "").strip()}


KIVA = Hallucination

__all__ = ["Hallucination", "KIVA", "EMBODIMENTS"]
