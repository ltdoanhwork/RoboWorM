"""RIGIS: VGGT-Omega multi-view consistency as a :class:`~metrics.base.BaseMetric`.

Wraps ``metrics/rigis/run.sh`` (code vendored from branch
``thanhnv/multiview-consistency-indep`` @8015d97, see ``metrics/rigis/VENDORED.md``; full write-up in
``docs/metrics/multiview_consistency_metric.md``). The launcher owns the interpreter
(``uv`` project + injected ``numba``), the 4.6 GB checkpoint and the model
service; this class only stages inputs and parses the CSV, so it runs from any
python with numpy.

Inputs (``Sample.pred``), per sample:
  * an episode dir holding ``view1/`` and ``view2/`` frame folders (already
    extracted), or
  * a mosaic mp4 (e.g. ``pred_all_views.mp4``) -- frames are extracted first
    with ``extract_frames.py`` using ``layout`` / ``fps``.
Reference video is not used: the metric is reference-free (it measures
whether the generated views agree on one rigid 3D scene).

Outputs (both in [0, 1], higher is better; the real-video ceiling is ~0.89,
not 1.0 -- always report the GT anchor next to a variant):
  geometric_consistency   depth + RGB agreement after cross-view reprojection
  pose_stability          temporal stability of the predicted camera poses
  disjoint_frame_ratio    fraction of frames with <2% covisible pixels (lower
                          is better; those frames score 0 in the mean)
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from ..base import REPO_ROOT, ExternalMetric, Sample, free_port, staged_names

HERE = Path(__file__).resolve().parent
FFMPEG_BUILD = "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/ffmpeg/ffmpeg-master-latest-linux64-gpl-shared"


class MultiviewConsistency(ExternalMetric):
    name = "rigis"
    launcher = "metrics/rigis/run.sh"
    higher_is_better = {
        "geometric_consistency": True,
        "pose_stability": True,
        "disjoint_frame_ratio": False,
    }

    def __init__(self, device: str = "cuda", views: str = "view1,view2",
                 frame_batch_size: int = 4, dtype: str = "float16",
                 layout: str = "2x2", fps: float = 2.0,
                 view_globs: str = "view_0.mp4,view_1.mp4",
                 ckpt: str | None = None, **config: Any) -> None:
        super().__init__(device=device, views=views, frame_batch_size=frame_batch_size,
                         dtype=dtype, layout=layout, fps=fps, view_globs=view_globs,
                         ckpt=ckpt, **config)
        self.views = views
        self.view_names = [v.strip() for v in views.split(",") if v.strip()]
        if len(self.view_names) < 2:
            raise ValueError(f"need >= 2 views, got {views!r}")

    def validate(self, sample: Sample) -> None:
        super().validate(sample)
        p = Path(sample.pred)
        if p.is_dir():
            missing = [v for v in self.view_names[:2] if not (p / v).is_dir()]
            if missing:
                raise ValueError(f"{p} is missing view folders {missing}")
        elif not p.is_file():
            raise FileNotFoundError(p)

    # ------------------------------------------------------------------ #
    def run_batch(self, samples: Sequence[Sample]) -> dict[str, dict[str, float]]:
        run_dir = self.new_run_dir()
        batch_root = run_dir / "data" / "batch"
        batch_root.mkdir(parents=True)
        names = staged_names(samples)
        for name, s in names.items():
            src = Path(s.pred).resolve()
            dst = batch_root / name
            if src.is_dir():
                dst.symlink_to(src, target_is_directory=True)
            else:
                self._extract(src, dst)

        env = {
            "MVC_DATA_ROOT": run_dir / "data",
            "MVC_OUT_ROOT": run_dir / "out",
            "VARIANTS": "batch",
            "VIEWS": self.views,
            "FRAME_BATCH_SIZE": self.config["frame_batch_size"],
            "MVC_DTYPE": self.config["dtype"],
            "MVC_PORT": free_port(),
            "MAX_EPISODES": 0,
        }
        if self.config.get("ckpt"):
            env["MVC_CKPT"] = self.config["ckpt"]
        self.run_launcher(env, run_dir)

        out_csv = run_dir / "out" / "batch.csv"
        if not out_csv.is_file():
            raise RuntimeError(f"launcher wrote no CSV at {out_csv} (log in {run_dir})")
        rows: dict[str, dict[str, float]] = {}
        with out_csv.open(newline="") as f:
            for row in csv.DictReader(f):
                s = names.get((row.get("name") or "").strip())
                if s is None:
                    continue
                rows[s.id] = {k: _f(row.get(k)) for k in
                              ("geometric_consistency", "pose_stability",
                               "disjoint_frame_ratio", "num_disjoint_frames")}
        print(f"[{self.name}] raw outputs kept in {run_dir}")
        return rows

    def _extract(self, video: Path, out_dir: Path) -> None:
        cmd = [sys.executable, str(HERE / "extract_frames.py"),
               "--video-path", str(video), "--out-dir", str(out_dir),
               "--layout", str(self.config["layout"]), "--fps", str(self.config["fps"]),
               "--view-names", ",".join(self.view_names[:2])]
        if self.config["layout"] == "per_view":
            cmd += ["--view-globs", str(self.config["view_globs"])]
        env = dict(os.environ)
        # /usr/bin/ffmpeg segfaults in a process tree that has imported torch
        # (gotoblas clash, see docs/metrics/hallucination_metric.md); the pfss build is immune.
        if Path(FFMPEG_BUILD, "bin").is_dir():
            env["PATH"] = f"{FFMPEG_BUILD}/bin:{env.get('PATH', '')}"
            env["LD_LIBRARY_PATH"] = f"{FFMPEG_BUILD}/lib:{env.get('LD_LIBRARY_PATH', '')}"
        r = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"frame extraction failed for {video}:\n{r.stderr[-2000:]}")


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")
