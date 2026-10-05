"""RIGIS: VGGT-Omega multi-view consistency metric.

Vendored from branch thanhnv/multiview-consistency-indep @8015d97
(``multiview_consistency/``, ``vggt_omega/``, ``pyproject.toml``; see VENDORED.md).
This dir is also the ``uv`` project the launcher runs in (``.venv`` here).

  run.sh             launcher (uv interpreter, model service, sweep)
  extract_frames.py  mosaic / per-view mp4 -> <episode>/view1, view2 frames
  aggregate.py       cross-variant table from the launcher's CSVs
  metric.py          :class:`MultiviewConsistency` (BaseMetric wrapper)

See docs/multiview_consistency_metric.md.
"""
from .metric import MultiviewConsistency

RIGIS = MultiviewConsistency

__all__ = ["MultiviewConsistency", "RIGIS"]
