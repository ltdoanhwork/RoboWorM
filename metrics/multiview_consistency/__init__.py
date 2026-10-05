"""VGGT-Omega multi-view consistency metric (branch thanhnv/multiview-consistency-indep).

  run.sh             launcher (worktree, uv interpreter, model service, sweep)
  extract_frames.py  mosaic / per-view mp4 -> <episode>/view1, view2 frames
  aggregate.py       cross-variant table from the launcher's CSVs
  metric.py          :class:`MultiviewConsistency` (BaseMetric wrapper)

See docs/multiview_consistency_metric.md.
"""
from .metric import MultiviewConsistency

__all__ = ["MultiviewConsistency"]
