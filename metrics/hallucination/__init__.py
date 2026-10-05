"""Kinescore embodiment-hallucination metric (branch quang/hallucination-official).

  run.sh          launcher (interpreter, store, ffmpeg workaround, sweep)
  aggregate.py    cross-variant table from the launcher's scored dirs
  probe_clips.py  pre-flight: frame size / count per clip
  map_to_dr.py    join hallucination rows onto the DR table
  metric.py       :class:`Hallucination` (BaseMetric wrapper)

See docs/hallucination_metric.md.
"""
from .metric import EMBODIMENTS, Hallucination

__all__ = ["Hallucination", "EMBODIMENTS"]
