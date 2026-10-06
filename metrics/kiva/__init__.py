"""KIVA: kinescore embodiment-hallucination metric.

kinescore is vendored from branch quang/hallucination-official @5225ce7 into
``src/kinescore`` (+ ``configs/``, ``docs/``, ``scripts/``; see VENDORED.md).

  run.sh          launcher (interpreter, store, ffmpeg workaround, sweep)
  aggregate.py    cross-variant table from the launcher's scored dirs
  probe_clips.py  pre-flight: frame size / count per clip
  map_to_dr.py    join hallucination rows onto the DR table
  metric.py       :class:`Hallucination` (BaseMetric wrapper)

See docs/metrics/kiva.md.
"""
from .metric import EMBODIMENTS, KIVA, Hallucination

__all__ = ["Hallucination", "KIVA", "EMBODIMENTS"]
