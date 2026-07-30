"""Quality & controllability metrics for the video-world physics benchmark.

WorldScore's non-dynamics aspects (all per-frame and reference-free, so no
matched-grid resampling is needed):

  subjective_quality  ``quality_control_probe.py``  clip_iqa+ (pyiqa clipiqa+)
                                                    + clip_aesthetic (pyiqa laion_aes)
  content_alignment   ``quality_control_probe.py``  clip_score — torchmetrics
                                                    CLIPScore(clip-vit-base-patch16), 0-100
  object_control      ``quality_control_probe.py``  GroundingDINO detection rate
                      ``object_control_viz.py``     annotated boxes on representative frames

Note: WorldScore exports two CLIPScore classes; the evaluator uses the *torchmetrics*
one (0-100), not pyiqa's (0-2.5, aliased ``IQACLIPScoreMetric`` and never used).

Submodules are intentionally NOT imported here: they pull pyiqa / torchmetrics /
GroundingDINO / spaCy, which only exist in the dreamgen venv. Import explicitly,
e.g. ``from metrics.quality.quality_control_probe import IQA, Detector``.
"""
