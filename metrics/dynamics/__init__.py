"""Dynamics / motion metrics for the video-world physics benchmark.

WorldScore's three "dynamics" aspects, plus our paired-GT (reference-based) variants:

  motion_accuracy    ``motion_accuracy_gdino.py``   GroundingDINO(arm+object) -> SAM2 -> SEA-RAFT
                                                    foreground-vs-background flow (our version; the
                                                    WorldScore-original SAM2-only probe and the
                                                    smoothness viz were archived to
                                                    tmp/20261005_refactor/misc/metrics/dynamics/)
  motion_magnitude   ``motion_mag_smooth_probe.py`` SEA-RAFT median flow
  motion_smoothness  ``motion_mag_smooth_probe.py`` VFIMamba mid-frame interpolation (SSIM/MSE)

  actor flicker      ``actor_flicker.py``           FB-filtered photometric residual (NR pipeline)

  paired-GT flow     ``physics_metrics.py``         flow EPE / magnitude ratio vs GT
                     ``flow_backend.py``            SEA-RAFT wrapper used by the above

Submodules are intentionally NOT imported here: most pull heavy optional deps
(SAM2, SEA-RAFT, VFIMamba, GroundingDINO) that are only present in the dreamgen venv.
Import them explicitly, e.g. ``from metrics.dynamics.flow_backend import SeaRaftFlow``.
"""
