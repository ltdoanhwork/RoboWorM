"""Execution-grounded (expensive) tier: recover the embodiment, then check it.

`metrics/sim2real` scores the geometry of hand-object interaction without a
simulator. This module goes after the part that proxy cannot reach: whether the
behaviour in a generated video corresponds to a **robot configuration that the real
robot could have been in**, which is RoboWM-Bench's headline robot failure mode
("the end effector may appear to reach the target in the video, while the recovered
joint configuration yields a mismatched end-effector pose under the robot's true
forward kinematics").

Modules
-------
``episode_index.py``
    Resolves any generated mp4 under ``sampling_dataset`` to the action-labelled GT
    episode in ``datasets/``. Eight naming conventions coexist in the tree; 99.9% of
    62,181 prediction mp4s resolve. Everything else here depends on this mapping.

``identifiability_probe.py``
    Measures whether robot configuration is recoverable from pixels at all, per
    embodiment, using frozen CLIP features and closed-form ridge. Run before
    committing to an IDM: a negative result here costs hours, not GPU-days.

``kinematics.py``
    FK and joint limits from an MJCF, per embodiment. The menagerie Franka Panda
    reproduces DROID's own end-effector column to 3.3e-08 m, so FK residual is a
    usable error signal rather than a calibration offset.

``checkers.py``
    The metrics: ``fk_consistency`` (the paper's headline robot failure),
    ``kinematic_feasibility`` (limits / velocity / acceleration vs GT-calibrated
    ceilings), ``trajectory_agreement`` (end-effector path error in metres).

``run_checker_controls.py``
    Oracle + six synthetic violations proving each checker fires on its own term and
    not on the others. 7/7 pass.

Read ``README.md`` for the measured numbers and what they rule in and out.
"""
