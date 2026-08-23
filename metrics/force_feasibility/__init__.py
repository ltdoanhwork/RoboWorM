"""Force-feasibility tier: dynamics, not geometry.

``metrics/sim2real`` scores the *geometry* of hand-object interaction (did the object
move untouched, did it follow the grasp, did the gripper pass through it) in units of
object diameters, with no simulator. ``metrics/embodied_exec`` checks the *robot's*
configuration against forward kinematics. Neither can reach mass, friction or
gravity, so neither can answer the question a physics benchmark is actually named
after: is this motion consistent with Newton's laws?

This module answers it inside a real physics engine, but *not* by rolling the scene forward.
MuJoCo (``mj_forward``) is used for what it is uniquely good at here -- building the contact
set: which geoms touch, where, with what surface normals and tangent frames -- in a proxy
scene fitted automatically from the video. The dynamics question is then posed directly on
the observed trajectory: given this contact set, could forces inside their friction cones
have produced the acceleration the video shows?

That distinction is deliberate and was arrived at by measurement, not preference. An earlier
version did drive the object with a PD controller inside ``mj_step`` and reported the injected
force; it failed its own controls, because the controller's gains became a tuned constant that
dominated the answer (see README, "What the first design got wrong"). Not integrating forward
is what removes every hand-set constant, and it is also what makes the measurement possible
at all -- a rollout diverges from the video after a few frames, so there would be nothing left
to compare against.

Modules
-------
``proxy_scene.py``
    Fits the smallest scene in which the motion can be asked to obey Newton: one
    support plane, one convex primitive for the object, two kinematically-driven
    finger bodies. No per-task assets -- the deliberate departure from
    RoboWM-Bench's digital twin, whose cost scales with the number of tasks.

``newton.py``
    The measurement, in two parts, because measuring showed one number cannot cover both
    regimes.

    *Airborne or supported:* ask whether **any** set of contact forces inside their
    friction cones, plus gravity, produces the observed acceleration and angular
    acceleration (non-negative least squares over cone generators, full 6-D Newton-Euler).
    The residual is reported as a dimensionless multiple of ``m g`` (and ``m g L`` for the
    torque rows). Here feasibility genuinely discriminates: free fall scores 0.000 while a
    hover at the same height scores 1.000.

    *Held:* feasibility **cannot** discriminate, and this is a theorem rather than a
    tuning problem. Two opposing frictional contacts give a wrench matrix of rank 6 --
    measured worst relative residual 2.3e-15 over 500 random wrenches -- so every motion
    of a grasped object is explainable, torque included. Adding torque balance did not
    change that. What stays meaningful is ``grip_demand``: the minimum contact force the
    *gripper's own* contacts must supply (LP over the same cone coefficients, support
    contacts free) to explain the motion, in units of the object's weight. An object resting
    on a table costs 0.000, holding the same object in the air costs 1.250, and asserting
    200 m/s^2 on it costs 26.7 -- which no hand applies.

    Two invariances worth stating plainly: both quantities are *exactly* invariant to the
    object's unknown mass (verified to six decimals from 0.01 kg to 5 kg), and contact
    stiffness never enters. Friction is the only unknown left, and it is swept rather than
    calibrated. There are no tuned constants.

    Orientation feeds the torque rows and is treated as measurable only when it is:
    angular velocity comes from single-step SO(3) logarithms (a finite difference on
    rotation matrices was 62% low at 150 rad/s^2), and rotation beyond pi/2 per frame is
    flagged via ``orientation_aliased_frac`` and dropped rather than reported as a small
    residual.

``run_newton_controls.py``
    Controls with closed-form answers -- 11/11 pass, matching analytic values to
    three decimals, noise floor 0.00000. Includes the specificity case geometry
    cannot handle (free fall touches nothing yet scores 0.000, while a hover at the
    same height scores 1.000) and the pair that shows what this tier adds over the
    cheap one: an identical object lift scores 0.000 when a gripper holds it and
    1.051 when the gripper is 0.4 m away.

``run_grip_controls.py``
    Controls for the held regime -- 11/11 pass. The feasibility column is flat ~0.000 while
    grip demand spans 0 to 26.7 across the same cases, which is the argument for having
    both. Includes the frame that dominates real episodes and that the first version could
    not score at all: an object resting on the table with fingers touching costs the fingers
    0.000, while an airborne hold of the same object costs 1.250. Also asserts what must
    *not* be answered: an unmeasurably fast spin and a mid-clip angular reversal are
    flagged, and a free-falling object reports ``None`` for grip demand rather than 0.

``lift_to_3d.py``
    Vision bridge: ``sim2real`` masks plus MoGe point maps to metric 3D pose. Object
    orientation is recovered by PCA with each frame resolved against the previous one over
    the 24 rotations of a cube (raw per-frame PCA flips axes, which reads as enormous
    angular acceleration). Validated on synthetic clouds: 0.0% error on an asymmetric box
    even with 2 mm of noise, and a cube is flagged 100% unmeasurable -- correct, since a
    cube has no orientation observable from its point cloud.

Read ``README.md`` for the measured numbers.
"""

from .proxy_scene import Observation, ProxyParams, build_proxy_xml  # noqa: F401
from .newton import newton_feasibility, newton_residual  # noqa: F401
