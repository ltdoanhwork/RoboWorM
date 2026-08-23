"""Proxy scenes fitted from vision, not authored per task.

RoboWM-Bench builds a digital twin: every task needs its own scene assets, so the
cost grows with the number of tasks. This module takes the opposite trade. It never
tries to *reproduce* the scene -- it fits the smallest scene in which the observed
motion could be asked to obey Newton: one support plane, one convex primitive for
the manipulated object, and two kinematically-driven finger boxes.

Everything here is derived from quantities the cheap tier already produces (object
mask extent, gripper mask centroids) plus a single metric scale factor. Nothing is
hand-authored per task or per embodiment, which is the whole point: the same three
primitives are fitted automatically for every episode in ``sampling_dataset``.

The unknowns we cannot see (mass, friction) are deliberately left as *free
parameters* rather than calibrated constants. Feasibility is then "does there exist
a plausible parameter setting under which this motion is Newtonian", which is a
weaker and therefore safer claim than "this is what the real object weighed".
"""

from __future__ import annotations

from dataclasses import dataclass, field

G = 9.81


@dataclass
class ProxyParams:
    """Geometry we can see, and the physics we cannot.

    Sizes are half-extents in metres (MuJoCo convention). ``mass`` and ``mu`` are
    the free parameters swept by the feasibility search; their defaults are only
    the centre of the sweep, never a calibrated value.
    """

    obj_half: tuple[float, float, float] = (0.03, 0.03, 0.03)
    obj_type: str = "box"
    finger_half: tuple[float, float, float] = (0.01, 0.004, 0.02)
    support_z: float = 0.0
    mass: float = 0.2
    mu: float = 0.6
    timestep: float = 0.002
    # A plane is infinite in MuJoCo; the visual size only bounds the checker board.
    support_size: tuple[float, float, float] = (2.0, 2.0, 0.05)


def build_proxy_xml(p: ProxyParams) -> str:
    """Emit the MJCF for one episode's proxy scene.

    The object gets a free joint (six DoF, fully dynamic). The two fingers are
    ``mocap`` bodies: MuJoCo treats them as infinitely stiff kinematic drivers, so
    they impose exactly the observed gripper motion while still generating real
    contact forces on the object. That asymmetry is intentional -- we are asking
    what the *object* does given the gripper, not simulating the robot.
    """
    ox, oy, oz = p.obj_half
    fx, fy, fz = p.finger_half
    sx, sy, sz = p.support_size
    if p.obj_type == "sphere":
        obj_geom = (f'<geom name="obj" type="sphere" size="{ox:.6f}" mass="{p.mass:.6f}" '
                    f'rgba="0.95 0.55 0.15 1" friction="{p.mu:.4f} 0.005 0.0001"/>')
    else:
        obj_geom = (
            f'<geom name="obj" type="box" size="{ox:.6f} {oy:.6f} {oz:.6f}" '
            f'mass="{p.mass:.6f}" rgba="0.95 0.55 0.15 1" '
            f'friction="{p.mu:.4f} 0.005 0.0001"/>'
        )
    return f"""<mujoco model="proxy">
  <option timestep="{p.timestep}" gravity="0 0 -{G}" integrator="implicitfast"/>
  <default>
    <geom condim="4" solref="0.01 1" solimp="0.95 0.99 0.001"/>
  </default>
  <!-- Everything below this line is appearance only: lights, textures and rgba do not
       enter the contact solver, so the rendered scene is the same model that is measured.
       They exist because a scene nobody can look at cannot be checked for the failures
       that actually matter (object sunk through the plane, fingers not where the video
       shows them, object size implausible in metres). -->
  <visual>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
    <rgba haze="0.9 0.92 0.95 1"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.84 0.86"
             rgb2="0.92 0.94 0.96" width="300" height="300"/>
    <material name="grid_mat" texture="grid" texrepeat="14 14" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0.4 -0.4 {p.support_z + 1.0:.6f}" dir="-0.3 0.3 -1" diffuse="0.7 0.7 0.7"/>
    <geom name="support" type="plane" pos="0 0 {p.support_z:.6f}" size="{sx} {sy} {sz}"
          material="grid_mat"
          friction="{p.mu:.4f} 0.005 0.0001"/>
    <body name="object" pos="0 0 {p.support_z + oz:.6f}">
      <freejoint name="obj_free"/>
      {obj_geom}
    </body>
    <body name="finger0" mocap="true" pos="0 -0.2 {p.support_z + fz:.6f}">
      <geom name="f0" type="box" size="{fx:.6f} {fy:.6f} {fz:.6f}" rgba="0.20 0.45 0.85 0.85"
            friction="{p.mu:.4f} 0.005 0.0001"/>
    </body>
    <body name="finger1" mocap="true" pos="0 0.2 {p.support_z + fz:.6f}">
      <geom name="f1" type="box" size="{fx:.6f} {fy:.6f} {fz:.6f}" rgba="0.20 0.45 0.85 0.85"
            friction="{p.mu:.4f} 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""


@dataclass
class Observation:
    """What the video says happened, in metres and seconds.

    ``obj_pos`` is the manipulated object's centre, ``finger_pos`` the two driven
    contact bodies. All are (T, 3); ``fps`` converts frame index to time. Held
    separately from :class:`ProxyParams` because the observation is measured while
    the parameters are searched.

    ``obj_quat`` (T, 4) as (w, x, y, z) is optional and enables the angular half of the
    Newton-Euler balance. Without it only linear force balance is checked, which is
    strictly weaker: two opposing contacts can supply unbounded normal force, so *any*
    translation of a grasped object is force-feasible. Rotation is what remains
    constrained, and it is where the signal is for held objects.
    """

    obj_pos: object
    finger_pos: object
    fps: float
    obj_quat: object = None
    scale_source: str = "unknown"
    notes: str = ""
    meta: dict = field(default_factory=dict)
