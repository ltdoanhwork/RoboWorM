"""Photoreal robosuite rendering of the measured trajectory.

`render_sim.py` draws the physics proxy: a plane, a box, two finger boxes. That picture is
faithful to what the metric computes, and for exactly that reason it looks nothing like the
video -- the metric does not need the scene to look like anything. But an abstraction nobody
recognises cannot be sanity-checked by a reader, so this module renders the same trajectory
inside a real robosuite scene: the actual GR1 humanoid, a textured table arena, and a bottle
mesh, from a camera that frames the workspace the way the source video does.

Nothing here feeds the metric. It is a viewer, and the physics numbers are unchanged by it
(the proxy XML is untouched). The trajectory, object size, and support height all come from the
same cached lift, so what is on screen is what was measured -- rendered legibly.

The robot's pose comes from the GT parquet joints through validated FK, which is only available
for ground truth. For a prediction there are no joints, so the arm is not drawn; the object and
the contact bodies still are.

Needs its own venv (`metrics/.venv_vis`): robosuite 1.5.2 pins MuJoCo <= 3.2.x because it calls
`MjData.qM`, removed in MuJoCo 3.11 which the inference venvs use.

Usage:
    MUJOCO_GL=egl metrics/.venv_vis/bin/python -m metrics.force_feasibility.render_robosuite \
        --lift /tmp/sim_rg/ep002346_gt_lift.npz --joints /tmp/ep002346_joints.npy \
        --out /tmp/rs_vis/ep002346
"""

from __future__ import annotations

import argparse
import os

import numpy as np


TABLE_TOP_Z = 0.80          # arena is built with this top surface height
GR1_BASE_Z = 0.90           # measured: puts GR1's hands at z=0.804, level with the table
GR1_BASE_X = -0.55


def build_scene(obj_half, robot: str = "GR1FixedLowerBody"):
    """A textured table arena with a GR1 standing at it, and a bottle for the object.

    Heights are not guessed. `TableArena(table_offset=(0,0,0.8))` puts its top surface at
    z = 0.800 (verified from `geom_xpos + geom_size` of `table_collision`), and
    `GR1FixedLowerBody`'s origin is at the hips rather than the feet, so at base z = 0 its hands
    sit at **z = -0.096**, under the floor. Sweeping base height gives hands at z = 0.804 for
    base z = 0.90, which is what is used -- the robot then stands at the table like a person
    instead of being buried beneath it.
    """
    from robosuite.models import MujocoWorldBase
    from robosuite.models.arenas import TableArena
    from robosuite.models.objects import BottleObject
    from robosuite.models.robots import create_robot

    world = MujocoWorldBase()
    world.merge(TableArena(table_full_size=(0.8, 0.8, 0.05),
                           table_offset=(0.0, 0.0, TABLE_TOP_Z)))

    rb = create_robot(robot, idn=0)
    rb.set_base_xpos([GR1_BASE_X, 0.0, GR1_BASE_Z])
    world.merge(rb)

    obj = BottleObject(name="obj")
    world.merge_assets(obj)
    world.worldbody.append(obj.get_obj())
    return world, rb, obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lift", required=True, help="cached *_lift.npz")
    ap.add_argument("--joints", help=".npy of (T, n_arm) GT joint angles; omit to hide the robot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera", default="frontview")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--robot", default="GR1FixedLowerBody")
    a = ap.parse_args()

    import imageio.v3 as iio
    import mujoco

    z = np.load(a.lift, allow_pickle=True)
    obj_pos = z["obj_pos"]
    finger = z["finger_pos"]
    half = tuple(z["obj_half"])
    support_z = float(z["support_z"])
    T = len(obj_pos)

    # Recentre the measured trajectory onto the table. The lift frame's origin is wherever MoGe
    # put the camera, which has no meaning in the arena, so only relative motion is preserved:
    # the object's mean horizontal position maps to the table centre and its support height maps
    # to the table top. This is a viewer, so a rigid placement is enough; the metric never sees
    # these coordinates.
    centre = np.nanmean(obj_pos, axis=0)
    shift = np.array([0.0, 0.0, TABLE_TOP_Z - support_z])
    obj_w = obj_pos - np.array([centre[0], centre[1], 0.0]) + shift

    world, rb, obj = build_scene(half, a.robot)
    model = world.get_model(mode="mujoco")
    data = mujoco.MjData(model)

    jname = None
    for cand in ("obj_joint0", "obj_joint", "obj"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, cand) >= 0:
            jname = cand
            break
    if jname is None:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                 for i in range(model.njnt)]
        jname = next((n for n in names if n and n.startswith("obj")), None)
    if jname is None:
        raise RuntimeError("could not find the object's free joint in the built scene")
    adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)]

    Q = np.load(a.joints) if a.joints and os.path.exists(a.joints) else None
    idx: list[int] = []
    n_use = 0
    if Q is not None:
        # GT's 32 values are in `GR1_JOINT_NAMES` order (arms, legs, waist, head). Robosuite
        # prefixes the same MJCF joints with `robot0_`, so the mapping is done BY NAME. Writing
        # a contiguous block of qpos instead would silently put arm angles into leg joints --
        # the exact class of bug that `RobotSpec.joint_names` exists to prevent.
        # The joint order is read out of `kinematics.py` textually rather than imported: the
        # `metrics` package `__init__` pulls in torch, which this lightweight viewer venv
        # deliberately lacks. Reading the source keeps the two in sync without the dependency.
        import ast
        _kp = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "..", "embodied_exec", "kinematics.py"))
        _src = open(_kp).read()
        _ns: dict = {}
        for node in ast.parse(_src).body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in (
                    "GR1_ARM", "GR1_LEG", "GR1_JOINT_NAMES"):
                _ns[node.targets[0].id] = eval(compile(ast.Expression(node.value),
                                                       "<kin>", "eval"), {}, dict(_ns))
        GR1_JOINT_NAMES = _ns["GR1_JOINT_NAMES"]
        for jn in GR1_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_{jn}")
            idx.append(int(model.jnt_qposadr[jid]) if jid >= 0 else -1)
        idx_arr = np.array(idx)
        keep = idx_arr >= 0
        n_use = int(keep.sum())
        print(f"[rs] driving {n_use}/{len(GR1_JOINT_NAMES)} GR1 joints from GT, matched by name")
        if n_use < len(GR1_JOINT_NAMES):
            missing = [n for n, k in zip(GR1_JOINT_NAMES, keep) if not k]
            print(f"[rs] not in this robosuite model: {missing}")
    else:
        print("[rs] no joints supplied -- object only")

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, a.camera)
    frames = []
    with mujoco.Renderer(model, a.height, a.width) as r:
        for t in range(T):
            mujoco.mj_resetData(model, data)
            data.qpos[adr:adr + 3] = obj_w[t]
            data.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
            if n_use:
                qt = Q[min(t, len(Q) - 1)]
                for k, ad in enumerate(idx):
                    if ad >= 0 and k < len(qt):
                        data.qpos[ad] = qt[k]
            mujoco.mj_forward(model, data)
            r.update_scene(data, camera=(cam_id if cam_id >= 0 else -1))
            frames.append(r.render())

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    vid = np.stack(frames)
    iio.imwrite(f"{a.out}.mp4", vid, fps=16)
    iio.imwrite(f"{a.out}_mid.png", vid[T // 2])
    print(f"[rs] {a.out}.mp4  {a.out}_mid.png")
    os._exit(0)


if __name__ == "__main__":
    main()
