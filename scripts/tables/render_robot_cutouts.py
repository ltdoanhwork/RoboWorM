#!/usr/bin/env python3
"""Render clean, background-free (transparent PNG) pictures of the benchmark robots.

Why render instead of downloading press photos: there is NO openly-licensed cutout
photo available for these four robots (Wikimedia Commons only has cluttered trade-fair
shots of the Franka, and nothing at all for ALOHA / GR1-T2 / MMK2). Vendor product
renders are copyrighted and unsafe for a paper. Every one of these robots however
ships an official, openly-licensed 3D description, so we render those ourselves:

  single_arm  Franka Emika Panda   MuJoCo Menagerie franka_emika_panda   Apache-2.0
  bimanual    ALOHA (2x ViperX)    MuJoCo Menagerie aloha                BSD-3 (Trossen)
  humanoid    Fourier GR1-T2       FFTAI/Wiki-GRx-Models GR1T2.urdf      GPL-3.0
  humanoid    AIRBOT MMK2          discoverse-dev/robot_descriptions     BSD

Alpha comes from MuJoCo's segmentation buffer (background id == -1), so the cutout is
pixel-exact -- no chroma keying, no halo.

Setup (one-off, clones only the needed subdirectories, ~120 MB):
  bash scripts/tables/fetch_robot_models.sh

Usage:
  MUJOCO_GL=osmesa python3 scripts/tables/render_robot_cutouts.py
  MUJOCO_GL=osmesa python3 scripts/tables/render_robot_cutouts.py --robot franka --azimuth 150
  MUJOCO_GL=osmesa python3 scripts/tables/render_robot_cutouts.py --width 1600 --height 1600
  ... --flat            # flat silhouette-ish shading (schematic look)
  ... --white           # also write a white-background copy next to each PNG
"""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path("/tmp/robot_models")  # populated by scripts/tables/fetch_robot_models.sh
OUT_DIR = REPO / "assets" / "figures" / "robots"
BUILD = Path("/tmp/robot_cutout_build")


# --------------------------------------------------------------------------------------
# Robot spec
# --------------------------------------------------------------------------------------


@dataclass
class Robot:
    key: str  # output filename stem
    label: str  # human readable name
    embodiment: str  # which benchmark embodiment it belongs to
    path: Path  # MJCF or URDF
    source: str  # where the model came from
    license: str  # license of the model
    azimuth: float = 135.0
    elevation: float = -15.0
    distance: float | None = None  # None -> auto-fit from model extent
    lookat_z: float | None = None  # None -> centre of the model bounding box
    fit_scale: float = 1.25  # padding factor when auto-fitting the camera
    urdf_meshdir: str = ""  # relative mesh dir to inject when importing a URDF
    pkg_roots: dict = field(default_factory=dict)  # ROS package name -> dir with meshes/
    qpos: dict = field(default_factory=dict)  # joint name -> angle, to pose the robot


def robots() -> list:
    return [
        Robot(
            key="single_arm_franka",
            label="Franka Emika Panda",
            embodiment="Single-Arm",
            path=MODEL_ROOT / "menagerie/franka_emika_panda/panda.xml",
            source="MuJoCo Menagerie (google-deepmind/mujoco_menagerie)",
            license="Apache-2.0",
            azimuth=140,
            elevation=-12,
            qpos={
                "joint1": 0.0, "joint2": -0.5, "joint3": 0.0, "joint4": -2.0,
                "joint5": 0.0, "joint6": 1.6, "joint7": 0.785,
            },
        ),
        Robot(
            key="bimanual_aloha",
            label="ALOHA (2x ViperX 300)",
            embodiment="Bimanual",
            path=MODEL_ROOT / "menagerie/aloha/aloha.xml",
            source="MuJoCo Menagerie (google-deepmind/mujoco_menagerie)",
            license="BSD-3-Clause (Trossen Robotics)",
            # az 90 looks down the table axis -> BOTH arms visible and separated
            azimuth=100,
            elevation=-22,
            # lift both arms into the classic ALOHA "ready" pose
            qpos={
                "left/waist": 0.0, "left/shoulder": -0.75, "left/elbow": 0.9,
                "left/wrist_angle": 0.6,
                "right/waist": 0.0, "right/shoulder": -0.75, "right/elbow": 0.9,
                "right/wrist_angle": 0.6,
            },
        ),
        Robot(
            key="humanoid_fourier_n1",
            label="Fourier N1",
            embodiment="Humanoid (single-view)",
            # Fourier N1 stands in for the GR1 used by the dataset: same vendor
            # (Fourier Intelligence) and the same humanoid morphology, but shipped
            # under Apache-2.0 instead of the GR1 models' GPL-3.0, so the render is
            # safe to publish. Native MJCF -> no URDF patching needed.
            path=MODEL_ROOT / "menagerie/fourier_n1/n1.xml",
            source="MuJoCo Menagerie (google-deepmind/mujoco_menagerie)",
            license="Apache-2.0 (Fourier Intelligence)",
            # az 155 is the best 3/4 view: both arms read clearly and the near arm
            # does not occlude the torso (az 135 does, az 170 goes flat/frontal)
            azimuth=155,
            elevation=-8,
            # arms forward in a manipulation-ready pose, legs slightly bent
            qpos={
                "left_shoulder_pitch_joint": -0.45,
                "left_shoulder_roll_joint": 0.18,
                "left_elbow_pitch_joint": -0.9,
                "right_shoulder_pitch_joint": -0.45,
                "right_shoulder_roll_joint": -0.18,
                "right_elbow_pitch_joint": -0.9,
                "left_hip_pitch_joint": -0.12,
                "left_knee_pitch_joint": 0.25,
                "left_ankle_pitch_joint": -0.13,
                "right_hip_pitch_joint": -0.12,
                "right_knee_pitch_joint": 0.25,
                "right_ankle_pitch_joint": -0.13,
            },
        ),
        Robot(
            key="humanoid_mmk2",
            label="AIRBOT MMK2",
            embodiment="Humanoid (multi-view)",
            path=MODEL_ROOT
            / "rdesc/humanoid/Airbot/airbot_mmk2_description/urdf/airbot_mmk2.urdf",
            source="discoverse-dev/robot_descriptions",
            license="BSD",
            azimuth=125,
            elevation=-14,
            urdf_meshdir="../meshes/",
            # MMK2 pulls its arm links from the sibling airbot_play package
            pkg_roots={
                "airbot_mmk2_description": MODEL_ROOT
                / "rdesc/humanoid/Airbot/airbot_mmk2_description",
                "airbot_play_description": MODEL_ROOT
                / "rdesc/manipulator/Airbot/airbot_play_description",
            },
            # lift the column and bring both arms forward, as in the dataset episodes
            qpos={
                "slide_joint": 0.35,
                "head_pitch_joint": 0.35,
                "left_joint1": -0.3, "left_joint2": -1.2, "left_joint3": 1.6,
                "right_joint1": 0.3, "right_joint2": -1.2, "right_joint3": 1.6,
            },
        ),
    ]


# --------------------------------------------------------------------------------------
# URDF -> MuJoCo-importable copy
# --------------------------------------------------------------------------------------


def prepare_urdf(rb: Robot) -> Path:
    """Copy the URDF and inject the <mujoco> compiler block MuJoCo needs.

    Two MuJoCo-specific gotchas handled here:
      1. Without `discardvisual="false"` MuJoCo throws away the visual meshes and you
         render a box-only collision skeleton instead of the real robot.
      2. MuJoCo's URDF parser DISCARDS the directory part of every `<mesh filename>`
         and looks the basename up in `meshdir`. So `package://` rewriting is not
         enough -- every mesh (plus its .mtl) has to be flattened into one directory.

    Working copy layout (per robot, so robots can't pick up each other's meshes):
        <build>/<key>/urdf/<name>.urdf
        <build>/<key>/meshes/<basename>   -> symlink to the real mesh
    """
    src = rb.path
    work = BUILD / rb.key
    if work.exists():
        shutil.rmtree(work)
    (work / "urdf").mkdir(parents=True)
    flat = work / "meshes"
    flat.mkdir()

    # collect every mesh directory this robot may reference
    roots = (
        {k: Path(v) for k, v in rb.pkg_roots.items()}
        if rb.pkg_roots
        else {"_self": src.parent.parent}
    )
    seen = {}
    for pkg, root in roots.items():
        root = root.resolve()
        if not root.exists():
            raise FileNotFoundError(f"model root missing for {pkg}: {root}")
        for f in root.rglob("*"):
            if f.suffix.lower() not in (".obj", ".stl", ".mtl", ".dae", ".png", ".jpg"):
                continue
            link = flat / f.name
            if link.exists():  # first package wins; report so mis-resolution is visible
                seen.setdefault(f.name, []).append(str(f))
                continue
            link.symlink_to(f)
    if seen:
        print(f"    [note] {rb.key}: {len(seen)} duplicate mesh basenames, first wins")

    root_el = ET.fromstring(src.read_text())
    _inject_mujoco_visuals(root_el, brighten=True)

    dst_urdf = work / "urdf" / src.name
    # ET writes the URDF as .xml-ish text; MuJoCo picks the parser from the extension,
    # so the file must keep its .urdf name.
    dst_urdf.write_bytes(ET.tostring(root_el))
    return dst_urdf


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


STUDIO_LIGHTS = (
    # (pos, diffuse) — key / fill / rim / top, no shadows (cutout has no floor)
    ((1.6, -1.4, 2.6), (0.85, 0.85, 0.85)),
    ((-1.8, 1.2, 1.8), (0.45, 0.45, 0.50)),
    ((0.0, 2.4, 0.6), (0.30, 0.30, 0.32)),
    ((0.0, 0.0, 3.0), (0.40, 0.40, 0.40)),
)


def _inject_mujoco_visuals(root_el: ET.Element, brighten: bool):
    """Add studio lights (+ a light default geom colour) into a URDF's <mujoco> block.

    URDF imports carry no lights and no materials, so MuJoCo falls back to a single
    headlight and the robot renders as a dark grey blob. MjSpec cannot parse URDF, so
    the lights are injected as MuJoCo-specific extension XML instead.
    """
    mj = root_el.find("mujoco")
    if mj is None:
        mj = ET.Element("mujoco")
        root_el.insert(0, mj)
    comp = mj.find("compiler")
    if comp is None:
        comp = ET.SubElement(mj, "compiler")
    comp.set("meshdir", "../meshes/")
    comp.set("discardvisual", "false")
    comp.set("balanceinertia", "true")
    comp.set("fusestatic", "false")

    if brighten:
        # URDF meshes usually arrive with no material -> give them a light grey
        dflt = ET.SubElement(mj, "default")
        ET.SubElement(dflt, "geom").set("rgba", "0.82 0.83 0.85 1")

    wb = ET.SubElement(mj, "worldbody")
    for pos, diff in STUDIO_LIGHTS:
        li = ET.SubElement(wb, "light")
        li.set("pos", " ".join(str(v) for v in pos))
        li.set("dir", " ".join(str(-v) for v in pos))
        li.set("diffuse", " ".join(str(v) for v in diff))
        li.set("specular", "0.15 0.15 0.15")
        li.set("castshadow", "false")


def build_model(rb: Robot, brighten_dark: bool = False):
    import mujoco

    path = prepare_urdf(rb) if rb.path.suffix == ".urdf" else rb.path
    if not path.exists():
        raise FileNotFoundError(
            f"{path}\nRun `bash scripts/tables/fetch_robot_models.sh` first."
        )
    model = mujoco.MjModel.from_xml_path(str(path))
    if brighten_dark:
        _lighten_materials(model)
    return model


def _lighten_materials(model, floor: float = 0.55):
    """Raise very dark materials towards mid-grey.

    ALOHA's Menagerie model is near-black (rgba ~0.1) which reads as a flat silhouette
    in print. This lifts anything darker than `floor` while keeping relative shading.

    Only `mat_rgba` and geoms WITHOUT a material are touched: MuJoCo's default
    `geom_rgba` is (0.5, 0.5, 0.5, 1) and rewriting it on a geom that has a material
    overrides that material — which is what turned the Franka from white to grey.
    """
    rgb = model.mat_rgba[:, :3]
    lum = rgb.mean(axis=1)
    dark = lum < floor
    if dark.any():
        scale = floor / np.maximum(lum[dark], 1e-3)
        rgb[dark] = np.clip(rgb[dark] * scale[:, None], 0, 1)

    no_mat = model.geom_matid < 0
    grgb = model.geom_rgba[:, :3]
    glum = grgb.mean(axis=1)
    gdark = no_mat & (glum < floor)
    if gdark.any():
        scale = floor / np.maximum(glum[gdark], 1e-3)
        grgb[gdark] = np.clip(grgb[gdark] * scale[:, None], 0, 1)


def pose(model, data, rb: Robot):
    import mujoco

    for name, val in rb.qpos.items():
        try:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                data.qpos[model.jnt_qposadr[jid]] = val
        except Exception:
            pass
    mujoco.mj_forward(model, data)


def scene_bbox(model, data):
    """World-space bounding box of all visible geometry.

    Using `geom_xpos` alone (geom CENTRES) badly underestimates the extent for long
    meshes -- MMK2's 1.4 m lift column then overflows the frame. Mesh geoms are sized
    from their actual vertices; primitives from geom_size.
    """
    import mujoco

    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for g in range(model.ngeom):
        pos = data.geom_xpos[g]
        mat = data.geom_xmat[g].reshape(3, 3)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[g]
            if mid < 0:
                continue
            v0 = model.mesh_vertadr[mid]
            n = model.mesh_vertnum[mid]
            verts = model.mesh_vert[v0 : v0 + n].reshape(-1, 3)
            world = verts @ mat.T + pos
            lo = np.minimum(lo, world.min(axis=0))
            hi = np.maximum(hi, world.max(axis=0))
        else:
            r = float(np.max(model.geom_size[g])) or 0.01
            lo = np.minimum(lo, pos - r)
            hi = np.maximum(hi, pos + r)
    if not np.isfinite(lo).all():
        return np.zeros(3), np.ones(3)
    return lo, hi


def render_cutout(
    rb: Robot, width: int, height: int, flat: bool, ss: int = 2, brighten: bool = True
):
    """Render one robot; returns an RGBA image with a pixel-exact alpha cutout."""
    import mujoco

    model = build_model(rb, brighten_dark=brighten)
    data = mujoco.MjData(model)
    pose(model, data, rb)

    W, H = width * ss, height * ss  # supersample, then downscale for clean edges
    # the offscreen framebuffer defaults to 640x480 -> must be grown before renderering
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)
    model.vis.global_.offheight = max(model.vis.global_.offheight, H)
    renderer = mujoco.Renderer(model, H, W)

    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    # a bare robot: no floor, no shadows, no reflections, no helper decorations
    opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = True
    for f in ("mjRND_SHADOW", "mjRND_REFLECTION", "mjRND_SKYBOX", "mjRND_HAZE"):
        try:
            renderer.scene.flags[getattr(mujoco.mjtRndFlag, f)] = 0
        except AttributeError:
            pass
    if flat:
        try:
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1
        except AttributeError:
            pass

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.azimuth = rb.azimuth
    cam.elevation = rb.elevation

    # auto-fit: centre on the geometry and back off proportionally to its true extent
    lo, hi = scene_bbox(model, data)
    centre = (lo + hi) / 2
    extent = float(np.linalg.norm(hi - lo))
    cam.lookat[:] = centre
    if rb.lookat_z is not None:
        cam.lookat[2] = rb.lookat_z
    cam.distance = rb.distance if rb.distance else max(extent * rb.fit_scale, 0.5)

    renderer.update_scene(data, cam, opt)
    rgb = renderer.render()

    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, cam, opt)
    seg = renderer.render()
    mask = (seg[..., 0] >= 0).astype(np.uint8) * 255  # -1 == background

    rgba = np.dstack([rgb, mask])
    im = Image.fromarray(rgba, "RGBA")
    if ss > 1:
        im = im.resize((width, height), Image.LANCZOS)
    return im.crop(im.getchannel("A").getbbox() or (0, 0, im.width, im.height))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="all", help="key substring, or 'all'")
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--supersample", type=int, default=2)
    ap.add_argument("--azimuth", type=float, default=None)
    ap.add_argument("--elevation", type=float, default=None)
    ap.add_argument("--flat", action="store_true", help="flat/schematic shading")
    ap.add_argument(
        "--no-brighten",
        action="store_true",
        help="keep original (often near-black) material colours",
    )
    ap.add_argument("--white", action="store_true", help="also write white-bg copies")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)

    manifest, failures = [], []
    for rb in robots():
        if args.robot != "all" and args.robot not in rb.key:
            continue
        if args.azimuth is not None:
            rb.azimuth = args.azimuth
        if args.elevation is not None:
            rb.elevation = args.elevation
        try:
            im = render_cutout(
                rb,
                args.width,
                args.height,
                args.flat,
                args.supersample,
                brighten=not args.no_brighten,
            )
        except Exception as e:
            failures.append((rb.key, str(e)[:200]))
            print(f"[FAIL] {rb.key}: {str(e)[:200]}")
            continue

        dst = OUT_DIR / f"{rb.key}.png"
        im.save(dst)
        if args.white:
            bg = Image.new("RGB", im.size, "white")
            bg.paste(im, mask=im.getchannel("A"))
            bg.save(OUT_DIR / f"{rb.key}_white.png")
        cover = float((np.asarray(im)[..., 3] > 0).mean())
        manifest.append(
            {
                "file": dst.name,
                "robot": rb.label,
                "embodiment": rb.embodiment,
                "size": f"{im.width}x{im.height}",
                "alpha_coverage": round(cover, 3),
                "model_source": rb.source,
                "model_license": rb.license,
                "camera": {"azimuth": rb.azimuth, "elevation": rb.elevation},
            }
        )
        print(f"[ok] {dst.relative_to(REPO)}  {im.width}x{im.height}  robot={cover:.1%} of frame")

    if manifest:
        (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[ok] {(OUT_DIR / 'manifest.json').relative_to(REPO)}")
    if failures:
        print("\nFailures:")
        for k, e in failures:
            print(f"  {k}: {e}")


if __name__ == "__main__":
    main()
