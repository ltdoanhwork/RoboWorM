# Adding a robot

## How it fits together

KIVA scores a video without knowing the robot's real pose. A *reader* fills that gap: it looks
at the frames and predicts the 3-D positions of a fixed set of keypoints on the robot (joints,
gripper tips). KIVA then checks those keypoints. Bones between them should keep their length
(rigidity), and they should move smoothly (jerk).

A reader only works for the robot and camera view it was trained on. A new world model on a
robot that already has a reader needs nothing from this page; a new robot, or a new view of an
existing one, needs a new reader.

To train one you need real videos where the robot's pose is known. Teleop datasets have this:
every frame comes with the logged robot state. kinescore turns that state into keypoint labels,
in one of two ways:

- the corpus logs joint angles: forward kinematics on the robot's URDF gives each keypoint;
- the corpus logs the end-effector pose: each keypoint is a fixed offset from it, and no URDF
  is needed.

So adding a robot is four pieces of work. Write the code that turns the logged state into
keypoints. Tell kinescore where the data is and which camera to read. Train. Fit thresholds on
real clips the reader did not train on.

```
teleop corpus (video + robot state)                         step 1
        |
robot state -> keypoints (URDF + FK, or end-effector)       steps 2-4
        |
reader entry in cells.yaml (robot, corpus, camera, view)    step 5
        |
kinescore data / cache / train  ->  reader checkpoint       step 6
        |
kinescore calibrate on real val clips  ->  thresholds       step 7
        |
kinescore score on generated clips
```

The running example is ALOHA bimanual (`aloha_bimanual.bimanual_sv.sv1_16x9`): two 6-DOF arms,
12 predicted joints, 18 keypoints. A reader id is always `<robot>.<corpus>.<view>`.

## 0. Check what the corpus logs

Open one parquet and look at `observation.state`. If it holds joint angles, follow every step.
If it holds an end-effector pose, skip the URDF and forward kinematics (steps 2 and 3). Galaxea
A1X is such a robot: its corpus logs `(x, y, z, roll, pitch, yaw, gripper_width)`, and
`robots/a1x_ee/spec.py` (117 lines) builds four keypoints with:

```python
P = pos.unsqueeze(2) + torch.einsum("btij,kj->btki", rot, offsets)
```

## 1. Get the data

Training data lives in LeRobot layout under
[`doanh25032004/vgp_datasets`](https://huggingface.co/datasets/doanh25032004/vgp_datasets),
as `{embodiment}/singleview/{makovian,non_makovian}/`. Add your folder to the `train` source in
`configs/sources.yaml`, then pull:

```yaml
train:
  repo: doanh25032004/vgp_datasets
  include: ["bimanual/*", "humanoid/*", "<your_embodiment>/*"]
  dest: corpus
```

```bash
kinescore pull --what train      # -> $KINESCORE_DATA_ROOT/corpus/
```

## 2. Put the URDF under `$KINESCORE_ASSETS`

Path: `$KINESCORE_ASSETS/<robot>/urdf/*.urdf`. Expand any xacro first. `pytorch_kinematics`
reads a single tree, so a two-arm robot needs both arms merged under one root:
`aloha_bimanual.urdf` is two Interbotix vx300s arms under a synthetic `world` link.

## 3. Write `robots/<robot>/`

Three files turn the logged state into keypoints:

```
src/kinescore/robots/aloha/
    constants.py   joint names, keypoint links, ACTUATED_LINKS, EE_LINK
    fk.py          pytorch_kinematics chain, gripper scatter, rest bone lengths
    spec.py        the RobotSpec: keypoints, bone pairs, rigid-bone mask
```

`constants.py` is declared by hand from the URDF and the corpus. It lists the predicted joint
names in the order the corpus stores them (read `meta/modality.json`, not the URDF order) and
the links whose origins become keypoints. The order of that keypoint tuple is the keypoint
index everywhere downstream. `ACTUATED_LINKS` are links moved by a joint you do not predict,
such as fingers, and `EE_LINK` is the tool-centre point on each side.

`fk.py` mostly wires up the helpers in `robots/base.py`:

```python
chain = pk.build_chain_from_urdf(open(urdf_path, "rb").read())
assert_keypoints_in_urdf(set(chain.get_frame_names(exclude_fixed=False)),
                         {"left": KEYPOINTS_LEFT, "right": KEYPOINTS_RIGHT}, ee_link=EE_LINK)
pred_chain_idx = build_pred_chain_index(chain.get_joint_parameter_names(),
                                        LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS)
lo, hi, vel, eff = read_joint_limit_arrays(urdf_path, pred_joints)
pairs = consecutive_bone_pairs(num_keypoints)
lengths = rest_pose_bone_lengths(keypoints_at_rest, pairs)
```

`scatter_predicted(q_flat, pred_chain_idx, n_chain_joints)` places the predicted joints into the
chain and leaves the others at rest. ALOHA's grippers are not in `q`: they arrive as `aux`
`(B, T, 2)` in `[0, 1]` and are mapped onto the prismatic finger joints (0.021 m closed,
0.057 m open). A wrong joint name, a joint missing from the chain or a keypoint link missing
from the URDF raises an error when `fk.py` loads.

`spec.py` decides which bones rigidity checks. A bone is dropped when either end is in
`ACTUATED_LINKS` or when it is shorter than 1 mm at rest, because its length can legitimately
change. ALOHA keeps 10 of 16 bones: per arm, the three bones touching a finger are dropped.

```python
mask = structural_rigid_bone_mask(self.keypoint_links, self.bone_pairs,
                                  self.bone_lengths, ACTUATED_LINKS)
warn_dropped_bones("AlohaSpec", self.keypoint_links, self.bone_pairs, self.bone_lengths, mask)
self.rigid_bone_pairs = self.bone_pairs[mask]
```

Nothing checks `ACTUATED_LINKS` for you. Run once and compare the `warn_dropped_bones` output
with the URDF: a finger bone left in makes every grasp look like a rigidity violation.

## 4. Register the robot

`robots/__init__.py`, with a lazy import:

```python
def _build_aloha_bimanual(**kwargs) -> RobotSpec:
    from kinescore.robots.aloha.spec import AlohaSpec
    return AlohaSpec(**kwargs)

_REGISTRY.register("aloha_bimanual", _build_aloha_bimanual)
```

`configs/robots.yaml`; `train` refuses to run if forward kinematics returns a different
keypoint count:

```yaml
aloha_bimanual:
  embodiment: bimanual
  keypoints: 18
```

## 5. Declare the view and the reader

The view is the clip size the reader expects. Add one to `configs/views.yaml` only if your
generated clips have a size not listed there:

```yaml
sv1_16x9:
  n_views: 1
  packing: none
  panel: [768, 432]
```

The reader entry in `configs/cells.yaml` says which robot, which corpus columns and which
camera to train on:

```yaml
_corpus_bimanual_sv: &bimanual_sv
  corpus: bimanual_sv
  adapter: lerobot
  root: ${KINESCORE_DATA_ROOT}/corpus/bimanual/singleview
  cameras: [cam_high]
  joint_field: observation.state
  joint_columns: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]   # 6 and 13 are the grippers
  gripper_columns: [6, 13]

readers:
  aloha_bimanual.bimanual_sv.sv1_16x9:
    robot: aloha_bimanual
    view: sv1_16x9
    train: *bimanual_sv
```

## 6. Train

```bash
R=aloha_bimanual.bimanual_sv.sv1_16x9
kinescore data  --reader $R                  # 90/10 scene-disjoint train/val split
kinescore cache --reader $R --device cuda    # DINOv3 tokens
kinescore train --reader $R --device cuda    # 6000 steps; reports val_mm
```

The checkpoint lands in `$KINESCORE_CKPT_DIR/$R.pt`, and the held-out real clips in
`$KINESCORE_DATA_ROOT/trees/$R/videos/val`.

## 7. Calibrate, then score

Fit the thresholds on those held-out real clips and keep the file next to the others in
`thresholds/`. Scoring the same real clips afterwards should give a low rate (about 5% at the
95th percentile); a much higher rate points at the reader or the rigid-bone mask.

```bash
kinescore calibrate --model $KINESCORE_CKPT_DIR/$R.pt \
  --real $KINESCORE_DATA_ROOT/trees/$R/videos/val --out metrics/kiva/thresholds/$R.json
kinescore score --model $KINESCORE_CKPT_DIR/$R.pt \
  --threshold metrics/kiva/thresholds/$R.json --videos /path/to/clips
```

## 8. Hook into the benchmark

For a robot whose embodiment already exists, point `run.sh` at the new reader with
`HAL_CKPT_<EMB>=<ckpt> HAL_THR_<EMB>=<thresholds>` (for example `HAL_CKPT_BIMANUAL`). For a new
embodiment, add it to `EMBODIMENTS`, `thr_for`, `ckpt_for` and `tag_for` in `run.sh`, and to
the `EMBODIMENTS` tuple in `metric.py`.
