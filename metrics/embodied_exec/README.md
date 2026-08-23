# `metrics/embodied_exec` — the execution-grounded (expensive) tier

`metrics/sim2real` checks the *geometry* of hand-object interaction without a
simulator. This module targets what that proxy cannot reach: whether the behaviour
in a generated video corresponds to a robot configuration the real robot could
actually have been in. That is RoboWM-Bench's headline robot failure mode —

> the end effector may appear to reach the target in the video, while the recovered
> joint configuration yields a mismatched end-effector pose under the robot's true
> forward kinematics

— and it is invisible to any pixel- or mask-level metric.

**Status: infrastructure built and validated on ground truth; never yet run on a
generated video.** The gate is the pose decoder (component 3 below) — without it there
are no joint angles for a prediction, and all three predicates need them. Read the
scope table before planning on this tier.

## What the paper does, and which parts of it scale

RoboWM-Bench: video → action (IDM / hand retargeting) → **execute in a digital twin**
→ per-step checkers → success rate. Splitting that into independent pieces shows only
one of them fails to scale, and it is not the one people expect:

| component | scales with | our cost | verdict |
|---|---|---|---|
| robot model (URDF/MJCF) + FK | **embodiment** | Panda verified 3.3e-08 m; ALOHA + **GR1 (robosuite)** GT-within-limits 0.000%; **Airbot MMK2 missing** | mostly available |
| video → action (IDM) | **embodiment** | 6,688 action-labelled episodes on disk | **not built — the gate** |
| checkers | nothing | pure geometry; 7/7 controls pass on GT | free |
| **digital twin** (mesh + 6D pose + arena) | **task/scene** (~100+) | 0 meshes, 0 intrinsics | does not scale |

So the twin is dropped and the robot's own kinematics becomes the grounding. A handful
of robot models covers every embodiment; scene reconstruction never would. The paper
could afford twins because Lightwheel — a real-to-sim asset company — is a co-author,
and even then they shipped 8 + 8 + 6 tasks × 10 configs.

One honest limit up front: FK-consistency needs **joint angles**, and the groups differ
more than the embodiment names suggest.

| group | joint angles | independent EE column | robot model |
|---|---|---|---|
| `single_arm/multiview` | 7-D `joint_position` | **yes** (`cartesian_position`) | Panda, verified to 3.3e-08 m |
| `single_arm/singleview` | **none** (7-D cartesian only) | — | Panda |
| `bimanual` (sv + mv) | 14-D `qpos(0:14)` | **no** | ALOHA (unverifiable, see below) |
| `humanoid/multiview` | 36-D | no | **missing** (Airbot MMK2) |
| `humanoid/singleview` | 44-D | no | **GR1, from robosuite** — GT within limits (see below) |

Two consequences worth stating loudly:

**"humanoid" is two different robots, not one.** `humanoid/singleview` is a legged
humanoid — `left_arm 7 + left_hand 6 + left_leg 6 + neck 3 + right_arm 7 +
right_hand 6 + right_leg 6 + waist 3` = 44-D, with 6-DoF dexterous hands — while
`humanoid/multiview` is Airbot MMK2, a wheeled dual-arm manipulator with 36-D state
and no `modality.json`. They need *two* models. `mujoco_menagerie` has neither (it
ships `fourier_n1`, not GR1, and no Airbot), which is why this group was previously
blocked — but **robosuite ships GR1**, so `humanoid/singleview` is now unblocked
(`checkpoints/mjcf_robosuite/gr1/robot.xml`, fetched by
`scripts/bootstrap_mjcf_assets.sh`). Airbot MMK2 remains missing. The legs also explain
why this group scores worst in the identifiability probe: it is being scored on joints
an ego camera cannot see. And 6-DoF hands mean the two-finger `aperture` predicate does
not transport there unchanged.

### GR1 (robosuite) vs the dataset — measured

The MJCF has exactly 32 body hinges = `waist 3 + head 3 + arm 7×2 + leg 6×2`, matching
every group in the dataset's own `modality.json` (the two 6-DoF hands are separate
files, `fourier_{left,right}_hand.xml`, and are excluded: 44 − 12 = 32).

Checked on **all 260 episodes / 77,890 frames** of real GT against the MJCF ranges:

| quantity | value |
|---|---|
| values out of range | 7,087 of 2,492,480 (frame, joint) pairs = **0.284%** |
| joints implicated | **only 4 of 32** — `r_shoulder_roll` 4.89%, `l_shoulder_roll` 3.74%, `l_elbow_pitch` 0.36%, `r_elbow_pitch` 0.10% |
| joints that never violate | **28 / 32** |
| excess magnitude | p50 0.0035, p99 0.0208, **max 0.0468 rad (2.68°)** |

Every implicated joint has a bound of exactly `0.000` and the worst excess is under 3°.
A wrong joint *order* within a group would produce excursions of order 1 rad, not 0.03,
so this is model conservatism at the boundary — the same pattern as `PANDA_LIMIT_SLACK`.
Slack sweep: 0.02 rad leaves 1.13% of excesses, 0.03 leaves 0.085%, **0.05 rad (2.9°)
leaves 0.000%**. Applied to those four joints only, GT now measures **0.000%**, which is
what makes a prediction-side violation meaningful.

### A latent bug this exposed: joint id ≠ qpos address

`Kinematics` used a single index array for both `jnt_range` lookups and `qpos` writes.
Those coincide only for a chain of 1-DoF hinges based at `qpos 0` — true for Panda and
ALOHA, **false for any floating-base robot**. GR1's free joint occupies `qpos 0..6`, so
measured: `jnt_ids = [14, 15, …]` while `qpos_index = [20, 21, …]`, off by exactly 6.
The old code would have read entirely wrong joint ranges with no error raised.

Fixed by naming the joints (`RobotSpec.joint_names`) and letting MuJoCo resolve both
indices. `RobotSpec.state_dims` was added for the same reason: GR1's model dims are not
a leading contiguous slice, because the two hands are interleaved between the body
groups, so `joint_slice` cannot express them. Panda re-verified after the change at
**3.32e-08 m** — no regression.

**Two levels of evidence, and the difference matters.** `validate_group` now reports
`evidence: fk` or `evidence: limits`:

| group | evidence | GT violation | detail |
|---|---|---|---|
| `single_arm/multiview` | **fk** | 0.000% | FK residual 3.32e-08 m — pins joint order, base frame and units at once |
| `humanoid/singleview` | limits | 0.000% | GR1, after 0.05 rad slack on 4 joints |
| `bimanual/singleview` | limits | 0.000% | ALOHA arm joints (gripper dims excluded, unit mismatch) |
| `bimanual/multiview` | limits | 0.000% | same model |

`limits` can **refute** a bad mapping but cannot **confirm** a good one — it says nothing
about the base frame or the joint order within a group. Do not read a `limits` pass as
equivalent to the FK-validated case.

**Only `single_arm/multiview` can validate its own robot geometry.** It is the one group
carrying joint angles *and* an independently measured end-effector column, which is what
made the 3.3e-08 m check possible. Elsewhere joint angles are recoverable by slicing
`observation.state` per `modality.json`, but with no independent EE column there is no
way to confirm the MJCF's *link lengths*.

Joint **limits**, however, are verifiable everywhere without an EE column, because a
real robot cannot violate its own mechanical range: measure the GT violation rate and a
non-zero result convicts the model. Applied to bimanual vs the ALOHA MJCF over 40
episodes / 33,545 frames, this validated the arm and caught **two** traps.

⚠️ **Trap 1 — qpos layout mismatch (this one produced a wrong conclusion first).**
ALOHA's MJCF has **16** qpos (two finger joints per side); the dataset has **14** dims
(one gripper value per side). Slicing `jnt_range[:14]` silently misaligns every dim
after index 5 — MJCF 6 and 7 are *both* left fingers, so `right/wrist_rotate` lands at
index 13. An explicit dataset→qpos map (`joint_index_map`) is mandatory. With it:

```
GT joint-limit violation (gripper dims excluded) : 0.000%  mean and worst
per-dim raw violations : [0]*6 + [31307] + [0]*6 + [31211]
                          ^ 12 arm joints: zero    ^ only the 2 gripper dims
```

All **12 arm joints sit strictly inside** ALOHA's ranges (`L_shoulder` GT
`[-1.459,+0.569]` in MJCF `[-1.850,+1.257]`; `R_elbow` GT `[-0.150,+1.523]` in
`[-1.763,+1.606]`), so ALOHA's arm model matches the real bimanual robot and
`kinematic_feasibility` is sound there.

⚠️ **Trap 2 — gripper unit mismatch.** The two gripper dims violate on ~93% of frames,
and it is units, not physics:

```
GT gripper   [-0.002, 1.000]   normalised open/close
MJCF gripper [ 0.000, 0.041]   metres of finger travel
```

Run without excluding them, bimanual reports a meaningless 99.49% violation rate. They
are excluded via `limit_check_dims` until a normalised→metres calibration is validated —
which is also a prerequisite for the `aperture` predicate on this embodiment.

### Which errors cancel and which do not

A wrong MJCF does not damage every predicate equally, and the rule is simple:

- **Cancels:** both sides of the comparison pass through the same MJCF — i.e. prediction
  vs paired GT. A systematic geometric bias shifts both equally and largely differences
  out, the same reasoning that lets `metrics/sim2real` reach an exact 1.000 oracle.
- **Does not cancel:** one side comes from the MJCF and the other from vision. Exactly
  one predicate is in this class — **`aperture`**, which compares gripper opening (MJCF,
  in cm) against object width (pixels → cm). Here a model error produces *wrong verdicts*,
  not just noise.

| predicate | affected on bimanual? |
|---|---|
| `kinematic_feasibility` (limits) | no — 12/12 arm joints verified |
| `fk_consistency` | systematic offset, largely cancels under paired-GT |
| **`aperture`** | **yes, seriously** — needs the 0–1 → metres calibration first |

`single_arm/singleview` keeps only a 7-D cartesian state, so FK-consistency and
joint-limit feasibility are not computable there at all; the code reports `skipped`
rather than scoring something else.

## Measured: the whole benchmark is addressable (99.9%)

"Applies to everything" is a claim, so `episode_index.py` measures it instead of
assuming. It maps every prediction mp4 to its GT episode parquet:

```
dataset index   : 1608 GT episodes
mp4 seen        : 62183   (output/ only; input/ holds GT copies)
parsed          : 62181
resolved to GT  : 62136   (99.9% of parsed)
by pattern      : {1: 1560, 2: 4509, 3: 5103, 4: 2272, 5: 36686, 6: 3867, 7: 1200, 8: 6984}
```

Eight naming conventions coexist in the tree, accumulated as models were added:

| # | layout | where |
|---|---|---|
| 1 | `{split}/iter_*/{view}/episode_N.mp4` | single_arm sv dreamgen |
| 2 | `{split}/iter_*/episode_N/{role}.mp4` | single_arm sv dreamdojo |
| 3 | `{split}/episode_N.mp4` | single_arm mv dreamgen |
| 4 | `{split}/iter_*/{task}__episode_N.mp4` | bimanual sv dreamgen |
| 5 | `{split}/iter_*/{task}/episode_N/{role\|seg}.mp4` | bimanual/humanoid mv dreamgen |
| 6 | `{split}/episode_N/{role}.mp4` | ctrlworld |
| 7 | `{split}/iter_*/{task}__episode_N/{role}.mp4` | bimanual sv dreamdojo |
| 8 | `{split}/episode_{task}__N/{role}.mp4` | bimanual/humanoid mv ctrlworld |

Patterns 4/7 put the task *before* the id; pattern 8 puts it *after*. Two further
traps had to be handled:

- **Episode ids are not globally unique.** `episode_000004` exists under many
  bimanual task dirs, so the key is `(embodiment, view_type, split, task, id)`.
- **Ids are written with 4 or 6 digits** (`episode_0023` vs `episode_000023`) by
  different scripts, so they are compared as integers. Before this fix one group
  resolved at 48.5%.

The 45 stragglers are genuine orphans, not resolver bugs:
`fastercache/single_arm/.../dreamdojo/non_makovian` contains `episode_0122`–`0129`
while that split's GT stops at `episode_000119`.

```bash
python3 metrics/embodied_exec/episode_index.py --min-coverage 0.99 \
  --out /tmp/ee_index.jsonl --roles pred pred_strip bare seg
```

## Measured: every action space is an absolute position target

Checked over all 1,608 episodes, not a sample:

| group | action dim | state dim | `mean abs(a_t - s_t)` |
|---|---|---|---|
| single_arm/mv | 8 | 8 | 0.0427 |
| single_arm/sv | 7 | 7 | **0.0015** |
| bimanual/sv+mv | 14 | 42 | 0.0628 |
| humanoid/mv | 36 | 36 | **0.000000** |
| humanoid/sv | 44 | 44 | 0.0401 |

Nothing here is a velocity or a delta — `action` is a position setpoint, and for
humanoid multiview it is a **verbatim copy of the state** on all 248 episodes. So
"recover the action" is really "pose the robot from pixels", and the per-step action
follows by differencing. That is the better-posed problem, and it is why the probe
measures `state_abs` and `state_delta` separately.

It also means an IDM here needs no action-space reverse engineering per embodiment;
it needs a **pose decoder**, and the paper's two-consecutive-frames IDM input is a
design choice we can revisit rather than inherit.

## Measured: pose is recoverable *within* an episode, not *across* episodes

Frozen CLIP ViT-B/32 + closed-form ridge (a **lower bound** on a trained IDM), split
by episode, `datasets/` GT video only:

| group | `state_abs` across-ep | `state_abs` within-ep | `state_delta` across-ep | `state_delta` within-ep |
|---|---|---|---|---|
| **bimanual/singleview** | −0.557 | **+0.820** (max +0.947) | −0.232 | **+0.609** |
| humanoid/singleview | −0.025 | +0.600 (max +0.859) | −0.123 | +0.587 |
| humanoid/multiview | −0.130 | +0.171 (max +0.614) | −0.055 | +0.520 |
| single_arm/multiview | −0.280 | +0.663 (max +0.906) | −0.182 | +0.098 → +0.467 @ h=30 |

(within-episode at horizon 30 for the top three, horizon 5 → 30 sweep for single_arm.)

Across-episode training R² is 0.77–0.85 while test R² is ≈ 0 **everywhere**: the
features fit but do not transfer. Within-episode, **every group is positive** and
bimanual reaches +0.820 / +0.609 — its `cam_high` really is a fixed overhead camera,
which is exactly the condition the paper relies on. `humanoid/multiview` is the weakest
(`state_abs` +0.171) and it is also the group whose 36-D state has no `modality.json`
to trim, so it is scored on all dims including ones the camera cannot see; that is a
known follow-up, not a blocker.

The cause of the across-episode collapse is measurable, and for DROID it is not a
modelling failure but an **ill-posed problem**:

```
camera_extrinsics.exterior_1_left over 60 episodes (xyz + rpy)
  WITHIN  episode std : ~1e-6          <- camera is bolted down during an episode
  ACROSS episode std  : [0.13 0.47 0.13 0.21 0.09 0.82]
  ACROSS episode range: [0.69 1.44 0.53 0.95 0.72 2.88]
```

Yaw varies by 0.82 rad std (2.88 rad range, ~165°) and position by up to 1.44 m
between episodes. Pixels therefore **do not determine base-frame pose** unless the
camera is known. This is exactly why the paper fixes the camera per scene and
calibrates it with FEEPE — and why their protocol cannot be lifted onto DROID
unchanged.

### Why within-episode is the right regime, not a cop-out

Because `episode_index.py` resolves 99.9% of the tree, every generated video comes
with the GT episode it is being compared against. The decoder never has to transfer
across episodes: it is **fit on the paired GT episode and applied to the prediction**,
which holds the camera fixed by construction (within-episode extrinsics std ~1e-6).
The metric is a comparison against that episode's own GT, in the same spirit as the
paired-GT differencing that makes `metrics/sim2real`'s oracle control reach 1.000.

The probe splits alternate frames rather than prefix/suffix, so test poses are
interpolations — it measures "are the pixels informative about pose", not "can it
extrapolate in time".

`state_delta` is horizon-limited, and the sweep is unambiguous (single_arm/multiview,
within-episode):

| horizon (frames @ 15 fps) | `state_abs` | `state_delta` |
|---|---|---|
| 5 | +0.663 | +0.098 |
| 15 | +0.497 | +0.177 |
| 30 | +0.476 | **+0.467** |

Over 5 frames the motion is at the noise floor; by 30 frames (2 s) displacement is
clearly readable and `state_delta` nearly matches `state_abs`. The paper's IDM takes
**two consecutive frames** — at our frame rates that is the worst end of this curve,
so the horizon is a parameter to choose from this table rather than inherit. These are
the numbers a trained IDM has to beat.

```bash
CUDA_VISIBLE_DEVICES=0 models/dreamgen/.venv/bin/python \
  metrics/embodied_exec/identifiability_probe.py \
  --groups single_arm/multiview bimanual/singleview humanoid/singleview \
  --episodes 20 --stride 10 --horizon 30 --alpha 1000 --out /tmp/ident.json
```

Note the probe restricts the target to the dims a camera can see, read off each
group's `meta/modality.json` (`TARGET_DIMS`): bimanual state is
`qpos(0:14) + qvel(14:28) + effort(28:42)`, and joint velocity and torque are not
readable from two frames while dominating the variance. Runtime is dominated by CLIP
over long episodes — bimanual episodes are 800–1000 frames, so 20 episodes take ~19 min
on one A100.

## Measured: the Panda MJCF *is* DROID's robot (FK residual 3.3e-08 m)

This is the load-bearing check for the whole design. If forward kinematics from a
public MJCF does not reproduce the dataset's own end-effector column, then FK residual
is a calibration offset and worthless as a metric. Measured over **all 260**
`single_arm/multiview` episodes, both splits:

```
max |FK(joint_position) - observation.state.cartesian_position|
  mean over episodes  = 3.3e-08 m
  worst episode       = 4.2e-08 m
ground-truth joint-limit violation rate = 0.000%   (both splits, 130 eps each)
```

3e-08 m is floating-point noise: `mujoco_menagerie`'s `franka_emika_panda/panda.xml`
matches DROID in joint order *and* base frame with no fitting.

⚠️ **The limit check needed one measured correction, and finding it is why this was
run over all 260 episodes rather than a sample.** A 15-episode sample reported a
1.03% GT violation rate (worst episode 15.5%), which would have quietly invalidated
the checker. Sweeping all 74,116 frames localised it exactly:

```
frames with violation : 138 / 74116  (0.186%)
per-joint counts      : [0 0 0 0 0 138 0]   <- all joint6, nothing else
joint6 MJCF range     : [-0.0175, 3.7525] rad
worst excess          : 0.485 rad, only 3 episodes involved
```

Every violation is joint6, whose MJCF lower bound (−0.0175 rad) is tighter than the
physical robot revision DROID was recorded on. This is a model-vs-hardware revision
difference, not DROID violating physics, so `PANDA_LIMIT_SLACK` widens that single
joint by 0.5 rad and leaves the other six at zero. GT then measures **0.000% on all
260 episodes**, which is the property the checker depends on: real trajectories always
pass, so a violation belongs to the prediction.

```bash
bash scripts/bootstrap_mjcf_assets.sh   # -> checkpoints/mjcf
models/dreamgen/.venv/bin/python -m metrics.embodied_exec.kinematics \
  --groups single_arm/multiview --split makovian --max-episodes 130
```

Note `single_arm/singleview` stores a 7-D cartesian state with **no joint angles**, so
FK-consistency is not computable there; the module reports `skipped` rather than
silently scoring something else.

## Measured: the checkers fire, and only on what they should

Following `metrics/sim2real`, no checker is trusted until an oracle and a set of
synthetic violations prove sensitivity *and* specificity. 20 episodes, ceilings
calibrated from GT at p99.5:

| control | `fk_resid_max` | `limit_frac` | `vel_exceed` | `acc_exceed` | `ee_err_mean` | `path_ratio` |
|---|---|---|---|---|---|---|
| **oracle** | **3.3e-08** | **0.000** | 0.033 | 0.030 | **0.000000** | **1.000000** |
| freeze | 3.3e-08 | 0.000 | 0.000 | 0.000 | 0.289 | **0.000** |
| overshoot | 3.3e-08 | **0.319** | 0.490 | 0.344 | 0.256 | 2.762 |
| jitter | 3.3e-08 | 0.000 | 0.999 | **1.000** | 0.076 | 24.985 |
| time_reverse | 3.3e-08 | 0.000 | 0.033 | 0.030 | **0.198** | 1.000 |
| **fk_desync** | **0.396** | 0.000 | 0.033 | 0.030 | 0.181 | 1.295 |
| limit_break | 3.3e-08 | **1.000** | 0.023 | 0.024 | 0.908 | 0.956 |

`OVERALL: PASS` (7/7). What the table shows:

- **The oracle is exact.** `ee_err = 0.000000`, `path_len_ratio = 1.000000`,
  `limit_frac = 0.000`. The checkers add no bias of their own.
- **`fk_desync` is the paper's headline failure and it is caught cleanly.** Same joint
  trajectory, end-effector claim taken from a shifted point on the path — exactly "the
  end effector appears to reach the target while the joint configuration says
  otherwise". `fk_residual` jumps 7 orders of magnitude (3.3e-08 → 0.396 m) while
  `limit_frac` stays at 0.000.
- **Specificity is real, and `time_reverse` proves it.** Playing a trajectory backwards
  changes nothing about which configurations are visited, so `limit`, `vel` and `acc`
  read *identically to the oracle* and only `ee_err` moves. A term that fired here
  would be measuring the wrong thing.

⚠️ **`vel_exceed_frac` is 0.033 on the oracle, not 0, and that is by construction.**
The ceilings are the p99.5 of GT itself, so ~0.5% of GT samples per joint exceed them
by definition. Read velocity/acceleration terms as *excess over GT*, never as absolute
violations — the same GT-relative framing `metrics/sim2real` needs for its violation
terms. Ceilings are calibrated from data rather than a datasheet specifically so this
ports to the next embodiment without hunting down spec sheets.

```bash
models/dreamgen/.venv/bin/python -m metrics.embodied_exec.run_checker_controls \
  --episodes 20 --out /tmp/ee_controls.json
```

## Still to build

| # | component | blocked on |
|---|---|---|
| 3 | pose decoder — joint angles from a *generated* video, conditioned on the paired GT episode | nothing technically; this is the real gate (see below) |
| 4b | Airbot MMK2 model (`humanoid/multiview`) | not in `mujoco_menagerie` **nor robosuite** — must be sourced elsewhere. GR1 is now solved via robosuite. |
| 6 | batch driver over the resolved index + per-method summary | 3 |

⚠️ **Component 3 is not one item among several — it gates the entire tier.** Every
number in this README was computed from joint angles read out of the **GT parquet**.
All three predicates (`fk_consistency`, `kinematic_feasibility`, `aperture`) need the
joint angles *of the generated video*, and nothing in the repo produces those yet. So
the tier has never been run on a single generated frame; it is validated infrastructure
waiting on its input.

Realistic scope, in order of confidence:

1. **`single_arm/multiview`** — the only group whose robot model is verified against its
   own data. Do this one end to end first: pose decoder → three predicates → run on real
   predictions and check the metric separates `dense` from a known-bad method. If it
   cannot separate them there, stop; the other groups are strictly harder.
2. **bimanual** — works in principle (joints via `qpos(0:14)`, ALOHA MJCF fetched) but
   the model cannot be validated, so report it a tier weaker.
3. **`humanoid/singleview`** — GR1 now available (robosuite) and GT sits at 0.000% inside
   its limits, so this group joins bimanual's tier: runnable, model consistent but not
   FK-validated. The legs remain scored-but-invisible to an ego camera, and the 6-DoF
   hands still break the two-finger `aperture` predicate.
4. **`humanoid/multiview`** — still blocked on the Airbot MMK2 model, not on effort.

`scripts/bootstrap_mjcf_assets.sh` pulls ALOHA alongside Panda from `mujoco_menagerie`,
plus GR1 and the gripper models from robosuite into `checkpoints/mjcf_robosuite`.
ALOHA's joint list (`left/waist … left/right_finger`, `right/waist … right/right_finger`)
lines up with bimanual's `qpos(0:14)` — 2 × (6 arm + 1 gripper). So 4b is done for
bimanual and for `humanoid/singleview`; what is missing for both is a way to *validate*
the model, not the model itself.

## Verified capabilities and one that failed

Worth recording because the failure was assumed to be a free win:

- **Gripper aperture in centimetres — works, exactly.** `qpos` 0.00 / 0.02 / 0.04 maps
  to fingertip separation 0.00 / 4.00 / 8.00 cm. This is what makes the `aperture`
  predicate possible: fingers closed to 2 cm cannot be holding an 8 cm object, and no
  mask-based metric can see that.
- **Self-collision from raw contact counts — does NOT work.** `mj_forward` reports 10
  contacts for an ordinary pose and 0 for an extreme twisted one, because adjacent links
  touch by design. Usable self-collision detection needs the adjacent-link pairs filtered
  out first; the raw count is worse than useless (it is anti-correlated). Do not add
  self-collision as a predicate without that filtering.

## Dependencies

One new package: `mujoco` (3.11.0), for FK and joint ranges only — no simulation
rollout. Verified not to disturb the existing `models/dreamgen/.venv`:

```
numpy 1.26.4   transformers 4.51.3   torch 2.6.0+cu124   mujoco 3.11.0
```

`numpy` is untouched, which matters: `metrics/quality/README.md` records that a numpy
upgrade breaks cosmos-predict2/DreamGen. For the same reason the ridge in
`identifiability_probe.py` is solved in closed form rather than by installing
scikit-learn. CLIP comes from the local `checkpoints/openai/clip-vit-base-patch32`.

Robot models are fetched by `scripts/bootstrap_mjcf_assets.sh` from two upstreams, kept
as separate sparse clones (a few MB each):

| root | upstream | provides | override |
|---|---|---|---|
| `checkpoints/mjcf` | `mujoco_menagerie` | `franka_emika_panda`, `aloha` | `$MJCF_ROOT` |
| `checkpoints/mjcf_robosuite` | `robosuite` | `gr1` (+ 15 gripper MJCFs) | `$RS_MJCF_ROOT` |

Only the MJCFs are checked out — **robosuite itself is not installed**, deliberately: the
tier needs the asset files, not the framework, and `metrics/quality/README.md` records
that dependency churn (particularly `numpy`) breaks cosmos-predict2/DreamGen. Both roots
are on `MJCF_SEARCH`, so `resolve_mjcf("gr1/robot.xml")` just works.

The gripper MJCFs are not used by this module yet; they are the intended replacement for
the two placeholder finger boxes in `metrics/force_feasibility/proxy_scene.py`, whose
approximate geometry is the largest source of false contacts there.
