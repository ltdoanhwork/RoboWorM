# `metrics/force_feasibility` — dynamics, not geometry

The tier that answers the question this benchmark is named after: **is the observed
motion consistent with Newton's laws?**

## Why this exists

| tier | question it answers | can reach mass / friction / gravity? |
|---|---|---|
| `metrics/sim2real` | did the object move untouched, follow the grasp, keep its shape? (geometry, object-diameter units, no simulator) | no |
| `metrics/embodied_exec` | is the *robot's* joint configuration consistent with its forward kinematics? | no |
| **`metrics/force_feasibility`** | **could any admissible contact force have produced this motion?** | **yes** |

The first two tiers are geometric. `sim2real`'s `levitation` predicate says "the object
moved while nothing touched it", which has to special-case gravity by hand — a falling
object also touches nothing. Neither tier can distinguish an object falling at
9.81 m/s² from one drifting sideways in mid-air.

## The measurement

At each frame, solve a feasibility problem:

> does there exist a set of contact forces, each inside its friction cone, that
> together with gravity produces the acceleration we observed?

```
minimise  || A·λ − (m·a_obs − m·g) ||    subject to   λ ≥ 0
```

`A`'s columns are friction-cone generators (`n + μ·t_j`, pyramid with 4 tangents,
under-approximating the true cone so the result is conservative) built from the contact
normals MuJoCo detects at the observed configuration. Solved with `scipy.optimize.nnls`.
The residual is reported as **`|r| / (m·g)`** — the leftover force in units of the
object's own weight.

### Two properties that make this the right formulation for video

**Mass cancels exactly.** The problem is homogeneous of degree one in mass, so the
normalised residual is invariant. Measured, identical to six decimals:

| object mass | `levitation` residual |
|---|---|
| 0.01 kg | 1.000000 |
| 0.05 kg | 1.000000 |
| 5.00 kg | 1.000000 |

Object mass is unknowable from a video. Here it does not enter. Neither does contact
stiffness. **Friction is the only unknown left, and it is swept, not calibrated** — a
trajectory is charged only when no plausible `μ` explains it.

**There are no tuned constants.** This matters because it is the second design; the
first one had three.

## What the first design got wrong (and why it was replaced)

The initial version drove the object along the observed path with a PD controller
inside `mj_step` and reported the injected force. It failed for reasons worth recording:

1. **Three hand-tuned constants** — tracking bandwidth, damping, force cap.
2. **Controller lag leaked into the residual.** An honest free fall scored **0.767**
   instead of 0, purely from tracking error (`track_err = 0.179 m`).
3. **It depended on penetration depth to produce grip force.** Measured: a
   kinematically-driven rigid finger generates almost no normal force — the proxy grasp
   slipped with 4 mm of overlap and `ncon` showed *only* `support–obj` contacts during
   the lift. Fixing that needed an explicit contact stiffness — another constant, and
   one that would have been read off noisy monocular depth in the real pipeline.

The friction-cone formulation has none of these. MuJoCo still does the physics that
matters (collision detection, contact frames at the observed configuration); what was
dropped is forward integration, which is what made the answer depend on gains.

## Controls: 11/11 pass, matching closed-form values

`python -m metrics.force_feasibility.run_newton_controls`

```
control                     mean     max  lift_bias    horz  ncon    expect      got
rest_on_table              0.000   0.000      0.000   0.000   4.0     0.000    0.000
freefall_true              0.000   0.000     -0.000   0.000   0.0     0.000    0.000
grasp_lift_valid           0.000   0.000      0.000   0.000   8.0     0.000    0.000
ghost_follow               1.051   1.051      1.051   0.000   0.0     1.051    1.051
levitation                 1.000   1.000      1.000   0.000   0.0     1.000    1.000
freefall_slow              0.700   0.700      0.700   0.000   0.0     0.700    0.700
drift_airborne             1.021   1.021      1.000   0.204   0.0     0.204    0.204
slide_beyond_friction      1.365   1.365     -0.853   1.066   4.0     0.582*   0.582
teleport                   2.120   6.600      1.000   1.305   0.0         -    6.600
```

`* best over the μ sweep.` **Noise floor = 0.00000.** Every expectation is derived on
paper, so the residual has to land on a number, not merely rank correctly:

- `freefall_slow` at `a = 0.3g` needs `0.7·m·g` of support → **0.700**
- `drift_airborne` at `a_h = 2.0` needs `a_h/g` → **0.204**
- `ghost_follow` lifting at `a = 0.5` needs `1 + a/g` → **1.051**
- `slide_beyond_friction` is the *perpendicular distance to the cone*,
  `(a_h − μg) / (g√(1+μ²))` — a projection, not a subtraction → **0.582**

### The two pairs that carry the argument

**Specificity, which geometry cannot get right:** `freefall_true` touches nothing and
scores **0.000**; `levitation` hovers at the same height, also touching nothing, and
scores **1.000**. Gravity is not special-cased anywhere — it falls out.

**What this adds over the cheap tier:** `grasp_lift_valid` and `ghost_follow` are the
*same object trajectory*. The only difference is where the gripper is. Geometrically
both satisfy `sim2real`'s `grasp_follow` (object and gripper move together in the
image). Dynamically they differ by exactly one body weight: **0.000 vs 1.051**.

Note `grasp_lift_valid` is feasible at *every* `μ`, correctly: opposing fingers may
squeeze arbitrarily hard, so normal force is unbounded and the cone always reaches.
The friction sweep therefore only bites on single-contact cases like sliding — where it
moves the verdict a lot (`slide_beyond_friction`: 0.582 at μ=1.5 vs 2.303 at μ=0.2).

## Scale sensitivity — why the vision pipeline is viable

The tier needs metres; the cheap tier produces object-diameter units. Measured effect of
scaling the whole trajectory by `s` (i.e. getting the metric scale wrong by that factor):

| `s` | `freefall_true` | `levitation` | `ghost_follow` |
|---|---|---|---|
| 0.50 | 0.500 | **1.000** | 1.025 |
| 0.70 | 0.300 | **1.000** | 1.036 |
| 1.00 | 0.000 | **1.000** | 1.051 |
| 1.40 | 0.400 | **1.000** | 1.071 |
| 2.00 | 1.000 | **1.000** | 1.102 |

**Severe violations are scale-invariant.** With no contact, the residual is
`|s·a_obs − g| / g`; for a hovering object `a_obs = 0`, so it reads 1.000 for *any* `s`.
`ghost_follow` moves only 7% across a 4× scale error. The reason is structural: without
contact the object *must* fall, and "not falling" is a scale-invariant observation.

Only the borderline case — distinguishing a fall at `g` from a fall at `0.8g` — needs an
accurate scale, where the error is linear (`|s−1|`). So the failures video models
actually make are measurable **before** the metric-scale problem is solved.

## Metric scale: what is available

| group | metric source | status |
|---|---|---|
| `single_arm/multiview` | `observation.state.cartesian_position` (metres; span 0.35 × 0.30 × 0.56 m) | available |
| `single_arm/singleview` | `observation.state` only, no EE column | needs FK (`metrics/embodied_exec`) or a fitted factor |
| `humanoid/singleview` | `observation.state` only | same |
| others | — | not yet checked |

Planned: fit a single scalar per episode by regressing tracked gripper displacement
(pixels) against paired-GT end-effector displacement (metres), resolved through
`metrics/embodied_exec/episode_index.py` (99.9% of 62,181 prediction mp4s map to their
GT episode). One free parameter, fitted from data, no hand-set constant.

## Vision bridge: measured on real episodes (and what it revealed)

`lift_to_3d.py` turns the cheap tier's masks into a metric 3D trajectory; `metric_scale.py`
supplies the scale from robot kinematics. Run over 6 real `humanoid/singleview` episodes
(`run_episodes.py`), the bridge exposed four bugs worth recording, because each was found
by a measurement rather than by inspection.

**1. Averaging plane normals before orienting them cancels gravity.** RANSAC returns an
arbitrary sign per frame. The GT and prediction of the *same static episode* came out with
opposite verticals (`[0.008,-0.77,-0.64]` vs `[-0.006,+0.81,+0.58]`). Fixed by orienting
each normal against the camera-up hint first; `normal_coherence` now reports 1.000.

**2. A gripper centroid is not a contact point.** Segmentation returns the whole arm:
measured, the gripper mask is ~25x the object's area, its centroid sits 1065 px (8.4 object
diameters) from the object, while the two masks' nearest boundaries are **0.0 px apart** --
they touch. Using the centroid reported "no grasp" on every frame of a successful pick
(`unsupported_frac = 0.97`), so the whole episode read as levitation. Taking the gripper
pixels nearest the object instead moved contact detection from 0.00 to **0.78** and dropped
the GT residual from 1.023 to **0.207**.

**3. Rotating an extent vector is not a valid operation.** `|R @ extent|` collapsed the
object's height to 9 mm. The point cloud has to be rotated *first*, then measured per axis.

**4. The support plane must not be projected by hand.** An earlier version mixed camera and
gravity frames and put the surface 0.42 m from the object. Anchoring `support_z` to the
object's own lowest observed position keeps both quantities in one frame by construction.

### The degeneracy this tier had to be rescued from

With those fixed the pipeline ran end to end -- and the result was still meaningless. Across
6 episodes:

```
residual_excess ≈ −0.999 × Δcontact_frac + 0.008        r = −0.9996
```

Slope −1.000, correlation −0.9996. The score was **a restatement of whether the proxy
detected contact**, wearing the clothes of a force measurement. The cause is structural:
in these clips the object is supported nearly throughout (`|a| = 3.6 m/s²` ≪ g), so the only
question left is contact-or-not. Contact → the friction cone explains everything → residual
≈ 0. No contact → the object must fall → residual ≈ 1.

Fix: report `unexplained_mean_held`, the residual **restricted to frames where a contact was
found**, since only those frames can separate "unphysical" from "unobserved". After the fix:

| quantity | corr with Δcontact_frac | verdict |
|---|---|---|
| `residual_excess` (all frames) | **−0.996** | degenerate |
| `held_excess` (contact frames) | **+0.195** | decoupled |

`run_episodes.py` recomputes this correlation on every run, so the failure cannot recur
unnoticed.

### Metric scale from robot kinematics (robosuite GR1)

The object-size prior is now a fallback only. `metric_scale.py` runs FK on the paired GT
joint trajectory and matches path lengths, giving one fitted scalar per episode:

| episode | FK path length | scale | EE body chosen |
|---|---|---|---|
| 000829 | 0.2543 m | 0.6772 | `right_hand_eef` |
| 001065 | 0.3721 m | 0.1580 | `right_hand_eef` |
| 002133 | 0.6428 m | 0.4012 | `right_hand_eef` |
| 002346 | 0.8494 m | 0.0940 | `left_hand_eefd` |
| 003085 | 1.0524 m | 0.8642 | `right_hand_eef` |
| 003151 | 0.4987 m | 0.4990 | `right_hand_eef` |

The manipulating hand is picked by path length, not named per episode (GR1 ep003151: right
hand 0.4987 m vs left 0.0661 m; ep002346 correctly selects the left). This is what adding
GR1 to `embodied_exec` from robosuite unlocked -- `humanoid/singleview` had no robot model
in mujoco_menagerie, so before that the scale could only be guessed.

### Current verdict on real video — and a retracted headline

The first paired run reported "prediction needs 3.6x the grip force of GT, 5/5 episodes".
**That result was wrong**, and the visualisation is what exposed it: the rendered panels showed
the same mustard bottle as `obj_half = (0.010, 0.020, 0.017)` m in GT and
`(0.021, 0.051, 0.044)` m in the prediction — a 2.4x size disagreement about one physical
object.

Chasing that gave the mechanism. Grip demand is **proportional to the metric scale**, because
acceleration carries a factor of scale:

| scale s (m/unit) | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|
| grip demand | 2.52 | 3.80 | 6.35 | 11.44 |

Each side had been scaled independently by its own gripper path length, giving GT/pred scale
ratios from **0.20x to 2.35x** on the same episode. Correlation between the grip ratio and the
scale ratio: **r = +0.943**. The headline was mostly a unit mismatch.

The fix is the constraint that was being ignored: the *same physical object* has one size, so
GT is lifted first and its largest extent anchors the prediction's scale
(`scale_source = paired_object_extent`). On episode 002346 the prediction's grip demand drops
from **4.584 to 0.492**, against GT's 0.425 — no longer a violation, just agreement. The audit
is now computed on every run (`corr_gripratio_vs_scaleratio`, `scale_ratio_max_dev`) so this
cannot silently return.

What the tier can honestly say, with shared scale, is below. The lesson is more useful than the
retracted number: **a paired metric is only as meaningful as the shared units behind it**, and a
residual table cannot reveal a units bug — a picture of the scene can.

One further correction, for the same reason. The first version of the automatic audit compared
`scale_m_per_unit` between the two sides and reported r = +0.61, implying leftover contamination.
That was itself wrong: MoGe returns depth in a *per-video* arbitrary scale, so metres-per-unit is
a ratio of two different unit spaces and comparing it is meaningless. The audit now checks the
physical invariant — object extent in metres — which matches to **max deviation 0.0e+00**.

### Current verdict on real video (shared scale)

6 humanoid singleview makovian episodes, 5 with a scorable grasp:

| episode | GT | pred | excess | ratio |
|---|---|---|---|---|
| 000829 | n/a | 0.602 | n/a | n/a |
| 001065 | 0.000 | 0.508 | +0.508 | — |
| 002133 | 0.140 | 0.276 | +0.136 | 1.98 |
| 002346 | 0.425 | 0.492 | +0.067 | 1.16 |
| 003085 | 0.667 | 0.833 | +0.165 | 1.25 |
| 003151 | 0.685 | 0.703 | +0.017 | 1.03 |

| quantity (held frames) | GT | pred | excess | pred worse |
|---|---|---|---|---|
| feasibility residual | 0.0007 | 0.0000 | −0.0007 | 67% |
| **grip demand** | 0.384 | 0.562 | **+0.179 ± 0.172** | **100%** (5/5) |

Feasibility still says nothing, exactly as the rank-6 force-closure result predicts: on held
frames the question has no content, and its excess is negative (the prediction looks *more*
physical, which is not a claim worth making).

Grip demand keeps a consistent direction — **5/5 episodes have pred ≥ GT** — but the honest
effect size is **+0.179, about 1.04σ**, not the 3.6x originally claimed. Two of the five
episodes (002346 at +0.067, 003151 at +0.017) are within noise. The degeneracy audit is clean
(correlation with the change in contact fraction **−0.145**, so this is not a contact detector
in disguise), and object size now matches to 0.0e+00, so the remaining difference is not a
units artefact. With n = 5 this is a *direction*, not a demonstrated effect; it needs more
episodes before it can carry weight.

Two caveats visible in the same run. `aliased_frac_mean = 0.994`: orientation from monocular
PCA is almost never certifiable here, so the torque rows are mostly off and grip demand is
driven by translation. And episode 000829 has no scorable grasp in GT at all.

### Why torque balance did not fix held motion — and what does

The plan above said torque balance was the missing constraint. It was implemented, and then
measured, and the measurement said the plan was wrong.

Two opposing frictional face contacts give a wrench matrix of **rank 6**. Over 500 random
6-D wrenches the worst relative NNLS residual was **2.3e-15**. That is force closure: *every*
wrench is explainable, torque rows included. So the vacuity of feasibility on held objects was
never about missing the angular half — the span already covered all of R⁶. No amount of extra
geometry could have fixed it, because the question itself was wrong. Adding torque is still
worth having (it is what makes the *airborne* rotation cases decidable, and it is now
implemented and controlled), but it cannot discriminate a grasp.

The question that stays meaningful is **how hard the fingers must squeeze**:

```
grip_demand(t) = min  sum_{gripper cols} lambda   s.t.  W(t) lambda = wrench_needed(t),  lambda >= 0
```

an LP over the *same* non-negative friction-cone coefficients, divided by `m g`. Only the
gripper's own cone columns are charged; support contacts stay in the equality constraint at zero
cost. That detail is load-bearing, not cosmetic: the first version simply skipped any frame with
table contact, and the real-data run came back `n/a` on 5 of 6 episodes, because "object resting
on a table while a hand arrives" *is* the typical frame. With the weighting, resting costs the
fingers **0.000** while an airborne hold of the same object costs **1.250** — the table doing the
work and the hand doing the work are now different numbers instead of one missing number.

Finite, monotone in the demanded acceleration, and dimensionless. A normal pinch is ~1.25 m·g; a
video model asserting 200 m/s² of lateral acceleration on a held object demands 26.7 m·g,
which no hand applies.

Controls: **11/11 pass** (`run_grip_controls.py`).

| held control | feasibility | grip demand | reading |
|---|---|---|---|
| resting on table, fingers touching | 0.000 | **0.000** | the table carries it |
| held airborne, still | 0.000 | **1.250** | now the hand carries it |
| accel 2 m/s² | 0.000 | 1.505 | plausible |
| accel 20 m/s² | 0.000 | 3.798 | firm |
| accel 200 m/s² | 0.000 | **26.734** | no hand does this |
| spin-up 20 rad/s² | 0.068 | 1.285 | plausible |
| spin-up 80 rad/s² | 0.082 | 1.336 | plausible |
| free fall, no grasp | 0.000 | **None** | nothing to squeeze — not 0 |

Note the feasibility column: **flat ~0.000 while grip demand spans 0 → 26.7**. That is the whole
argument in one column.

Verified invariances: grip demand is exactly mass-invariant (3.798420 at 0.01 / 0.05 /
5.00 kg), and moves the right way with friction (μ 0.2 → 15.19, 0.8 → 3.80, 1.5 → 2.03).

### Orientation: measured, or declared unmeasurable

Torque needs orientation, which the vision bridge now estimates by PCA of the object's own
point cloud, with each frame's axes resolved against the previous frame over the 24 rotations
of a cube (raw per-frame PCA flips sign and axis order, which reads as an enormous angular
acceleration). Validated on synthetic clouds: an asymmetric box recovers ω to **0.0% error**
even with 2 mm of point noise; a **cube is flagged 100% unmeasurable**, which is correct — a
cube has no orientation observable from its point cloud. Only objects with aspect ratio > 1.25
are attempted.

Sampled orientation also has a hard limit, and it is enforced rather than smoothed over.
A finite difference on rotation matrices was measured **62% low at 150 rad/s²**, so ω is taken
from single-step SO(3) logarithms instead — exact (0.0000%) for every frame it certifies.
Rotation beyond π/2 per step is flagged and dropped:

| α (rad/s²) | certified frames | ω error | α error |
|---|---|---|---|
| 2 – 175 | all | **0.0000%** | **0.0000%** |
| 250 – 300 | partial | 0.0000% | 0.0000% |
| ≥ 1000 | **0** (flagged) | n/a | n/a |

An unmeasurably fast spin reports `orientation_aliased_frac = 1.00` and the angular rows are
switched off **for those frames only**. This is per-frame by necessity, not by taste: the first
version dropped aliased frames entirely, and on real video that discarded everything — 11 of 12
gt/pred passes came back with `aliased_frac >= 0.97` and `n_frames_scored = 0`. PCA axes on
monocular point clouds jitter far more than a rigid object does, and force balance never needed
orientation in the first place. `torque_frac_scored` reports how much of the clip actually
carried usable rotation, so a small number is visible rather than silent.

Before the aliasing guard existed at all, an absurd spin scored *lower* than a merely
implausible one.

One more bug the controls caught: the wrench system mixes newtons and newton-metres, and with
L = 3 cm the torque block is ~30x smaller in magnitude, so NNLS effectively ignored it (an
80 rad/s² spin scored 0.190 while 40 rad/s² scored 0.260 — non-monotone). Rows are now
non-dimensionalised (force by `m g`, torque by `m g L`) **before** solving.

### What is still genuinely ambiguous

A mid-clip reversal of angular velocity is *not* resolvable: rotating by −0.4 rad and by
+5.88 rad produce identical frames. The tier flags it instead of picking one. This is asserted
by a control, so it cannot silently become a number later.



| step | state |
|---|---|
| proxy scene fitted from primitives, no per-task assets | done |
| friction-cone feasibility core, zero tuned constants | done |
| controls with closed-form expectations | done — 11/11, floor 0.00000 |
| mass invariance | done — exact |
| scale-sensitivity characterisation | done — severe violations invariant |
| metric scale from paired GT | done — GR1 FK via robosuite, 6/6 episodes |
| vision → 3D trajectory (reuse `sim2real` masks/tracks) | done — 4 bugs found by measurement |
| degeneracy audit (score vs contact detection) | done — decoupled, re-checked every run |
| torque balance (Newton–Euler, 6-D wrench) | done — and it did **not** fix held motion (rank 6) |
| **grip demand** for held objects | done — 11/11 controls, mass-exact, 0 → 26.7 range |
| orientation from vision + aliasing guard | done — 0.0% on asymmetric objects, cubes flagged |
| real gripper geometry instead of proxy boxes | pending (15 MJCFs fetched) |
| batch driver across methods / embodiments | partial (`run_episodes.py`, one split) |
| **real-data result** | done — grip demand direction 5/5, effect +0.179 (~1σ, n=5) |
| **visualisation of the fitted scene** | done — `render_sim.py`, EGL headless; caught the scale bug |

## Visualising the scene that is being measured

`render_sim.py` renders the same MuJoCo model `newton.py` queries, driven by the same lifted
trajectory, with MuJoCo's contact points and contact forces drawn, beside the source video
frame and the per-frame score:

```bash
MUJOCO_GL=egl python -m metrics.force_feasibility.render_sim \
    --tracks metrics/sim2real_out/.../episode_002346/tracks.npz \
    --out /tmp/sim_vis/ep002346 --which both
```

Rendered output for two episodes is committed under
[`metrics/force_feasibility_out/vis/`](../force_feasibility_out/vis/) (see its README), including
the cached lifts — so those panels re-render in ~18 s with **no GPU and no MoGe pass**.

This exists because it is the only check that can catch the class of bug that a residual
table cannot — and it immediately did, finding the 2.4x object-size disagreement that
invalidated the first headline result. Reading a panel, in order of how badly each failure
corrupts the measurement:

1. **the object sits on the plane**, neither floating nor sunk (a wrong `support_z` turns
   every resting frame into a violation),
2. **fingers touch when the video shows contact** (the score is conditioned on contact),
3. **object size is plausible in metres, and matches between GT and pred** (it sets the
   moment arm and, through the metric scale, the magnitude of grip demand itself).

Notes: EGL and OSMesa both work headless (GLFW does not); glyph sizes are scaled to the
object because MuJoCo's defaults are sized for human-scale robots and drown a 2 cm bottle;
lifting is cached to npz, so the ~2 min MoGe pass is paid once and re-renders take ~10 s.
All appearance changes to the XML (lights, textures, rgba) were verified not to touch the
physics: 22/22 controls still pass and grip demand is unchanged to every printed digit
(3.798420).

## Honest limitations

- **Grip demand is proportional to the metric scale**, so it is only comparable between two
  videos that share one. Measured: 2.52 / 3.80 / 6.35 / 11.44 for s = 0.5 / 1 / 2 / 4 on an
  identical trajectory. This is enforced by anchoring both sides to the same object extent,
  and audited every run (`obj_size_ratio_max_dev`). It also means the *absolute* value carries
  the scale's error; only the paired comparison is trustworthy.
- **n = 5 is not a result.** The 5/5 direction on real episodes is suggestive at ~1σ, with two
  of five episodes inside noise. It needs more episodes and more embodiments.
- **Acceleration is a second derivative**, so tracker noise is amplified. Boundary
  frames are excluded rather than estimated (a one-sided stencil put a spurious
  0.5 m·g on an honest free fall). `smooth` is exposed rather than hard-coded because
  the right amount depends on the tracker; paired-GT differencing is what cancels the
  bias.
- **Feasibility is vacuous for held objects, by construction, and no fix exists within
  feasibility.** Two opposing frictional contacts have rank 6 (worst residual 2.3e-15 over 500
  random wrenches), so grasps are always feasible in both force and torque. Use
  `grip_demand_*` on held frames and `unexplained_*` on airborne/supported frames. The two
  answer different questions and neither substitutes for the other.
- **Orientation is only available for asymmetric objects.** Near-cubic and near-spherical
  objects have no PCA-observable orientation and are flagged, not guessed, so the torque rows
  and any rotation-driven grip demand are simply unavailable there.
- **Rotation faster than π/2 per frame is unrecoverable.** Flagged via
  `orientation_aliased_frac` and excluded. A mid-clip angular-velocity reversal is genuinely
  ambiguous with a near-2π wrap and is likewise flagged rather than scored.
- **Object mass and inertia are proxies.** Grip demand is exactly mass-invariant (verified),
  but the inertia is that of a uniform box fitted to the point cloud, so torque-driven grip
  demand inherits that shape assumption.
- **The proxy is a convex primitive on a plane.** Concave objects, multi-object stacks
  and articulated targets are approximated; the friction cone under-approximation keeps
  the direction of error conservative, but a genuinely feasible motion involving
  geometry the proxy cannot represent will be charged.
- **Contact detection uses proxy geometry**, so a contact the real scene had but the
  proxy misses becomes a false violation. This is the main risk when the vision stage
  lands, and the reason the paired-GT difference — not the raw residual — should be the
  reported metric.
