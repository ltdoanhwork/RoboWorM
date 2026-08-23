# real2sim — reconstructing the scene that is measured

Status: **blocked on segmentation quality, upstream of anything here.** What follows is what
was built, what was measured, and the diagnosis that stopped it. Nothing in this module should
be used for a reported number yet.

## Why this module exists

`force_feasibility` applies Newton to a *proxy* scene: a box for the object, two boxes for the
fingers. That needs no per-object asset, which is why it was chosen. The objection to it is
fair — real contact geometry and real inertia need the real shape — and single-image-to-3D
models now make the shape obtainable automatically, so the "we would need per-object assets"
argument no longer holds.

## What works: object mesh from one masked frame

`external/TripoSR` (vendored, two local patches) turns the object's own masked crop into a
watertight mesh. Measured on `humanoid_singleview/makovian/episode_002346` (mustard bottle):

| quantity | value |
|---|---|
| reconstruction time | **2.0 s** on one A100 |
| mesh | 48 128 verts / 96 244 faces, **watertight** |
| bbox (TripoSR units) | 0.470 x 0.504 x 0.978 (aspect 0.48 : 0.52 : 1.0) |
| frame chosen | 101, picked automatically as max area x (1 - gripper occlusion); occlusion **0.000** |

Watertight matters: it is the precondition for MuJoCo to use the mesh as a collision geom and
for `trimesh` to give a defensible inertia tensor, so this output is directly usable by the
force tier rather than being a picture.

Two patches were needed, both recorded in the vendored source:

- `tsr/models/isosurface.py` — `torchmcubes` is a CUDA extension needing a local build;
  `skimage.measure.marching_cubes` is already installed and is the same algorithm. The two
  differ in axis convention, so the `[2, 1, 0]` permute is applied only for the original
  backend. Getting that wrong silently mirrors every mesh.
- `tsr/utils.py` — `rembg` is imported at module level but only used by `remove_background`,
  which this pipeline never calls: the masks the metric already computes are a better matte
  here, because they are the same masks the physics is measured from.

## What does not work: metric scale, and now we know why

Reconstruction gives shape in a normalised unit cube. It cannot give **metres** — and metres
is what the force tier is most sensitive to, since grip demand is proportional to scale. Three
rulers were tried, each replacing the last after it was measured and found wanting:

| ruler | how it failed |
|---|---|
| gripper **path length** | biased by tracker jitter: +20 mm of noise inflates it **7.05x** (radius of gyration: 1.009x) |
| gripper **radius of gyration** | jitter-invariant, but matched the mask centroid to FK's *end effector*, while the mask is the whole arm — 2.65x inflation, mustard bottle came out 37.5 cm |
| **static arm extent** (`probe_static_scale.py`) | plausible sizes at last (2.2-13.7 cm, all graspable) but **not reproducible** — see below |

The static ruler measures the arm's 3D size per frame, so it needs no differencing (immune to
the jitter that killed path length) and no assumption about which material point the mask
centroid is. Its unknown is where the mask stops proximally, so all contiguous distal subsets
of the FK chain are computed and the answer is reported as their median with their spread as
the uncertainty. That part behaved: **candidate subsets agree to within 8-26%**, so the unknown
cut-off is not a large error term.

### The test it failed, and the diagnosis

Two episodes in this set manipulate **the same object** — `episode_000829` and
`episode_003151` are both "green bok choy". Same object, same robot, independent episodes, so
they must return the same size. They returned **9.1 cm and 2.2 cm: a 4.1x disagreement.**

Decomposing the extent into image-plane versus depth axis localises part of it:

| episode | arm extent XY | arm extent **Z** | Z/XY | obj/arm ratio (XY) |
|---|---|---|---|---|
| 000829 | 0.857 | 0.461 | 0.54 | 0.241 |
| 003151 | 1.757 | **2.051** | **1.17** | 0.056 |

The `obj/arm` ratio is **dimensionless** — scale cancels out of it — yet it differs by 4.3x.
So the error is not only in the ruler's calibration; something about the measured quantity is
wrong. Rendering the masks says what
(`sim2real_out/r2s/mask_compare.png`, `obj_mask_compare.png`):

- the **"gripper" mask covers both arms plus background**. Its extent is therefore partly the
  *distance between the two arms*, an episode-specific pose rather than a fixed length. In
  `003151` it reaches 10.5% of the image and spans 2.05 units in depth — wider in depth than
  in the image plane — because it unions two arms at two different depths.
- the **object mask is on the wrong object in both episodes.** Both instructions say "green bok
  choy". `000829` masks the **yellow corn**; `003151` masks a **small red item** at the table
  edge. The bok choy is plainly visible in both and is segmented in neither. The referent
  selection that exists for exactly this problem (`select_referent` / `pick_referent`) was
  **not run** — `referent_pick` is empty in the stored metadata.

### Isolating how much of the error is the ruler versus the masks

Progressive fixes, each measured on the same pair, with the pass criterion being agreement to
better than 1.5x:

| what is measured | disagreement |
|---|---|
| whole arm mask, stored object mask | **4.87x** FAIL |
| nearest connected component of arm mask, stored object mask | **2.34x** FAIL |
| nearest component + **the same physical object** in both (yellow corn, HSV-segmented) | **1.52x** borderline |

That last row is the informative one: measuring one real object rather than two different ones
returns **8.4 cm and 5.6 cm**, both plausible for a corn cob. So the static ruler is
**approximately sound**, and the bulk of the original 4.87x was the masks — the wrong object in
one direction and a two-armed blob in the other. The residual 1.5x is MoGe's per-episode
geometry plus the component split, which is the error term left to reduce.

## What the diagnosis actually found (three of my hypotheses were wrong)

The chain from instruction to measured object was traced end to end. Three successive
conclusions were drawn and each was overturned by the next measurement, so they are recorded
here in order — the retractions are the useful part.

**Hypothesis 1 — "the object mask is on the wrong object."** Read off the rendered overlays by
eye. **Wrong.** Measuring the *colour* of the masked pixels settles it: the instruction says
"green bok choy", and the masked pixels are **73%** green (median hue 47) in `000829` and
**65-71%** green (hue 52) in `003151`, both inside the bok-choy hue band 35-85. As a control,
`002346` ("yellow mustard bottle") reads hue 21-25 with 0% green — also correct. The masks are
on the right objects; the red overlay had made the greens unreadable to the eye.

**Hypothesis 2 — "GroundingDINO never proposes the object."** Drawn from the `no_contact` veto
firing on all 4 candidates. **Wrong.** Sweeping thresholds and prompts and then matching box
coordinates against HSV-detected green blobs shows the detector puts a box on the bok choy at
conf **0.43** (`000829`, box `[181,300,293,383]` vs green blob `[186,314,269,370]`) and **0.30**
(`003151`, box `[117,291,162,334]` vs blob `[122,294,157,328]`) — and in both episodes **that
box is the one the pipeline selected.** Detection and selection were both right.

**Hypothesis 3 — "then the contact measurement must be broken."** Also wrong, and this is where
the real defect is. Recomputing the gap from the *stored* masks gives contact fractions of
**0.47** (`000829`) and **1.00** (`003151`) — not zero. Splitting the arm mask into connected
components explains the discrepancy:

| `episode_003151` arm component | motion (px/frame) | gap (object diameters) | contact frac |
|---|---|---|---|
| component nearest the object | **1.7** | 0.00 | **1.00** |
| component with the most motion | **22.5** | 10.35 | 0.00 |

**The arm that is holding the object is nearly still (1.7 px/frame); the arm that moves a lot
(22.5) never touches it.** So `pick_acting_arms`' criterion — keep the arm with the most motion
energy — selects the wrong arm precisely when the manipulating hand has already grasped and is
holding steady. My threshold fix (0.4 -> 0.65) made this *worse*: by dropping the second arm
more aggressively it discarded the contacting one, which is why the re-run reported
`contact = 0.00` for every candidate while the stored masks (which unioned both arms) reported
1.00.

This is a genuine conflict between episodes, not a threshold to tune: the docstring's original
tuning case (`000829`: acting 0.212 vs idle 0.101) wants motion to select, and `003151` wants
the opposite. Motion energy is the wrong criterion. **Contact with the candidate object is the
criterion that is correct in both** — but it cannot be used to pick the arm and then also be
the evidence that the pick was right, so the two need to be resolved jointly (e.g. choose the
(arm, object) pair that maximises contact, rather than choosing the arm first).


## The final answer: neither episode supports the measurement, for different reasons

Joint resolution was implemented (`resolve_arm_object_pair`) and re-run. It works mechanically —
every object candidate is now paired against every arm candidate, and the log carries `arm=...` —
but it does not rescue these episodes, and pushing further would be tuning against noise. Two
independent checks settle each one.

**Timing check.** GT joint angles say when a grasp happens: the manipulating hand's finger joint
sweeps ~1.4 rad, starting at **67%** of `000829` and **59%** of `003151`, and both grasp windows
fall **entirely inside** the video segment (37/37 and 38/38 frames). So a grasp really is present
in the data. Comparing the vision gap before versus during that window:

| episode | gap before grasp | gap during/after | verdict |
|---|---|---|---|
| `000829` | median 0.430 | median **0.362** (min 0.357) | never crosses the 0.35 contact threshold, and the episode's minimum gap (0.212) occurs at **frame 0**, before any grasp. The hand never approaches *this* object. |
| `003151` | median **0.000** | median **0.000** | zero from **frame 0 onward**, i.e. before the grasp as well. The object and arm masks simply overlap throughout; `contact = 1.00` is adjacency, not evidence of grasping. |

**Scale-invariance check.** The gap is normalised by object diameter, so it should not depend on
resolution — confirmed: 0.416 (stored 2304x1296) versus 0.425 (native 768x432) on `000829`. The
`contact = 0.00` from the re-run and the `contact = 0.47` computed from stored masks differ only
because the latter included the *unioned* second arm; the measurement itself is consistent.

So `000829` is an **instruction/video mismatch** — the robot grasps something, but not the bok
choy that both the detector and the pipeline correctly identified — and `003151` gives a
**degenerate contact signal**, since masks that touch from the first frame say nothing about when
a grasp begins. Neither has a segmentation bug left to fix, and neither can serve as the
same-object reproducibility test the metric scale needs. That test remains open, not passed.

### What changed in the code, and what was reverted

| change | status |
|---|---|
| `pick_acting_arms` threshold 0.4 -> 0.65 | **reverted to 0.4.** It made things worse: dropping the second arm more confidently discarded the *contacting* arm on `003151`. The function is now documented as a fallback only. |
| `no_contact` veto in the referent ranking | **kept.** It is what surfaced this whole diagnosis instead of silently reporting a number. |
| `resolve_arm_object_pair` (joint arm+object selection) | **added.** Removes the circular tie-break; costs `n_arm + n_obj` propagations, not their product. |


## The screening gate: which episodes can support a measurement at all

Two episodes were spent discovering, one at a time, that they could not support the measurement.
Both failures are detectable *before* any 3D lifting or scale fitting, so they are now a gate
(`screen_episodes.py`) rather than a per-episode discovery. The two gates are independent by
construction — A reads GT joint angles and never looks at the video, B reads only vision and never
looks at the robot — so agreement is evidence and disagreement is a diagnosis.

- **Gate A (kinematic)**: the manipulating hand's finger joint must sweep a real range (> 0.5 rad)
  and the resulting grasp window must lie inside the video segment.
- **Gate B (vision)**: the object-to-arm gap must *close* during that window **and** be non-zero
  at frame 0. A gap already at zero on the first frame is mask adjacency, which cannot distinguish
  holding from standing next to.

Neither threshold is invented here: A's is the midpoint of the joint's own observed range, B's is
the `tau_gap` the metric already uses.

| episode | hand | joint range | grasp@ | gap f0 | gap before | gap during | contact during | A | B | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| `000829` | right | 1.41 | 68% | 0.212 | 0.430 | 0.362 | 0.00 | Y | n | reject — gap never closes |
| `001065` | right | **0.33** | 98% | 0.354 | 0.000 | 0.000 | 1.00 | **n** | Y | reject — no real grasp in GT |
| **`002133`** | right | 0.97 | 17% | 2.021 | 0.000 | 0.000 | 1.00 | Y | Y | **USABLE** |
| **`002346`** | left | 1.37 | 20% | 0.773 | 0.258 | **0.000** | 0.72 | Y | Y | **USABLE** |
| `003085` | right | 1.43 | 19% | 0.157 | 0.196 | 0.408 | 0.10 | Y | n | reject — gap *widens* |
| `003151` | right | 1.43 | 60% | **0.000** | 0.000 | 0.000 | 1.00 | Y | n | reject — degenerate from frame 0 |

**2 of 6 are usable.** `002346` shows the signature of a genuine grasp: gap 0.773 before, 0.000
during, exactly at the kinematic window.

### The gate was cross-validated, not just written

It agrees with three results derived independently and earlier, by different means:

| independent finding | gate verdict |
|---|---|
| `002346` is the episode whose TripoSR mesh succeeded (frame 101, occlusion 0.000) | **USABLE** — match |
| `000829`'s hand never approaches the detected object (gap analysis) | **reject** — match |
| `003151`'s masks overlap from frame 0 (colour + gap analysis) | **reject** — match |

It also rejects `001065` for a reason nobody fed it: joint range 0.33 rad, i.e. the hand never
really closes. That is the "slice of toast" episode the `pick_referent` docstring already records
as the known residual failure (a dragged crumb outscoring the target, false-clean 0.975). The gate
catches it from GT kinematics alone.



Every scale estimator tried computed a different statistic of the same unreliable input, which is
why each failed differently. Isolating the error showed the masks dominate: measuring one real
object with the arm split into components brought a 4.87x disagreement down to 1.52x and returned
plausible centimetres (8.4 cm / 5.6 cm for a corn cob).

But the two episodes chosen as the reproducibility test turn out to be unusable as a test — one is
an instruction/video mismatch, the other has degenerate (frame-0) mask overlap. So the honest
status is: the ruler is *plausible but unvalidated*, and validating it needs episodes that
actually pass a grasp-signal sanity check first.

Ordered by what the measurements support, the next steps are:

1. **Screening is implemented and cross-validated** (`screen_episodes.py`, 2/6 usable, agrees with
   3 independent prior findings and catches a 4th failure unprompted). Apply it before any tier
   runs, so no measurement is reported from an episode that cannot support one.
2. **The same-object test still needs a screened pair.** Verified against the instructions: the two
   usable episodes manipulate different objects (`002133` a light red fruit, `002346` the yellow
   mustard bottle), so the reproducibility test cannot run on this 6-episode set. Two alternative
   validations *are* available and do not need more episodes:
   - **A known real dimension — attempted, and it does not hold up.** `002346`'s instruction says
     "yellow mustard bottle", which suggested the YCB item 006_mustard_bottle (19.5 cm) as an
     absolute reference. Running the ruler gives **6.2 cm** (2D pixel-ratio cross-check: 9.0 cm),
     a 2-3x shortfall in the *same* direction from both methods, which rules out the 3D lifting
     step as the cause. But **looking at the mask invalidates the reference, not the ruler**: the
     tracked object is a small orange squeeze bottle being lifted to an upper shelf, not the YCB
     mustard bottle (the white "MAYO" jar on the table is a different object again). The 19.5 cm
     was imported from the object's *name*, not measured from this scene, so the test says nothing.
     **Lesson: an absolute reference must be identified from the pixels, not from the
     instruction's noun.**
   - **A shared static referent — the remaining option.** `002133` and `003085` both place into the
     same wire basket, so the basket's measured size is comparable across the two scenes and tests
     the ruler independently of any grasp (and independently of naming, since it is the same
     physical object in both videos). This is the test to run next.
3. **Only then** attach the reconstructed mesh to the force tier. The mesh part is ready and
   measured (2.0 s, watertight); it is the metres that gate it.

## Is real2sim blocked? No — the calibration was there the whole time

Everything above treats metric scale as something to *estimate* from an indirect ruler (path
length, radius of gyration, static extent), and every such ruler failed. The `probe_arm_split`
entry below even records the direct approach as impossible: prompting SAM with FK-projected hand
points "needs the ego camera's intrinsics and its pose in the robot frame, and this dataset records
neither".

**That was wrong, and it was the most expensive mistake in this file.** Both pieces exist, just not
in the dataset metadata:

| piece | where it actually is |
|---|---|
| intrinsics | the GR1 MJCF defines camera **`robotview`** on body `head_pitch`, `fovy=50`, `pos=[0.13, 0, 0.09]` |
| extrinsics per frame | `modality.json` puts **`neck` at state dims 19:22** — the three head joints are recorded, so the camera pose is driven by GT rather than assumed |

`probe_reprojection.py` tests the chain with no hand annotation: forward-kinematic the body, project
`left_eef`/`right_eef` through `robotview`, and compare against the *hand mask centroid*, which came
from an entirely separate detection+tracking pipeline. Two independent pipelines agreeing on where
the hand is in the image is the evidence; predicting the image centre is the null model.

| stage | median error | as % of the 2643 px diagonal |
|---|---|---|
| null model (predict image centre) | 736 px | 27.8% |
| raw MJCF camera, no correction | 498 px | 18.8% |
| **per-episode offset removed** | **139 px** | **5.3%** |
| **ONE shared 4-param correction, all 6 episodes** | **172 px** | **6.5%** |

Two things matter here. First, after removing a mere translation the residual is **5.3%** with
FK-vs-mask correlation **r_x = +0.87** (0.98 on four of six episodes) — the projection genuinely
tracks the hand's motion, so the kinematic chain and camera mounting are essentially correct.
Second, that offset is **not** a per-episode fudge: one correction fitted jointly over all 241
observations from all 6 episodes (`dpos = (+0.065, +0.041, +0.178) m`, `fovy 50 -> 59.3 deg`)
reaches 172 px, close to the per-episode-offset bound. Nor is it explained by pose — neck and waist
angles are ≈ 0 (≤ 0.044 rad) in every episode, far too small to move a projection by hundreds of
pixels.

The remaining ~170 px has a visible cause: `r2s/reproj.png` shows the FK markers landing on the
correct hand and following it, over an image with obvious **fisheye distortion** (the table edge
bows). A pinhole model cannot absorb that, which both explains the residual and says what to fit
next — one shared distortion coefficient, still no per-episode parameters.

**So "real2sim is nearly impossible here" is not what the data says.** What is impossible is
recovering scale from indirect motion statistics; that failed six different ways for the reasons
recorded above. With a calibrated camera, scale is not estimated at all — FK is already in metres,
so a pixel's depth falls out of the projection. The next step is to finish the calibration (add the
shared distortion term, with this same reprojection test as the accept criterion), not to invent a
seventh ruler.

## Files

- `probe_static_scale.py` — static arm-extent ruler, with the shift control showing the time
  alignment is only ~1.6 sigma above chance (hence a median ratio, which needs no alignment,
  rather than a regression over time).
- `probe_arm_split.py` — compares arm-separation strategies on the same-object test. Its strategy 3
  (SAM prompted with FK-projected hand points) is left without a number, on the stated grounds that
  the camera calibration is unavailable — **that reasoning is now refuted** by
  `probe_reprojection.py`; the calibration exists and strategy 3 is implementable.
- `probe_reprojection.py` — projects FK hand positions through the MJCF's `robotview` camera and
  compares against the independently-tracked hand mask. Shows one shared 4-parameter correction
  brings all 6 episodes to 6.5% of the image diagonal, i.e. the calibration chain is real.
- `probe_detector.py` — threshold/prompt sweep proving the detector *does* propose the object
  (conf 0.43 / 0.30) and rendering every box so "found but ranked low" is distinguishable from
  "never found".
- `probe_candidate_pool.py` — reproduces the pipeline's detection path to show which candidates
  survive `nms` / `_drop_huge` / `max_object_boxes` / `prefilter`.

## Honest limitations

- The shift control is reported, not hidden: correlations of +0.7 against a null of
  +0.38 +- 0.19 do **not** establish frame alignment between video and parquet rows.
- The FK-row-to-video-frame map is proportional, an assumption, because the video is
  temporally resampled relative to the parquet.
- Prediction videos have no joint angles, so FK-anchored scale exists only for GT. Whatever
  scale is settled on must be transferred to the prediction through the shared object, as
  `force_feasibility.run_episodes` already does.
