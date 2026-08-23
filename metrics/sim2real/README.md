# `metrics/sim2real` — object-interaction physics without a simulator

Does the hand-object interaction in a generated video obey physics? RoboWM-Bench
([2604.19092](../../uploads/2604.19092-0.md)) answers this by converting the video
back to actions and **executing them in a physically grounded simulator**, then
running per-step checkers on the sim state. That is the gold standard and also the
expensive part: it needs a digital twin of every scene, a video-to-action policy,
and a sim rollout per episode — for three embodiments and 257 Ctrl-World episodes
alone.

This module keeps the *checkers* and drops the *simulator*. The predicates
(contact, levitation, grasp-follow, penetration, object integrity) are evaluated
directly on tracked 2D masks, in units of object diameters, with the **paired GT
video processed through the identical pipeline** so that tracker bias cancels.

The whole thing needs one GPU pass of GroundingDINO + SAM2 per episode (~35 s) and
no new dependency beyond what `metrics/dynamics` already uses.

## Files

| file | role | models used |
|---|---|---|
| `track_backend.py` | strip → per-view masks for object / target / gripper; mask cache | GroundingDINO + SAM2 (+ background subtraction) |
| `lift_backend.py` | 2D (default) / `dav2` / `moge` depth backends, geometry helpers | optional Depth-Anything-V2 or MoGe |
| `interaction_metrics.py` | the predicates, agreement terms, step-level nodes, score | numpy / OpenCV only |
| `interaction_viz.py` | panel (GT vs pred strip + gap curve + contact timeline) and violation clip | OpenCV only |
| `interaction_probe.py` | one episode end-to-end (CLI) | all of the above |
| `compute_sim2real_batch.py` | whole method tree + summary (CLI) | all of the above |
| `negative_controls.py` | synthetic clips with known violations | numpy / OpenCV only |
| `run_negative_controls.py` | validation table, verdict per control (CLI) | none (reads cached masks) |
| `mine_violations.py` | rank real episodes per violation term, re-render evidence (CLI) | none (reads cached masks) |
| `compare_methods.py` | per-term deltas vs dense on the shared valid episode set (CLI) | none (reads metric JSONs) |
| `render_comparison.py` | worst vs clean episode in one video, live HUD (CLI) | none (reads cached masks) |

Shell wrappers: `scripts/run_sim2real_interaction.sh` (one episode),
`scripts/run_sim2real_interaction_batch.sh` (whole tree). Both `cd` into
`metrics/WorldScore` so the vendored SAM2 hydra configs resolve, exactly like
`scripts/run_motion_accuracy_gdino.sh`.

## Pipeline

```
gt_all_views.mp4 / pred_all_views.mp4      # Ctrl-World 3-view strip, 320x192/view
  └─ split into views, upscale 3x          # SAM2 outputs at input res; gaps are
                                           #   measured in pixels, so upscaling
                                           #   sharpens the boundary
  ├─ GroundingDINO on GT frame 0           # object + target boxes, from the
  │    └─ SAM2 video propagation           #   instruction's noun phrases; the SAME
  │                                        #   boxes seed GT and pred (frame 0 is
  │                                        #   the shared conditioning frame)
  └─ background subtraction (per video)    # gripper = whatever moves vs the
                                           #   arm-free first frames
  → track_series(): gap(t), centroid(t), area(t), aspect(t), overlap(t)
  → contact(t) per view, AND-gated across the 2 exterior cameras
  → violation terms + agreement-vs-GT terms + step nodes → score
```

### Why the gripper is not detected with GroundingDINO

The first attempt prompted GroundingDINO for the arm on frame 0 and got background
clutter, because **in these episodes the arm is not in frame 0 yet** — the videos
start before the robot enters. Background subtraction against the arm-free opening
frames is both more reliable and free, and it works on blurry predictions where a
detector gives up. It only works for *static* cameras, hence the wrist view is
excluded from the contact decision.

### Why contact is ternary, not boolean

`gap` is nan when a mask is missing, but the two causes are opposites:

| situation | meaning | state |
|---|---|---|
| object visible, arm visible, all views agree | measure the gap | 1 or 0 by `tau_gap` |
| **object not visible** (usually the arm occludes it) | no evidence | **-1 unknown** |
| **object visible, arm absent** | evidence nothing touches it | **0 no contact** |
| **views disagree** (one sees a touch, another does not) | conflicting evidence | **-1 unknown** |

All of these were bugs found by the controls and by looking at the rendered panels:

- Treating "object occluded" as *no contact* made **real GT video score a 48%
  levitation rate** — the object was being carried while one camera could not see it.
- Then treating "arm absent" as *unknown* made the `teleport_object` control score a
  **perfect** levitation term, because every frame was skipped for lack of evidence.
- Treating a **split vote** as *no contact* made `episode_008843` report levitation on
  frames where the primary view measured `gap = 0.00` diameters — a contradiction
  visible right on the panel (`LEVITATION` printed next to `gap pred=0.00`). One view
  seeing a touch is real evidence, so a split can block a contact claim but must never
  license a violation claim.

Levitation is only claimed on frames where the absence of contact is *known*.

## The terms

All lengths are divided by the clip's median object diameter (`sqrt(area)`), so
numbers are comparable across episodes, resolutions and object sizes.

### Violations (is this possible at all?)

| term | measures | fires when |
|---|---|---|
| `levitation` | object translates while nothing touches it, excluding downward (gravity) motion | the Fig. 3 failure: fingers graze the object yet it gets lifted |
| `grasp_follow` | median `abs(v_obj - v_grip) / abs(v_grip)` during contact | the object does not move with the gripper holding it |
| `object_presence` | fraction of frames the object is trackable | the object vanishes |
| `object_shape` | area / aspect coefficient of variation | the object melts or dissolves |
| `penetration` | 95th pct of gripper-over-object overlap | the gripper swallows the object |
| `chatter` | contact on/off transitions | fingers flicker in and out of the object |
Each is scored on its **GT-relative excess**, not its raw value. The tracker leaves
a residual violation signal even on real video (occlusion, mask jitter); differencing
against GT is what makes the oracle control reach exactly 1.000. Raw values are
still reported for diagnosis.

### Agreement (is it the physics of *this* episode?)

`contact_onset_err`, `contact_temporal_iou`, `contact_f1`, `obj_traj_err`,
`obj_travel_pred/gt`, `gap_curve_err`.

### Step-level nodes

RoboWM-Bench hand-specifies key nodes per task and checks them in sim. We read them
off the GT trajectory instead — `contact` → `lift` → `transport` → `release` — which
keeps the protocol automatic across all episodes and grounds every threshold in what
actually happened. Reported as `step_pass_rate` and `task_success`.

### The score

`sim2real_interaction_score` is a weighted mean of the normalised terms, and the
**per-term breakdown is always reported alongside it**. `metrics/dynamics/README.md`
records how WorldScore's single max-minus-max number misled us; a lone scalar here
would repeat that mistake.

## Validation: the negative controls

Cheap physics proxies are easy to fool yourself with. Each control edits the *cached
masks* of a real episode to inject a violation that is known by construction, then
asserts the specific sub-term collapses:

```bash
REPO=$PWD models/dreamgen/.venv/bin/python -m metrics.sim2real.run_negative_controls \
  --tracks /tmp/s2r_batch/dense/makovian/episode_000255/tracks.npz \
  --out /tmp/s2r_negctrl --viz
```

Measured on `episode_000255` (norm: 1.0 = clean, 0.0 = maximally violating):

| control | score | levit | grasp | presence | shape | chatter | c_timing | c_overlap | traj | gap | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| real (dense pred) | 0.715 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.465 | 0.400 | 0.432 | — |
| **oracle** (pred := GT) | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |
| freeze_object | 0.751 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.605 | **0.426** | 0.718 | PASS |
| teleport_object | 0.531 | **0.000** | — | 1.000 | 1.000 | 1.000 | — | 0.000 | 0.185 | — | PASS |
| no_gripper | 0.667 | 0.000 | — | 1.000 | 1.000 | 1.000 | — | **0.000** | 1.000 | — | PASS |
| shuffle | 0.585 | 1.000 | 0.670 | 1.000 | 1.000 | 0.000 | 0.000 | 0.365 | **0.433** | **0.000** | PASS |
| dissolve | 0.713 | 1.000 | 0.960 | 1.000 | **0.017** | 0.833 | 1.000 | 0.953 | 0.000 | 0.480 | PASS |

The table shows **sensitivity** (each control collapses the term it targets, in bold)
and **specificity** (`dissolve` drops `object_shape` to 0.017 while leaving presence,
chatter and timing near 1.0). The oracle row is the important one: pred := GT scores
exactly 1.000, so the pipeline adds no bias of its own.

`swap_gripper` needs a second episode's masks: pass `--donor_tracks <other>/tracks.npz`.

## Side-by-side evidence: violating vs clean

`render_comparison.py` plays the worst- and best-scoring episode in ONE video,
through the same tracker and the same overlays, with a live HUD under each row (gap
curve + playhead, contact state, flag). A violation number alone gives no sense of
what "clean" looks like on the same tracker, so the pair is the actual evidence.

```bash
REPO=$PWD models/dreamgen/.venv/bin/python -m metrics.sim2real.render_comparison \
  --work_root /tmp/s2r_batch/dense --term score --out /tmp/s2r_viz
```

Verified pair (dense, 122 episodes scored):

| | episode | score | what the video shows |
|---|---|---|---|
| WORST | `makovian/episode_008843` "Put the black and white mugs inside the drawer" | **0.549** | at t=137 GT still holds the green mug in the gripper; the prediction has **lost the mug entirely** (`gap pred = --`, object untrackable), and `object_presence` / `object_shape` / `trajectory` are all 0.0 |
| CLEAN | `makovian/episode_003056` "Open the top drawer" | **0.997** | object and gripper track together throughout, `gap pred = GT = 0.00`, contact states identical |

It also writes a still of the most incriminating frame, selected by *which term
actually failed* (levitation frame → middle of the longest object-lost run → max gap
error), plus the per-episode panels. Pass `--worst_ep`/`--clean_ep` to pin specific
episodes; those work straight off `tracks.npz`, before the batch writes metrics.

## Mining real violations

The controls prove each term *can* fire; mining proves they fire on real generated
video and that a human agrees with what was flagged. `mine_violations.py` ranks every
measurable episode by each term and re-renders the panel + clip for the extremes
(cheap: masks come from `tracks.npz`, no GPU).

Verified example — `makovian/episode_002129`, "Put the mug on the table", ranked
worst on `grasp_follow`:

| quantity | pred | GT |
|---|---|---|
| `grasp_follow_ratio` | 0.704 | 0.073 |
| `grasp_follow_cos` | 0.896 | 0.999 |
| `object_present_frac` | 0.917 | 1.000 |

The panel confirms it by eye: at the last frame GT still shows the mug held in the
gripper, while the prediction has **lost the mug entirely** — only the arm remains,
and the frame is marked `?` (occluded/unknown) rather than counted as contact. So
`grasp_follow_excess = 0.63` and `object_presence` = 0.725 are describing exactly
what the video does, which is the property that makes the term trustworthy.

## Known traps

1. **`full_pred.mp4` is only the primary view.** Ctrl-World's 3-view strip is
   `pred_all_views.mp4` / `gt_all_views.mp4`. Using `full_pred.mp4` silently throws
   away the second exterior camera that the contact AND-gate depends on.
2. **Do not keep the extracted frames.** They are ~35 MB/episode (9 GB over the
   tree) and are derivable from the mp4s; only `tracks.npz` (~1 MB, bit-packed) is
   not. `rebuild_frames()` regenerates pixels on demand for visualisation.
3. **The bg-subtraction gripper mask is the whole arm**, so its centroid is dragged
   toward the base and barely moves. `grasp_follow` uses `gripper_tip()` — the arm
   pixel nearest the object — instead.
4. **Use a low percentile, not `min`, for the gap.** A plain minimum distance is
   decided by one pixel of SAM2 boundary noise. `boundary_gap` takes the 5th
   percentile via a distance transform (O(HW), not O(|A|·|B|)).
5. **Estimated depth is optional and off by default.** RoboWM-Bench's own appendix F
   measured that estimated depth (absolute *and* first-frame-aligned relative) stays
   far enough from ground truth that it did not improve their hand tracking. The
   two-camera AND-gate is our substitute; `--depth dav2|moge` exists for experiments.
6. **`SPLITS` for Ctrl-World are `makovian` / `non_makovian`**, and metadata lives in
   the shared dense input tree, not per method (sampling-dataset-layout rule 3).

## Validity gate: refusing to score what cannot be measured

A metric that silently rewards its own tracking failures is worse than no metric.
Two false-perfect modes were found by inspecting high scorers, and both are now
rejected outright (`valid: false`, `score: null`, excluded from summaries and
rankings):

| mode | symptom | example | reason string |
|---|---|---|---|
| tracking failure | object never detected in any view, so every informative term is `None` and the few defaults are 1.0 | `episode_000397` scored a **perfect 1.000** with `object_present_frac_gt = 0.00` | `gt_object_untracked`, `too_few_terms`, `gt_object_degenerate` |
| no interaction in GT | object tracked fine, but the arm never comes near it in GT, so no interaction term *can* be violated | `episode_001471` scored **0.984** with `gap_min_gt = 1.53` diameters and 0 GT contact frames | `no_gt_contact` |

The second is the subtler one: the track was perfectly good, the episode simply
contained no hand-object interaction to reproduce. Scoring it would reward a model
for the absence of the very thing being measured.

The gate checks: GT object trackable in >= `min_present_gt` of frames, >= 3 GT
contact frames, non-degenerate object scale, and >= `min_scored_weight` of the score
weight measurable. `valid_rate` is reported over all episodes; every other number is
averaged over valid ones only.

## Run commands

One episode, with panel and clip:

```bash
CUDA_VISIBLE_DEVICES=0 METHOD=dense SPLIT=makovian EPISODE=episode_000255 \
  OUT=/tmp/s2r_ep255 bash scripts/run_sim2real_interaction.sh
```

A whole method tree (resumable — episodes with `sim2real_metrics.json` are skipped;
`FORCE=1` retracks, `VIZ=1` renders a panel per episode):

```bash
CUDA_VISIBLE_DEVICES=0 METHOD=dense WORK_ROOT=/tmp/s2r_batch/dense \
  bash scripts/run_sim2real_interaction_batch.sh
```

Mine the violations and render the evidence (no GPU):

```bash
REPO=$PWD models/dreamgen/.venv/bin/python -m metrics.sim2real.mine_violations \
  --work_root /tmp/s2r_batch/dense --out /tmp/s2r_mined --top_k 5
```

Compare methods on the episodes valid for ALL of them:

```bash
REPO=$PWD models/dreamgen/.venv/bin/python -m metrics.sim2real.compare_methods \
  --roots dense=/tmp/s2r_batch/dense worldcache=/tmp/s2r_batch/worldcache \
          dicache=/tmp/s2r_batch/dicache itm=/tmp/s2r_batch/itm sito=/tmp/s2r_batch/sito \
  --baseline dense --out /tmp/s2r_compare
```

Outputs: per-episode `sim2real_metrics.json` + `tracks.npz` (+ `panel.png`,
`violation_clip.mp4` with `VIZ=1`), and `sim2real_summary.json` at the work root.
Without `WORK_ROOT` these land in `<episode dir>/sim2real/`, next to the existing
`metrics.json` / `physics_metrics.json`.
