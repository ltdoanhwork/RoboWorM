# Metric adaptation roadmap

Candidate metrics to adapt into PhyWorld. This is a roadmap, not a claim that
every proposed metric is already validated. Keep raw diagnostics separate from
composite scores until calibration against failures and human labels is complete.

## Reference types

| Type | Inputs | Question answered |
|---|---|---|
| **NR** (no-reference) | Method video, optional subject mask | Is this rollout internally stable, active, and visually coherent? |
| **DR** (dense-reference) | Method video + dense rollout from same episode | Did acceleration alter the dense rollout? |
| **GT-ref** | Method video + matched ground truth / privileged state | Is rollout correct relative to target scene state or action? |

Do not call an NR metric a fidelity metric: it cannot determine agreement with
dense or ground truth.

---

## Already adapted

### RBench — Motion-Aware Score (MAS)

| Source concept | PhyWorld adaptation | Type | Status |
|---|---|---|---|
| Subject-localized robot motion after camera compensation | **ActorMotion**: mean temporal median of ego-compensated SEA-RAFT flow inside arm ∪ object mask | NR | implemented |
| Avoid rewarding static smooth clips | **ActorMotionActiveRate**: fraction of subject pairs with motion ≥ 0.25 px/source-frame | NR | implemented |
| Subject motion preservation relative to reference | **ActorMotionDrift**: `abs(log(actor_motion_method / actor_motion_dense))` | DR | implemented |

Implementation notes:

- Foreground mask: existing GroundingDINO + SAM2 arm ∪ object pipeline.
- Camera compensation: subtract `median(flow_bg)` vector before subject reduction.
- Flow is normalized by actor decoding stride, so reported units are
  px/source-frame.
- `ActorMotion`, `ActorMotionActiveRate`, and `ActorMotionDrift` are diagnostics;
  they are not averaged into PhyDyn or PhyFid.

Known limitation: if GroundingDINO misses the arm, the metric becomes object-only.
Always report arm/object detections, foreground coverage, and mask overlays.

---

## Adapt next

### RoboPhys-3D — motion quality

RoboPhys-3D decomposes motion quality into localized intensity, overall flow,
interpolation-based smoothness, and high-frequency flicker. Its motion
smoothness weighting is particularly relevant because unweighted interpolation
metrics reward frozen clips.

| RoboPhys-3D source metric | Proposed PhyWorld metric | Definition | Type | Priority |
|---|---|---|---|---|
| Dynamic Degree (top active flow pixels) | **ActorMotionPeak95** | Mean over pairs of p95 `||flow_fg - median(flow_bg)||` inside arm ∪ object mask | NR | P0 |
| Motion Smoothness, motion-weighted | **ActiveMotionContinuity** | `sum_t w_t * SSIM_t / sum_t w_t`, `w_t = clip(actor_motion_t / tau, 0, 1)` | NR | P0 |
| Temporal Flickering | **ActorFlicker** | Flow-warped photometric residual inside arm ∪ object mask; higher residual = shimmer/flicker | NR | P0 |
| Flow Score | Keep **KineticBudget** | Existing scene-wide mean median flow; retain as context, not quality score | NR | done |

Why these three:

- Mean ActorMotion misses a small, fast gripper/object movement. Peak95 restores
  sensitivity without using a single maximum-flow outlier.
- ActiveMotionContinuity prevents static clips from receiving a high continuity
  score solely because interpolation is easy.
- ActorFlicker catches high-frequency appearance noise that a VFI interpolation
  score may miss.

Suggested raw outputs: `actor_motion_peak95_raw`,
`active_motion_continuity`, `actor_flicker_raw`, per-pair values, foreground
coverage, and mask-validity flags. Do not add them to PhyDyn until failure-based
calibration establishes weights.

### RoboPhys-3D — camera trajectory geometry

| RoboPhys-3D source metric | Proposed PhyWorld metric | Definition | Type | Priority |
|---|---|---|---|---|
| Camera Control for fixed views | **CameraDrift** | Mean temporal `||median(flow_bg)||` after excluding arm ∪ object mask | NR | P0 |

Apply only to static external cameras. Report wrist cameras in a separate group;
depth parallax makes a global background-translation model insufficient.

### RoboPhys-3D — content consistency

| RoboPhys-3D source metric | Proposed PhyWorld metric | Definition | Type | Priority |
|---|---|---|---|---|
| Subject Consistency, SAM2 + DINO | **ActorAppearanceDrift** | DINOv2 cosine similarity on subject crops: adjacent frame for local drift and first-frame anchor for global identity drift | NR | P1 |
| Background Consistency | **BackgroundAppearanceDrift** | Adjacent-frame feature similarity outside subject mask | NR | P1 |
| Photometric Consistency | Fold into **ActorFlicker** | Subject-localized flow-warp residual rather than scene-wide AEPE | NR | P0 |
| Style / semantic consistency | Optional appendix diagnostics | CLIP/ViCLIP/Gram metrics | NR | P2 |

Do not put CLIP/ViCLIP metrics in the core VLM-free dynamics loop. They are useful
as appendix diagnostics, but can reward semantic plausibility while missing state
and trajectory errors.

---

## Requires stronger supervision or infrastructure

### 3D geometry

| RoboPhys-3D source metric | Possible PhyWorld adaptation | Required input | Status |
|---|---|---|---|
| Depth Accuracy | **DepthDrift** between method and dense/GT depth | Shared depth estimator or calibrated depth | Pilot only |
| Reconstruction-based geometry / reprojection | **ReprojectionError** | Shared 3D reconstruction for method and reference | Expensive; pilot main view first |
| Roundtrip Consistency | **ReturnViewConsistency** | Camera revisits initial pose | Task/camera dependent |

For a 3D claim, process method and reference through the same reconstruction
pipeline, then measure residual error relative to the reconstruction baseline.
Do not compare a reconstruction-derived score to raw video ground truth and call
the difference generation error.

### State and trajectory correctness

| RoboPhys-3D source metric | Possible PhyWorld adaptation | Required input | Type |
|---|---|---|---|
| Object Localization | Object AP / AR | GT per-frame masks or boxes | GT-ref |
| HSD / nDTW trajectory accuracy | **ActorTrajectoryFidelity** | Matched object and end-effector tracks | GT-ref or DR |
| Dynamic Consistency | Velocity / acceleration distribution distance | Same matched tracks | GT-ref or DR |

These are much stronger than NR motion magnitude, but require reliable GT masks,
poses, action trajectories, or dense tracks. Do not substitute a VLM judgment for
trajectory correctness.

### Execution and task metrics

| RoboPhys-3D source metric | Why not core PhyWorld now |
|---|---|
| Interaction quality | VLM-based contact, force, interpenetration, grasp judgment; not VLM-free or geometrically grounded enough alone |
| Physical commonsense | VLM Likert judgment; useful qualitative audit, weak as core score |
| Instruction following | Measures recognizable task compliance, not exact state correctness |
| Collision safety / task success | Requires inverse-dynamics decoder plus simulator replay and task success specification |

Keep these separate if introduced. A rollout can be conservative and collision-free
because it barely moves; pair any safety result with ActorMotion and trajectory or
task-completion evidence.

---

## Score policy

1. **Never assume dense > speedup in NR raw motion.** Speedup can suppress motion,
   exaggerate it, or add flicker. NR metrics diagnose the direction; they do not
   define dense as automatically better.
2. **Use DR metrics for closeness to dense.** MotionFidelity, KineticDrift, and
   ActorMotionDrift answer preservation questions.
3. **Keep opposed signals separate.** Continuity rewards stillness; ActorMotion
   and ActiveRate expose inactivity. Averaging them can erase both signals.
4. **Gate mask validity.** Report `actor_n_arm_dets`, `actor_n_object_dets`,
   foreground coverage, and overlays. Mark arm-missing samples as object-only.
5. **Calibrate before composite scoring.** Choose thresholds or weights from a
   held-out failure set and human labels, not convenience constants.

## Recommended implementation order

1. Add `CameraDrift`, `ActorMotionPeak95`, `ActiveMotionContinuity`, and
   `ActorFlicker` with raw per-episode diagnostics.
2. Add `actor_mask_valid` / `actor_mask_mode` (`arm_object`, `object_only`,
   `arm_only`, `invalid`) and prevent invalid samples from method means.
3. Run matched dense / WorldCache / DiCache / FasterCache samples covering static,
   moderate-motion, and active episodes. Inspect overlays before aggregate claims.
4. Calibrate NR thresholds against labeled failures.
5. Pilot shared-depth or reconstruction metrics only after NR mask validity is
   stable.
