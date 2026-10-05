# PhyWorld metrics — full inventory

Every metric currently in this repo, named for the paper where applicable.
WorldScore names survive only as lineage. JSON keys match the PhyWorld names
below when a PhyWorld mapping exists.

## Paper groups (PhyWorld)

| Group | Reference | Backend | VLM in score | Status |
|---|---|---|---|---|
| **PhyQual-NR** | none | CLIP, GroundingDINO | yes | probe ready (`metrics/quality/`) |
| **PhyDyn-NR** | none | SEA-RAFT, VFIMamba, arm∪object mask | no\* | probe ready (`metrics/dynamics/`) |
| **PhyFid-DR** | dense rollout, same episode | SEA-RAFT, VFIMamba | no† | code ready (`physics_metrics.py`) |
| **PhyBody-NR** | none | 3D-lifted body keypoints | no | scores in `hallucinate_bench/` (detector code external) |
| **PhyView-NR** | none | cross-view point tracking | no | scores in `data_for_web.csv` (detector code external) |

\*ActorDynamics currently seeds the foreground with GroundingDINO+SAM2; the
measured quantity is pure flow. †LookDrift is CLIP-derived and stays with PhyQual.

---

## 1. PhyQual-NR — quality branch (from WorldScore)

Source: `metrics/quality/quality_control_probe.py`. Reference-free, per-frame.
Env: `models/dreamgen/.venv`.

| PhyWorld | WorldScore | Model | What it compares | Our edit | Better |
|---|---|---|---|---|---|
| **SceneIntegrity** (`scene_integrity`) | `clip_iqa+` | CLIP RN50 + learned prompts | image vs good/bad poles → softmax | Drop WorldScore z-norm. Prefer min across views, not mean of a tiled grid. Map raw [0,1]×100. | ↑ |
| **LookScore** (`look_score`) | `clip_aesthetic` | CLIP ViT-L/14 + 5-layer MLP | image only, no text | Computed; **not** folded into PhyQual. Aesthetic ≠ physics. | ↑ (diagnostic) |
| **TaskLock** (`task_lock`) | `clip_score` | CLIP ViT-B/16 (torchmetrics) | image vs text, 100·cos | Text = episode **task instruction**, not a scene prompt. | ↑ |
| **EntityPresence** (`entity_presence`) | `object_detection` | GroundingDINO SwinT | localize phrases, detection rate | Persistence of arm + object nouns, not “new object appears”. | ↑ |

**PhyQual** = mean(SceneIntegrity, TaskLock, EntityPresence) on 0–100.

Trap: do **not** use pyiqa’s CLIPScore — WorldScore’s evaluator uses torchmetrics
ViT-B/16 (`100·cos`). Swapping silently breaks the scale.

---

## 2. PhyDyn-NR — dynamics branch (from WorldScore)

Source: `metrics/dynamics/`. Env: `models/dreamgen/.venv`, cwd `metrics/WorldScore`.

| PhyWorld | WorldScore | Raw return | Meaning | Our edit | Better |
|---|---|---|---|---|---|
| **KineticBudget** (`kinetic_budget`) | `motion_magnitude` | 1 float | mean over pairs of median \|flow\| (px/frame), **scene-wide** | Report **raw**; do **not** score higher=better. | report only |
| **ActorMotion** (`actor_motion`) | none — ours, adapted from RBench MAS | 1 float | mean over pairs of median \|flow_fg − median(flow_bg)\| **inside arm∪object mask only** — subject kinetic budget after ego-motion removal | New. Report **raw** (`actor_motion_raw`); retain `actor_motion_uncorrected_raw` plus background ego vector for audit. Optional soft-clip 0–100 map (`actor_motion_score`, tanh scale=2.0). | report only (raw); diagnostic map, not "higher always better" unbounded |
| **ActorMotionActiveRate** (`actor_motion_active_rate`) | none — ours | 1 float in [0,1] | fraction of frame-pairs whose ego-compensated subject median flow is ≥0.25 px/frame | No-reference temporal coverage. Read with ActorMotion(raw): high mean + low active rate signals a burst/spike, not sustained useful motion. Matched grid required; report raw, do not fold into PhyDyn. | report only |
| **ActorDynamics** (`actor_dynamics`) | `motion_accuracy` | 1 float | WorldScore: max(obj)−max(bg) | Foreground = **arm ∪ object**; use **median(fg)−median(bg)**. Soft-clip to 0–100. | ↑ |
| **MotionContinuity** (`motion_continuity`) | `motion_smoothness` | mse, ssim, lpips | VFIMamba odd-from-even | Primary = **SSIM**×100; MSE/LPIPS diagnostic. Never without KineticBudget. | ↑ |

**PhyDyn** = mean(ActorDynamics_mapped, MotionContinuity_mapped). KineticBudget and
ActorMotion and ActorMotionActiveRate sit beside the mean, not inside it — Continuity rewards stillness (high
SSIM on a frozen clip), ActorMotion penalizes it; averaging them blindly cancels the
signal. Use ActorMotion≈0 + Continuity≈99 as a **smooth-but-inactive** flag (see
`tables/wave0.md`, episode 000255).

ActorMotion vs. RBench's Motion-Aware Score (MAS): same question — "does the robot
actually move, not just the camera/background" — but ego-compensated flow-on-mask instead of
GDINO+SAM2+CoTracker as a second pipeline. We already run GDINO+SAM2 for ActorDynamics'
arm∪object mask, so ActorMotion reuses that mask's `fg_flow_ego_med` per frame at zero
extra inference cost — it's a field on the existing `motion_accuracy_gdino` probe
output, not a new probe. No VLM in the score path (bg-sub or the existing GDINO+SAM2
mask both work; CoTracker not required).

Files:

| file | role |
|---|---|
| `motion_mag_smooth_probe.py` | KineticBudget + MotionContinuity |
| `motion_accuracy_gdino.py` | ActorDynamics (arm∪object) **and** ActorMotion (`mean_fg_flow_med` on the same mask) |
| `motion_accuracy_probe.py` | WorldScore original (object-only) — comparison only |
| `flow_backend.py` | SEA-RAFT loader for paired metrics |
| `physics_metrics.py` | flow vs ref (GT or dense) |

Protocol: flow is **px/frame** — only compare on matched (H, W, fps, T). Per pair,
subtract `median(flow_bg)` as a robust global camera-motion estimate before subject
reduction. Static external cameras are target protocol; report wrist cameras separately
because depth parallax defeats global translation compensation. If actor decoding uses
stride > 1, divide flow by stride before scoring so values remain px/source-frame.

ActorMotionActiveRate uses a fixed 0.25 px/frame threshold on this matched grid. It is
a no-reference coverage diagnostic, not a correctness or task-success score; calibrate
the threshold again if resolution or sampling rate changes.

---

## 3. PhyFid-DR — dense as reference (ours, same backends)

For acceleration methods. Reference = **dense pred** of the same episode, never GT.

| PhyWorld | Definition | Better |
|---|---|---|
| **MotionFidelity** (`motion_fidelity`) | mean EPE ‖flow_method − flow_dense‖ | ↓ |
| **KineticDrift** (`kinetic_drift`) | \|log(mag_method / mag_dense)\| , scene-wide | ↓ |
| **ActorMotionDrift** (`actor_motion_drift`) | \|log(actor_motion_method / actor_motion_dense)\| — same formula as KineticDrift, subject-only magnitude | ↓ |
| **ContinuityGap** (`continuity_gap`) | SSIM_dense − SSIM_method | ↓ |
| **LookDrift** (`look_drift`) | \|SceneIntegrity_method − SceneIntegrity_dense\| (CLIP → group with PhyQual) | ↓ |

**PhyFid** = mean of min-max mapped {MotionFidelity, KineticDrift, ContinuityGap}
to 0–100 (higher = closer to dense). LookDrift and ActorMotionDrift reported beside,
not in the mean — ActorMotionDrift on a near-zero-motion episode is a noisy log
ratio (see `tables/wave0.md`, episode 000255: largest drift, 0.808, is the most
static clip — expected, not a bug).

Also available (appendix / older trial): GT-ref `flow_epe` via
`scripts/compute_physics_metrics_ctrlworld.py` — not the main speedup table.

---

## 4. PhyBody-NR — embodiment integrity (ours)

Per 16-frame segment. Scores live in
`sampling_dataset/data_for_web/hallucinate_bench/300-prune/{robot}/segments.json`.
Detector code is **not** in this repo (external); outputs + human labels are.

| PhyWorld | Source key | Units / reduce | Meaning | Better |
|---|---|---|---|---|
| **BodyRigidity** (`body_rigidity`) | `rigidity.value` | mm, median | link lengths stay constant (no stretch/break/dissolve) | ↓ raw / ↓ ratio |
| **BodyJerk** (`body_jerk`) | `jerk.value` | mm/s³, max | motion attainable (no snap/teleport) | ↓ raw / ↓ ratio |

Each segment also has `ratio` (= value / per-source calibrated threshold) and
`violated` (ratio ≥ 1). Score on **ratio**, not raw mm — comparable across
embodiments; 1.0 is the violation boundary. Failures stay localized per segment.

---

## 5. PhyView-NR — cross-view coherence (ours)

Multiview only. Scores live in `sampling_dataset/data_for_web/data_for_web.csv`
(300 clips, human-labeled). Detector code external; CoTracker overlays for raters
are in `methods/human_assessment/` + `tmp/20261005_refactor/web/scripts/build_assessment_multiview.py` (archived).

| PhyWorld | CSV column | Meaning | Better |
|---|---|---|---|
| **ViewGeometry** (`view_geometry`) | `geometric_consistency` | views agree on one scene geometry | ↑ |
| **ViewPoseStability** (`view_pose_stability`) | `pose_stability` | agreement holds over time | ↑ |
| **ViewDisjointRate** (`view_disjoint_rate`) | `disjoint_frame_ratio` | fraction of frames no single scene explains | ↓ |

`combined_score` in the CSV = mean(geometric_consistency, pose_stability).
On the 300 human-rated clips: AUC(good vs bad) = **0.987** for that pair,
0.965 pose alone, 0.870 geometry alone, **0.599** disjoint alone. Disjoint is a
rare hard-failure flag (median clip = 0) — report beside the score, do not average in.

---

## 6. Pixel baselines (always on)

Source: `metrics/video_metrics.py`. Written into per-episode `metrics.json`.
Not PhyWorld; kept as the standard reconstruction baseline.

| Metric | What | Better |
|---|---|---|
| PSNR | peak signal-to-noise vs GT | ↑ |
| SSIM | structural similarity vs GT | ↑ |
| LPIPS | perceptual distance vs GT | ↓ |

---

## Other metric modules in the repo (not PhyWorld paper columns)

These exist under `metrics/` and may feed future tables or appendix claims. They
are **not** folded into PhyQual / PhyDyn / PhyFid / PhyBody / PhyView.

### 6a. Sim2real interaction — `metrics/sim2real/`

Hand–object physics **without a simulator**. Masks from GroundingDINO+SAM2 /
bg-subtraction; lengths in object diameters. Paired GT cancels tracker bias.

**Violations (reference-free / GT-excess):**

| Term | Measures |
|---|---|
| `levitation` | object translates with no contact (excl. falling) |
| `grasp_follow` | \|v_obj − v_grip\| / \|v_grip\| during contact |
| `object_presence` | object trackability deficit vs GT |
| `object_shape` | area/aspect CV excess vs GT |
| `penetration` | gripper-over-object overlap excess |
| `chatter` | contact on/off transitions excess |

**Agreement (action-conditioned only):** contact onset / temporal IoU / F1,
object trajectory error, gap-curve error.

**Step nodes:** contact → lift → transport → release (derived from GT).

Headline: `sim2real_interaction_score` (+ always the per-term breakdown).
Text-conditioned models: charge physics group only, not agreement.

### 6b. Embodied execution — `metrics/embodied_exec/`

Kinematic grounding the mask tier cannot reach. Status: **checkers validated on
GT; not yet run on generated video** (needs video→joint pose decoder).

| Checker | Measures |
|---|---|
| `fk_consistency` | recovered joints → FK EE vs observed EE |
| `kinematic_feasibility` | joint limits / velocity / acceleration ceilings |
| `trajectory_agreement` | EE path pred vs GT |

### 6c. Force feasibility — `metrics/force_feasibility/`

Newton residual: does any friction-cone contact force explain observed
acceleration? Residual = \|r\| / (m·g); mass cancels. Also `grip_demand`.
Controls pass; real video path still depends on lift/scale quality.

### 6d. Real2sim reconstruction — `metrics/real2sim/`

TripoSR mesh + metric-scale probes for a real collision geom. Status:
**blocked on segmentation / scale reproducibility** — do not report numbers yet.

---

## Paper tables (suggested)

| Table | Columns | Who |
|---|---|---|
| 1 PhyQual-NR | SceneIntegrity, TaskLock, EntityPresence, PhyQual | every method + dense |
| 2 PhyDyn-NR | KineticBudget (raw), ActorMotion (raw), ActorDynamics, MotionContinuity, PhyDyn | every method + dense |
| 3 PhyFid-DR | MotionFidelity, KineticDrift, ActorMotionDrift, ContinuityGap, PhyFid | speedups only |
| 4 PhyBody / PhyView | BodyRigidity ratio, BodyJerk ratio, ViewGeometry, ViewPoseStability (+ Disjoint as flag) | where applicable |

LookScore / LookDrift: appendix. PSNR/SSIM/LPIPS: baseline table or appendix.
Sim2real / embodied_exec / force: separate physics-interaction section if claimed.

---

## Protocol

- Optical flow is **px/frame**. Matched (H, W, fps, T) only.
- NR comparable inside one (embodiment, model, view) cell.
- Dense = NR ceiling and DR reference. Speedup scoring above dense on NR is a red flag.
- Write `phyworld.json` beside the episode; never overwrite `metrics.json`.
- PhyWorld names in JSON / scripts / paper; WorldScore names only in comments.

## Known failure modes

- **EntityPresence** on tiled multiview strips: abstract instruction nouns
  (“control”, “button”) → false negatives. Score primary view or filter nouns.
- **MotionContinuity** saturates (~98–99) on short low-fps clips — this is exactly
  why ActorMotion exists: Continuity alone cannot tell a moving robot from a frozen
  one. Always read ActorMotion (raw) next to it, never Continuity in isolation.
- **KineticBudget** scene-wide median is tiny on multiview strips (static table
  dominates) — report only; use KineticDrift for comparisons.
- **ActorMotion** inherits ActorDynamics' mask failure modes (empty/low-confidence
  arm∪object mask on a bad GDINO detection reads as near-zero motion even if the
  robot moves — check `actor_fg_cov_pct` isn't degenerate, e.g. <1%, before trusting
  a low ActorMotion as "inactive" rather than "mask missed the arm").
- **ActorMotionDrift** blows up on near-static episodes (small absolute change, large
  log ratio) — same shape as KineticDrift; read beside ActorMotion(raw), not alone.
- **ActorDynamics** with object-only WorldScore mask inverts on robot video
  (arm in “background”) — always use arm∪object + median−median.
- Dense-vs-dense MotionFidelity must be ~0 (self-check).
