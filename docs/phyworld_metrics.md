# PhyWorld metrics

Paper-facing definition of every PhyWorld metric: what it measures, how it is
computed, and — for the seven adapted from WorldScore — what we changed and why.

PhyWorld evaluates video world models where **the embodiment and the setting are
known in advance**. That assumption is what lets us name one failing property per
metric instead of producing a single opaque quality number, and it is what most of
the algorithm edits below exploit.

Names in this document are the JSON keys and the paper columns. WorldScore names
survive only as lineage.

## Groups

| Group | Reference | Backend | VLM in the score |
|---|---|---|---|
| **PhyBody-NR** — embodiment integrity | none | 3D-lifted body keypoints | no |
| **PhyView-NR** — cross-view coherence | none | cross-view point tracking | no |
| **PhyDyn-NR** — motion plausibility | none | SEA-RAFT, VFIMamba | no |
| **PhyFid-DR** — fidelity to dense | dense rollout, same episode | SEA-RAFT, VFIMamba | no |
| **PhyQual-NR** — perceptual and semantic | none | CLIP, GroundingDINO | **yes** |

The first four are the physics core and are VLM-free. PhyQual is CLIP- and
GroundingDINO-based; it is reported as its own group and must never be described
as "derived from segmentation, tracking and geometry".

## Adapted from WorldScore (seven metrics)

A rename alone would not make these ours. Each one carries a substantive
algorithmic change, listed in the last column.

### Quality branch

| PhyWorld | WorldScore | Model | What it compares | Our edit |
|---|---|---|---|---|
| **SceneIntegrity** | `clip_iqa+` | CLIP RN50 + learned prompts | image vs good/bad poles → softmax probability | Drop WorldScore's z-normalization — those constants are fit to their camera-path dataset and are meaningless on fixed-camera robot video. Score **per view and take the minimum** across views instead of averaging a tiled grid: one melted view is the failure, and a mean over three views hides it. |
| **LookScore** | `clip_aesthetic` | CLIP ViT-L/14 + 5-layer MLP | image only, no text | Computed but **deliberately excluded from every aggregate**. "Pretty photo" is not physical fidelity, and a smeared frame can be more aesthetic than a correct one. Diagnostic column only. |
| **TaskLock** | `clip_score` | CLIP ViT-B/16 | image vs prompt, 100·cos | The reference text is the episode's **actual robot task instruction** from metadata, not a WorldScore scene description. This turns a scene-description check into an instruction-adherence check, which is the only form that means anything when the setting is fixed and known. |
| **EntityPresence** | `object_detection` | GroundingDINO SwinT | localize arm + object, detection rate | WorldScore asks whether a newly prompted object **appears**. We ask whether the arm and the manipulated object **stay present** for the whole clip. Queries are the known embodiment phrases plus object nouns parsed from the instruction; the score is the fraction of expected phrases detected, averaged over frames. |

### Dynamics branch

| PhyWorld | WorldScore | Returns (raw) | Meaning | Our edit |
|---|---|---|---|---|
| **KineticBudget** | `motion_magnitude` | 1 float | mean over frame pairs of median \|flow\| | Kept as a measurement but **removed from the score**. WorldScore treats more motion as better, which pays a model for hallucinating movement. It is reported raw in px/frame; its scored form is KineticDrift below. |
| **ActorDynamics** | `motion_accuracy` | 1 float | mean over frames of max(obj flow) − max(bg flow) | Two changes. Foreground is **arm ∪ object**, not object alone: on a fixed camera the arm sits in WorldScore's "background" and is the fastest thing in frame, which inverts the metric. And **median − median** replaces max − max, because a max is decided by a single pixel of flow noise. |
| **MotionContinuity** | `motion_smoothness` | 3 floats (mse, ssim, lpips) | mean over interpolated triples | VFIMamba reconstructs odd frames from even ones; **SSIM is the single reported number**, with MSE and LPIPS kept as diagnostics. Never read it without KineticBudget: a static clip is trivially smooth. |

## Derived from the same backends (dense as reference)

New metrics, not in WorldScore. They exist because the benchmark also measures
**acceleration methods**, where the question is not "does this look real" but
"did the speedup preserve what dense produced". The reference is the dense
rollout of the same episode on a matched grid, never the ground-truth video.

| PhyWorld | Definition | Better |
|---|---|---|
| **MotionFidelity** | mean endpoint error between the method's flow field and dense's | lower |
| **KineticDrift** | \|log(mag_method / mag_dense)\| — symmetric, so under- and over-moving are penalized alike | lower |
| **ContinuityGap** | SSIM_dense − SSIM_method; positive means jerkier than dense | lower |
| **LookDrift** | \|SceneIntegrity_method − SceneIntegrity_dense\| (CLIP-derived, so grouped with PhyQual) | lower |

## Ours from the start

Not WorldScore-derived. These are the two signals the suite is built around.

| PhyWorld | Measures | Computed from |
|---|---|---|
| **BodyRigidity** | robot links keep a constant length — the body does not stretch, break, or dissolve | 3D-lifted body keypoints, per 16-frame segment, mm |
| **BodyJerk** | body motion stays physically attainable — no snap or teleport | 3D-lifted body keypoints, per 16-frame segment, mm/s³ |
| **ViewGeometry** | simultaneously generated views agree on one scene geometry | cross-view point correspondences |
| **ViewPoseStability** | that agreement holds over time, not just at the first frame | cross-view point correspondences |
| **ViewDisjointRate** | fraction of frames no single scene can explain | cross-view point correspondences |

Body metrics are scored on the **ratio** of the raw value to a per-source
calibrated threshold, not on the raw millimetres: mm are not comparable across
embodiments, ratios are, and 1.0 is the violation boundary by construction. Each
segment carries its own `violated` flag, so a failure stays localized to a
timespan instead of being averaged away.

`ViewGeometry` and `ViewPoseStability` are averaged; `ViewDisjointRate` is
reported separately. On the 300 human-rated multiview clips the pair separates
"good" from "bad" with **AUC 0.987**, against 0.965 for pose stability alone and
0.870 for geometry alone. `ViewDisjointRate` reaches only 0.599 alone and is 0.0
for the median clip — it is a rare hard-failure flag, not a continuous quality
term, and averaging it in changed no ranking while pushing every clip toward the
ceiling.

## Protocol

- Optical flow is in **px/frame**. Any comparison — to dense or across methods —
  is only valid on a matched (H, W, fps, T) grid.
- No-reference numbers are comparable within one (embodiment, model, view) cell,
  not across resolutions or models.
- Dense is the no-reference ceiling and the dense-reference target. A speedup
  scoring above dense on a no-reference metric is a red flag, not a win.
- Write `phyworld.json` beside the episode. Never overwrite `metrics.json`.

## Known failure modes

- **EntityPresence on tiled multiview strips.** Instruction nouns include
  abstract words ("control", "button") and the detector runs on the whole grid.
  A visually clean episode scored 8/100 in the Wave 0 run. Score the primary view
  only, or filter abstract nouns.
- **MotionContinuity saturates** (~98–99) on short, low-fps clips, so it cannot
  discriminate there. Do not let it drive a ranking without checking spread.
- **KineticBudget is dominated by the static scene** on multiview strips: the
  scene-wide median was ~0.05 px in Wave 0. It is a report, not a score.
