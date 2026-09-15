# PhyWorld NR metric redesign

## Goal

Each reported metric must answer one visible question. It must not silently
stand in for task success, physics correctness, or dense-reference fidelity.
Raw measurements remain separate; no grand mean is reported.

## Core metrics

| Group | Metric | Visible question | Definition | Validity / exclusion |
|---|---|---|---|---|
| Visual integrity | **VisualIntegrity** | Does every part of clip remain visually usable? | p10 of CLIP-IQA+ over generated frames. Report mean as context, never replace p10. | All views. Compare within view family. |
| Condition grounding | **TextConditionAlignment** | For text-conditioned model, does rendered content agree with supplied instruction? | Mean CLIP image--text score. | Only DreamGen/CtrlWorld text-conditioned rows. Not NR-only; never applied to image-only models. |
| Condition grounding | **ConditionEntityCoverage** | For text-conditioned model, are instruction entities visibly detected? | Per-frame GroundingDINO recall over curated expected entities, then temporal mean. | Curated task ontology/synonyms required. Do not call this generic `EntityPresence`. |
| Actor activity | **ActorActivity** | Does robot/object subject move rather than freeze? | Ego-compensated median flow inside arm union object mask; active rate is auxiliary. | Require arm mask. External and wrist views reported separately. |
| Actor activity | **ActorMotionPeak95** | Is there localized fast gripper/object movement? | Mean p95 ego-compensated subject-flow magnitude. | Same subject-mask validity. Diagnostic beside ActorActivity, not a second quality score. |
| Temporal dynamics | **ActiveMotionContinuity** | When subject moves, is motion temporally coherent? | VFIMamba interpolation SSIM weighted by localized actor activity. | Require nonzero activity weight and valid arm mask. |
| Temporal dynamics | **ActorFlicker** | Does subject appearance shimmer after motion alignment? | Forward/backward-consistent flow-warped photometric residual inside subject mask. Lower is better. | Exclude occlusions, invalid flow, object-only masks. Human-calibrate before paper use. |
| Camera stability | **CameraDrift** | Does supposedly static external camera drift? | Mean magnitude of robust background flow vector. | Static external views only. Wrist views N/A. |

## Context only

| Old metric | New role | Reason |
|---|---|---|
| `scene_integrity` | Backward-compatible mean image-quality field. `visual_integrity` becomes core. | Mean conceals short collapse. |
| `entity_presence` | Replace with condition-scoped `ConditionEntityCoverage`. | Detector output depends on prompt wording and is not persistence. |
| `task_lock` | Rename `TextConditionAlignment`; condition-scoped. | CLIP similarity is not task completion. |
| `actor_dynamics` | Retire from headline table. | Redundant with localized ActorActivity; max-flow form is outlier-prone. |
| `kinetic_budget` | Scene-motion context only. | Background/table dominate; it cannot say robot moved. |
| `motion_continuity` | Backward-compatible global diagnostic. `ActiveMotionContinuity` becomes core. | Frozen clips get falsely high global continuity. |
| `actor_motion_active_rate` | Auxiliary statistic of ActorActivity. | Same latent construct, useful to identify bursts. |

## Validation gate

Do not call metric validated from model output alone. Build stratified visual set
before a paper table:

1. Sample 150 clips across model, embodiment, view, and each metric decile.
2. Three raters label independent binary/ordinal targets: image collapse;
   required-entity visibility; text-condition match; robot active; active motion
   smoothness; subject shimmer; external-camera drift.
3. Score rank correlation for ordinal targets and AUROC/F1 for binary targets,
   with bootstrap confidence intervals. Report inter-rater agreement first.
4. Freeze threshold/direction on development half; evaluate once on held-out half.
5. Drop, relabel, or restrict any metric with weak held-out association or a
   dominant confound. In particular, keep ActorFlicker experimental until it
   distinguishes visual shimmer from occlusion/specular changes.

## Required reporting

- No pooled ranking across external and wrist cameras.
- Show masks, camera-drift validity, detection phrase list, and score coverage.
- Plot ActorActivity against ActiveMotionContinuity. Their joint quadrants expose
  inactive-smooth and active-broken rollouts; their arithmetic mean hides both.
- Condition metrics have an explicit `text_condition_applicable` flag and are
  omitted, not zero-filled, for image-conditioned methods.
