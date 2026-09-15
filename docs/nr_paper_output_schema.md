# Dense-NR v3 paper output schema

Protocol: `dense-nr-v3.1-closed-world`. Text Alignment uses only the task clause
(`text_condition_prompt`); DreamGen layout prose is stripped. Subject masks are
seeded from the most confident arm detection among four frames and propagated both ways. Every number is computed per view-clip.
Aggregate only inside same model, embodiment, camera mount (`static`, `head`,
`wrist`), and task split. Report median with 95% bootstrap CI; include valid `n`
and exclusion rate. GT view-clips (`method: gt`, Ctrl-World 5 fps inputs) run
through the identical protocol and are the plausibility anchor for every column.

| Paper column | Stored field | Scale | Direction | Claim permitted |
|---|---|---|---|---|
| Visual Integrity | `visual_integrity` | CLIP-IQA+ frame p10 × 100, [0,100] | high | Few low-quality/collapsed frames. Not semantic correctness. |
| Text Alignment | `text_condition_alignment` | CLIPScore [0,100] | high | Text-condition agreement only. N/A for image-conditioned methods. |
| Actor Activity | `actor_motion_raw`, `actor_motion_active_rate` | px/source-frame, mount-specific: static = subject-mask p95 flow (uncorrected); head = subject p95 after background-translation removal, pairs with background < 10% invalid; wrist = whole-frame median flow (camera rides end effector). Active-pair fraction [0,1] at >= 1 px/source-frame. `actor_motion_median_raw` is diagnostic only | descriptive | Amount and temporal extent of robot movement. Compare to GT within mount; never higher-is-better globally. |
| Active Motion Continuity | `active_motion_continuity` | activity-weighted clip(1 − MSE_VFI / MSE_copy) × 100, [0,100] | high | Motion explainable by interpolation, normalized by motion amount. N/A without valid arm / active pairs, or when `active_motion_continuity_weight_sum` < 3 activity-equivalent triples. |
| Actor Flicker | `actor_flicker_raw` | flow-aligned RGB residual [0,1] | low | Subject shimmer after alignment. Experimental until human-calibrated. |

## Validity gates (a metric is `None`, never imputed)

- `actor_validity != robot_detected` → all subject metrics N/A.
- `actor_tracked_pair_frac < 0.5` → subject masks were lost by SAM2 over most of
  the clip; motion, active rate, and continuity are N/A. Do **not** use
  `actor_motion_valid_pair_frac` for this: the probe pre-filters its own
  `per_frame`, so that field is ~1.0 even when 1 of 163 pairs survived.
- `active_motion_continuity_weight_sum < 3` → too little activity to score
  continuity. The weighted mean is scale-free, so without this floor one marginal
  triple scores the same as a hundred (a 0.35-weight wrist clip scored AMC 83.1).
- ActorActivity is comparable only inside one `(model, embodiment, camera_mount)`
  stratum: `source_interval` is 1 for GT/Ctrl-World, 2–3 for DreamDojo, 3 for
  DreamGen, so px/source-frame spans different real time windows.
  `actor_motion_px_per_second` is the cross-fps-safe variant.

## Dropped in v3

- `camera_drift_raw`: v2 high tail (drift > 3 px) came from subject masks covering
  ~98% of frame (towel/box tasks), not camera motion. Not emitted.
- VFI global SSIM continuity: rank correlation −0.68 with actor motion in v2, so it
  measured motion amount. Kept only as `motion_continuity_global_ssim` diagnostic.

## Explicitly withheld columns

`ConditionEntityCoverage` and `TaskStateConsistency` are not emitted in v2
paper tables until closed-world task registry supplies canonical object IDs,
visibility phases, and task-state predicates for all 57 task templates. Existing
GroundingDINO phrase recall remains `condition_entity_coverage_proxy`, a debugging
field only. Do not relabel it as entity persistence.

## Closed-world calibration

For each task template, registry must define:

```yaml
task_id:
  relevant_entities: [canonical_object_id]
  visibility_expected: [phase ids]
  active_expected: [phase ids]
  final_predicates: [task-specific visual/state check]
```

Then derive two additional bounded columns:

- `ConditionEntityCoverage`: percentage of expected-visible entity/phase pairs
  observed (`0–100`, high better).
- `TaskStateConsistency`: percentage of applicable final predicates satisfied
  (`0–100`, high better). This is task completion evidence, not CLIP alignment.

Thresholds for activity, flicker, and predicate detectors freeze
from held-out human labels. Before that, publish native-unit distributions, not
invented 0–100 quality maps.
