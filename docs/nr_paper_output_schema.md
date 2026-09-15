# Dense-NR v2 paper output schema

Protocol: `dense-nr-v2-closed-world`. Every number is computed per generated
view-clip. Aggregate only inside same model, embodiment, camera family, and task
split. Report median with 95% bootstrap CI; include valid `n` and exclusion rate.

| Paper column | Stored field | Scale | Direction | Claim permitted |
|---|---|---|---|---|
| Visual Integrity | `visual_integrity` | CLIP-IQA+ frame p10 × 100, [0,100] | high | Few low-quality/collapsed frames. Not semantic correctness. |
| Text Alignment | `text_condition_alignment` | CLIPScore [0,100] | high | Text-condition agreement only. N/A for image-conditioned methods. |
| Actor Activity | `actor_motion_raw`, `actor_motion_active_rate` | ego-compensated px/source-frame; active-pair fraction [0,1] at >= 0.25 px/source-frame | descriptive | Amount and temporal extent of robot movement. Never higher-is-better globally. |
| Active Motion Continuity | `active_motion_continuity` | activity-weighted VFI SSIM ×100, [0,100] | high | Temporal coherence conditional on actor motion. N/A without valid arm / active pairs. |
| Actor Flicker | `actor_flicker_raw` | flow-aligned RGB residual [0,1] | low | Subject shimmer after alignment. Experimental until human-calibrated. |
| Static Camera Drift | `camera_drift_raw` | robust background-flow px/source-frame | low | Fixed-camera instability. N/A for wrist views. |

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

Thresholds for activity, flicker, camera drift, and predicate detectors freeze
from held-out human labels. Before that, publish native-unit distributions, not
invented 0–100 quality maps.
