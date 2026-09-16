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
| Visual Integrity | `visual_integrity` | CLIP-IQA+ frame p10 × 100, [0,100] | **descriptive only** | **No ranking claim.** Failed the GT-is-ceiling audit (E1, 2026-09-16): pred beats co-located GT on dreamdojo (+0.90, p=5e-05) and 4view (+0.52). Motion/smoothness/sharpness explanations all rejected → domain bias, not quality. Report as a descriptive statistic with this caveat or omit. |
| Text Alignment | `text_condition_alignment` | CLIPScore [0,100] | **paired only** | Dropped by E2 on 2026-09-16, **reinstated the same day** — the prompt-shuffle control (n=100, +0.088 [−1.043, +0.930]) was itself confounded, because swapping in another task's instruction also swaps caption style/length, and CLIPScore's per-caption bias (≈7 pts) dwarfs image-text agreement (≈2). The double-centered clip-by-prompt matrix (`scripts/_exp_text_matrix.py`) shows real discrimination on every embodiment (centered diagonal +1.40 / +2.53 / +1.64, all CIs excluding 0; retrieval top-1 37.5% vs 6.2% chance). **Report paired deltas only**: pred and its own GT are scored against the same prompt (83/83 pairs), so the bias cancels exactly, and GT wins RESOLVED (ctrlworld −0.552 [−1.019, −0.302]; 4view −0.209 [−0.703, −0.066]). Never quote an unpaired median — it pools different captions. Enforced by `PAIRED_ONLY` in `scripts/table_dense_nr.py`. |
| Actor Activity | `actor_motion_raw`, `actor_motion_active_rate` | px/source-frame, mount-specific: static = subject-mask p95 flow (uncorrected); head = subject p95 after background-translation removal, pairs with background < 5% invalid (E4; was 10); wrist = whole-frame median flow (camera rides end effector). Active-pair fraction [0,1] at >= 1 px/source-frame. `actor_motion_median_raw` is diagnostic only | descriptive | Amount and temporal extent of robot movement. Compare to GT within mount; never higher-is-better globally. |
| Active Motion Continuity | `active_motion_continuity` | activity-weighted clip(1 − MSE_VFI / MSE_copy) × 100, [0,100] | high | Motion explainable by interpolation, normalized by motion amount. N/A without valid arm / active pairs, or when `active_motion_continuity_weight_sum` < 3 activity-equivalent triples. |
| Actor Flicker | `actor_flicker_raw` | flow-aligned RGB residual [0,1] | low | **DROPPED — must not appear in any paper table.** Two independent failures: (E7) it *improves* under blur at every severity, incl. realistic (−0.001 [−0.0020, −0.0004]), and does not react to frame-dup/smoothing/blocking; (E3) the prescribed per-pixel fix was built and run — 9 candidate normalizations, and **none** passes `\|rho(motion,·)\| < 0.25` on all three mounts. Kept in the record as experimental. See E3/E7. |

## Validity gates (a metric is `None`, never imputed)

- `actor_validity != robot_detected` → all subject metrics N/A.
- `actor_tracked_pair_frac < 0.5` → subject masks were lost by SAM2 over most of
  the clip; motion, active rate, and continuity are N/A. Do **not** use
  `actor_motion_valid_pair_frac` for this: the probe pre-filters its own
  `per_frame`, so that field is ~1.0 even when 1 of 163 pairs survived.
- `active_motion_continuity_weight_sum < 3` → too little activity to score
  continuity. The weighted mean is scale-free, so without this floor one marginal
  triple scores the same as a hundred (a 0.35-weight wrist clip scored AMC 83.1).
- `n_active_pairs < max(3, 0.05 × valid pairs)` → too few pairs actually reach
  `ACTIVE_PX = 1`. The absolute floor alone is length-blind, so on long clips it is
  nearly free: three clips with 79/177/245 valid pairs cleared it with 3–7 active pairs
  at `active_rate` 0.02–0.04 and reported **AMC 71 / 71 / 82**. The 5% companion floor
  removes that class (3 → 0 clips) and costs 15 clips, all ≤ 0.67 px/source-frame; at
  15% it would start deleting genuinely moving clips.
- **Head pairs with `bg_cov_pct < 5` are invalid** (was 10 until E4, 2026-09-16). The
  band [5,10) is only 5.66% of head pairs and moving the threshold changes the head
  `actor_motion_raw` median by 0.00% on a fixed clip set — the gate does not bias
  values, it **discards whole clips**, and 5 recovers 15 head clips (+6.2%). 20 does
  shift values (per-clip max +34%).
- **These gates are applied at table time, not at scoring time.** `actor_per_frame`
  stores every per-pair input and `continuity_per_triple` every per-triple gain/copy
  error, so `table_dense_nr.subject_metrics()` recomputes ActorActivity, active rate and
  AMC exactly from the record (validated: max abs deviation 7e-15 vs stored values over
  1,520 clips). Changing a gate therefore needs **no re-scoring** and does not fork the
  protocol version — records from any v3.x aggregate under one consistent set of gates.
  `actor_flicker_raw` is the exception: no per-pixel *pixel arrays* are stored, so the
  statistic itself cannot be recomputed from a record. It is however now stored
  **alongside 8 alternative normalizations and 4 per-pair diagnostics**
  (`actor_flicker_variants`: `flownorm`, `gradnorm`, `unexplained`, `shiftmin`,
  `sub_{0.05,0.1,0.2,0.4}`, plus `copy`/`flow`/`grad`/`align_pred`), all computed in the
  same probe pass. That made the E3 candidate comparison — and a further zero-GPU
  search over 7 derived candidates — reconstructible from the record without re-scoring,
  the same principle as the gates above.
- ActorActivity is comparable only inside one `(model, embodiment, camera_mount)`
  stratum: `source_interval` is 1 for GT/Ctrl-World, 2–3 for DreamDojo, 3 for
  DreamGen, so px/source-frame spans different real time windows.
  `actor_motion_px_per_second` is the cross-fps-safe variant.
- **On `head` mounts, AMC and VisualIntegrity are motion-confounded** (E1 criteria 3-4
  at full n: `rho(actor_motion_raw, ·)` = **−0.541** and **−0.553**, vs a <0.35 bar;
  static/wrist pass). Head is the only mount using ego-translation removal. Report those
  two columns per mount with head flagged, or exclude head — never pool head into a
  single AMC number. `table_dense_nr.CONFOUNDED_MOUNTS` stamps the warning on head rows
  of **both** the GT-delta table (T2) and the accel-vs-dense table (T3).
- **VisualIntegrity is descriptive-only for MODEL ranking but claim-eligible for the
  ACCELERATION comparison.** Its GT-is-ceiling failure (E1) is an insensitivity to the
  generative-vs-real gap, and that bias cancels when a method clip is differenced against
  its own dense counterpart; E7 measured VI as the most sensitive *correct* detector of
  the spatial artifacts accel methods actually produce (blur −6.79, block −12.18, still
  resolved at realistic severity −0.93 [−1.23, −0.28]). Encoded as
  `CEILING_FAILURE_CANCELS_VS_DENSE` in `table_dense_nr.py`.
- **An acceleration method's reference is the DENSE prediction of the same episode+view,
  never GT** (PhyFid-DR). Pairing is on
  the method-free `pair_key`, keyed by `(method, model)` — an accel row carries the *same*
  `model` as its dense counterpart, so keying by model alone silently overwrites the
  dense row and pairs the wrong clip.
- **The headline accel table is T3b (`table3b_accel_pooled.csv`), pooled per method; T3
  (`table3_accel_vs_dense.csv`) is the per-cell breakdown only.** Measured on the full
  9-method wave: 30 clips/method spread over (model × embodiment × mount) gives 9–17
  cells at n=1..7, and exactly 1 of 124 rows resolves. Pooling is legitimate here because
  every delta is *within* one episode+view+geometry, so no absolute level crosses
  embodiments — the same reason `NOT_CROSS_EMBODIMENT` does not apply to T3b. Both tables
  are built from one `accel_deltas()` call, so they cannot disagree.
- **A claim requires the bootstrap CI to STRICTLY exclude 0 *and* Wilcoxon p < 0.05
  *and* n >= 8** — one shared `resolved_flag()` for T2/T3/T3b. The CI alone is not enough:
  a discrete tie-heavy column can produce a CI bound exactly on 0 (Active Rate,
  `[−0.059, 0.000]`), and a skewed delta distribution can give a CI excluding 0 with
  p=0.262 (pisa VisualIntegrity). Below n=10 the normal-approximation Wilcoxon is
  unavailable and the CI decides alone.
- **External validation of the accel columns**: pixel-space sharpness (E7b, no NR model
  involved) rank-correlates with the pooled NR deltas at rho +0.67 (VisualIntegrity),
  +0.70 (AMC), +0.63 (ActorActivity), and every resolved accel delta is negative. Quote
  this when asked whether the NR metrics measure real degradation.

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
