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
- **A paired delta requires the two clips to be the same MEASUREMENT, not just the same
  episode+view.** `comparable_pair()` drops a pair unless `source_fps`, `effective_fps`
  match exactly and sampled length is within 10%. Found 226/1836 accel pairs violating it,
  216 of them the entire `dreamgen/single_arm` cell (dense 277 frames @ 16 fps vs accel 93
  @ 10 fps at the same relative path), which was faking a +10 AMC gain for all 9 methods
  and caused the one remaining ceiling violation in T3b. Applies to T2 as well as T3/T3b.
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

## Reporting scale for the single main-results slot (2026-09-17)

The 5 raw columns are on incompatible scales (VI 13–60, AMC 19–87, TextAlign 16–36,
ActorActivity 0–27 px/frame, ActiveRate 0–1), so a reader cannot weigh "−1.22 VI"
against "−0.005 ActiveRate", and 5 matrices do not fit one slot.

### Upstream conventions (checked)

- **CLIPScore** is *defined* as `max(100·cos(E_I, E_C), 0)`, bounded 0–100 (TorchMetrics).
  Our 16–36 range is the correct native scale; do not rescale it.
- **CLIP-IQA+** returns a softmax probability over ("Good photo.", "Bad photo."), i.e.
  natively 0–1. The source paper only ever reports SROCC/PLCC, never a per-image score,
  so there is **no published convention** for reporting the raw value. Our ×100 is a
  presentation choice, not a standard.
- **WorldScore** (ICCV'25, the lineage of this benchmark) hits the same wall and solves it
  by **linear normalisation to empirical bounds then ×100**; for the four metrics with no
  natural bounds (content alignment, subjective quality, motion accuracy/magnitude) it
  uses **z-score rescaling so the surveyed models land in 25–75**.

### Why WorldScore's recipe is NOT copied here

WorldScore normalises **absolute** scores to rank models. Every claim here is a **paired
delta against the method's own dense counterpart**, which is already self-normalising —
adding empirical bounds would layer an arbitrary transform on top of a difference. And
their z-score trick fixes the bounds to the surveyed model set, so adding one method
silently moves every published number.

### Options rejected by measurement, not by taste

- **GT-headroom** ("what fraction of the dense→GT gap does this method throw away") is the
  obvious candidate since we uniquely have both anchors. Measured per config the gap is
  0.2–2.6 points and **often the wrong sign** (VI: dense is *above* GT by +0.53 on
  ctrlworld/bimanual; AMC by +3.93 on ctrlworld/humanoid), and **5 of 13 configs have no
  GT anchor at all**. A near-zero, sign-flipping denominator makes the ratio meaningless.
- **Averaging the composite over cells to rank methods.** Spearman vs the pooled T3b
  ordering is **+0.883, not +1.0** — it moves `radial` from best (T3b dVI −0.07) to 6th,
  because methods are not run on identical config sets and thin cells get equal weight.
  Rank methods with T3b (pooled over *pairs*); use the grid to show *where* a method
  breaks.

### What is reported

`Retention % = median over paired clips of (accel / dense)` per column, then the mean over
the three **directional** columns (VI, AMC, TextAlign). Dense = 100 by construction, so
one number per cell is readable and unit-free. Two rules:

1. **ActorActivity and ActiveRate never enter the composite.** They are descriptive — a
   method that moves the arm *more* is not better. They are a separate behaviour check,
   and that is what isolates ITM/SiTo's distinct failure (ActorActivity retention down to
   **40%** on ctrlworld/single_arm: the motion itself is destroyed, not just blurred).
2. **`n < 8` gets its own symbol (`?`/`*`), never `ns`.** 11 of 37 unresolved cells are
   unresolved *only for lack of data*; printing them as "no effect" asserts an untested
   null. Before this fix the whole `DreamGen/Bimanual/SV` row read `ns` for all 9 methods,
   which looks like "all methods are free here" — it actually means "n=1–4, nothing tested".

Rendered candidates live in `scripts/table_main_options.py` (retention, loss index, win
rate, two-row per-metric, severity tiers, transposed).

## Storing BOTH views: unpaired absolute + paired delta (`scripts/table_paper_main.py`)

Both are kept, because **scoring is already unpaired** — `run_dense_nr.py` scores every
clip independently and writes one JSON record per clip. Pairing is only a join on
`(pair_key, model)` plus a subtraction, performed at table time. Verified on this run:
for all **1,610** comparable pairs the value used by the paired table is byte-identical to
the value the unpaired table reads off disk (0 differ), e.g.

```
dicache  humanoid/.../episode_000004/grid_top
  dense JSON visual_integrity = 26.0063
  accel JSON visual_integrity = 23.1634
  paired delta                = -2.8429   <- a subtraction, not a re-score
```

So keeping both costs **0 GPU-seconds** and they can never disagree about a clip.
Cross-checked against the existing T3b table: 9 methods x 5 columns, **0 mismatches** on
delta / n / resolved.

### This mirrors the field standard, with one addition

PAB, FasterCache (ICLR'25 Tab.1), TeaCache (CVPR'25 Tab.1) and Δ-DiT all report:

```
Method | MACs | Speedup | Latency | VBench (reference-free) | LPIPS / SSIM / PSNR (vs dense)
dense  | 6.30 |   1x    | 192.07  | 78.79%                  |   -   /   -   /   -
```

Their LPIPS/SSIM/PSNR are computed *against the original model's own output on the same
prompt* — already a paired comparison, then averaged, with `-` on the dense row. VBench is
the unpaired column (mean over ~72–100 prompts per dimension, then min–max normalised +
weighted for the aggregate). Mapping: their VBench = our absolute NR medians; their
PSNR-vs-dense = our paired NR deltas. What we add is a bootstrap CI, a Wilcoxon p and an
n>=8 floor, which those tables do not carry.

Measured from their own published numbers, the unpaired column barely separates methods
while the paired one does:

| base model | VBench spread across methods | PSNR spread | paired separates |
|---|---|---|---|
| Open-Sora 1.2 | 2.19 pt (2.8%) | 10.79 pt (39.9%) | **14.4x** |
| Open-Sora-Plan | 2.48 pt (3.1%) | 8.30 pt (35.0%) | **11.3x** |
| Latte (FasterCache) | 1.04 pt (1.3%) | 9.23 pt (32.7%) | **24.2x** |
| Latte (TeaCache) | 4.27 pt (5.5%) | 13.44 pt (60.8%) | **11.0x** |

On Latte all five methods sit inside VBench 76.01–77.05 with no error bars, while PSNR
ranges 18.98–28.21. Same phenomenon we measured locally (dropping pairing loses 63% of
resolved effects); those papers absorb it by letting the paired column do the
discriminating.

### The unpaired table is emitted PER CONFIG and must not be pooled for ranking

Rebalancing the config mix does **not** rescue it (Visual Integrity, 9 methods):

| unpaired ranking basis | Spearman vs paired ranking |
|---|---|
| pooled over ALL configs | **+0.883** |
| pooled over the 6 configs shared by all 9 methods | **+0.833** (worse) |

Restricting to the common 6 configs halves n (dicache 252 -> 108); the noise that adds
exceeds the config-mix bias it removes, so the dominant error is **between-episode
variance inside one config**, not the mix. Concretely: pooled-over-all promotes svg1 to
best (absolute 29.30) though its paired delta ranks 7th (-0.48), and common-6 demotes
radial from best (-0.07) to 5th.

Therefore pooled unpaired rows are written but tagged `pooled_unpaired_is_confounded=True`;
the per-config rows are the readable ones, and the paired table stays the only ranking and
claim surface. Config coverage for the record: dicache/worldcache 13 configs, fastercache
8 (no `ctrlworld_4view_grid` at all), itm/sito skip every dreamdojo config; only 6 configs
are shared by all 9 methods.

Outputs: `<run>/tables/paper_unpaired.csv` (516 rows, incl. dense and GT-ceiling rows) and
`<run>/tables/paper_paired.csv` (464 rows, per-config + pooled).

### The paper grid: rows = method x model family, cols = speedup + every metric

`scripts/table_paper_main.py` also emits the layout meant for the paper slot — one block
per method, split into its model families, with **all five metrics as columns** next to
speedup, so no single metric hogs the table:

```
method       model         speedup   VisInteg  TextAlign   ActorAct    ActRate        AMC     n
DiCache      Ctrl-World      1.11x     -1.53*    -0.092*      -0.02     +0.000     -1.58*   132
             DreamGen        2.28x     -1.57*    -0.672*      -0.10*    -0.066*      -0.01    60
             DreamDojo          --      +0.08         --      -0.04     -0.005      -0.14    35
ITM          Ctrl-World      1.25x     -6.76*    -0.828*      -1.12*    -0.082*     -5.54*   108
             DreamGen        0.96x     -5.48*    -1.902*      -0.27*    -0.110*    -10.16*    60
```

Model **families**, not raw model fields: `ctrlworld` and `ctrlworld_4view_grid` are the
same model under two view layouts, so they collapse to one Ctrl-World line.

Pooling over embodiment / mount / view inside a family is safe **for the paired view and
only for it**: each delta is accel minus its own dense clip, so it carries no absolute
level and the NOT_CROSS_EMBODIMENT problem (humanoid AMC ~18 points below the others)
cannot leak in. The absolute variant does mix those levels, so it is written separately
with the confound flag and TextAlign withheld.

Speedup is joined per `(method, family)` from `metrics/reports/speedup_summary_full.csv`
using only rows with measured timing; `timing_source=unavailable` cells print `--` rather
than borrowing a number from another family. Coverage is genuinely partial — Ctrl-World has
no measured timing for PISA/Radial/SVG1/SVG2/FasterCache, DreamDojo none for
DiCache/WorldCache/FasterCache — so the `--` are real gaps, not formatting.

Checked while building it: the DreamDojo column shows several POSITIVE VisualIntegrity
deltas (dicache +0.08, worldcache +0.21, radial +0.18), which looks like the render-mismatch
bug signature. It is not — all are unresolved noise around zero (p = 0.896 / 0.322 / 0.220,
every CI spanning 0) and `comparable_pair` had already rejected the only 3 length-mismatched
dreamdojo pairs. No ceiling violation.

Outputs: `paper_by_method_model_paired.csv` and `paper_by_method_model_absolute.csv`.
