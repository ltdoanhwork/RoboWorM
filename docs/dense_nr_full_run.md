# Dense NR full run — 2026-09-13

## Scope and execution

The frozen manifest at `results/dense_nr_v1/manifest.json` contains 3,714 prediction
videos, expanded into 9,126 view-clips. It covers DreamGen, DreamDojo, Ctrl-World,
and Ctrl-World 4-view-grid, across embodiments and both dataset splits. Input/GT,
comparison panels, intermediate segments, and the inactive DreamGen single-arm
grid quadrant are excluded. This is a first full evaluation pass, not a completed
paper result.

The `dense_nr_full` tmux session runs three persistent workers on GPU 0. Workers
load their metric models once and process disjoint manifest slices sequentially.
They write independent, atomic JSON results; resuming skips successful records.
Existing inference jobs are preserved. GPU utilization depends on decode,
detection, tracking, and flow stages and cannot be guaranteed at 100% continuously.

```bash
bash scripts/run_dense_nr.sh --workers 3
tail -f results/dense_nr_v1/worker_0.log
```

Each worker log records START/DONE plus scores. Failed clips receive a traceback
under `results/dense_nr_v1/errors`; workers continue to the next clip. A worker
startup failure is recorded by its log and nonzero exit code. Inspect error files
and coverage before interpreting any finished queue as full metric coverage.

## Verified launch snapshot

At handoff, `dense_nr_full` and all three workers were live on GPU 0. GPU
utilization reached 99–100% across repeated samples, with approximately 26 GB
allocated. Five view-clips had completed with all six metric fields and no error
records; one was object-only and excluded from subject-based plots. The queue was
still running. This snapshot is not a final completion count.

The figure exporter was exercised on those real partial results; PDF, SVG, PNG,
CSV, and coverage JSON were produced. The six-panel PNG was visually inspected.
Numerical tests passed for identical frames, brightness flicker, inconsistent
flow, exact translation, and empty subject overlap. Python compilation, shell
syntax, and diff whitespace checks passed.

## Protocol

- Six selected metrics: VisualIntegrity, EntityPresence, TaskLock, ActorMotion,
  MotionContinuity, and experimental ActorFlicker.
- Prediction views are evaluated separately, resized to 256×192, with integer
  temporal subsampling targeting approximately 5 FPS. Actual source/effective FPS,
  selected source frame indices, and interval are retained in every record.
  Aspect-ratio resampling and temporal subsampling limit interpretation; this pass
  can miss flicker above the sampled temporal bandwidth.
- Exclude the initial frame conservatively. Internal conditioning boundaries in
  segmented outputs are not yet reconstructed; this remains a protocol limitation.
- Quality probes run every fifth analysis frame; continuity uses all available
  odd-frame interpolation triples; motion/flicker use all adjacent analysis pairs.
- ActorMotion records analysis px/frame, px/source-frame, and analysis px/second.
  Compare strata with compatible view, resolution, and frame rates.
- Subject masks are seeded by GroundingDINO and propagated by SAM2. Missing robot
  detection is recorded as object-only or invalid; figures exclude these records
  from robot-motion/flicker distributions. Wrist views remain separate, since
  global translation compensation does not remove rotation or parallax.
- Missing task instructions yield TaskLock N/A. The current manifest has 756
  DreamDojo view-clips and one DreamGen view-clip without extracted task text.
  EntityPresence on those clips checks embodiment phrases only.
- ActorFlicker is an experimental motion-aligned RGB absolute-residual proxy,
  averaged over valid subject pixels and frame pairs. Forward/backward flow
  consistency, image bounds, and next-frame subject masks gate valid pixels.
  It is not an exact RoboPhys-3D implementation or a validated high-frequency
  temporal metric. Lighting changes and flow errors can affect it.
- The existing MotionContinuity backend uses global image SSIM, not a local-window
  SSIM implementation. Retain this implementation identity in paper reporting.
- Fixed random/model determinism across repeated runs has not been established.

## Paper figure preparation

After workers exit, the queue invokes:

```bash
models/dreamgen/.venv/bin/python scripts/plot_dense_nr.py --run results/dense_nr_v1
```

The same command can produce interim figures while workers run. Exports include
vector PDF/SVG, 300-DPI PNG, exact plotting CSV, and metric coverage JSON.

- Six-panel box plots for each model/embodiment/split, with separate view
  distributions and valid sample counts. Incomplete runs are labeled partial.
- Motion-versus-continuity diagnostic scatter, using analysis px/second.
  Pooled views are explicitly labeled as diagnostics, not model rankings.
- Coverage plot to expose missing values rather than imputing them.

Do not claim dense outperforms acceleration methods from dense-only results.
Do not use raw motion magnitude as an unbounded higher-is-better quality score.
Before publication, visually audit low/high cases and validate flicker against
human judgments; numerical smoke tests alone do not establish metric validity.

## v2 paper protocol queue

`dense_nr_v2_queue` is queued behind the frozen v1 job and launches
`results/dense_nr_v2` (now archived in `tmp/20261005_refactor/misc/worldscore_results/`) only after v1 exits. v2 fixes source-frame normalization
for activity thresholds, adds p10 VisualIntegrity, localized motion p95,
activity-weighted continuity, and static-camera drift. It omits text scores for
image-conditioned methods. Its paper columns and units are frozen in
[nr_paper_output_schema.md](nr_paper_output_schema.md).

Entity coverage and task-state success remain withheld from v2 headline tables:
their closed-world task registry has not yet been authored for all 57 templates.
The existing phrase detector value is retained only as a debugging proxy.

## v3 (2026-09-15) — supersedes v1/v2

v1 (3,153/9,126) and v2 (2,232/9,126) were stopped; partial results stay on disk.
v2 audit on 2,232 bimanual clips found:

- Camera drift tail was mask coverage (~98% subject mask), not camera motion. Dropped.
- Global-SSIM continuity correlated −0.68 with actor motion. Replaced by
  motion-normalized VFI gain over frame copy.
- Ego compensation on static cameras subtracted deformable-object motion. Static
  mounts now use uncorrected subject flow; head/wrist mounts require >= 25% background.
- `actor_motion_active_rate` was computed but not saved. Now saved.
- Triple/pair alignment used list position although probe skips pairs. Now keyed by pair.
- 17/19 v2 errors were CUDA OOM from v1+v2 sharing GPU 0. Workers now retry OOM.

v3 adds GT view-clips from Ctrl-World inputs (all three embodiments, 5 fps) to the
manifest ahead of predictions, scored with the identical protocol.

```bash
# smoke validation (paired GT/pred episodes + every group)
bash scripts/run_dense_nr.sh --out results/dense_nr_v3_smoke --workers 3 --smoke-paired-episodes 4
# full run
bash scripts/run_dense_nr.sh --out results/dense_nr_v3 --workers 3
```

## v3.1 (2026-09-15) — code review + div30 job audit

v3.1 fixed four v3-smoke findings (static p95 + 1 px active threshold, wrist
whole-frame median, task-clause-only CLIPScore, best-of-four arm seed frame with
bidirectional SAM2 propagation). The diversity run

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_dense_nr.sh --out results/dense_nr_v3_div30 \
  --workers 3 --smoke-paired-episodes 10 --smoke-per-group 30
```

is 1,047 view-clips over 28 `model x embodiment x camera_mount` groups (~11 h at
40 s/clip effective on 3 workers, 120 s median per clip). It was audited at 62
completed clips, 0 errors, 3 live workers, GPU 0 at 99%.

### Confirmed working in v3.1

- **All four v3-smoke fixes hold on GPU.** Seed frame is chosen per clip (observed
  0, 29, 40, 41, 58, 60, 62, 82, 83, 87, 123, 124 — 25/39 non-zero), robot
  detection is 100% (`actor_validity: robot_detected` on every clip, vs 8% failure
  in v3 smoke). Static-mount motion moved from ~0.05 to a median 4.7-5.3
  px/source-frame with active rate 0.6. CLIPScore now sees only the task clause
  (`fit small gear shaft`), scoring 23-31 for both GT and pred.
- **GT is the ceiling on every column, on paired episode-views**: GT better in
  87% of pairs for ActiveMotionContinuity (median +2.76), 70% for VisualIntegrity
  (+1.00), 72% for ActorFlicker (-0.0010). This is the plausibility anchor v3 was
  built for, and it holds.
- **ActiveMotionContinuity is decorrelated from motion amount**: rho = -0.27 vs
  actor motion, where the v2 global-SSIM continuity it replaced is still -0.54 on
  the same clips (kept as `motion_continuity_global_ssim`). The redesign works.
- **p95 is doing real work and is not a mask-area artifact**: p95/median ratio is
  6.1x median (up to 77x) on static mounts, and `corr(fg_cov_pct, ratio) = +0.03`
  — the ratio is driven by localized gripper motion, not by how big the mask is.
- **Resumability and OOM retry**: atomic `.tmp`-then-replace writes, per-id result
  files, `dest.exists()` skip, 4-attempt OOM backoff. 0 errors in 62 clips.

### Bugs found in the audit and fixed

1. **`plot_dense_nr.py` silently fell back to the v1 column set.** It selected V2
   keys only on the exact string `dense-nr-v2-closed-world`, so `v3.1` records got
   V1 keys: the CSV and all box plots emitted `entity_presence` (always `None`
   since v3) and `motion_continuity` (removed in v2) at 0 valid observations, while
   `visual_integrity` / `active_motion_continuity` / `actor_motion_active_rate`
   were not plotted at all. This runs automatically when the queue finishes, so the
   whole 11 h run would have produced v1-shaped figures. Now keyed on schema
   (`'visual_integrity' in r`), verified: coverage reports all six v3 keys at 56/56.
2. **`actor_motion_valid_pair_frac` can never fail its own gate.** It is
   `len(motion) / len(per_frame)`, but the probe already dropped unusable pairs
   before returning `per_frame`, so the value is 1.00 on every clip including one
   where SAM2 tracked **1 of 163** pairs (0.6%). Added `actor_tracked_pair_frac`
   measured against the analysis pairs that actually exist, and gated `motion_ok`
   on it at `MIN_VALID_PAIR_FRAC`.
3. **AMC weight sum was unbounded below.** The weighted mean is scale-free, so one
   marginal triple scores the same as a hundred. The near-static wrist clip above
   (motion 0.35 px, active rate 0.00, `weight_sum` **0.35** over 81 triples) was
   reported as AMC **83.1** — the highest continuity in the cohort — while its
   unweighted mean gain was 48.6. Added `MIN_WEIGHT_SUM = 3.0`; that clip becomes
   `None`, healthy clips (`weight_sum` 16-79, median 41) are unaffected.

Regression-checked on the stored per-clip records: the anomaly clip now yields
`tracked_frac 0.006 -> motion_ok False` and `weight_sum 0.35 < 3.0 -> AMC None`,
while `weight_sum 34.8 / tracked 1.000` stays valid.

### Known limitations left open (not bugs)

- **Cross-fps comparability.** `actor_motion_raw` is px/source-frame and the
  analysis grid is fixed at 5 fps, so `source_interval` differs per model: GT and
  both Ctrl-World variants 5 fps/interval 1, DreamDojo 10 fps/interval 2 (single_arm
  15/3), DreamGen 16 fps/interval 3. Division by `interval` normalizes the unit but
  the underlying displacement is measured over a 3x longer real interval for
  DreamGen. Only compare ActorActivity within one `(model, embodiment, mount)`
  stratum, as the schema already requires; `actor_motion_px_per_second` is the
  cross-fps-safe field.
- **ActorFlicker still tracks motion** (rho = +0.45, was +0.75 in v3 smoke).
  Remains `experimental`.
- **`MIN_BG_COV_PCT = 10` is currently inert on the audited mounts.** Wrist uses
  whole-frame median (no gate) and static is uncorrected; measured background
  coverage is p1 = 14.8% on wrist and median 70.7% on static. The gate only binds
  on `head` mounts, which start at manifest index 80 and had not been reached.
- **The GT anchor is Ctrl-World 5 fps inputs only**, so the paired GT/pred
  comparison exists for `ctrlworld*` rows; DreamGen/DreamDojo rows have no
  same-episode GT partner and are compared to GT only within-stratum.
- **DreamDojo has no instruction** for all 90 of its clips, so TextAlignment is
  N/A there by design (`text_condition_applicable`).
- **4 `(pair_key, model)` collisions** in the manifest (DreamDojo singleview
  `full_pred`, two DreamGen wrist rows). `id` is unique so nothing is overwritten,
  but those `pair_key`s cannot be used for pairing.
- Manifest order groups all `ctrlworld*` bimanual first, so early results are
  bimanual-only; per-group conclusions cannot be drawn until the shards pass
  index 539 (DreamGen) and 599 (DreamDojo).

## v3.2 (2026-09-15) — power analysis, per-model GT anchor, paper tables

Scoping pass driven by one question: *which of these numbers can actually go in a
paper?* The answer changed the protocol.

### The unpaired comparison has no power at this n; the paired one does

Measured on the 166 v3.1 clips, GT vs pred, same metric, same clips, two tests:

| metric | paired GT−pred | unpaired medians |
|---|---|---|
| VisualIntegrity | **+1.00, 95% CI [+0.56, +1.60]** → resolved | GT 32.53 [31.04, 33.60] vs pred 31.66 [29.71, 33.11] → **CIs overlap** |
| ActiveMotionContinuity | **+1.78, 95% CI [+0.01, +3.05]** → resolved | GT 60.92 [58.24, 65.33] vs pred 59.63 [57.43, 61.17] → **CIs overlap** |

True effects are 1–3 points while the bootstrap CI half-width of a median at n=30
is 1.3–2.2 (VisualIntegrity) and 2.7–3.9 (AMC). Resolving a 1-point effect by
comparing independent medians needs n≈100 per stratum; the paired delta resolves it
at n=20. **So the paper's headline column must be the paired GT-delta, not a
cross-model ranking of medians.** Reporting bare medians as a ranking would be
claiming differences the data cannot support.

### The old GT anchor could not reach 2 of 4 models

`GT_SOURCES` drew GT from `input/multiview/ctrlworld*` (5 fps). Episode-id overlap
with the other models: DreamGen **0/210** clips, DreamDojo **0/90**. The one test
with power was unavailable for the two headline models.

Fix: score the **co-located GT** that already sits next to every prediction
(`full_gt.mp4` / `gt_all_views.mp4`) as `gt_<model>`. Verified with `ffprobe` that
the sibling matches the prediction in **width, height, fps AND frame count**
(DreamGen 768x432@16 both, 535 frames both; Ctrl-World 1280x192@5, 165; DreamDojo
640x480@15, 565) — so `prepare()` samples the identical frames and the delta
carries no fps/crop caveat. This also retires the cross-fps limitation for the
paired column. Pairing coverage went 0% → **100% ctrlworld / 100% 4view_grid /
58% dreamgen / 32% dreamdojo**, with **0** `(pair_key, model)` collisions.

### Two pairing bugs that were silently merging clips

`row_for` built `pair_key` with `next(x for x in reversed(rel.parts) if x.startswith('episode_'))`:

1. **DreamGen** stores `<task>/episode_000004/full_pred.mp4` → key was the bare
   `episode_000004`, which 48 bimanual tasks reuse. Different tasks collided.
2. **DreamDojo** stores `<task>__episode_000039/` → no part startswith `episode_`,
   so it fell through to `rel.stem` and **every** DreamDojo clip keyed on
   `full_pred`. This was the source of the 4 collisions noted in v3.1.

Both fixed by a shared `episode_key()` that returns `<task>/<episode>`, now also
used by `task_of` so the two cannot drift apart.

### Sample selection is now clip-balanced, not episode-balanced

`smoke_manifest` took a fixed episode count per group, but views-per-episode ranges
from 1 (DreamDojo singleview) to 4 (grid/strip) — DreamDojo got 14 paired clips vs
Ctrl-World 65 at the same setting. Power tracks paired *clips*, so it now adds
episodes until a clip target is met: **34–57 paired clips for all four models**.

### Manifest is walked in id order (fail-fast)

The manifest is grouped, so in manifest order `head` mounts start at index 80,
DreamGen at 539 and DreamDojo at 599 — three code paths that had never touched a
GPU after 8 h of running. Sorting by sha256 `id` is a deterministic, stateless
shuffle covering all 5 models and 3 mounts within 56 clips and all 28 groups within
140. Confirmed live: DreamGen and DreamDojo both scored within the first 8 clips.

### Paper tables: `assets/scripts/table_dense_nr.py`

Emits two tables, deliberately separated by how much they can claim:

- **T1 descriptive** — median + 95% bootstrap CI + n + exclusion rate per
  `(model, embodiment, camera_mount)`. Never pooled across mounts or fps. Read as
  "what the output looks like", not a ranking.
- **T2 paired** — pred minus its own co-located GT, median + CI + `gt_better_frac`
  + Wilcoxon signed-rank, plus an explicit **`resolved`** flag (CI excludes 0). A
  claim is only permitted on a resolved row.

Gates are re-applied **post-hoc** from the stored record (`actor_tracked_pair_frac`
or `actor_per_frame` vs `sampled_source_frames`, and `..._weight_sum`), so v3.1 and
v3.2 clips aggregate identically and a gate change needs no re-scoring. Per-column
`digits` is required: ActorFlicker lives at ~0.01 and printed every delta as `0.00`
at 2 decimals, hiding two resolved effects.

### Operational gotcha: `pkill -f run_dense_nr.py` leaves the supervisor

The queue is a supervisor that respawns shards, and `nohup bash -c ... &` adds a
wrapper, so pattern-killing the workers let the supervisor start a **second** set —
10 workers wrote into one results dir. Launch with `setsid` and kill the whole
process group (`kill -- -$PGID`); verify with
`ps -eo pid,ppid,pgid,args | grep run_dense_nr.py` that exactly one supervisor and
`--workers` shards remain.

### Capacity

Per-worker GPU footprint is ~13.9 GB, so 5 workers fit in 80 GB (~70 GB observed
peak 44.7 GB) versus the previous 3. 1,573 clips at ~120 s/clip median ≈ **10.5 h**.

### Next wave (decided, not yet run): acceleration methods

Full NR scoring of the 9 acceleration trees is **not feasible**: 80,174 prediction
videos ≈ 173,913 view-clips ≈ **48 days** on one GPU at the measured 24 s/clip
(5 workers). Agreed scope is **30 clips per (method, embodiment, camera_mount)** ≈
5,670 clips ≈ **1.6 days**.

Design constraint carried over from v3.2: the claim must be a **paired delta against
the `dense` prediction of the same episode+view** (not against GT), matching the
PhyFid-DR rule that a speedup method's reference is the dense pred. The unpaired
median comparison has no power at n=30, and an acceleration method's expected effect
is *smaller* than the GT−pred gap, so pairing is mandatory rather than optional.
Prerequisite: `dense` must be scored for exactly those episode+views, so this wave
runs only after the current dense+GT run lands.

### v3.2 audit at 155/1573 clips — 3 findings

First audit with DreamGen/DreamDojo and the per-model anchors actually on GPU.

**PASS — mount logic and both v3.1 gates work.** Robot detected in 155/156. The
`tracked_pair_frac < 0.5` gate fired on 2 real tracking failures (0.15 and 0.48) and
the `weight_sum < 3` gate on 9 clips, all at scoring time. Motion medians order as
physics demands: `head` >> `static` > `wrist` (13.3 / 3.7 / 2.7 px per source frame
for ctrlworld) — head mounts sweep the whole frame, wrist mounts ride the effector.

**PASS — AMC is decorrelated from motion amount, and it is the normalization that
does it.** `rho(motion, AMC)` = −0.35 static / −0.24 head / **−0.02 wrist**, while the
v2 diagnostic `motion_continuity_global_ssim` on the *same* clips is −0.65 / −0.58 /
−0.51. Keeping that column is what proves the fix is real and not a reweighting.

**BUG 1 (fixed) — the new anchors were marked not-text-conditioned.**
`TEXT_CONDITIONED_MODELS` is matched by exact model name, so `gt_dreamgen`,
`gt_ctrlworld`, ... all evaluated `text_condition_applicable = False`:
TextAlignment was `None` on **every** anchor, which would have left the paired
TextAlignment column permanently empty while looking like a legitimate N/A. The
anchors are real videos of the same instructed task. Fixed with `text_conditioned()`,
which strips the `gt_` prefix. Pred-side values were never affected (ctrlworld 25.00,
4view 25.05, dreamgen 25.59, gt 25.47 — all in the healthy 23-31 band, so the v3.1
CLIPScore prompt fix still holds).

**BUG 2 (fixed) — `MIN_WEIGHT_SUM` cannot separate "few strong" from "many weak".**
A `gt_dreamgen` wrist clip reported **AMC 78.3 at active_rate 0.000** (motion 0.113 px
over 88 pairs). Weights are `min(motion/ACTIVE_PX, 1)`, so 88 × 0.113 = weight_sum
4.95 — above the floor, though **no pair was ever active**. This is the v3.1 AMC-83
bug resurfacing just under the threshold. Raising the floor is the wrong fix: a
genuinely active clip (17.6 px, 11/11 pairs active) also sums to only 5.0, so
`MIN_WEIGHT_SUM = 8` would delete the real signal and keep the artifact. Added an
independent `MIN_ACTIVE_PAIRS = 3` count of pairs at or above `ACTIVE_PX`, stored as
`actor_n_active_pairs` so it is reproducible post-hoc. Effect: drops the 2 rate=0
clips (1% of data), keeps the 11-active-pair one, anomaly count → 0.

**Known, documented, not fixed — ActorFlicker tracks motion.**
`rho(motion, actor_flicker_raw)` = **+0.77 static / +0.44 head / +0.55 wrist**, worse
on static than the +0.45 measured in v3.1. A fast-moving subject is not a flickering
one, so this column measures motion amount more than shimmer. It is already labelled
`experimental ... until human-calibrated` in the schema, so it must **not** be quoted
as a quality result in the paper — report it as a diagnostic distribution only, or
drop it. Fixing it properly needs the residual normalized by local flow magnitude,
which is a metric change, not a gate change.

**Sample-size caveat.** At 155 clips only 1 GT/pred pair had *both* halves scored
(forward and reverse queues reach the two members of a pair at different times), so
CHECK 1 (GT-is-the-ceiling) is not yet testable and cannot be judged from partial
results. It must be re-run at completion.

## v3.2 run landed (2026-09-16) — E1 audit: **VisualIntegrity fails GT-is-ceiling**

Run `dense_nr_v3_div30` finished: **1568/1573 clips**, 5 workers, ~21h wall,
`worker_exit_codes: [0,0,0,0,0]`. The 5 misses were 1 corrupt GT clip
(`ValueError: Invalid FPS`) + 4 CUDA OOM from 5 concurrent workers peaking
together; the 4 OOMs were re-queued afterwards at `--workers 2` and pass.

### E1 verdict: FAIL on VisualIntegrity, PASS on the rest

`CHECK 1` (GT must beat pred on the same episode+view) is the blocking gate.
Paired, co-located GT, so geometry/fps/frame-count are identical (verified:
median sampled-frame count and `source_fps` match exactly per model).

| model | n | median dVI (pred−GT) | GT better |
|---|---|---|---|
| dreamgen | 47 | **−2.02** | 77% ✅ |
| ctrlworld | 67 | −0.02 | 52% ✗ coin-flip |
| ctrlworld_4view_grid | 74 | **+0.52** | 39% ✗ |
| dreamdojo | 34 | **+0.90** | 24% ✗ |

Only DreamGen behaves. For DreamDojo the paired delta is **+0.90
[+0.27, +1.35], p=5.3e-05** — a *resolved, significant* result that the
generation looks better than real video. That cannot go in a paper as a
quality metric.

**Three mechanistic hypotheses were tested and all REJECTED:**

1. *Pairing is wrong / different clips.* No — sampled-frame counts and
   `source_fps` are identical within every pair.
2. *Pred moves less, so it is "easier".* No — `dMotion` is ≈0
   (−0.09 px for 4view static) while `dVI` is +0.78.
3. *Pred is smoother/blurrier and CLIP-IQA rewards clean frames.* No —
   `dSSIM` is only +0.07, and objective sharpness (variance of Laplacian,
   25 pairs) says pred is **blurrier** (median dSharp −3.9, pred sharper in
   only 9/25) while still winning VI by +0.97. Spearman(dSharp, dVI)=+0.31.

A visual mid-frame check on the largest dVI case (+4.75, `insert_washer_shaft`)
confirms pred is **not** perceptually better than GT — it is visibly softer on
two of four views. So CLIP-IQA+ is simply **not discriminative at this effect
size** on robot video: its pred−GT deltas are ~1 point with inconsistent sign
across models, i.e. domain bias, not quality.

**Consequence for the paper.** Do not present VisualIntegrity as a
quality/ranking column. Options, in order of preference:
(a) drop it; (b) keep it only as a descriptive "domain" statistic with the
GT-fails-ceiling caveat printed; (c) replace CLIP-IQA+ with a metric that
passes the ceiling test. Do NOT report the DreamDojo +0.90 as a win.

### What does pass

- **AMC**: no model beats GT with a resolved positive delta on any stratum
  where n is meaningful; GT-better rates are 52–61%, i.e. GT is at worst tied.
  Directionally correct, weak effect.
- **ActorFlicker**: GT better in 51–89%, deltas ~1e-4 = at noise floor. Still
  confounded with motion (rho=+0.70 static, +0.47 head) → E3 stands.
- **Gates fire as designed**: `visual_integrity` 1568/1568 valid,
  `active_motion_continuity` 1448/1568, flicker gate dropped 17.
  `MIN_ACTIVE_PAIRS=3` cut the v3.1 anomaly class from many to **3** clips.
- **TextAlignment**: applicable on all text-conditioned models incl. the new
  `gt_*` anchors (the v3.2 `text_conditioned()` fix works); DreamDojo is 0/117
  as expected (no prompt). Medians are 24.6–25.4 across *all* models and GT —
  a 0.8-point spread → E2 (saturation) is now the top-priority experiment.

### Sample-size reality check (affects the n=30/group decision)

The paired table stratifies by embodiment × mount, which **splits the 30/group
budget**: realized paired n is 1–34 per cell (e.g. `dreamgen humanoid head
n=1`, `ctrlworld humanoid head n=3`). Cells with n<8 produce no usable CI.
Either report at model×mount level pooled over embodiment, or raise the
sampling budget for the thin cells before claiming anything per-embodiment.

## E2 executed (2026-09-16) — ~~TextAlignment DROPPED~~ **REVERSED: kept as paired-only**

> **This section's conclusion was overturned the same day.** The shuffle design is
> caption-confounded; the double-centered matrix control shows CLIPScore *does*
> discriminate, and in the paired table the bias cancels exactly. See
> **"E2 reversed"** below. The measurement below is correct; the inference from it
> was not.

`scripts/_exp_text_shuffle.py` (+ `_exp_text_shuffle.sh` for the venv/PYTHONPATH).
Each clip scored twice on the SAME analysis frames: once with its own instruction,
once with an instruction from a **different task in the same embodiment**.
Calls `iqa.frame_scores()[2]` directly — never `score_qual` — so no GroundingDINO
cost and, crucially, no generic-caption fallback. Clips without a real prompt are
excluded up front: **1397/1571 clips have a real prompt** (the 174 excluded are
DreamDojo, which has no text).

| arm | median CLIPScore |
|---|---|
| matched (own instruction) | 24.781 |
| shuffled (other task's instruction) | 24.766 |
| **paired delta** | **+0.088 [−1.043, +0.930]** (n=100) |

Per model the delta is noise in both directions (`gt_ctrlworld` +3.13,
`gt_dreamgen` −2.74, `ctrlworld` −0.80), i.e. no consistent effect.

**Verdict: interpretation (b).** CLIPScore returns ~24.8 whether or not the caption
describes the video, so it cannot rank models — and it cannot even tell a *wrong*
task from the right one. Both pre-registered drop conditions are met (gap < 1.0
AND CI includes 0), so per the E2 protocol the column is **removed from the paper**.

Enforced in `assets/scripts/table_dense_nr.py`:
- `DROPPED` dict → the column is out of `COLUMNS`, its stale
  `table2_text_condition_alignment.tex` is deleted on every run, and the reason is
  printed at the top of the report and stored in `provenance.json`.
- `CEILING_FAILED` dict → `visual_integrity` is kept but can **never** be marked
  `RESOLVED` against GT, with the caveat printed in the report and as a `%` comment
  in its `.tex`. (Update 2026-09-20: the column's `direction` was originally also
  forced to `None`. It is now `True` — WorldScore's own CLIP-IQA+ polarity, higher =
  better quality — because polarity and permission-to-rank are separate things. The
  `RESOLVED` ban and the printed caveat are unchanged; only the `direction` line in
  the `.tex` now reads `higher is better` instead of `descriptive`.)

`run_dense_nr.py` still computes both metrics — the data is retained, only the
paper claim is withdrawn.

### What actually survives as a paper claim

After both audits, all `RESOLVED` rows come from **3 columns**: AMC, ActorFlicker
and ActorActivity(rate). Every one of them says GT ≥ pred, which is the correct
direction. Note ActorFlicker's resolved rows are ~5e-3 in magnitude and it is
still motion-confounded (rho +0.70) → **E3 is now the blocking experiment** for
that column; AMC is the only clean quality claim standing.

> **Updated after the E2 reversal and the criteria-3-5 pass:** TextAlignment is back
> as a **4th** claim column (paired only, GT better and RESOLVED on both ctrlworld
> variants). Conversely AMC is *not* clean everywhere — it is motion-confounded on
> **head** mounts (rho −0.54), as is VisualIntegrity (−0.55), so those two must be
> reported per mount with head flagged or excluded. Net: AMC (off head) + TextAlignment
> (paired) are the two defensible quality claims; ActorFlicker awaits E3;
> VisualIntegrity stays descriptive.

Final clip count after re-queueing the OOM failures at `--workers 2`:
**1572/1573** (the single remaining miss is the corrupt-FPS GT clip).

## E1 criteria 3-5 + E3/E4/E5 audit (2026-09-16, second pass)

Independent pass over the same completed run. Criteria 1-2 reproduce the section
above (VisualIntegrity GT-better **48.6%** of n=222 pairs, +0.030 [−0.182, +0.224];
AMC 55.1% of n=187, −0.400 [−1.041, +0.152]). Three criteria the first pass did not
score, and they change what is publishable.

### Criteria 3-4 FAIL on head mounts — the motion confound is not only ActorFlicker's

| rho vs `actor_motion_raw` | static | head | wrist | bar |
|---|---|---|---|---|
| `active_motion_continuity` | −0.325 (n=660) | **−0.541** (n=208) | −0.053 (n=568) | <0.35 |
| `visual_integrity` | −0.029 (n=702) | **−0.553** (n=209) | +0.297 (n=609) | <0.35 |

Both fail, and both fail *only on head*. The v3.1 note "AMC is decorrelated from
motion (rho −0.27)" was measured before head data existed; at full n the head
mount is at −0.54 for **both** columns. So on head views AMC and VisualIntegrity
both partly measure "how much the camera/subject moved", not quality. Head is also
the only mount using ego-translation removal. **Either report AMC/VI per mount with
head flagged, or exclude head from those two columns** — do not pool head into a
single AMC number.

### Criterion 5 FIXED: the AMC-78 artifact class is now 0 clips (was 3)

`MIN_ACTIVE_PAIRS = 3` is an absolute count, so it is nearly free on long clips: the
3 survivors had 79 / 177 / 245 valid pairs and cleared the gate with 3-7 active pairs
at `active_rate` 0.02-0.04, reporting **AMC 71.0 / 71.0 / 82.0**. Added a
length-aware companion `MIN_ACTIVE_FRAC = 0.05` (`n_active >= max(3, 0.05 * n_valid)`)
in `run_dense_nr.py` and, because gates are recomputed post-hoc, in
`table_dense_nr.gated()` — so all 1,572 already-scored clips are fixed **without
re-scoring**.

Calibrated on the 1,451 AMC-valid clips (drop = clips losing AMC, FP = a dropped clip
that actually moves at >= 1 px/source-frame):

| min_frac | kept | dropped | artifacts left | dropped with motion >= 1px |
|---|---|---|---|---|
| 0.00 (old) | 1451 | 0 | **3** | 0 |
| **0.05** | **1436** | **15** | **0** | **0** |
| 0.10 | 1396 | 55 | 0 | 0 |
| 0.15 | 1355 | 96 | 0 | **4** ← starts eating real motion |

0.05 is the exact knee: every one of the 15 clips it drops is <= 0.67 px/source-frame
(genuinely static), and no clip at >= 1 px is touched. Verified: criterion 5 now
reads `0 clips`, and paired AMC n only moves 191 → 187.

**Two reconstruction traps in that gate** (both cost a wrong answer on the first
attempt, both now handled in `table_dense_nr.n_active_pairs()`):
`actor_n_active_pairs` is **missing on 142 clips** (scored before the field existed) —
treating missing as 0 marks vigorously moving clips (rate 0.95, motion 9.5 px) as
inactive. And the reconstruction denominator is **not** `len(actor_per_frame)`:
`active_rate` is averaged over the bg-coverage-gated motion dict, which is strictly
smaller on head mounts — using per_frame made all 9 head clips look inconsistent.

### E4 answered with ZERO GPU — move `MIN_BG_COV_PCT` to 5, at table time

`actor_per_frame` already stores per-pair `bg_cov_pct` and `fg_flow_ego_p95`, and the
gate is applied *after* the probe, so it can be re-applied exactly post-hoc. **The
"re-score ~40 head clips at 3 settings" step in `dense_nr_experiments.md` is wasted
GPU time.** On all 241 head clips:

| `MIN_BG_COV_PCT` | usable head clips | None rate | median ActorActivity |
|---|---|---|---|
| 5 | **224** | 0.071 | 6.79 |
| 10 (current) | 209 | 0.133 | 6.25 |
| 20 | 183 | 0.241 | 6.62 |

On a fixed clip set the change 10 → 5 moves the head median by **0.00%** (max 0.00%),
far inside the <10% pass bar — because the discarded pairs sit at `bg_cov < 5`
(band [5,10) is only 5.66% of head pairs). The gate does not bias values, it
**discards whole clips**: 5 recovers **15 head clips (+6.2%)** for free. `thr=20` is
where it starts to matter (per-clip max +34%). **Verdict: set it to 5.**

**Applied at table time, NOT by re-scoring.** The change touches 29/1572 records
(1.84%) — 14 whose value moves, 15 recovered from `None`. Re-scoring 1,573 clips (~9 h)
for that, and splitting the results dir into two protocol versions right before E5 has
to pair method clips against dense clips, would be strictly worse. So
`table_dense_nr.subject_metrics()` now recomputes `actor_motion_raw`,
`actor_motion_active_rate` and `active_motion_continuity` from the stored per-pair /
per-triple records, and both new gates are applied there. **Validated exact** against
the constants the run itself used (1,520 clips): max abs deviation **7.1e-15** on
`actor_motion_raw`, **0.0** on `active_rate`, 9.9e-14 on AMC; the single disagreement is
the recompute correctly rejecting a stored AMC of 36.3 that had 0 active pairs. Only
1 of 1,572 records lacks `actor_per_frame`, and it has no stored value either — nothing
is lost. `PROTOCOL` therefore **stays v3.2**, and `MIN_BG_COV_PCT` / `MIN_ACTIVE_FRAC`
are now one-line switches in `table_dense_nr.py` verifiable without GPU:

```
bg gate 5  (now default)  head usable 224/241  head median 6.904  AMC valid 1447  artifacts 0
bg gate 10 (v3.2 value)   head usable 209/241  head median 6.253  AMC valid 1436  artifacts 0
```

`actor_flicker_raw` is the one column that cannot be recomputed (no per-pixel inputs
stored) — which is exactly why E3 still needs GPU.

Limitation that no amount of compute fixes: the paired GT-delta half of the E4
criterion has at most **n=12 head pairs (thr 5) / n=9 (thr 10)** in the entire
manifest (+0.014 [−0.857, +0.623] vs −0.694 [−2.442, +1.810]) — record as a
limitation, do not wait for more data.

### E3 pre-check: confirms per-pixel is mandatory (still needs GPU)

Reproduced on the full run, and the "obvious fix" is confirmed harmful:

| mount | rho raw | rho ÷motion | rho log-OLS residual (pair level) |
|---|---|---|---|
| head | +0.473 | −0.784 | **−0.071** ✅ |
| static | +0.695 | −0.794 | **+0.061** ✅ |
| wrist | +0.364 | −0.778 | **−0.257** ✗ |

New information: the confound is almost purely **multiplicative** (log-log slope
0.19-0.27), and the best possible *pair-level* correction (regress log flicker on log
motion, keep the residual) already brings head/static well inside the |rho|<0.25 bar
— but **wrist still fails at −0.257**. So no post-hoc transform of the stored
aggregates can pass criterion 1; the residual must be normalized by local flow
magnitude **per pixel inside the probe**, exactly as specified. E3 remains GPU work.

### E5 prerequisite verified — the pairing is there, budget is 9.4 h

Not 48 days: the constraint is that a method clip must pair with an **existing dense
NR result**, and dense results exist for the 1,573-clip div30 sample only.

- Method mp4s whose relative path also exists in `dense`: 100% for worldcache,
  dicache, itm, sito, svg2; 99.6-99.9% pisa/radial; 97.1% svg1; 89.3% fastercache.
- Against clips that **already have a dense NR result**: **0 of 63
  (method × embodiment × mount) groups is short of 30** pairable clips.
- Realistic budget **1,890 clips ≈ 49 GPU-h (≈9.9 h wall on 5 workers)**.

Remaining work for E5 is only the code change (parameterize `inventory()` by method
tree, restrict the sampler to episode+views with a dense result) — safe now that the
queue has exited.

### Operational notes

- **`queue_finished.json` existing does NOT mean the run finished.** At 07:37 on
  09-15 the file was already present with `worker_exit_codes: [-15,-15,-15,-15,-15]`
  — a SIGTERM'd queue. The real completion at 20:11 wrote `[0,0,0,0,0]`. Any gate on
  that file must check the exit codes.
- 4 of the 5 run errors were `CUDA OOM`, caused by the **superseded `dense_nr_v2`
  queue still running on the other host's GPU 0** (protocol `dense-nr-v2-closed-world`,
  which this doc records as stopped at v3). It had been holding 73 GB / 100% util for
  2 days and reached 4,019/9,126 before being stopped with `kill -- -<pgid>`. All 4
  OOM clips re-scored cleanly afterwards → **1572/1573, 1 permanent error (0.06%)**.

## E2 reversed (2026-09-16) — TextAlignment is NOT saturated; it is caption-biased

The E2 drop rested on a single-shuffle control, which is confounded: replacing a
clip's instruction with another task's changes the caption's **style and length**, and
CLIPScore's per-caption bias (≈7 points) is far larger than the image-text agreement it
is probing (≈2). What exposed it: an independent 80-clip replication reproduced the
pooled ≈0 gap (+0.35 [−0.96, +1.95]) but with strata that saturation cannot produce —
predictions +1.96 [+0.16, +2.93] and bimanual +1.66 [+0.42, +3.38] RESOLVED **positive**,
humanoid **−3.18 [−5.05, −0.96] RESOLVED negative**. A text-blind metric returns the
same number for both arms; it does not reliably *prefer the wrong prompt*. The humanoid
pool mixes long imperative teleop instructions (→ ~20) with short descriptive captions
(→ ~31), so the shuffle was largely comparing registers.

Deciding test: `scripts/_exp_text_matrix.py` (+ `.sh`) — an N×N clip-by-prompt matrix
per embodiment, **double-centered** (subtract row and column means) so each caption's
bias and each clip's bias vanish, plus a bias-free retrieval rank of the true prompt.

| embodiment | raw diag vs off | **centered** diag [CI] | centered off | retrieval top-1 (chance 6.2%) | median rank /16 |
|---|---|---|---|---|---|
| bimanual | 26.94 vs 24.42 | **+1.398 [+0.291, +3.023]** | −0.044 | 37.5% | 3.0 |
| humanoid | 27.28 vs 24.97 | **+2.527 [+1.586, +3.658]** | −0.128 | 43.8% | 2.0 |
| single_arm | 25.19 vs 23.47 | **+1.635 [+1.203, +2.012]** | −0.186 | 31.2% | 2.0 |

All three RESOLVED; mean retrieval top-1 **37.5% = 6× chance**. CLIPScore discriminates.

**The paper table was never exposed to the bias.** In a pair, the prediction and its own
GT are scored against the **same prompt** — verified **83/83 pairs, 0 mismatches** — so
the caption bias cancels *exactly* in the delta, and the paired result passes the
ceiling test VisualIntegrity failed:

| model | n | paired delta (pred − own GT) | |
|---|---|---|---|
| ctrlworld | 20 | **−0.552 [−1.019, −0.302]** | RESOLVED, GT better |
| ctrlworld_4view_grid | 27 | **−0.209 [−0.703, −0.066]** | RESOLVED, GT better |
| ALL | 83 | −0.301 [−0.444, −0.118] | GT better |

**Code change.** `text_condition_alignment` removed from `DROPPED` (now empty) and
restored to `COLUMNS` in `assets/scripts/table_dense_nr.py`, registered in a new `PAIRED_ONLY`
dict: emitted in T2, **excluded from T1**, because an absolute median pools different
captions and would invite the between-model ranking the data cannot support.
`table2_text_condition_alignment.tex` is emitted again; `provenance.json` records
`paired_only_columns`.

**Methodological lesson (applies to E3 and E6 too).** Never test a metric's sensitivity
with a manipulation that also moves a nuisance variable the metric is known to be
sensitive to. Either match the nuisance (same caption → paired) or remove it
(double-centering). The same trap is why "divide flicker by motion" is wrong in E3.


## E7 (2026-09-16) — **do the NR metrics penalise acceleration artifacts?**

Before scoring 9 accel trees, the metrics must be shown to move the RIGHT WAY on
accel-style degradation. Otherwise we repeat the VisualIntegrity mistake at scale:
a metric that *rewards* the artifact would rank a lossy fast method above dense.

`scripts/_exp_accel_sanity.py` synthesises 4 artifacts from 10 real dense pred clips
(4 models × 3 mounts) and scores them with the **unmodified** scorer via
`--reuse-manifest` (no second metric implementation). A degraded clip is
definitionally worse, so a "high is better" metric MUST drop.

| artifact (mimics) | AMC | ActorFlicker | VisualIntegrity | globalSSIM(v2) |
|---|---|---|---|---|
| `dup2` frame reuse (step caching) | **−39.3 ✅** | −0.004 ns | +0.17 ns | −1.93 ✅ |
| `smooth3` temporal smoothing | +7.8 ns | −0.000 ns | −1.50 ns | **+1.52 ✗** |
| `blur` spatial blur (pruning) | **+3.6 ✗** | **−0.005 ✗** | **−6.79 ✅** | **+1.08 ✗** |
| `block` 8× block artifact (sparse attn) | −4.1 ns | −0.002 ns | **−12.18 ✅** | **+0.81 ✗** |

✅ = penalises with 95% bootstrap CI excluding 0; ✗ = *rewards* the artifact,
CI excludes 0; ns = CI spans 0 (metric does not react at n=10).

**A first pass of this table judged by the sign of the median alone and concluded
"ActorFlicker fails all four". With CIs that is wrong** — only the `blur` column is
significant for ActorFlicker. Recording the correction because the sign-only reading
is exactly the trap this experiment exists to catch.

**Findings.**

1. **AMC is right about frame reuse and blind-to-fooled elsewhere.** `dup2` −39.3
   [−47.2, −7.8] is a large, correct, resolved penalty — caching-style frame reuse is
   the artifact AMC detects best. But `blur` gives **+3.64 [+1.92, +9.06]**: a
   *resolved reward* for a degradation. That is structural, not a bug: AMC =
   `1 − MSE_VFI/MSE_copy`, and blurring makes a clip easier to interpolate, which is
   what AMC rewards. `smooth3` (+7.8) and `block` (−4.1) have CIs spanning 0 at n=10.
2. **ActorFlicker only reacts to blur, and in the wrong direction**
   (−0.005 [−0.0069, −0.0026] = "improves"). For `dup2`/`smooth3`/`block` it does not
   react at all. So it is either wrong or blind on every artifact tested — no
   configuration in which it is a usable quality signal. With rho +0.70 vs motion and
   ~5e-3 deltas, **do not report it**; renormalising cannot fix a metric the artifact
   itself suppresses.
3. **VisualIntegrity is the best artifact detector for spatial degradation**
   (blur −6.79 [−11.8, −4.2], block −12.18 [−18.0, −10.9], both resolved and correct)
   but does not react to `dup2` or `smooth3`. Note this is the *opposite* of its
   GT-ceiling failure: VI is sensitive to spatial corruption but insensitive to the
   generative-vs-real gap. It stays descriptive-only for model ranking, yet it is
   genuinely informative for accel artifacts — see the recommendation below.
4. **globalSSIM (the v2 metric) is rewarded by 3 of 4 artifacts** (all resolved), a
   strong retroactive justification for replacing it with AMC in v3.

### Which artifact do the REAL methods have? (`scripts/_exp_accel_character.py`)

Paired pixel-space measurement, 24 clips per method, no GPU:

| method | sharpness ×dense | temporal ×dense | AMC risk |
|---|---|---|---|
| dicache | 0.993 | 1.024 | none |
| worldcache | 0.989 | 1.033 | none |
| fastercache | 0.973 | 1.031 | none |
| pisa | 0.987 | 0.997 | none |
| radial | 1.006 | 1.035 | none |
| svg1 | 0.974 | 0.967 | none |
| **svg2** | **0.932** | 1.055 | blur-like |
| **sito** | **0.758** | 1.004 | blur-like |
| **itm** | **0.741** | 1.127 | blur-like |

**So the AMC blur-blindness only threatens 3 of 9 methods** — the two token-pruning
methods (SiTo, ITM) and SVG2. That matches the pruning insight rules, which report
SiTo/ITM smearing at aggressive settings. Caching (dicache/worldcache/fastercache)
and PISA/Radial/SVG1 are within 1–3% of dense sharpness, and *no* method duplicates
frames (`d_dup_rate` = 0.0000 everywhere), i.e. nobody triggers AMC's big correct
penalty either.

Severity bound: the synthetic `blur` that produced the +3.6 AMC bias destroys
sharpness to **0.072×**, ~10× more severe than SiTo/ITM's 0.74–0.76×. A calibrated
milder point (`blur_real`, sigma 0.45 → 0.684×, still slightly worse than SiTo) is
being scored to bound the real bias rather than extrapolate from the extreme.

### Severity calibration: is the AMC blur bias big enough to matter? **No.**

The +3.64 AMC reward above came from `sigma=1.6` blur, which destroys sharpness to
**0.072×** the original — roughly 10× more severe than anything a real method does.
Two milder points were scored to find the bias at *realistic* severity (same 10
sources, same unmodified scorer):

| blur point | sharpness ×orig | AMC bias | ActorFlicker bias | VisualIntegrity |
|---|---|---|---|---|
| `blur` σ1.6 | 0.072× | **+3.64 [+1.92, +9.06]** ✗ | −0.005 ✗ | −6.79 ✅ |
| `blur_mild` σ0.8 | 0.244× | **+1.55 [+0.26, +4.59]** ✗ | −0.002 ✗ | −2.28 ✅ |
| **`blur_real` σ0.45** | **0.684×** | **+0.50 [−0.17, +1.00] ns** | −0.001 ✗ | −0.93 ✅ |
| real SiTo | 0.758× | — | — | — |
| real ITM | 0.741× | — | — | — |
| real SVG2 | 0.932× | — | — | — |

**`blur_real` (0.684×) is already blurrier than the worst real method (ITM 0.741×),
and at that severity the AMC bias is +0.50 with a CI spanning 0 — i.e. not
detectable.** The bias only becomes significant below ~0.25× sharpness, a regime no
method in this benchmark reaches. Extrapolating the σ1.6 number to real methods
would have overstated the problem by ~7×.

VisualIntegrity, by contrast, still gives a clean resolved penalty even at
`blur_real` (−0.93 [−1.23, −0.28]) — it is the sensitive detector for exactly the
artifact AMC is blind to.

### Verdict for the acceleration wave

1. **AMC is usable for the accel comparison.** It correctly and strongly penalises
   frame reuse (−39.3), and its blur blind-spot is quantified as non-significant at
   real severity. Caveat to print: on SiTo/ITM/SVG2 a *small* positive AMC delta
   (≲ +0.5) must not be read as "better than dense".
2. **Report AMC together with a sharpness ratio.** `scripts/_exp_accel_character.py`
   already computes it cheaply (no GPU) and it is what makes the AMC number
   interpretable for pruning methods. This directly answers "the metrics should
   reflect the nature of each method".
3. **ActorFlicker stays out** — it is the only column that is *significantly* wrong
   at every blur severity including the realistic one.
4. **VisualIntegrity is worth reporting for accel** (descriptive for model ranking,
   but a valid *within-method* artifact detector vs its own dense counterpart, where
   the generative-vs-real bias that broke E1 cancels).
5. No method duplicates frames (`d_dup_rate` = 0.0000 for all 9), so AMC's strongest
   correct penalty will not fire — expect small AMC deltas overall.

### Tooling added for the wave

`run_dense_nr.py` now inventories any method tree: `--methods dense dicache pisa …`
(`_pred_rows()` is one walk reused per tree, so accel rows are built by exactly the
same rules as dense). Two correctness details:

- **`id` is namespaced by method.** An accel tree mirrors dense's relative paths, so
  `sha256(rel+view)` collided and an accel result would have *overwritten* the dense
  result file. `dense`/`gt` keep the bare identity, so the 1,572 existing results stay
  valid — verified: all 1,573 div30 ids are still present with identical
  `path`/`pair_key`.
- **`pair_key` is deliberately method-free**, which is what pairs an accel clip to its
  dense counterpart. Verified on `--methods dense dicache pisa`: 0 duplicate ids,
  8,226 pair_keys carrying both dense and dicache.
- Co-located GT is only emitted from the dense tree (accel trees ship identical GT
  copies; scoring them again would duplicate GPU work and collide on `pair_key`).
- A few accel episodes lack `metrics.json` (interrupted runs) so their view layout is
  unknown; it is taken from the dense counterpart at the same relative path, which is
  authoritative since the layout must match for the clips to be comparable.

## Publication readiness audit (2026-09-16, third pass) — 2 enforcement gaps closed

Question asked: *are the numbers ready for the paper, and what is left to check?*
Two things were documented but **not enforced anywhere in code**, so the emitted
tables still carried claims they should not.

### Gap 1: `RESOLVED` was granted at n=3 — now requires n >= 8

A percentile bootstrap CI of the median at n=3 resamples three numbers; it "excludes
0" whenever all three share a sign, which happens ~25% of the time by chance. Two
v3.2 rows were RESOLVED at n=3 (`ctrlworld/humanoid/head`, `dreamgen/humanoid/wrist`),
and one AMC row at n=3. Added `MIN_PAIRS_FOR_CLAIM = 8` to `table_dense_nr.py`,
applied to **both** the paired-GT table and the accel-vs-dense table.

Effect: **20 -> 13 resolved claims.** The delta and CI are still printed, only the
claim flag is withdrawn.

### Gap 2: the head-mount motion confound is now printed on the row

E1's second pass measured rho = −0.541 (AMC) and −0.553 (VisualIntegrity) vs
`actor_motion_raw` **on head mounts only**, and prescribed "report per mount with head
flagged" — but nothing flagged it. Added `CONFOUNDED_MOUNTS`, which stamps
`!! CONFOUNDED (rho ...)` on every head row of those two columns and records it in
`provenance.json` as `<key>_confound_warning`.

Good news: **AMC has no resolved claim on head at all**, so no current AMC claim is
affected. One TextAlignment claim is on head (`4view/humanoid/head`, n=12, −0.354) and
is now visibly flagged.

### The AMC ceiling violation was investigated and is NOT a metric failure

`ctrlworld_4view_grid / humanoid / static` is RESOLVED at **+1.34** (pred beats GT),
the one wrong-signed AMC claim. Three candidate mechanisms were tested on those exact
7 pairs:

| hypothesis | measurement | verdict |
|---|---|---|
| E7's blur inflation | sharpness pred/GT = **0.964×** | rejected — E7 needs <= 0.684× for even +0.50 |
| temporal smoothing | temporal ratio = **0.971×** | rejected — same ~3% magnitude |
| GT sensor noise depresses GT-AMC | Spearman(GT motion, dAMC) = **+0.082, p=0.3** | rejected |

And the population-level check is reassuring: across all 163 GT/pred pairs, median
dAMC is **negative in all four GT-motion quartiles** (−0.87 / −0.46 / −0.21 / −0.05)
with pred winning only 40–48%. So **AMC does satisfy GT-is-ceiling globally**; the
+1.34 cell is a local n=8 fluctuation, not the structural failure VisualIntegrity had
(where pred beat GT at p=5e-05 on a whole model).

### The real remaining blocker: humanoid is sample-starved, not metric-broken

All **7** claims lost to the n>=8 floor are humanoid. The cause is stratification, not
missing video:

| embodiment | paired AMC n | cells (model×mount) | cells with n<8 |
|---|---|---|---|
| bimanual | 106 | 6 | **0** |
| single_arm | 49 | 3 | **0** |
| **humanoid** | **34** | **8** | **6** |

Humanoid has a third of bimanual's pairs spread over *more* cells (it has both grid
layouts and 3 mounts). Two options, both cheap since nothing needs re-scoring:
(a) pool humanoid over mount (or over the two ctrlworld layout variants) to clear
n>=8; (b) score more humanoid episodes — the manifest has plenty
(`div30` sampled only a fraction of the 19,128-clip inventory).

### Status: what is publishable now

- **3 claim columns**, 13 resolved rows, all with n>=8: TextAlignment (paired-only,
  4 rows), AMC (4 rows), ActorActivity + ActiveRate (5 rows). 12 of 13 are in the
  correct direction (GT >= pred).
- **VisualIntegrity**: descriptive only (E1), never resolved, caveat printed.
- **ActorFlicker**: dropped (E7), `.tex` auto-deleted.
- **Acceleration**: metric behaviour is validated (E7) and the tooling is in place,
  but **the wave has not been run** — that table does not exist yet.

## E3 executed (2026-09-16) — the per-pixel flicker fix was BUILT and RUN; **it fails, so DROP is now proven from both sides**

E7 had already dropped ActorFlicker on artifact *sign*. E3 was still worth executing
because "the metric is confounded" and "the confound is unfixable" are different
claims, and the paper is stronger making the second one. The per-pixel
renormalisation specified in E3 is now implemented and measured. **It fails.**

### Built: 9 candidate statistics in ONE probe pass

`metrics/dynamics/actor_flicker.py::residual_variants` computes, per frame-pair:
`raw` (v1), `flownorm` (r/(1+m) per pixel), `gradnorm` (r/|∇I|), `unexplained`
(Σr/Σ|I₁−I₂|), `shiftmin` (best residual within ±1 px), `sub_K` for
K ∈ {0.05,0.1,0.2,0.4} (r − K·|∇I|·m), plus 4 diagnostics (`copy`, `flow`, `grad`,
`align_pred`). One pass is deliberate: the optical flow and SAM2 masks are the GPU
cost, the candidates are numpy on arrays that already exist. That makes the *choice*
of statistic a zero-GPU decision — the same principle that made the E4 gate a
table-time decision.

**Back-compatibility verified: `actor_flicker_raw` reproduces the stored v3.2 value
with relative deviation 0.0e+00 on all 53 clips.** The baseline did not move under
the refactor, so the candidate comparison is against a fixed reference.

### Result: criterion 1 fails for every candidate (53 clips, n=19/19/10 per mount)

| candidate | rho static | rho head | rho wrist | C1 | C2 (GT better) |
|---|---|---|---|---|---|
| raw (v1) | +0.786 | −0.067 | +0.575 | FAIL | 80.8% |
| flownorm | +0.321 | +0.115 | +0.272 | FAIL | 73.1% |
| gradnorm | +0.830 | −0.430 | +0.612 | FAIL | 76.9% |
| **unexplained** | **−0.079** | **+0.139** | +0.405 | FAIL (wrist only) | 76.9% |
| shiftmin | +0.791 | −0.333 | +0.623 | FAIL | 73.1% |
| sub_0.4 | +0.419 | +0.115 | +0.389 | FAIL | 73.1% |

C2 passes for all nine — which is exactly why C1 exists. C2 alone cannot separate a
flicker metric from a motion metric, because GT also moves less erratically than pred.

### Insight 1 — the confound is `copy`, not `flow`. The original E3 prescription aimed at the wrong quantity.

With per-pair diagnostics stored, the driver is directly measurable (5,254 pairs):

| mount | rho(flow, r) | **rho(copy, r)** | rho(grad, r) | rho(grad·flow, r) |
|---|---|---|---|---|
| static | +0.787 | **+0.904** | +0.125 | +0.792 |
| head | +0.589 | **+0.861** | +0.527 | +0.638 |
| wrist | +0.692 | **+0.889** | +0.242 | +0.711 |

The residual is best explained by **how much the image changed at all**, not by
displacement and not by gradient. So correcting for *misalignment* (`gradnorm`,
`sub_K`, `shiftmin`) cannot work: the residual is dominated by scene change flow
cannot follow at all — occlusion, appearance change, codec noise. Only `unexplained`,
which divides by observed change, moves the needle, and it is the one that nearly
passed. **E3's instruction "normalize by local flow magnitude per pixel" is what
`flownorm` implements literally, and it fails.** Anyone revisiting this should
normalize by observed temporal change instead.

A synthetic unit test now isolates the mechanism (`scripts/test_actor_flicker.py`):
with **exact** flow every candidate is already motion-independent (fast/slow ratio
1.00), but with an 8% flow-estimation error and **zero** flicker, `raw` inflates
**5.02×** from 1 px to 6 px of displacement. The confound is flow *error*, not motion.

### Insight 2 — a post-hoc "winner" appeared, and two checks proved it was noise

A zero-GPU search over 7 further candidates built from the stored diagnostics
(`scripts/_exp_e3_posthoc.py`) produced one full pass: `ratio_med` =
median(r/max(copy, 0.01)), rho −0.128/+0.079/+0.191, GT better 69.6%, spread 1.85×.
It did not survive:

1. **It passes only at that exact arbitrary floor.** Sweeping it: 0.005 → rho −0.698,
   **0.01 → −0.128**, 0.02 → +0.475. A statistic whose confound flips sign under a
   ±2× change of a nuisance constant is fitting noise.
2. **Multiple comparisons.** Permuting motion labels within mount: the probability
   that **at least one of the 16 screened candidates passes C1 under the null is
   0.87**.

### Insight 3 — criterion 1 was itself under-powered as written, in a way that favours false passes

`|rho| < 0.25` is an **equivalence** test, so *small* n makes it easier to pass:

| n per mount | P(false FAIL \| rho_true = 0) | P(false PASS \| rho_true = 0.35) |
|---|---|---|
| 10 (head, selection run) | 0.45 | 0.38 |
| 19 (static/wrist) | 0.29 | 0.34 |
| 60 | **0.05** | 0.22 |

The pre-registered "~60 clips spanning all three mounts" is **~19/mount** — not
enough to conclude either way. This is a general lesson for the remaining
experiments: an equivalence bar needs its n justified in the *opposite* direction
from a significance test. A 305-clip out-of-sample confirmation run at **60/mount**
was therefore launched (`--exclude-run` holds out the 54 selection clips, since
screening 16 candidates on the same clips is what produced the 0.87 false-pass rate).
Note `head` has only **9** pred clips with a scored GT partner in the entire dense
run, so C2 can never be powered on head; C1 can.

### Kept in the code deliberately

`residual_variants` and the stored `actor_flicker_variants` stay (cost: a few numpy
ops). They make this drop auditable and any future re-examination zero-GPU.
`actor_flicker_raw` remains the reported (experimental, non-quotable) value, so
nothing downstream changed. `scripts/_exp_accel_sanity.py` now flattens the variant
dict onto each record, so re-running E7 re-tests **every** candidate against the
artifact sign in one pass — the decisive test, since a statistic that improves under
a definitionally worse video cannot be repaired by any normalization.

## E5 prepared (2026-09-16) — code complete, and a **silent pairing bug** found before it could corrupt a table

### 1. Sampler: `scripts/_exp_e5_build_manifest.py`

`inventory(out, methods=…)` / `_pred_rows(tree, method, …)` already parameterise by
method tree, so no change was needed there. What was missing is the sampler: it
restricts candidates to clips whose dense counterpart **already has a scored result**
(dense NR results exist only for the 1,573-clip div30 sample, not all 10,465 dense
videos), samples 30 per (method, embodiment, mount) with `spread_sample` over
(task, split), and emits **only** method rows — GT is not re-emitted because the accel
trees ship byte-identical copies and the dense partners are already scored.

```
dense reference: 1572 scored records, 733 pairable pair_keys
worldcache/dicache 864 pairable -> 210    itm/sito  615 -> 210
pisa 676, radial 674, svg1 702, svg2 673 -> 210 each
fastercache 358 -> 210
groups: 63, short of 30: 0
TOTAL 1890 clips ~= 49 GPU-h (~9.9 h wall on 5 workers)
```

Two corrections to earlier numbers in this doc: the budget is **49 GPU-h, not 12.6**
(1,890 × 94 s; the 9.9 h figure is *wall* on 5 workers), and it is **7 groups per
method, not 9** — only 7 (embodiment, mount) combinations exist in the dense NR
sample (bimanual static/wrist, single_arm static/wrist, humanoid head/static/wrist).

### 2. BUG: `paired_table` would have paired the WRONG clip once accel data arrived

```python
by_key[r['pair_key']][r['model']] = r      # BUG
```

`pair_key` is deliberately method-free — that is what pairs an accel clip to its dense
counterpart — and an accel row carries the **same `model`** as its dense counterpart
(a `worldcache` clip of the ctrlworld tree has `model == 'ctrlworld'`). So the first
time an accel run is aggregated, the accel row **overwrites the dense row** for that
episode+view and the GT delta is computed on the wrong clip: no error, no warning, a
plausible-looking number. Fixed to key by `(method, model)`.

Verified latent-only today: the dense-only run has **0 collisions** on
`(pair_key, method, model)` and **0** cases of one `(pair_key, model)` carrying two
methods, so **no published dense number is affected**. (11 `pair_key`s do carry two
different dense models — `ctrlworld` and `ctrlworld_4view_grid` — which is harmless
since those differ in `model`. That near-miss is what exposed the bug.)

A second, smaller bug found by the same test: `write_csv` took its fieldnames from
`records[0]`, but rows are not homogeneous — `<key>_confound_warning` only appears on
flagged rows — so it raised `ValueError` as soon as a flagged row appeared later in
the list. Now takes the union of keys in first-seen order.

### 3. T3 accel-vs-dense table: `accel_table()` in `table_dense_nr.py`

Pairs each method clip against the **dense prediction** at the same `pair_key`
(PhyFid-DR reference rule, *not* GT). Cells = (method, model, embodiment, mount);
bootstrap CI, Wilcoxon p, `resolved` flag under the same `MIN_PAIRS_FOR_CLAIM = 8`
floor as T2, `CONFOUNDED_MOUNTS` stamped on head rows, `ACCEL_CAVEATS` (AMC
blur-blindness) printed inline, and unpaired accel clips counted and reported rather
than dropped silently. `--also <dense_run>` aggregates the reference run.

One deliberate difference from T2, per **E7 verdict 4**: VisualIntegrity's
GT-is-ceiling failure does **not** carry over here, because the generative-vs-real
bias it is blind to *cancels* when a method clip is differenced against its own dense
counterpart — and E7 measured VI as the most sensitive correct detector of exactly the
spatial artifacts accel methods produce (blur −6.79, block −12.18, still resolved at
realistic severity −0.93 [−1.23, −0.28]). `CEILING_FAILURE_CANCELS_VS_DENSE` encodes
that, so a VI *accel* delta may carry a claim while a VI *model ranking* still may not.

Validated end-to-end by injecting a known −0.8 VisualIntegrity shift into 120
synthetic `worldcache` records: T3 recovered −0.66 … −0.99 per cell, resolved exactly
the cells with n ≥ 8, and printed the head confound. Regression-checked on the real
dense run: **13 resolved claims unchanged, `table1`/`table2` byte-identical, and T3
does not appear when no accel data is present.**

### Run it

```bash
python scripts/_exp_e5_build_manifest.py
bash scripts/run_dense_nr.sh --out metrics/WorldScore/results/accel_nr_v1 \
     --workers 5 --reuse-manifest
python assets/scripts/table_dense_nr.py --run metrics/WorldScore/results/accel_nr_v1 \
     --also metrics/WorldScore/results/dense_nr_v3_div30
```

Columns E5 may report: **VisualIntegrity** (claims allowed vs dense, see above),
**AMC** (with the blur caveat + sharpness ratio from `_exp_accel_character.py`),
**ActorActivity** descriptive, **TextAlignment** paired-only. **ActorFlicker must not
appear** — it improves under every synthetic artifact, so it would credit the very
methods it should penalise.

## Humanoid top-up (2026-09-16) — claims 13 → 19, and a NEW embodiment-level finding

Pooling over mount was **rejected after measuring it** (the user's first choice, but
the data says no): humanoid mounts disagree in sign and scale (wrist +10.6, static
+1.7, head −0.3), head is the motion-confounded mount, and pooling produced exactly
**one** claim which was *wrong-signed* (+1.24, pred beats GT). Pooling would have
manufactured the same class of artifact we just removed from VisualIntegrity.

Instead: `scripts/_topup_humanoid.py` scored **120 more humanoid clips** (60 pred + 60
co-located GT), allocated per (model × mount) cell to reach n≥12, drawn only from
pair_keys where BOTH halves exist and excluding everything div30 already had. 120/120
scored, 0 errors, ~35 min at 5 workers. Merged into
`metrics/WorldScore/results/dense_nr_v3_pooled` (symlinked results, merged manifest,
1692 clips) so gates and tables see one population.

| column | div30 | pooled+topup |
|---|---|---|
| TextAlignment (paired) | 4 | **7** |
| AMC | 4 | **5** |
| ActorActivity | 3 | **5** |
| ActiveRate | 2 | 2 |
| **total resolved** | **13** | **19** |

Every humanoid AMC cell is now n≥7 (was 6 cells below 8), so per-mount reporting is
possible without pooling.

### NEW: AMC is not comparable ACROSS embodiments — humanoid GT sits ~18 points lower

The top-up exposed something the smaller sample hid: humanoid is the **only**
embodiment where pred beats GT on AMC.

| embodiment | n pairs | median dAMC | pred wins | GT AMC | pred AMC |
|---|---|---|---|---|---|
| bimanual | 92 | −0.82 | 37% | 65.9 | 65.2 |
| single_arm | 43 | −0.81 | 44% | 67.1 | 64.8 |
| **humanoid** | **79** | **+1.43** | **67%** | **47.9** | **49.3** |

The cause is the **GT side**, not the prediction: humanoid GT clips score AMC ~48
while bimanual/single_arm GT score ~66. Ruled out as explanations:

- **Motion amount.** Binning GT-only clips by motion, humanoid GT is ~46–49 in
  *every* bin while bimanual is 63–68 in every bin:

  | GT motion bin | bimanual | humanoid | single_arm |
  |---|---|---|---|
  | 0–1.5 px | 68.4 (n=70) | **48.2** (n=67) | 68.9 (n=48) |
  | 1.5–3 px | 67.9 (n=84) | **45.8** (n=62) | 62.9 (n=20) |
  | 3–6 px | 68.1 (n=75) | **49.4** (n=102) | 69.5 (n=44) |
  | 6+ px | 63.3 (n=62) | **48.3** (n=117) | 47.8 (n=59) |

  Within-embodiment rho(motion, AMC) is small for humanoid (−0.079).
- **Geometry / fps.** Identical analysis grid (256×192) and the same
  source/effective-fps mixes across embodiments.

So GR1 humanoid video is *intrinsically less interpolable* (scene detail and
non-linear ego/arm motion), which lowers the VFI-gain denominator for GT and pred
alike. **Consequence: never compare absolute AMC across embodiments, and read the
humanoid GT-delta sign with care** — a small positive humanoid dAMC is the local
baseline, not evidence the model beats reality. The paired delta stays valid *within*
an embodiment, which is how it is reported.

One humanoid AMC claim (`ctrlworld/humanoid/head`, n=8, +4.17) is both wrong-signed
and on the confounded head mount; it is auto-flagged `!! CONFOUNDED` and should not be
quoted.

## Acceleration PILOT (2026-09-16) — flow validated, E7 predictions confirmed

Pilot on 2 methods chosen to span the two E7 regimes: **dicache** (caching, sharpness
0.993× dense) and **sito** (token pruning, 0.758× = the blur-like regime where AMC is
known to be blind). `scripts/_pilot_accel.py` selected 30 paired clips per method
whose **dense counterpart was already scored**, so the pilot cost only 60 accel clips
(~35 min, 5 workers, 0 errors) instead of 120. Aggregated into
`dense_nr_v3_pooled` (1752 clips) so `accel_table()` finds each dense partner.

### Result, pooled per method (accel − its own dense clip)

| method | VisualIntegrity | AMC |
|---|---|---|
| dicache | **−1.16 [−2.21, −0.03]** resolved | −0.79 [−2.25, +0.32] ns |
| sito | **−4.88 [−6.61, −3.52]** resolved | **−5.60 [−11.94, −3.77]** resolved |

**Every E7 prediction holds:**

1. **VisualIntegrity ranks the two methods correctly and in the right direction** —
   sito (the blurrier method) loses 4× more VI than dicache, matching the measured
   sharpness ratios (0.758× vs 0.993×). This validates E7 verdict 4: VI *is* usable
   for accel comparison even though it failed GT-is-ceiling, because the
   generative-vs-real bias cancels against the clip's own dense counterpart.
2. **AMC did NOT get fooled by sito's blur.** E7 warned a blur-like method could show
   a spuriously *positive* AMC delta; instead sito is −5.60, i.e. AMC correctly
   penalises it. The blur bias (+0.50 at 0.684× sharpness, non-significant) is an order
   of magnitude smaller than the real degradation sito causes, so it does not flip the
   sign. Good news for the wave: AMC is safe to report for pruning methods.
3. **dicache is near-dense on AMC** (ns), consistent with 0.993× sharpness and with the
   caching rules that call it quality-neutral at the tuned threshold.

### The operational lesson for E5: pool per method, do NOT stratify to model×mount

30 clips per method fragment into 15+ (model × embodiment × mount) cells at n=1–5, and
**not one cell reaches the n≥8 claim floor** — the per-cell table is unreadable. Pooled
per method the same 30 clips resolve immediately. So E5 should either
(a) report accel pooled per method (as above) with per-cell numbers as an appendix, or
(b) budget ~8× more clips to make per-cell claims. Recommendation: (a), plus per-cell
only for the one or two model families a method is actually tuned for.

Also note the pilot's dense side came free from the reference run — always select accel
clips whose dense partner is already scored (`_pilot_accel.py` does this automatically:
804 such candidates existed for dicache, 614 for sito).

## E3 confirmation landed (2026-09-16) — the powered run **passes** a candidate, then three non-statistical tests kill it

The 305-clip out-of-sample run (60/mount, `--exclude-run` holding out the 54 selection
clips) finished 305/305 with 0 errors. It did not confirm the interim read — it
reversed it, twice, and that is the most transferable result of the whole exercise.

### At full power, `unexplained` passes both pre-registered criteria

| candidate | rho head | rho static | rho wrist | C1 | C2 (GT better) | GT spread |
|---|---|---|---|---|---|---|
| raw (v1) | +0.405 | +0.728 | +0.361 | FAIL | 58.4% FAIL | 4.56× |
| flownorm | −0.055 | +0.556 | +0.169 | FAIL | 61.6% FAIL | 2.86× |
| **unexplained** | **−0.094** | **+0.232** | **+0.135** | **PASS** | **84.8% PASS** | 1.86× |
| gradnorm | +0.796 | +0.750 | +0.399 | FAIL | 65.6% PASS | 4.18× |
| shiftmin | +0.679 | +0.722 | +0.373 | FAIL | 57.6% FAIL | 4.53× |
| sub_0.4 | +0.368 | +0.570 | +0.222 | FAIL | 68.0% PASS | 3.11× |

### THE headline: three sample sizes, three verdicts, same statistic

| n per mount | `unexplained` static rho | `unexplained` C2 | verdict |
|---|---|---|---|
| 19 (selection, 53 clips) | **−0.079** | 76.9% | FAIL (wrist +0.405) |
| ~44 (interim, 131 clips) | **+0.272** | **55.6%** | FAIL |
| 60 (full, 305 clips) | **+0.232** | **84.8%** | **PASS** |

`raw`'s C2 moved the other way over the same growth: **80.8% → 58.4%**. Two of these
three readings would have been published as a conclusion had the run been stopped
early. `|rho| < 0.25` is an **equivalence** bar, so unlike a significance test it gets
*easier* to pass as noise grows — the exact opposite of the intuition that a small
sample is conservative. Every remaining equivalence-style criterion in this benchmark
("indistinguishable from dense", "not confounded") needs its n justified before it is
read.

### Three tests that are not significance tests, and all three fail it

1. **Arbitrary-constant sensitivity.** `unexplained` divides by `max(copy, 0.01)`.
   Sweeping that hardcoded floor: 0.005 → rho −0.386, **0.01 → −0.129**,
   0.02 → +0.324, 0.05 → +0.706 (C2 also decays 74% → 56%). It passes only in a narrow
   window around the value that happened to be in the code — the identical failure mode
   that had already killed `ratio_med`.
2. **Leave-one-model-out.** Drop `ctrlworld`: static **+0.310**, wrist **+0.323**.
   Drop `ctrlworld_4view_grid`: head **−0.303**. The pass is a property of the model
   *mix* in the sample, not of the metric.
3. **Multiple comparisons.** Permuting motion labels within mount, P(≥1 of the 9 probe
   candidates passes C1 under the null) = **1.00** (0.87 on the selection set). At this
   n the bar carries no information in either direction.

### The decisive test: artifact sign (E7 protocol re-run with the variants flattened in, 70 clips)

| artifact (mimics) | ActorFlicker v1 | flick:flownorm | **flick:unexplained** |
|---|---|---|---|
| `dup2` frame reuse (**caching**) | −0.0038 ns | −0.0004 ns | **−0.0425 [−0.1347, −0.0025] ✗ REWARDS** |
| `smooth3` temporal smoothing | ns | ns | +0.0984 ns |
| `blur` σ1.6 (pruning) | −0.0049 ✗ | ns | −0.0494 ns |
| `blur_mild` σ0.8 | −0.0024 ✗ | −0.0009 ✗ | ns |
| `blur_real` σ0.45 (realistic) | −0.0010 ✗ | −0.0003 ✗ | ns |
| `block` 8× (sparse attn) | ns | ns | +0.1237 ✅ |

**`unexplained` "improves" when every second frame is duplicated** — exactly the
artifact the step-caching family (WorldCache/DiCache/FasterCache) produces, and the one
AMC detects best (−39.3). A flicker column that rewards frame reuse would credit the
most aggressive cache settings for being the most degraded. It is also blind to blur at
all three severities. Meanwhile `flownorm` — the literal reading of the original E3
prescription ("normalize by local flow magnitude") — is *significantly wrong* on blur
at realistic severity.

So the candidate that survived the statistics fails the mechanism check, and the
candidate the spec asked for fails the artifact check.

### Final verdict: DROP, proven three independent ways

1. v1 is motion-confounded (rho +0.73 static at full n);
2. no correction is stable — all 16 candidates either stay confounded, or pass only at
   one arbitrary constant / one model mix;
3. both survivors are directionally wrong on the artifacts the benchmark exists to
   measure.

That is a far stronger paper statement than "we suspected it was confounded", and it
cost ~11 GPU-h total. `residual_variants` and the stored `actor_flicker_variants` stay
in the code so this is auditable and any future re-examination is zero-GPU;
`scripts/_exp_accel_sanity.py` now flattens them automatically, so any new candidate is
artifact-tested in the same pass.

### Method note worth reusing

The artifact-sign test should be run **first**, not last. It needed 70 clips (~1.8 h)
versus 305 (~5 h) for the powered correlation test, it needs no equivalence-bar power
analysis, and it answers the question that actually matters for an acceleration
benchmark: *does the metric move the right way when the video gets worse?* Both E3
survivors were eliminated by it in one pass.

## E5 full wave — all 9 acceleration methods scored (DONE)

210 clips (7 remaining methods × 30 paired), 5 workers, 87 min wall, **0 errors**.
Dense counterparts were already in `dense_nr_v3_pooled`, so nothing dense was rescored.
Pooled results dir now holds 1962 records: `gt=888, dense=804, 9 × accel=30`.

```bash
bash scripts/run_dense_nr.sh --out /tmp/accel_wave --methods dense worldcache \
  fastercache pisa radial svg1 svg2 itm --inventory-only
mv /tmp/accel_wave/manifest.json /tmp/accel_wave/full_inventory.json
python scripts/_pilot_accel.py --out /tmp/accel_wave --methods worldcache fastercache \
  pisa radial svg1 svg2 itm --target 30 \
  --reference-run metrics/WorldScore/results/dense_nr_v3_pooled
bash scripts/run_dense_nr.sh --out /tmp/accel_wave --workers 5 --reuse-manifest
cp -n /tmp/accel_wave/results/*.json metrics/WorldScore/results/dense_nr_v3_pooled/results/
```

### The per-cell grid is unusable for accel, as the pilot predicted

30 clips/method spread over the real (model × embodiment × mount) grid gives **9–17
cells at n=1..7** — 124 rows in `table3_accel_vs_dense.csv`, of which exactly **1** is
RESOLVED. This is not a sampling accident: the accel effect is smaller than the GT–pred
gap, and a bootstrap CI on n≤7 cannot exclude 0 for effects this size. Per-cell reporting
would need ~10× the budget.

So `table_dense_nr.py` gained **T3b (`table3b_accel_pooled.csv`, the headline table)**:
the same paired deltas, pooled per method. Pooling is legitimate *here specifically*
because every delta is within one episode+view+geometry (accel vs its own dense clip),
so no absolute level is compared across embodiments — which is exactly why the
`NOT_CROSS_EMBODIMENT` restriction on absolute AMC does not bite. `accel_deltas()` is
shared by T3 and T3b, so the headline can never disagree with its own breakdown. T3 is
still written as the breakdown (it is where a method that helps one mount and hurts
another would show up).

### Headline result (n=30/method, pooled, delta = accel − its own dense clip)

| method | VisualIntegrity | AMC | ActorActivity | TextAlign | sharpness ×dense (E7b) |
|---|---|---|---|---|---|
| itm | **−5.50** R | **−3.89** R | −0.43 R | −1.26 R | 0.741 |
| sito | **−4.88** R | **−5.60** R | −0.41 R | −0.87 R | 0.758 |
| svg2 | **−1.71** R | **−2.15** R | −0.06 | −0.16 | 0.932 |
| dicache | −1.16 R | −0.79 | −0.35 R | −0.13 | 0.993 |
| worldcache | −0.74 R | −0.56 | −0.01 | −0.53 R | 0.989 |
| svg1 | −0.63 R | +0.16 | −0.07 | −0.22 | 0.974 |
| pisa | −0.68 ns | −0.24 | −0.01 | −0.14 | 0.987 |
| fastercache | −0.29 ns | −0.35 | −0.08 | −0.82 R | 0.973 |
| radial | +0.05 ns | +0.98 | −0.01 | +0.02 | 1.006 |

R = RESOLVED. Every resolved delta is **negative** — no method beats its own dense
baseline on any NR column, which is the correct ceiling behaviour and the strongest
sanity signal in the whole run.

### External validation: pixel sharpness predicts the NR ranking

E7b measured sharpness (variance of Laplacian) in pixel space, with no NR model
involved. Spearman rank correlation of that ratio against the pooled NR deltas:

| column | rho vs sharpness ratio |
|---|---|
| VisualIntegrity | **+0.667** |
| AMC | **+0.700** |
| ActorActivity | +0.633 |

Positive = blurrier method loses more. Two fully independent measurements agree on the
ordering, and the three methods E7b flagged as blur-like (itm/sito/svg2) are exactly the
three that top every degradation column. rho is not +1.0 only because the six near-dense
methods sit within 0.973–1.006 sharpness, where their rank order is noise. **This is the
E7 prediction confirmed at full scale, and it is the argument that these NR metrics
measure acceleration damage rather than an artifact of the protocol.**

Note AMC did **not** get fooled by blur on the real methods: sito/itm/svg2 are penalised
−2 to −6, an order of magnitude beyond the +0.50 (ns) blur reward E7 bounded. The
ACCEL_CAVEAT stays printed, but it does not bite at real severity.

### Two statistics bugs found while reading this table (fixed)

Both would have put false claims in the paper, and both were only visible once 9 methods
were on screen at once:

1. **CI bound sitting exactly on 0 counted as "excludes 0".** `accel_table` used
   `(lo > 0) == (hi > 0)`, which is True when `hi == 0.0`. Active Rate is discrete and
   tie-heavy, so this produced `svg1 delta 0.000, CI [−0.059, 0.000] → RESOLVED` with
   p=0.619. Now requires strict `lo > 0 or hi < 0`.
2. **Bootstrap CI and Wilcoxon could disagree, and only the CI was consulted.** `pisa`
   VisualIntegrity: median −0.68, CI [−1.25, −0.27], but **p=0.262** — a median shift
   carried by part of the sample while the signed-rank sum is not significant. A claim
   now requires both (`ALPHA = 0.05` vetoes the CI; below n=10 Wilcoxon is unavailable
   and the CI alone decides, as before).

Net effect: 5 spurious RESOLVED flags removed (pisa VI; dicache/svg1/fastercache/pisa
Active Rate). All three tables now share one `resolved_flag()` definition, so T2 gets the
same protection. T2's resolved counts were unchanged by the fix.

### Lesson

Read the *whole* table before trusting the flag. Both bugs were invisible at 1–2 methods
(the pilot) and obvious at 9, because a `RESOLVED p=0.62` row sitting next to a
`RESOLVED p=4e-05` row is self-evidently wrong. Scale the comparison up before scaling
the claim up.

## Stratified full-`sampling_dataset` wave (DONE) — and the render-mismatch bug it exposed

The user asked to "run full `sampling_dataset`". Full inventory over all 10 method trees is
**87,327 view-clips**. Measured throughput is **3.0 clips/min** (10 workers, A100-80GB,
GPU-bound: 100% util, 80/82 GB; 5→10 workers only bought 2.1→3.0, so more workers do
nothing) ⇒ **~474 h ≈ 20 GPU-days**, and it buys almost nothing statistically: the pooled
accel claims already resolve at n=30 (p to 1e-06) and a bootstrap CI narrows as 1/√n, so
full-n is 250× compute for 0.06× CI width and **no new claim**.

What was actually missing was the **per-cell** tables (1 of 124 rows resolved). Those are
cheap: capping each cell brings 153/154 accel cells to n≥8. So we ran a stratified wave
(`scripts/_stratify_full_nr.py`, accel-target 12 / dense-target 40) = **1,609 clips, 0
errors**, pooled dir now **3,571 records**, all `dense-nr-v3.2-closed-world`.

Result: **per-cell went from 1/124 resolved to 208 resolved over 136 rows, 134 at n≥8** —
the per-cell breakdown is now a real table. Pooled n rose to 93–228/method (p to 1e-29).

`_stratify_full_nr.py` is dependency-aware, which is the only non-obvious part: an accel
clip is worthless without its dense partner and a dense clip is worthless without its GT
anchor, so it prefers rows whose partner is already scored and pulls the partner in when it
must. Measured effect: 1,566 accel clips needed only **698 distinct dense partners (2.24×
reuse), 662 already scored** ⇒ only +11 paid partners. 100% of selected accel clips ended
up pairable.

### OOM is a real ceiling at 10 workers (fixed)

10 workers × ~8 GB = the whole 80 GB card, and **20 of the first 154 clips died with OOM
after all 4 retries (13% loss)**, concentrated in multiview 4-view-grid clips whose SAM2
propagation peaks far above the median. The 16-clip probe never showed it. Fixes:
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in `run_dense_nr.sh` plus **7 workers**
(VRAM 57/80 GB). A later 42-clip OOM batch was fully recovered at **3 workers → 0 errors**,
confirming these were contention, not bad clips. Final: **1,609/1,609, 0 errors.**

### THE BUG: the dense tree is not always the same render as the accel tree

`fastercache` came out **AMC +1.23 RESOLVED = better than dense**, a ceiling violation.
Chasing it found something much worse than one method:

1. Not blur/smoothing — per-model character showed dreamgen fastercache at sharpness
   **1.002×**, temporal 0.993×, dup 0.000 vs dense (i.e. indistinguishable).
2. Not motion smoothing — inside the top-decile motion region, fastercache had the
   *smallest* jerk reduction (0.936×) while dicache/worldcache/radial cut it more
   (0.855–0.876×) yet scored AMC *negative*. Mechanism ruled out.
3. The tell: on `dreamgen/single_arm`, AMC was positive for **all nine methods**
   (+3.2 to +17.1; radial, a near-no-op, showed AMC 72.4 vs dense 43.6).

Cause: at the *same relative path*,
`sampling_dataset/dense/single_arm/output/multiview/dreamgen/` holds **277 frames @ 16 fps**
while every accel tree holds **93 frames @ 10 fps**. The dense tree there was generated
with a different rollout config. `pair_key` cannot catch this — the clips genuinely are the
same episode+view, they were just **rendered differently** — and AMC/motion columns are
per-source-frame quantities, so a 3× length + fps difference moves them regardless of
method.

Spread: **226/1836 accel pairs (12.4%)**, of which `dreamgen/single_arm` is **216/216 =
100%**, plus 3 dreamdojo/single_arm, 6 dreamgen/humanoid, 1 dreamgen/bimanual.

Fix = `comparable_pair()` in `table_dense_nr.py`: require identical `source_fps` and
`effective_fps` and length within 10%. **All three fields were already stored, so the gate
is post-hoc and needed zero rescoring** — the same design that made the earlier gate
changes free. Applied to T2 (pred vs GT) as well as T3/T3b, and the rejected counts are
printed per method.

Effect: **fastercache AMC +1.23 RESOLVED → +0.37 ns**, and T3b now has **26 resolved
deltas, every single one negative — zero ceiling violations.** The 13 remaining positive
cells live only in T3 (the per-cell breakdown, not a claim table), 10 of them humanoid at
n=8–12, i.e. the already-documented `NOT_CROSS_EMBODIMENT` AMC anomaly plus small-n noise.

### Headline after the gate (pooled, delta = accel − its own dense clip)

| method | VI | AMC | TextAlign | ActorActivity |
|---|---|---|---|---|
| itm | **−6.31** R | **−7.05** R | −1.25 R | −0.42 R |
| sito | **−5.36** R | **−5.06** R | −0.97 R | −0.39 R |
| svg2 | −1.68 R | −2.66 R | −0.17 R | +0.01 ns |
| dicache | −1.22 R | −0.99 R | −0.16 R | −0.04 R |
| worldcache | −0.89 R | −1.07 R | −0.14 R | −0.00 ns |
| pisa | −0.63 R | −0.31 ns | −0.24 R | −0.03 ns |
| svg1 | −0.48 R | −0.76 ns | −0.21 R | −0.05 R |
| fastercache | −0.28 ns | +0.37 ns | −0.08 ns | −0.04 R |
| radial | −0.07 ns | −0.47 ns | −0.04 ns | +0.01 ns |

Ordering is unchanged from the n=30 wave — which is the point: the extra 1,609 clips
tightened CIs and **exposed a data bug**, but did not move any conclusion.

### Lessons

1. **"Run everything" is usually the wrong instrument.** Compute the CI-vs-n curve first:
   here 1.8% of the full budget bought the entire missing table, and the other 98% bought
   0.06× CI width.
2. **A paired key proves identity, not comparability.** Same episode + same view can still
   be a different *render*. Any paired benchmark across separately-generated trees needs an
   explicit same-measurement gate (fps + length at minimum), not just a join key.
3. **A ceiling violation is a bug report.** Three mechanism hypotheses were killed by
   measurement before the real cause showed up; the giveaway was that the anomaly held for
   *all nine methods* in one cell, which no method-specific story can explain.
4. **Store provenance fields even when unused** (`source_fps`, `effective_fps`,
   `sampled_source_frames`). They turned a 20-GPU-day re-scoring problem into a
   post-hoc filter.

## Main-results matrix (2026-09-17) + the AMC over-smoothing gate

`assets/scripts/table_main_results.py` builds the paper's main grid: **rows = (model,
embodiment, view_type), columns = the 9 acceleration methods**, one table per metric.
It imports every gate, delta and `resolved` decision from `table_dense_nr.py` rather
than reimplementing them, so it cannot disagree with T3b about the same pairs.

Aggregating over **mount** (not over model or embodiment) is what makes the cells
readable: 88 non-empty cells, **77 at n>=8** (88%). Mount is not the axis a reader
cares about, and mount-level cells were too thin to scan as a matrix. The cost is
reported, not hidden: cells with >=34% head-mount clips carry the `CONFOUNDED_MOUNTS`
marker for AMC/VI.

### What the matrix immediately exposed: 4 cells where accel BEAT dense

The pooled T3b table has one row per method, so a per-cell ceiling violation averages
away. At (model x embodiment x view) granularity, 4 cells had an accel method beating
its own dense counterpart with the CI excluding 0 and p down to 2e-05:

| cell | delta | p |
|---|---|---|
| fastercache dreamgen/humanoid/MV | AMC **+7.02** | 4e-05 |
| pisa ctrlworld_4view/humanoid/MV | AMC **+4.35** | 2e-05 |
| svg1 ctrlworld_4view/humanoid/MV | AMC **+2.57** | 6e-04 |
| radial dreamdojo/single_arm/SV | VI **+0.18** | 0.023 |

`comparable_pair()` had already cleared all of them — identical `source_fps`,
`effective_fps`, frame count, analysis geometry **and** mount — so this is NOT the
render-mismatch bug found earlier. Measuring the pixels of those exact pairs found the
real mechanism, and it is **second-order temporal jerk** (how far the middle frame sits
from the linear interpolation of its neighbours), which is precisely AMC's VFI-gain
denominator:

| cell (n=10 pairs) | dAMC | sharpness | temporal | **JERK** |
|---|---|---|---|---|
| fastercache dreamgen/humanoid/MV | +7.02 | x0.918 | x0.897 | **x0.859** |
| pisa ctrlworld_4view/humanoid/MV | +3.80 | x0.941 | x0.913 | **x0.896** |
| svg1 ctrlworld_4view/humanoid/MV | +2.79 | x0.970 | x0.918 | **x0.905** |
| *control:* itm dreamgen/humanoid/MV | −4.72 | x0.759 | x1.376 | **x1.275** |
| *control:* sito dreamgen/bimanual/MV | −9.77 | x0.746 | x1.001 | **x0.965** |

Every positive-AMC cell has **JERK < 1** (output more predictable than dense); every
negative one sits at **x0.96–x1.28**. So a positive AMC delta measures over-smoothing,
not quality.

**This is a larger effect than the existing `ACCEL_CAVEATS` blur bias** (+7.02 vs the
+0.50 measured at realistic blur in E7), because it is driven by *temporal* smoothing
which E7's synthetic spatial blur never reproduced. The caveat's "a positive AMC delta
<= ~+0.5 must not be read as better" bound was therefore too small to catch this.

### Enforcement

New `NO_IMPROVEMENT_VS_DENSE` set in `table_dense_nr.py` (holds `active_motion_continuity`):

- `resolved_flag()` refuses to claim a positive delta on such a column vs dense.
- `artifact_positive()` reports it instead, so the cell is **flagged, never silently
  dropped**: `!` in the console, `\textit{...}$^{\S}$` + a footnote in LaTeX, and an
  `artifact_positive` column in the CSV.

Result: 3 AMC cells moved from RESOLVED to artifact-flagged; **T3b changed 0 fields**
(the pooled numbers were already all negative), so this adds enforcement at per-cell
granularity without disturbing the published headline.

The remaining `radial dreamdojo/single_arm/SV` VI **+0.18** is left as a resolved
positive on purpose. It is statistically real (p=0.023, CI [+0.04, +0.68]) but
**1.8% of the Visual Integrity IQR and ~15x smaller than the median resolved negative
effect** (2.73). Rather than invent a threshold to suppress it, it is reported: at this
magnitude "radial is indistinguishable from dense on VI" is the honest reading, which
matches its pooled row (−0.07 ns).

### Speedup was deliberately NOT merged into the main table

`metrics/reports/speedup_summary_full.csv` has the right granularity
(model/embodiment/view) but only **29/108 cells carry a speedup number**, and
`timing_source` shows they are transcribed from the tuning-notes rules, not measured
per-episode like the NR deltas (71–228 pairs each). Joining them would put a
hand-copied single number next to a bootstrapped distribution in the same table. The
pooled join (9 methods) is sound and worth reporting on its own:

- **No speed/quality trade-off**: Spearman(speedup, dVI) = **+0.32** — the *faster*
  methods lose *less* quality. DiCache (1.50x) and WorldCache (1.47x) are both fastest
  and among the cleanest; ITM/SiTo cost ~7x more quality for 1.13–1.15x. SVG2 is worst
  on both axes (0.97x = slower than dense, and −1.68/−2.66).
- **NR agrees with PSNR**: Spearman(PSNR-retention, dVI) = **+0.55**, vs dAMC = **+0.60**
  — a second independent check after the pixel-sharpness rho +0.67.
- Caveat: `FasterCache PSNR-retention = 1.004` (better than dense) is the same *class*
  of signal as the render-mismatch bug; the PSNR pipeline has no `comparable_pair`
  equivalent, so it may still be affected.

### Usage

```bash
models/dreamgen/.venv/bin/python assets/scripts/table_main_results.py \
  --run metrics/WorldScore/results/dense_nr_v3_pooled
```

Writes `tables/main_<metric>.{csv,tex}` for all 5 columns. Legend: `[x]`/bold =
resolved, `x*`/dagger = n<8 (shown, not claimable), `x!`/§ = artifact-positive,
`--` = method not run on that configuration, `#`/ddagger = head-mount confound.
