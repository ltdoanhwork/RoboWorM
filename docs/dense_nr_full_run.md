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
`results/dense_nr_v2` only after v1 exits. v2 fixes source-frame normalization
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

