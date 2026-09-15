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
