# Dense-NR: cover the full spec distribution

The authoritative distribution is **29,343 episodes** (dense 3,166 + caching 8,693 +
sparse attention 12,664 + token reduction 4,820). This is how the job gets there.

## The problem found 2026-09-21

The live `manifest.json` was built **2026-09-17** and had drifted from disk in BOTH
directions:

| | effect |
|---|---|
| **missing 591 view-clips** | 197 `single_arm / MV / dreamgen` **dense** episodes were re-rendered 09-19, after the manifest was built. That spec cell read **60/128** makovian and **0/129** non_makovian — the entire non_makovian half was invisible. |
| **counting 567 deleted view-clips** | mp4s removed since: fastercache 267, svg1 260, pisa 34, radial 6. This inflated `fastercache` to **2,620** against a spec of 2,361. |

So the old manifest could never satisfy the spec: it lacked rows the spec requires and
padded the count with rows that no longer exist.

`scripts/nr_distribution_audit.py` is the check. It collapses view-clips back to episodes
(via `pair_key` minus the trailing view segment) and diffs against the spec table, which
is why the drift is visible at all — the manifest is in **view-clips** (86,767) while the
spec is in **episodes** (29,343), and one 4-view ctrlworld episode is 4 rows.

## After the rebuild: exact match

```
method        class                    spec  manifest   delta
dicache       Acceleration caching     3166      3166      +0
fastercache   Acceleration caching     2361      2361      +0
worldcache    Acceleration caching     3166      3166      +0
dense         Dense                    3166      3166      +0
pisa          Sparse attention         3166      3166      +0
radial        Sparse attention         3166      3166      +0
svg1          Sparse attention         3166      3166      +0
svg2          Sparse attention         3166      3166      +0
itm           Token reduction          2410      2410      +0
sito          Token reduction          2410      2410      +0
TOTAL                                 29343     29343
```

All 12 dense grid cells `ok`, every per-class total exact:

```
Dense                   spec   3166   manifest   3166   +0
Acceleration caching    spec   8693   manifest   8693   +0
Sparse attention        spec  12664   manifest  12664   +0
Token reduction         spec   4820   manifest   4820   +0
```

## Why the previous wave could not cover it — two independent causes

Of the 34,532 spec view-clips the running wave omits:

| cause | clips |
|---|---|
| `--target 300` per-cell cap | **33,941** |
| manifest stale (rows absent entirely) | 591 |

### Cause 1 (33,941): `cell()` omits `view_type`, and the cap counts view-clips

`scripts/nr_table_coverage.py` keys its budget on `(embodiment, model, split)`. The spec
distribution is one row per `(embodiment, view, model)`. The missing **view** dimension
merges the MV and SV branches of the same model into ONE 300-clip budget — and because MV
episodes are 3–4 clips each while SV episodes are 1, MV consumes it:

```
bimanual / dreamgen / makovian / dense    pool = 600 MV + 150 SV clips,  cap 300
```

Result, per `(embodiment, view, model)` for dense:

| cell | episodes with nothing | partial (views missing) | complete |
|---|---|---|---|
| bimanual MV dreamgen | 45 | **251** | **4** |
| bimanual MV ctrlworld_4view_grid | 20 | 256 | 24 |
| bimanual SV dreamgen | **174** | 0 | 126 |
| bimanual SV **dreamdojo** | 0 | 0 | **300** |
| single_arm MV dreamgen | 197 | 0 | 60 |

Two distinct defects fall out of this:

1. **Unfair model-to-model comparison.** `bimanual/SV/dreamgen` kept 126 of 300 episodes
   while `bimanual/SV/dreamdojo` kept 300/300 — purely because DreamGen also has a
   multiview branch competing for the same budget and DreamDojo does not. A main-results
   table comparing the two on SV would be comparing very different sample sizes.
2. **Partially-scored multiview episodes.** The cap is on clips and
   `sorted(pool)[:target]` orders by `id` (a sha256 hash, i.e. effectively at random), so
   it slices *across* the views of one episode. `bimanual/MV/dreamgen` has **251 episodes
   with only some views scored and just 4 complete**. A 4-view episode with 2 views scored
   cannot be pooled against one with 3 different views scored, so per-episode and per-view
   aggregation both break.

This was not a crash — `--target 300` did exactly what it was asked (300 clips per cell,
for a tight per-cell CI). It is simply the wrong instrument for "cover the distribution",
and the missing `view_type` made even its own sampling uneven.

### Cause 2 (591): the manifest never contained the rows

197 `single_arm / MV / dreamgen` dense episodes were re-rendered 09-19; the manifest was
built 09-17. Those 591 view-clips are in neither the manifest nor the subset, so no amount
of running the current wave produces them. `run_dense_nr_grid.sh` now refuses to launch
when any video is newer than `manifest.json`.

## Will running full cover every video in the distribution?

**Predictions: yes, 100%.** Verified four ways on the rebuilt manifest.

**1. Episode counts per method — exact**

Every per-method total lands on the spec with `delta +0`, and per **cell × method** every
method that exists matches its dense cell exactly (no partial cells anywhere):

```
emb        vt  model         dense  dicach  worldc  faster    pisa  radial    svg1    svg2     itm    sito
bimanual   MV  ctrlworld       300     300     300       -     300     300     300     300     300     300
bimanual   MV  dreamgen        300     300     300     300     300     300     300     300     300     300
bimanual   SV  dreamdojo       300     300     300     300     300     300     300     300       -       -
bimanual   SV  dreamgen        300     300     300     300     300     300     300     300     300     300
humanoid   MV  ctrlworld       248     248     248       -     248     248     248     248     248     248
humanoid   MV  dreamgen        248     248     248     248     248     248     248     248     248     248
humanoid   SV  dreamdojo       216     216     216     216     216     216     216     216       -       -
humanoid   SV  dreamgen        260     260     260     260     260     260     260     260     260     260
single_arm MV  ctrlworld       257     257     257       -     257     257     257     257     257     257
single_arm MV  dreamgen        257     257     257     257     257     257     257     257     257     257
single_arm SV  dreamdojo       240     240     240     240     240     240     240     240       -       -
single_arm SV  dreamgen        240     240     240     240     240     240     240     240     240     240
```

The 9 `-` cells are structurally absent (never rendered), and they reproduce the spec's own
irregular totals exactly:

| method | absent cells | arithmetic |
|---|---|---|
| `fastercache` | all 3 `ctrlworld / MV` | 300+248+257 = 805 → 3166 − 805 = **2361** ✓ |
| `itm`, `sito` | all 3 `dreamdojo / SV` | 300+216+240 = 756 → 3166 − 756 = **2410** ✓ |

So the spec's 2,361 and 2,410 are not arbitrary — they are 3,166 minus precisely the cells
where those methods were never generated, and the manifest reproduces that to the episode.

**2. View completeness — 0 gaps.** Across all 33,624 (episode × method) combinations in the
spec, every one has its full view set (`len(view_index set) == n_views`). This is the defect
the `--target 300` subset had (251 partial-view episodes in `bimanual/MV/dreamgen` alone);
the full manifest has none.

**3. vs-dense pairing — perfect.** 0 acceleration `pair_key` lacks a dense counterpart, in
every one of the 14 model-cells. **67,639 (accel, dense) pairs** are formable. This is the
paper's main axis and it is complete.

**4. Files excluded from the manifest are correctly excluded.** Of 135,738 mp4 on disk the
manifest uses 36,587 files (→ 86,767 view-clips, 2.37 per file, since a multiview grid file
splits into 3–4 views). The 99,151 unused are:

| excluded | n | why |
|---|---|---|
| `input/**/view_*.mp4` under each method tree | 30,472 | GT input copied into all 10 method trees. GT is inventoried ONCE from `dense/input` (2,963 rows) instead of being scored 10x. |
| `full_gt.mp4` | 16,001 | GT source; enters as the `gt_<model>` anchor row, not as a separate clip |
| `seg_*.mp4` | ~23,600 | intermediate autoregressive segments; the assembled `full_pred.mp4` is what is scored |
| `gt_all_views.mp4` | 8,032 | GT grid composite (also consumed as the anchor) |
| `full_merged.mp4` | 7,400 | GT‖pred side-by-side composite, not a prediction |
| `flow.mp4`, `raw.mp4`, `data_for_web`, `augment`, `tmp` | 4,749 | diagnostics / non-benchmark trees |

### The one real gap: GT anchors missing on 6 of 12 spec rows

GT anchors are discovered as a **sibling file** (`full_gt.mp4` / `gt_all_views.mp4`) written
next to the prediction. The DreamGen/DreamDojo singleview trees are flat
`episode_XXXXXX.mp4` + `.txt` and emit no GT sibling, so:

| spec row | GT anchor |
|---|---|
| bimanual MV dreamgen / ctrlworld | ✓ 100% |
| humanoid MV dreamgen / ctrlworld | ✓ 100% |
| single_arm MV ctrlworld | ✓ 100% |
| single_arm SV dreamdojo | ✓ 100% |
| **bimanual SV dreamgen / dreamdojo** | **✗ 0%** |
| **humanoid SV dreamgen / dreamdojo** | **✗ 0%** |
| **single_arm MV dreamgen** | **✗ 0%** |
| **single_arm SV dreamgen** | **✗ 0%** |

Checked: no `full_gt.mp4` exists in those output trees, and their `input/` trees hold no
mp4 at all. So this is a **rendering gap, not a scoring or inventory gap** — no amount of
scoring produces it. Consequences:

* the "Real Video" ceiling row of the paper table is only available on 6 of 12 spec rows;
* **the vs-dense comparison is unaffected** (complete, see check 3), so every acceleration
  claim still stands on all 12 rows.

To close it, GT for those cells must be rendered next to the predictions (or extracted from
`datasets/` at the prediction's fps/resolution — the co-located sibling is used precisely
because it is byte-identical in geometry, so a re-extraction has to match fps, crop and
frame count or the delta acquires a resolution/fps caveat).

### Is GT needed at all for an NR run?

NR metrics are computed **per clip, with no reference**, so GT is not needed to *compute*
anything. It is only used at table time, for three separate jobs — and only one of them is
presentation:

| job | needs GT? | replaceable? |
|---|---|---|
| **accel vs dense** (the paper's main axis) | **no** | — already complete, 67,639 pairs |
| **ceiling test** = does the metric rank real video above generated? | yes | no, but it is a property of the METRIC, not of each cell |
| "Real Video" reference row | yes | it is a presentation choice |
| "how far is the model from real footage" claim | yes | only if the paper makes that claim |

**Why accel-vs-dense needs no GT**, column by column:

* `visual_integrity` — in `CEILING_FAILED`, so it may not rank models. But it is also in
  `CEILING_FAILURE_CANCELS_VS_DENSE`: the generative-vs-real bias cancels when a clip is
  compared to *its own dense counterpart*, so vs-dense claims are allowed (E7 verdict 4).
* `text_condition_alignment` — `PAIRED_ONLY` because CLIPScore has a ~7-point per-caption
  style/length bias against a ~2-point image-text signal. The bias cancels whenever both
  clips are scored against the **same prompt**, which a dense/accel pair satisfies by
  construction (same episode). Measured: **97.42%** of the 67,639 pairs carry an identical
  instruction; the 1,745 that do not are DreamDojo, which has no instruction at all and is
  correctly excluded from `TEXT_CONDITIONED_MODELS`, so no generic-caption fallback occurs.
* `actor_motion_raw`, `active_motion_continuity` — per-clip quantities; the delta is taken
  within one episode+view.

**Why the ceiling test still matters.** It is the only thing that can catch an NR metric
that rewards "generated-ness" instead of quality, and it already did: `visual_integrity`
scores predictions *above* their co-located GT (dreamdojo +0.90, p=5e-05; 4view +0.52).
Without GT, that column would have been reported as a model-ranking metric and would have
been wrong. But this is a statement about CLIP-IQA+, not about any particular cell, so the
6 cells that do have GT anchors are sufficient to establish it.

**Cost.** GT is 5,758 of the 53,695 clips left (10.7%): 3.42 → 3.05 days on 5 boxes.
So dropping GT saves ~0.37 days and makes the 6-cell anchor gap irrelevant — **if** the
paper's claims are confined to accel-vs-dense. Keep GT if the table needs a Real Video row.

Note independently: DreamDojo carries no instruction anywhere (6,288 clips), so the Text
Alignment column is N/A for all three DreamDojo rows by design, GT or not.

## DROP `--only-ids`

The whole manifest is **86,767** view-clips; the spec requires **83,804** of them
(the other 2,963 are legacy `gt`-model anchor rows). So the spec is ~97% of the manifest:
there is nothing meaningful to subset away, and **the whole manifest is a superset of the
spec**.

`--target 300` sampling was the wrong tool. It *samples* 300 clips per cell — that gives a
tight CI per cell but does **not** cover the distribution, and it forces every box to pass
the byte-identical id file or the stride partition stops covering the subset. Dropping it
removes that entire failure class.

The old subsets are stale (they name deleted ids and cannot name the 591 new ones):

```
manifest_grid_target300.json      <- do not use
manifest_table_subset.json        <- do not use
```

## Swapping the manifest is position-sensitive

Workers partition by **position**: `rows[k*W + s :: W*MACHINES]` over the id-sorted list.
Replacing `manifest.json` under a running box shifts every row's index, so the live boxes
silently stop forming a partition — some rows get scored twice, others never.

Clip `id` is `sha256(method:relative:view)[:20]`, a pure **content** hash, so it does not
depend on manifest order: **all 32,273 already-scored results survive the swap** (measured
orphans: 0). Only the in-flight position breaks. Hence `scripts/nr_promote_manifest.py`
refuses while any `worker_*.log` has been touched in the last 10 minutes.

## Runbook

### A 4-GPU box is FOUR machine indices, not one

The partition is `rows[MACHINE*W + shard :: W*MACHINES]` where `W` = workers per GPU. One
machine index owns exactly `W` shard slots = **one GPU's worth**. Running 4 GPUs under a
single index would need `W=32` there and `W=8` elsewhere — but `W` is a factor of the global
`stride`, so a box with a different `W` computes a different stride and the partition stops
partitioning: some rows owned by nobody, others twice, and **nothing errors** because every
write is idempotent.

So with **4 single-GPU boxes + 1 four-GPU box**: `MACHINES=8`, indices 0–3 on the single
boxes, 4–7 on the big box (one per GPU). `scripts/run_dense_nr_multigpu.sh` does that.

**Measured: 8 workers/GPU is safe on a 32-core box even with 4 GPUs.** Per worker on this
workload: **~0.93 core**, ~3.0 GB RSS; with 8 workers the GPU sits at **99–100% util** using
57 GB of 80 GB VRAM. So 4×8 = 32 workers needs ~30 of 32 cores and ~96 GB RAM — it fits, and
the work is **GPU-bound, not CPU-bound**, so cores are not the limiting resource. VRAM is
per-GPU so 57 GB/GPU is unaffected by running four.

Throughput measured at 2.25 clips/min/GPU:

| configuration | GPUs | days for 53k remaining |
|---|---|---|
| 4 single boxes only | 4 | 4.09 |
| **4 single + 4-GPU box** | **8** | **2.04** |

### Steps

**1. Stop every box** (all of them, not just one):

```bash
pkill -f 'run_dense_nr.py'      # on EACH box
```

**2. Rebuild + verify** (once, from any box):

```bash
NEW=metrics/WorldScore/results/_manifest_rebuild
models/dreamgen/.venv/bin/python scripts/run_dense_nr.py --out $NEW --inventory-only \
  --methods dense dicache worldcache fastercache pisa radial svg1 svg2 sito itm

models/dreamgen/.venv/bin/python scripts/nr_distribution_audit.py --run $NEW --with-gt
# every `delta` column MUST read +0 and TOTAL 29343 before you promote
```

**3. Promote** (refuses if any worker is alive):

```bash
models/dreamgen/.venv/bin/python scripts/nr_promote_manifest.py \
  --from metrics/WorldScore/results/_manifest_rebuild \
  --to   metrics/WorldScore/results/dense_nr_full
```

**4. Relaunch every box with the SAME `MACHINES` and the SAME `WORKERS`, no `--only-ids`.**
Count `MACHINES` in **GPUs**, not boxes:

```bash
# 4 single-GPU boxes -- one index each
MACHINES=8 bash scripts/run_dense_nr_grid.sh 0
MACHINES=8 bash scripts/run_dense_nr_grid.sh 1
MACHINES=8 bash scripts/run_dense_nr_grid.sh 2
MACHINES=8 bash scripts/run_dense_nr_grid.sh 3

# the 4-GPU box -- four consecutive indices, one per GPU
MACHINES=8 bash scripts/run_dense_nr_multigpu.sh 4 4
```

Two boxes on the same index is the silent failure: both do identical work and 1/N of the
manifest is never scored. Every write is idempotent, so **nothing errors to tell you**.
Verify with:

```bash
bash scripts/dense_nr_status.sh
models/dreamgen/.venv/bin/python scripts/nr_shard_probe.py \
  --run metrics/WorldScore/results/dense_nr_full
```

## Cost of full spec coverage

37.0% of the spec is already scored (31,005 of 83,804 view-clips), so **52,799 remain**:

| boxes | time |
|---|---|
| 4 | 100.9 h (4.2 days) |
| 5 | 80.7 h (3.4 days) |
| 6 | 67.3 h (2.8 days) |

At 2.18 clips/min/box (8 workers, 80GB A100). Scoring the whole manifest instead of just
the spec costs ~3% more and removes all subset coordination.

## Why `nr_shard_probe.py` exists, and its limits

You cannot read another box's `--machines` from here. Two inference attempts failed, and
both failure modes are worth knowing:

1. **gcd over a whole worker log.** Logs are opened in APPEND mode, so a relaunched box
   holds STARTs from every geometry it has ever run (this dir has both a `--machines 3`
   and a `--machines 5` wave). The gcd of a mixture is 1 = meaningless.
2. **Residues of recent result mtimes modulo a swept stride.** When every offset has an
   owner, *every* stride looks consistent, so the sweep cannot discriminate — it
   spuriously reported `MACHINES=16`.

The probe now parses only STARTs after each worker's last startup banner. It still needs
a few hundred STARTs per worker to be conclusive, so it is only reliable on a wave that
has been running a while. **The authoritative answer is what you typed on each box** —
record it.
