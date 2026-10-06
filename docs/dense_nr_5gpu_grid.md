# Dense-NR wave on 5 machines — 4 metrics, paper grid

Current state and the exact commands for the 5-box relaunch. Read the two warnings
first; they are both things that silently produce wrong output rather than erroring.

## State (2026-09-21)

| | |
|---|---|
| protocol | `dense-nr-v3.2-closed-world` |
| reported metrics | **4** — Visual Integrity, Text Alignment, Actor Activity, AMC |
| manifest | 86,743 clips |
| scored | ~31,400 (36%) |
| run dir | `metrics/WorldScore/results/dense_nr_full` (shared filesystem) |

Active Rate is still computed (it gates AMC) but is no longer a reported column —
see `docs/phyworld_nr_adapted_metrics.md`.

## WARNING 1 — you cannot just add 2 boxes to the running wave

The split is `rows[k*W + s :: W*MACHINES]`. The 3 boxes currently running use
`MACHINES=3, W=8` → stride 24, offsets 0–23. **Those 24 offsets already cover every
row of the manifest.** A 4th/5th box launched with `MACHINES=5` (stride 40, offsets
24–39) lands on rows the first three already own:

```
3-machine stride-24 covers   : 1200 / 1200 rows
extra m3,m4 stride-40 covers :  480
overlap (done twice)         :  480  = 100% of the extra boxes
```

So the 2 new boxes would do **zero** new work. Every box must be restarted with
`MACHINES=5`. Nothing already scored is lost — results are per-id files and a worker
skips ids that exist.

`scripts/metrics/run_dense_nr_full.sh extra` (the reverse walker) is also no longer the answer:
it was the 2-machine trick and `run_dense_nr.py` rejects `--reverse` with `--machines > 1`.

## WARNING 2 — the paper table does NOT need the whole manifest

The table is one cell per `(embodiment, model, split)` × method row. That needs only
`TARGET` clips per cell:

| target / cell | clips left | 5 boxes @2.18 clips/min/box |
|---|---|---|
| 30 | 12 | minutes — **effectively already done** |
| 100 | 2,714 | ~4 hours |
| 200 | 7,761 | ~12 hours |
| **300** | **19,692** | **~1.25 days** ← default |
| whole manifest | 54,737 | ~3.5 days |

The subset is an **id list over the existing manifest**, not a new manifest. Ids,
`pair_key`s and shard offsets are unchanged, so records already scored count towards
it and a later full wave simply resumes.

## Relaunch — run on each of the 5 boxes

Stop the old process group on each box that is currently running:

```bash
bash scripts/metrics/dense_nr_status.sh          # prints the PGID per machine
kill -- -<PGID>
```

Then, one box per index (`0`, `1`, `2`, `3`, `4` — **never the same index twice**):

```bash
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics
bash scripts/metrics/run_dense_nr_grid.sh 0      # box 0
bash scripts/metrics/run_dense_nr_grid.sh 1      # box 1
bash scripts/metrics/run_dense_nr_grid.sh 2      # box 2
bash scripts/metrics/run_dense_nr_grid.sh 3      # box 3
bash scripts/metrics/run_dense_nr_grid.sh 4      # box 4
```

Env knobs: `GPU=1` if the free card is not 0, `WORKERS=6` if the card is shared,
`TARGET=200` for a faster/looser table, `FULL=1` to ignore the subset and score the
whole manifest.

The wrapper rebuilds the id subset at launch (~30 s, CPU only) and writes
`manifest_grid_target<TARGET>.json` plus a coverage report in `logs/coverage_m<k>.txt`.
A box starting later gets a smaller list, which is correct — it is still a subset of
what the others are walking.

Monitor from any box:

```bash
bash scripts/metrics/dense_nr_status.sh
```

## The table

```bash
# absolute values (Real Video / Dense / 9 methods)
models/dreamgen/.venv/bin/python assets/scripts/table_paper_grid.py \
  --run metrics/WorldScore/results/dense_nr_full

# method - dense on the same clip (the defensible form for a claim)
models/dreamgen/.venv/bin/python assets/scripts/table_paper_grid.py \
  --run metrics/WorldScore/results/dense_nr_full --delta
```

Layout matches the KIVA sheet: one block per embodiment, rows = Real Video + each
model's Dense and 9 methods, columns = Markovian | Non-Markovian × 4 metrics. Writes
`tables/table_paper_grid_{absolute,delta_vs_dense}.csv`.

`?` = fewer than 8 clips in that cell. `-` = that method was never rendered for that
cell (scoring cannot fix it; the video does not exist).

### Read down a column, not across blocks

Every method row inside one block shares the same episode/mount mix, so comparing
methods **down** a column is like-for-like. Comparing **across** embodiment blocks is
not: Actor Activity is a per-mount statistic and AMC is barred cross-embodiment
(`NOT_CROSS_EMBODIMENT` — humanoid GT scores AMC ~48 vs ~66 for the others at matched
motion, which is a baseline offset, not quality).

## Coverage / what is structurally missing

```bash
models/dreamgen/.venv/bin/python scripts/metrics/nr_table_coverage.py \
  --run metrics/WorldScore/results/dense_nr_full --target 300
```

Prints scored/planned per cell, what is still needed, and the cells no amount of
scoring can fill. Known holes, all because the videos were never generated:

| method | missing cells |
|---|---|
| `fastercache` | ctrlworld (all), ctrlworld_4view_grid (all), + 1 single_arm split |
| `itm`, `sito` | dreamdojo (all embodiments) |
| `svg1/svg2/radial/pisa` | ctrlworld bimanual + humanoid |
| `gt` | dreamdojo bimanual/humanoid, dreamgen single_arm |

567 manifest rows point at videos that have since been deleted (255 of them
`fastercache/dreamdojo`). The coverage script drops them by default so the per-cell
target can actually be reached; `--keep-missing` restores the old behaviour.

## Do not rank methods on a pooled number without the common-support check

`table_dense_nr.py` pools each method over the cells **it** covers, and those differ
(6 cells / 386 clips for fastercache vs 11 / 891 for dicache; the 9 methods share
**zero** individual clips). `common_support_ranking()` re-ranks on the cells all 9
cover and prints under each T3b column. Measured:

| column | max rank shift | verdict |
|---|---|---|
| Visual Integrity | ±1 | **stable** — rankable |
| Text Alignment | ±4 | coverage-driven |
| Actor Activity | ±4 | coverage-driven |
| AMC | ±3 | coverage-driven |

`fastercache` drops 4 places on Text Alignment once it is forced onto the shared
cells. Only Visual Integrity supports a 9-way ranking as-is.
