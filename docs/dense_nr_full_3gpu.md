# Full dense-NR wave on 3 A100s — hand-off

Everything is prepared and smoke-tested. Each machine runs **one** command.

## URGENT (2026-09-19): quota was silently eating 6% of clips — restart m1 and m2

A `df` showing 383T free does not mean you can write: the fileset has its own quota, and
from 15:50 on 09-19 writes started failing with `OSError: [Errno 122] Disk quota exceeded`.
Measured over the following 76 minutes: **474 clips scored, 30 failed = 6.0% loss**,
projected **~4,100 clips lost** over the rest of the wave. It was invisible for three
reasons, all now fixed:

| symptom | why it hid | fix |
|---|---|---|
| clips vanishing | a row is owned by exactly ONE worker shard, so a failure is never revisited by the other forward machines | quota/`ENOSPC` is now retried 4x with 5/10/15-min backoff, like OOM |
| progress read high | the failed write left a 0-byte `<id>.<host>.<pid>.tmp` in `results/`, and `ls results \| wc -l` counted it as a scored clip | `dump()` unlinks its stub on failure; status counts `find -name '*.json'` |
| nothing in the status output | the only trace was 28 files under `errors/` | status now prints `failed (perm)`, `!! write stubs` and `!! QUOTA` |

`record_error()` was also added, because the old handler called `dump()` from inside the
`except` block — under quota that raises again, escapes the retry loop and **kills the
worker process**, which is how a machine silently drops from 8 workers to 3.

**Machine 0 has been restarted and is running the fix** (`RETRY QUOTA 1/3 in 300s` appears
in its worker logs; it now leaves no stubs). Machines 1 and 2 are still on the old code —
every stub left in `results/` belongs to their hosts. On each of those boxes:

```bash
bash scripts/metrics/dense_nr_status.sh --resume   # or: bash scripts/run_dense_nr_full_m<k>.sh
```

Requeue anything already lost (safe, idempotent — deletes only the error record, so the
row becomes pending again and the reverse walker or a restarted shard re-scores it):

```bash
R=metrics/WorldScore/results/dense_nr_full
find $R/results -name '*.tmp' -size 0 -delete
python3 -c "
import json,os;R='$R/errors'
q=[n for n in os.listdir(R) if 'Disk quota' in json.load(open(f'{R}/{n}')).get('traceback','')]
[os.remove(f'{R}/{n}') for n in q];print('requeued',len(q))"
```

Note the wave only needs ~12 GiB (147.6 KiB/record x 86,743), so this is other jobs on the
shared workspace, not the wave itself. If quota stays closed the run makes no progress —
retries turn loss into delay, not into throughput.

## Mid-wave audit at 20% (17,439 records) — the numbers reproduce the frozen run

Checked on 2026-09-19 with `assets/scripts/table_paper_main.py`, and the answer is that the
partial wave is already telling the same story as the 1,609-clip stratified run it
supersedes. Pooled Visual Integrity delta vs dense, 9 methods:

| method | old (1.6k run) | new (17.4k run) |
|---|---|---|
| ITM | −6.31 (n=168) | −6.31 (n=354) |
| SiTo | −5.36 (n=168) | −5.50 (n=325) |
| SVG2 | −1.68 (n=179) | −1.73 (n=367) |
| DiCache | −1.22 (n=227) | −1.14 (n=441) |
| WorldCache | −0.89 (n=228) | −1.06 (n=439) |
| PISA | −0.63 (n=180) | −0.67 (n=326) |
| SVG1 | −0.48 (n=174) | −0.31 (n=360) |
| Radial | −0.07 (n=180) | −0.35 (n=340) |
| FasterCache | −0.28 (n=106) | −0.25 (n=221) |

**9/9 sign agreement, Spearman ρ = +0.950.** The ranking is stable, so the remaining 80%
buys precision (narrower CIs, more `*` cells), not a different conclusion.

Three further checks:

* **Representativeness.** Coverage is 18.4–24.9% in *every* method, model, embodiment and
  camera mount — no branch is starved. This is the id-sorted walk doing its job: ids are
  sha256, so walking them in order is a deterministic shuffle, and any prefix is a
  random sample of the whole manifest.
* **Pairing efficiency is ~25% and will rise to ~100%.** A pair needs both the accel clip
  *and* its dense counterpart scored, and they have independent ids, so at 20% done only
  ~20% of accel clips have a partner (`no dense partner: 679–1272` per method). Paired `n`
  therefore grows faster than linearly from here.
* **GT-ceiling.** Reproduces the known E1 result: `visual_integrity` violates it
  (pred > GT on 73% of `ctrlworld_4view_grid` and 77% of `dreamdojo` pairs), which is why
  it stays descriptive-only and never carries a claim. `actor_motion_raw` passes
  everywhere (GT above pred); AMC is a wash (|median| ≤ 0.5).

Two hygiene bugs fixed while auditing:

* The run dir still held **128 records from `dense-nr-v3-closed-world`** (survivors of the
  hardlink pre-seed). They lack `actor_per_frame[*].frame_flow_med`, so `subject_metrics`
  raised `KeyError` and no table could be built at all. Added `T.load_records()`, which
  *filters* to one protocol instead of merely warning — a table mixing protocol versions
  is not a table, because the gates differ.
* The same loader skips `.tmp` stubs and unreadable files, so a quota squeeze can no
  longer corrupt a table build.

## Run this

| Machine | Command |
|---|---|
| A100 #1 | `bash scripts/metrics/run_dense_nr_full.sh 0` |
| A100 #2 | `bash scripts/metrics/run_dense_nr_full.sh 1` |
| A100 #3 | `bash scripts/metrics/run_dense_nr_full.sh 2` |
| **A100 #4+ (added later)** | **`bash scripts/metrics/run_dense_nr_full.sh extra`** |

Run from the repo root. Nothing else to set; the run dir, manifest and the already-scored
records are in place. If a box's free card is not GPU 0, or the card is shared:

```bash
GPU=1 bash scripts/metrics/run_dense_nr_full.sh 1      # different card
WORKERS=6 bash scripts/metrics/run_dense_nr_full.sh 1  # leave headroom on a shared card
```

**The one rule: never run the same index twice.** The split is index-based, so two
machines both on `m0` leave a third of the manifest unscored — and because every write is
idempotent, *nothing will error to tell you*. One index per machine, exactly once.

### Adding a GPU to an already-running wave

Use `run_dense_nr_full.sh extra`, **not** a 4th stride index. With 3 machines × 8 workers
the split is `rows[k*8 + s :: 24]`, so every offset 0–23 is taken; a 4th index would
duplicate an existing one. Re-sharding to `--machines 4` would mean restarting all three
running boxes, and the partition would be wrong until they all came back.

The extra box instead walks the id-sorted manifest **backwards**. That is nearly free
because stride sharding makes the finished set a *prefix*: measured at 20% done, deciles
0–10% were 100% complete, 10–20% were 73%, and everything past 30% was only ~2% (the
pre-seeded v3 records). Walking the first 40,000 rows from the tail hits just **2.1%**
already-done ids, and those are skipped instantly without touching the GPU. Real
duplication only starts when the two fronts meet, i.e. near the end.

Only **one** box may do this — there is a single backwards direction. A second extra GPU
needs a genuine re-shard: stop everything, relaunch with `MACHINES=5`.

## Scope and cost (measured, not estimated)

```
full inventory over all 10 method trees   86,743 view-clips
already scored (v3 runs, pre-seeded)       3,699
still to score                            83,095
```

**Sustained throughput, measured over 4.3 hours of real running: 2.02 clips/min per GPU**
(8 workers, A100-80GB). Confirmed independently by a 30-minute rolling window (2.20) and
by mean elapsed time (201s/clip mean, 139s median → 2.39 ideal).

Do NOT trust a short spot-check right after launch. Measuring the first 5 minutes gave
**4.80 clips/min**, 2.4x too high, because the fast clips in each worker's queue finish
first. Only a window of ≥30 minutes is honest.

| GPUs | clips/min | wall time for 83k |
|---|---|---|
| 1 | 2.02 | ~28 days |
| **3** | **~6.1** | **~9.3 days** |

`WORKERS=8` is the measured knee on an 80GB A100: throughput went 2.1 → 3.0 clips/min from
5 → 10 workers (sub-linear, GPU-bound), while 10 workers OOM-killed 13% of clips on
4-view-grid multiview, whose SAM2 propagation peaks far above the median. 8 workers plus
`expandable_segments:True` (already exported in `run_dense_nr.sh`) is the reliable point —
observed at 75GB/82GB with OOMs retried successfully and 0 lost clips.

Per-method breakdown of the 83,095:

```
gt 9,063 · dicache 8,874 · worldcache 8,874 · dense 7,662 · svg1 7,538 · pisa 7,312
radial 7,284 · svg2 7,278 · sito 7,278 · itm 7,278 · fastercache 4,654
```

## How the split works

Machine `k` runs workers `s = 0..W-1`, each taking `rows[k*W + s :: W*MACHINES]` of the
**id-sorted** manifest. That is an exact partition — verified arithmetically on the real
83,095 rows:

```
83,095 rows / 3 machines / 8 workers
  covered 83,095   duplicates 0   missing 0
  per-machine load 27,703 / 27,696 / 27,696
```

No lock, no queue server, no coordination. id-order also means each machine's partial
output stays representative of all models/mounts if you stop early (manifest order is
grouped by model, so manifest order would not).

This replaces the old `--reverse` 2-machine trick (two queues meeting in the middle and
redoing the overlap); `run_dense_nr.py` now refuses to combine `--reverse` with
`--machines`.

## Safety

* Every result is one atomic JSON named by the row's sha256 id, and a worker skips any row
  whose file already exists. Restarting a machine, adding a 4th later, or killing one
  mid-flight costs at most the clips in flight.
* `dump()` writes to a hostname+pid-unique temp name before rename, so 3 machines on the
  shared filesystem cannot interleave into one partial file.
* The 3,699 already-scored records are **hardlinked** into the run dir (not copied), so
  they are skipped for free — that is ~3.5 GPU-hours saved and, more importantly, the new
  wave is a strict superset of what the current tables were built from.
* OOM is retried 4 times with backoff before a clip is recorded as an error.
* `setsid` puts the worker tree in its own process group, so it survives an ssh drop. Each
  launch prints its own `kill -- -<PGID>` line.

## THE containers are ephemeral — plan for restarts

This is the single most important operational fact, measured on this wave:

```
machine 0  launched 17:43  killed 22:00  (4h17m, container ended)
machine 1  launched ~03:24 killed 05:39  (its own container ended)
```

Neither wrote a `queue_finished` marker, both worker logs stop mid-clip, and the tmux
server itself was gone (`/tmp/tmux-1479` missing) while the node had been up 9 days. So
`setsid` survives an ssh drop but **not** the container ending. A 9-day wave will be
interrupted many times.

**Nothing is lost when it happens** — every finished clip is an atomic JSON and workers
skip existing ids, so a restart resumes exactly where it stopped (verified: 1,775 clips
survived the first two kills, 0 errors). But somebody has to notice and relaunch.

Use the status/resume helper for that:

```bash
bash scripts/metrics/dense_nr_status.sh            # report only
MACHINE=1 bash scripts/metrics/dense_nr_status.sh --resume   # relaunch this machine if it died
```

It also catches a quieter failure mode: a worker that hits a **non-OOM** exception does
`break` and exits, so a machine can silently drop from 8 workers to 3 and keep running at a
third of the rate with nothing in the log announcing it. `--resume` restarts the whole
machine when the worker count is short (safe — skip-if-exists), which is the only way to
get the missing shards moving again. Run it a few times a day, or from cron.

## Monitor

```bash
bash scripts/metrics/dense_nr_status.sh                # everything below, in one command
```

Or by hand:

```bash
RUN=metrics/WorldScore/results/dense_nr_full
ls $RUN/results | wc -l                        # global progress across all machines
tail -f $RUN/logs/machine0_*.log               # this machine's launcher log
grep -c '^START' $RUN/worker_m0_*.log          # per-worker clip counts
ps -eo cmd | grep -c -- '--shard'              # should be 8 per machine
```

Rate check: `find $RUN/results -name '*.json' -newermt '30 minutes ago' | wc -l` divided by
30. Expect **~2.0 clips/min per machine**, so ~6 with all three. Measure over ≥30 minutes —
a 5-minute sample reads about 2.4x too high.

Note `START` lines are buried between SAM2 / VFI progress bars in the worker logs — that is
normal, not a hang. A clip takes 1.5–4 minutes.

## Stop

Each launch prints `stop : kill -- -<PGID>`. If lost:

```bash
pkill -f 'run_dense_nr.py --out .*dense_nr_full'
```

Killing is always safe — finished clips are already on disk, and a restart resumes.

## When it finishes

```bash
python3 assets/scripts/table_dense_nr.py     --run metrics/WorldScore/results/dense_nr_full
python3 assets/scripts/table_main_results.py --run metrics/WorldScore/results/dense_nr_full
python3 assets/scripts/table_paper_main.py   --run metrics/WorldScore/results/dense_nr_full
```

All gates (`MIN_ACTIVE_FRAC`, `comparable_pair`, `MIN_PAIRS_FOR_CLAIM`, …) are applied at
table time from stored per-pair records, so the new run is directly comparable to
`dense_nr_v3_pooled` with no re-scoring.

## Trap worth knowing

`run_dense_nr.sh` does `cd metrics/WorldScore`, so a **relative** `--out` lands in
`metrics/WorldScore/metrics/WorldScore/...`. That already happened once while building the
inventory. The wrappers always pass an absolute path; keep it that way if you edit them.

## Rebuilding the manifest (only if it is ever lost)

CPU-only, ~90s. Pass an absolute `--out`:

```bash
CUDA_VISIBLE_DEVICES= bash scripts/metrics/run_dense_nr.sh \
  --out "$PWD/metrics/WorldScore/results/dense_nr_full" --inventory-only \
  --methods dense dicache worldcache fastercache pisa radial svg1 svg2 sito itm
```
