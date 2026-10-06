# Dense-NR full wave — 10 GPUs, heterogeneous boxes

**For the agent on each box.** Read the whole file before running anything. The partition is
positional and silently wrong if boxes disagree, so the numbers below are not suggestions.

---

## 0. State of the world (measured 2026-09-21 09:5x)

| | |
|---|---|
| clips scored so far | **34,272** |
| of those, still valid under the new manifest | **34,272 (100%)** — nothing wasted |
| remaining | **52,453** |
| spec total | 86,767 view-clips = 29,343 episodes |
| progress | **39.5%** |

**Nothing needs re-scoring.** Each result is one atomic JSON named by a sha256 of the row's
identity, and a worker skips any row whose file already exists. Killing a box costs only
the clips in flight (≤ its worker count).

**Why we are relaunching at all:** the boxes currently running are locked to an
`--only-ids` subset of 51,072 rows. That subset is real work (it is a subset of the spec),
but it leaves **34,726 spec clips that no box will ever touch**, and once the subset is
exhausted (~15h) every box exits and idles. The relaunch drops `--only-ids` so the wave
covers the whole manifest.

---

## 1. Hardware and the one hard constraint

Wave topology: **6 single-GPU boxes (80 GB A100) + 1 box with 4 weaker GPUs = 10 GPUs.**

The split each worker computes is:

```
stride = WORKERS * MACHINES
offset = MACHINE * WORKERS + shard        # worker takes rows[offset::stride]
```

`WORKERS` is a **factor of the global stride**, so **every box in the wave must pass the
same `WORKERS` and the same `MACHINES`**, and all must walk the same row list. If they
disagree the union stops being a partition — some rows owned by nobody, others scored
twice — and **nothing errors**, because every write is idempotent. This has already
happened once on this run (30% of remaining clips orphaned, 24% done twice).

That clashes with heterogeneous GPUs: an 80 GB card fits 8 workers, a 20 GB card fits 2.

**Solution — slots.** Set `WORKERS` to the *weakest* GPU's capacity and let bigger GPUs
claim **several slots on the same card**. `WORKERS` stays uniform (partition stays exact)
while workers-per-GPU = `WORKERS × slots_on_that_GPU`, free to differ per box.

### Measured sizing (live 8-worker 80 GB box, this workload)

| resource | per worker |
|---|---|
| VRAM | **7.66 GB** (8 workers = 62.7 of 80 GB; SAM2 propagation is the peak) |
| CPU | **0.93 core** — work is GPU-bound, util sits at 99–100% |
| RAM | ~3.0 GB RSS |
| throughput | **0.48 clips/min** (3.84 clips/min per 8-worker 80 GB GPU) |

Workers a GPU can hold ≈ `floor(0.88 × VRAM_GB / 7.66)` (12% headroom for the SAM2 peak):

| VRAM | 20 GB | 24 GB | 40 GB | 48 GB | 80 GB |
|---|---|---|---|---|---|
| max workers | 2 | 2 | 4 | 5 | 9 → **use 8** (measured knee) |

---

## 2. FIRST: the quad box reports its VRAM

`WORKERS` for the whole wave is set by the weakest GPU, so before anyone launches, the
agent on the 4-GPU box runs:

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
nproc
```

Then pick `WORKERS` from the table above and read the matching plan:

| quad-box VRAM/GPU | WORKERS | slots per 80 GB box | slots per quad GPU | MACHINES |
|---|---|---|---|---|
| **20–24 GB** | **2** | 4 (= 8 workers) | 1 (= 2 workers) | **28** |
| 40 GB | 4 | 2 (= 8 workers) | 1 (= 4 workers) | 16 |
| 48 GB | 5 | 1 (= 5 workers) | 1 (= 5 workers) | 10 |
| 80 GB | 8 | 1 (= 8 workers) | 1 (= 8 workers) | 10 |

Rule: `WORKERS` = weakest GPU's capacity; an 80 GB box takes `floor(8 / WORKERS)` slots
(8 = its measured knee); `MACHINES` = 6 × that + 4.

All three viable rows were **verified to be an exact partition** against the real 86,767-row
manifest — covered 86,767, **0 duplicates, 0 missing**, per-slot load spread ≤ 2 clips:

```
W=2 MACHINES=28 stride=56: covered 86767/86767  dup=0  missing=0  load 3098..3100
W=4 MACHINES=16 stride=64: covered 86767/86767  dup=0  missing=0  load 5420..5424
W=8 MACHINES=10 stride=80: covered 86767/86767  dup=0  missing=0  load 8672..8680
```

Everything below assumes the **20–24 GB** row: **`WORKERS=2`, `MACHINES=28`**. If the quad
box reports something else, stop and say so — the whole table changes and every box must
be told the new numbers.

### Slot allocation (WORKERS=2, MACHINES=28)

| box | GPUs | workers/GPU | slots | **slot range** |
|---|---|---|---|---|
| A (80 GB single) | 1 | 8 | 4 | **0–3** |
| B (80 GB single) | 1 | 8 | 4 | **4–7** |
| C (80 GB single) | 1 | 8 | 4 | **8–11** |
| D (80 GB single) | 1 | 8 | 4 | **12–15** |
| E (80 GB single) | 1 | 8 | 4 | **16–19** |
| F (80 GB single) | 1 | 8 | 4 | **20–23** |
| G (4× ~20 GB) | 4 | 2 | 4 | **24–27** |
| | **10** | | **28** | |

56 workers total → **52,453 clips ÷ (0.48 × 56) ≈ 32.5 h ≈ 1.4 days.**

**Each box must be given exactly one slot range. Two boxes on the same range silently do
identical work and orphan the rest.** Assign A–G to physical boxes once and write it down.

---

## 3. Runbook

### Step 1 — stop every box (all 7, including any still running)

On **each** box:

```bash
pkill -f 'run_dense_nr.py'
sleep 3
# Zombie-safe check. `pgrep -cf run_dense_nr.py` is NOT reliable here: this cluster's
# container init does not reap children, so a box can carry dozens of `[python] <defunct>`
# entries from waves that ended days ago. Zombies hold no CPU/GPU/RAM but they DO get
# counted, so pgrep never prints 0 and you would think the box is still busy.
ps -eo stat=,args= | grep 'run_dense_nr.py' | grep -v grep | grep -vc '^Z'   # must be 0
nvidia-smi --query-compute-apps=pid --format=csv,noheader                   # must be empty
```

`nvidia-smi --query-compute-apps` is the authoritative test — a real worker always holds
~7.66 GB of VRAM, a zombie holds none.

> Do not use a broader pattern like `pkill -f dense_nr` — it matches the shell running it
> and cuts your own command. (It also matches any `dense_nr_spec` complement job.)

The manifest promotion in step 2 **refuses to run while any worker is alive** (the split is
positional, so swapping the manifest under a live worker repartitions the rows it already
claimed). If it complains, a box is still up.

### Step 2 — promote the rebuilt manifest (ONCE, on any box)

The live `manifest.json` is from 09-17 and is stale: it misses 591 view-clips from 197
`single_arm / multiview / dreamgen` dense episodes re-rendered on 09-19, and it still lists
567 clips in trees deleted on 09-19. A rebuilt manifest that matches the episode spec
exactly (all per-method deltas `+0`, total 29,343 episodes) is already staged.

```bash
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics
models/dreamgen/.venv/bin/python scripts/metrics/nr_promote_manifest.py \
  --from metrics/WorldScore/results/_manifest_rebuild \
  --to   metrics/WorldScore/results/dense_nr_full
```

This backs up the old manifest and renames every stale `--only-ids` file to `.stale` so the
flawed subset cannot be picked up again.

**One caveat on the liveness guard.** It decides a box is down if its `worker_*.log` has not
been touched for `--alive-minutes` (default 10). That is mtime-based, so it is immune to the
zombie problem above — but it also means a box whose container was killed mid-wave looks
"down" while its scheduler may still **restart it later**. That happened on this run: `m4`
went silent and stayed silent for 37 min. So:

- Make sure every box is stopped **at the orchestration layer** (Determined experiment /
  tmux session killed), not merely `pkill`ed, before promoting.
- If a stale box wakes up after the promote, it rejoins with the OLD pinned mode and
  `wave_mode.json` will **refuse** it — which is the intended outcome. Check
  `logs/machine*.log` for such a refusal and relaunch that box with its slot range.

### Step 3 — clear the wave-mode lock (ONCE)

The launcher pins `machines/workers/row-list` on first launch and refuses mismatches. The
old wave pinned `machines=5 workers=8 list=ids:…`, so the lock must be cleared:

```bash
rm -f metrics/WorldScore/results/dense_nr_full/wave_mode.json
```

### Step 4 — launch (one command per box)

```bash
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics
unset NUM_SHARDS SHARD_ID TARGET ONLY_IDS REVERSE
```

`NUM_SHARDS` / `SHARD_ID` / `TARGET` leak through tmux's global env and silently make a run
skip most episodes. Always unset them.

```bash
# box A  (80GB single GPU)
MACHINES=28 WORKERS=2 bash scripts/metrics/run_dense_nr_slots.sh  0 4

# box B
MACHINES=28 WORKERS=2 bash scripts/metrics/run_dense_nr_slots.sh  4 4

# box C
MACHINES=28 WORKERS=2 bash scripts/metrics/run_dense_nr_slots.sh  8 4

# box D
MACHINES=28 WORKERS=2 bash scripts/metrics/run_dense_nr_slots.sh 12 4

# box E
MACHINES=28 WORKERS=2 bash scripts/metrics/run_dense_nr_slots.sh 16 4

# box F
MACHINES=28 WORKERS=2 bash scripts/metrics/run_dense_nr_slots.sh 20 4

# box G  (4 GPUs -- one slot each, so 2 workers per GPU)
MACHINES=28 WORKERS=2 bash scripts/metrics/run_dense_nr_slots.sh 24 4 0,1,2,3
```

The script prints a per-GPU VRAM check and a box-wide core check before launching, and
**refuses** if the slots would not fit or the range falls outside `[0, MACHINES)`.

Expected output per box (example, box A):

```
  GPU 0   81920 MiB  slots=4 workers=8  need 61.3GB / usable 70.4GB  ok
  box: 8 workers total, ~7 of 32 cores, ~24GB RAM

=== slot 0 of 28  ->  GPU 0  (2 workers) ===
=== dense-NR FULL wave ===
  mode           : machine 0 of 28 (stride shard)
  workers        : 2   (GPU 0)
  subset         : none (whole manifest)      <-- MUST say this
  ...
```

**`subset : none (whole manifest)` is the line that proves the spec hole is being filled.**
If it names a file, stop: `ONLY_IDS`/`TARGET` leaked.

### Step 5 — verify the wave is one exact partition

After all boxes are up, on any box:

```bash
cat metrics/WorldScore/results/dense_nr_full/wave_mode.json
# expect exactly: machines=28 workers=2 list=full-manifest

MACHINES=28 bash scripts/metrics/dense_nr_status.sh
```

Then confirm the manifest matches the episode spec:

```bash
models/dreamgen/.venv/bin/python scripts/metrics/nr_distribution_audit.py \
  --run metrics/WorldScore/results/dense_nr_full --with-gt
```

Every per-method delta must be `+0` and the episode total `29343`.

---

## 4. Monitoring

```bash
MACHINES=28 bash scripts/metrics/dense_nr_status.sh
```

Health signals and what they mean:

| signal | healthy | meaning if not |
|---|---|---|
| `.tmp` stubs in `results/` | **0** | disk quota is truncating writes |
| `errors/` count | low (was 40 / 34k) | check `errors/` for a repeated cause |
| worker log mtime | < 5 min | **all of a box's logs frozen at the same age = the process group was killed** (ephemeral container) → relaunch that box with its own slot range |
| `OOM` lines | some are fine | retry logic handles them (one box absorbed 136 OOM + 28 quota with no data loss) |
| `SKIP-GONE` | expected | rows pointing at deleted trees; not a failure |

A dead box is relaunched with **its own slot range only** — never someone else's.

---

## 5. Things that will silently corrupt the run

1. **Two boxes on the same slot range** → identical work, rest of the manifest orphaned.
2. **Different `WORKERS` or `MACHINES` on any box** → not a partition. `wave_mode.json`
   catches this; do not pass `FORCE_MODE=1` to get past it.
3. **`TARGET` / `ONLY_IDS` set** (often leaked from tmux) → back to the 51k subset and the
   34,726-clip spec hole. Check the `subset :` banner line every launch.
4. **Old wrappers.** Do **not** use `run_dense_nr_grid.sh` (defaults `TARGET=300` →
   `--only-ids`), `run_dense_nr_full_m{0..4}.sh`, or `run_dense_nr_multigpu.sh` (assumes one
   slot per GPU with uniform `WORKERS`, which is exactly what fails on mixed hardware).
   **`run_dense_nr_slots.sh` is the only entry point for a forward slot**;
   `run_dense_nr_full.sh extra` is the only entry point for the single reverse walker.
5. **Promoting the manifest while workers are live** → they repartition mid-flight. The
   promote script refuses; do not `--force`.

---

## 5b. Three failure modes measured during the 2026-09-21 launch

These all happened for real. Expect them.

### (a) `det shell open` kills anything not in tmux — and multi-step commands get truncated

Launching via `setsid` inside `det shell open` looked fine (banner printed, PGID reported,
`still to score : 52453`) but **all six boxes were dead 2 minutes later with 0-byte logs**.
Confirmed: `worker song: 0`, `GPU: 0 proc`. Only `tmux` survives — as the skill says.

Worse, a **compound** remote command silently loses its tail. This form:

```bash
det shell open $ID -- "tmux kill-session -t nrwave; pkill -f run_dense_nr.py; sleep 3;
                       tmux new-session -d -s nrwave2 '...'; sleep 25; echo worker=..."
```

returned in 10 s (not 28 s), never ran `tmux new-session`, and left **all 6 boxes at 0
workers** — the `pkill` landed but the relaunch did not. It took a second round to notice.

**Rule: one action per `det shell open`.** Launch with nothing but the `tmux new-session`,
then verify in a *separate* call:

```bash
det shell open $ID -- "tmux new-session -d -s nr$S 'cd $P && MACHINES=6 WORKERS=8 \
  bash scripts/metrics/run_dense_nr_slots.sh $S 1 > $P/logs/nrwave_slot$S.log 2>&1; sleep 99999'"
sleep 40
det shell open $ID -- 'echo worker=$(ps -eo stat=,args= | grep run_dense_nr.py \
  | grep -v grep | grep -vc "^Z")'      # expect 9
```

Also: if a session with that name **already exists**, `tmux new-session -d` is a no-op and
the box silently stays down. Always `tmux kill-session` (as its own call) or use a new name.

For a box with no `tmux` (`thanhnv` has none, and no apt candidate), `setsid` as a plain
background job *does* survive — verified with a tick-probe (1 → 12 ticks written after the
SSH session closed):

```bash
det shell open $ID -- "cd $P && setsid bash -c 'MACHINES=6 WORKERS=8 \
  bash $P/scripts/metrics/run_dense_nr_slots.sh 5 1 > $P/logs/nrwave_slot5.log 2>&1; \
  sleep 99999' </dev/null >/dev/null 2>&1 & sleep 3; echo launched"
```

### (b) `$HOME` is on the shared FS, so 56 workers race on one CLIP download

`HOME=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge` — shared by **every** box. On first
launch all workers raced to fetch `~/.cache/clip/ViT-L-14.pt` (932 MB) and killed each
other's download:

```
RuntimeError: Model has been downloaded but the SHA256 checksum does not not match
```

**19 of 54 workers died** this way (m0 3, m1 2, m2 2, m3 4, m4 1, m5 2, rev 5). The file
later verified clean (`b8cca3fd…` matches `clip._MODELS`), so it self-heals — but the first
wave loses a third of its capacity.

**Prevention:** warm the cache once, before launching anything, and verify it:

```bash
models/dreamgen/.venv/bin/python -c "import clip; clip.load('ViT-L/14', device='cpu')"
sha256sum ~/.cache/clip/ViT-L-14.pt   # b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836
```

### (c) A `RETRY QUOTA` worker is asleep, not dead — do not "fix" it

Symptom: throughput halves and ~11 workers look frozen for 10–15 min. Their logs end with

```
RETRY QUOTA 3/3 in 900s <id>
```

That is the intended disk-quota backoff (5/10/15 min), **not** a crash. `df` showed 379 TB
free and a 5 MB write succeeded at 2.8 GB/s, so this is a **user/group quota** blip, not a
full filesystem. Left alone for 9 minutes it fully recovered: blocked workers **11 → 1**,
48 alive, and the reverse walker went 0.61 → 1.33 clips/min.

Diagnose before acting — the check that distinguishes the three cases:

```bash
# asleep in backoff (leave it):   RETRY QUOTA at the END of the log
# dead (relaunch the box):        log mtime old AND no RETRY QUOTA at the end
# healthy:                        log mtime < 5 min
tail -c 2000 <log> | grep -c 'RETRY QUOTA'
```

Do **not** `pkill` a quota-blocked worker: it is holding a clip it will finish, and killing
it is what led to the 0-worker incident in (a).

### (d) The cluster reclaims shells without warning — and that ORPHANS stride slots

At ~05:00 on 2026-09-22, **7 of 11 shells went `TERMINATED / allocation exited
successfully`** at the same moment (VLAq, S-MajRL, H2R_TEAM, vla, 2× VLA_doanh, thanhnv).
Not idle-timeout (`idle_timeout: null` on all of them) and not a crash — the worker logs just
stop mid-SAM2-propagate with no traceback, i.e. SIGKILL from outside.

This is the failure the slot design is most exposed to. A stride slot owns a fixed
`1/MACHINES` of the manifest and **nobody else ever visits it**, so losing slots 0, 1 and 5
of a 6-slot wave orphaned **16,069 unscored rows** — silently, because every write is
idempotent and nothing errors.

The check, and the fix (re-shard the survivors):

```bash
det shell list | grep -E 'RUNNING|TERMINATED'     # who is actually alive?

# count rows that lost their owner
# (slots that died) x (rows not yet in results/) -> orphans

# then: stop survivors, clear the lock, relaunch them as a COMPLETE partition
det shell open $ID -- 'pkill -f run_dense_nr.py'   # one call per box
rm -f <run>/wave_mode.json
MACHINES=3 WORKERS=8 bash scripts/metrics/run_dense_nr_slots.sh <0..2> 1
```

Verify the new geometry is exact before trusting it (`W=8 M=3` -> covered 86,767/86,767,
0 dup, 0 missing).

**Opportunistic boxes did NOT create orphans** — by design. Their slices are a private
`--only-ids` subset while the forward wave still owns every row, so the 10 that died just
stopped contributing (they were 19–40% through their slices). That is exactly the tradeoff:
borrowed capacity is free to vanish, stride slots are not.

### (e) A stray `'` in a shell comment killed every worker on every box

While adding the `libGL` fix to `run_dense_nr.sh`, an edit mangled the comment block into a
single line ending `...hn image's own newer libs in front where they exist.` The unclosed
quote made bash fail at line 16:

```
scripts/metrics/run_dense_nr.sh: line 16: unexpected EOF while looking for matching `''
```

Every worker died instantly on **all** boxes. What made it expensive to find: the launcher
still printed a perfect banner (`mode`, `workers`, `still to score`, `launched PGID`) because
it is a *different* script, and the worker logs showed no error — only a clean truncation
mid-SAM2, indistinguishable from the SIGKILL in (d). Two relaunch rounds were wasted
blaming tmux before running the worker in the foreground surfaced the real message.

**Never put an apostrophe in a comment in these scripts**, and after editing any `run_*.sh`:

```bash
bash -n scripts/metrics/run_dense_nr.sh && echo OK    # cheap, catches exactly this
```

Then confirm one worker actually reaches a clip, not just that a PGID was printed.

---

| | |
|---|---|
| topology | 6 single-GPU boxes as slots 0–5 (`MACHINES=6 WORKERS=8`) + `vla` GPU 1 as the reverse walker |
| `wave_mode.json` | `machines=6 workers=8 list=full-manifest` |
| throughput | **12.3 clips/min** healthy (≈1.8 per forward box + 1.3 reverse); 10.4 while quota-blocked |
| progress | 36,335 / 86,767 = **41.9%**, ~50.4k left → **≈68 h (2.8 days)** |
| errors | 42 total = **0.119%**, all `ValueError: Invalid FPS`; 33 are rows pointing at deleted trees (harmless), **9 are genuinely corrupt mp4s** (0 B, 48 B, a truncated 2 MB) |
| write stubs | **0** — the atomic-write fix holds even under quota pressure |

### Why `vla` is a reverse walker and not slots 4–7

`vla` is shared: **loadavg 181–284 on 32 cores** (someone else's `libero` jobs, 369 python
procs) and GPU 0 already held 19–47 GB. A stride slot must keep pace or it becomes the
straggler that defines the wave's end — modelled cost if it runs 3× slow: **32.5 h → 98.6 h**.
The reverse walker has no such coupling: it is its own backwards queue, so if it is slow it
simply contributes less (measured 1.3 clips/min ≈ 11% of the wave) and the six forward
boxes still cover every row.

**This is also why `REVERSE=1` is now exempt from the `wave_mode.json` lock** (see
`_run_dense_nr_full.sh`): it forces `MACHINES=1` internally, so its mode string could never
match a wave pinned at `machines=6`, and the guard would have blocked the one mechanism that
is genuinely safe to add mid-wave. It is exempt but **never pins** the mode — verified
across all four `(REVERSE, lock-exists)` combinations.

---

## 6. Why 4 metrics, and what "done" means

The wave scores 4 reported NR metrics per clip — Visual Integrity, Text Alignment, Actor
Activity, Active Motion Continuity — plus Active Rate, which is computed only as a **gate**
for AMC and is not reported. Results land as one JSON per view-clip in
`metrics/WorldScore/results/dense_nr_full/results/`.

Tables are built afterwards (no GPU needed):

```bash
models/dreamgen/.venv/bin/python assets/scripts/table_paper_grid.py \
  --run metrics/WorldScore/results/dense_nr_full
```

Only records whose protocol is `dense-nr-v3.2-closed-world` are used; older records are
filtered out, so mixing them in is harmless.
