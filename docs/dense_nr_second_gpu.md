# Join the in-flight dense NR run from a second GPU

Hand this file to an agent on the other GPU/machine. It adds capacity to the
**already-running** scoring job without restarting it and without duplicating work.

Repo (shared `pfss` filesystem, same path from both machines):

```
/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics
```

## What is already running (do not touch it)

| | |
|---|---|
| run dir | `metrics/WorldScore/results/dense_nr_v3_div30` |
| manifest | 1,573 view-clips (**already built — never rebuild it**) |
| owner machine | host `fef1dfa98121`, GPU 0, supervisor pgid `490353`, 5 workers, **forward** id order |
| protocol | `dense-nr-v3.2-closed-world` |
| throughput | ~26 s/clip aggregate → ~11 h alone |

Your job: run 5 more workers over the **same** manifest in **reverse** id order.
The two queues converge in the middle and stop. Expect to roughly halve the ETA.

## The one command

```bash
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics

setsid bash -c 'CUDA_VISIBLE_DEVICES=0 exec bash scripts/metrics/run_dense_nr.sh \
    --out results/dense_nr_v3_div30 \
    --workers 5 --reverse --reuse-manifest' \
  > results/dense_nr_v3_div30.queue.rev.log 2>&1 < /dev/null &
disown
```

Set `CUDA_VISIBLE_DEVICES` to **your** GPU index (it is the index on *your* host).
Do not pass `--smoke-paired-episodes` / `--smoke-per-group`: with
`--reuse-manifest` the manifest is read, not regenerated.

### Immediately verify (do not skip)

```bash
sleep 60
D=metrics/WorldScore/results/dense_nr_v3_div30
# exactly 1 supervisor + 5 shards, all sharing ONE pgid:
ps -eo pid,ppid,pgid,args | grep run_dense_nr.py | grep -v grep
echo "PGID=$(ps -o pgid= -p $(pgrep -f 'run_dense_nr.py --out results' | head -1) | tr -d ' ')"
cat results/dense_nr_v3_div30.queue.rev.log     # must NOT print "INVENTORY"/"SMOKE"
ls $D/worker_rev_*.log | wc -l                  # 5
grep -h '^START' $D/worker_rev_*.log | head -5
```

`INVENTORY` or `SMOKE` in your log means the manifest was rebuilt — **kill
immediately** (see below) and re-check that `--reuse-manifest` was passed.

## Why reverse, and why you must not reshard

Sharding is `rows[shard::workers]`. The owner already runs `--workers 5`, so
shards 0-4 **cover the entire manifest** — there is no free shard index to claim.
Passing `--workers 10 --shard 5..9` would not partition anything; it would just
re-walk clips the owner is also walking.

`--reverse` sorts the same id-ordered manifest backwards instead. Safety rests on
three properties already in the code:

1. `dest.exists()` skip — a clip finished by either machine is skipped by the other.
2. Per-id atomic writes (`tmp.replace(path)`), with the temp name now unique per
   `host.pid`, so a simultaneous double-score cannot interleave into a corrupt file.
3. Both directions cover all 1,573 clips, so **if your machine dies the owner still
   finishes 100% on its own.** Nothing is lost by joining or leaving.

Cost is bounded: only clips in flight when the two directions meet can be scored
twice (at most ~10, one per worker). The later atomic write wins; both are valid.

## Stopping

```bash
kill -- -<YOUR_PGID>            # your process group only
```

**Never `pkill -f run_dense_nr.py`.** The queue is a supervisor that respawns
shards; pattern-killing the workers leaves it alive and it starts a *second* set.
That exact mistake once put 10 workers on one GPU writing into one results dir.
`kill -- -<pgid>` takes the supervisor down with its children. Also make sure
you kill *your* pgid, not the owner's `490353`.

## Monitoring your half

```bash
D=metrics/WorldScore/results/dense_nr_v3_div30
ls $D/results | wc -l                              # shared total, /1573
ls $D/errors  | wc -l
grep -hc '^DONE' $D/worker_rev_*.log               # your throughput
grep -h 'OOM retry' $D/worker_rev_*.log | wc -l    # >0 => lower --workers
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```

- **Capacity**: ~13-23 GB per worker (varies with clip length); 5 workers peaked at
  75/80 GB on an A100-80GB. On a smaller GPU use `--workers 3` (~45 GB). There is a
  4-attempt OOM backoff, but sustained `OOM retry` lines mean too many workers.
- **Errors are expected to be rare and data-driven.** One known failure is real and
  not a bug: `ValueError: Invalid FPS` on GT clip `13e60537103761875e78`
  (`single_arm/input/multiview/ctrlworld/non_makovian/episode_015306/view_2.mp4`) —
  a broken source video. Errors land in `errors/<id>.json` with a traceback and do
  not stop the queue. Re-scoring after a fix = delete just that result/error file.

## Do NOT do these

- Do not run `plot_dense_nr.py` or `table_dense_nr.py`. The **owner's** forward
  queue runs the plot hook when it finishes; your `--reverse` queue deliberately
  skips it so the two cannot race on `figures/` and `coverage.json`.
- Do not edit `scripts/metrics/run_dense_nr.py`, `metrics/dynamics/*`, or anything the
  workers import. Both machines are executing this code live; a mid-run edit
  changes the protocol for later clips and silently mixes two protocols in one
  results dir.
- Do not delete or rewrite `manifest.json`, `results/`, or `errors/`.
- Do not use `models/DreamDojo/.venv` or `models/Ctrl-World/.venv`.
  `scripts/metrics/run_dense_nr.sh` already selects `models/dreamgen/.venv` and the
  vendored `PYTHONPATH` — always go through the `.sh`, never call the `.py` bare.

## Report back

When your queue exits (or if you stop it), report: clips you completed
(`grep -hc '^DONE' worker_rev_*.log`), shared `results/` and `errors/` counts, any
new error ids with the exception line, and peak GPU memory.

Background: [docs/dense_nr_full_run.md](dense_nr_full_run.md) (v3.2 section).
