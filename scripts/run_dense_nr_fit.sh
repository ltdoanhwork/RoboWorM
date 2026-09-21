#!/usr/bin/env bash
# Fill GPUs that are only PARTLY free, with a per-GPU worker count sized to the VRAM that
# is actually left on each one.
#
#   bash scripts/run_dense_nr_fit.sh '0:3 1:7 2:6 3:6'   # explicit plan
#   bash scripts/run_dense_nr_fit.sh auto 0 1 2 3        # size each GPU from free VRAM
#
# WHY W=1 (and why that is the whole trick)
#   run_dense_nr.py partitions with `rows[MACHINE*W + shard :: W*MACHINES]`, so W (workers
#   per machine index) is a FACTOR OF THE GLOBAL STRIDE. That is why every box in a wave
#   must use the same W, and why run_dense_nr_multigpu.sh has to claim one index per GPU:
#   an index is exactly W worker slots, i.e. one GPU's worth of 8.
#
#   With W=1 the formula collapses to `rows[MACHINE :: MACHINES]`: one machine index IS one
#   worker process. The number of workers on a given GPU then has NOTHING to do with the
#   partition -- it is just how many of the MACHINES indices this box happens to pin to
#   that CUDA device. So 3 workers on a crowded GPU and 7 on an empty one is still an exact
#   partition, which is impossible with W=8.
#
# WHY THAT IS NEEDED HERE (measured 2026-09-21 09:2x UTC on this host)
#   The GPUs are shared with jobs in other PID namespaces (visible in nvidia-smi, not in
#   ps, so they can neither be inspected nor killed from here). Free VRAM per card:
#     GPU0 24-30 GB | GPU1 62 GB | GPU2 52 GB | GPU3 52 GB
#   A worker holds GroundingDINO x2 + SAM2 (image+video) + SEA-RAFT + VFIMamba + IQA
#   ~= 7.1 GB RESIDENT (8 workers = 57 GB measured), so a uniform 8/GPU does not fit on
#   three of these four cards. Undersizing is not cosmetic: on a full card 13% of clips
#   died after all 4 OOM retries (see scripts/run_dense_nr.sh), and each retry costs
#   60/120/180 s of that worker's time.
#
# WHAT IT SCORES, AND WHY IT COLLIDES WITH NOTHING
#   The live subset wave (indices m3/m4, 8 workers each) is walking
#   dense_nr_full/manifest_grid_target300.json off the STALE 09-17 manifest. That list is
#   asserted disjoint from dense_nr_spec/manifest_complement.json (see
#   run_dense_nr_complement.sh), so this box takes its work from the complement and cannot
#   duplicate theirs -- no box has to be stopped and no manifest has to be promoted first.
#
#   A forward box is already walking the complement in ascending id order. Workers sort by
#   id, so instead of resharding that box (which would mean killing it), this one takes the
#   TAIL of the still-unscored complement: the two fronts approach each other and meet only
#   when the list is nearly done. The split point is proportional to worker counts
#   (FWD_WORKERS vs the total here) so both halves finish at about the same time. Coverage
#   does not depend on this box at all -- the forward box owns the whole list, so anything
#   left here is picked up there, and anything done here is skipped there for free.
#
# ENV
#   RUN          run dir for this box (default results/dense_nr_spec_b; results/ and
#                errors/ are symlinked into dense_nr_full, so records land in ONE place)
#   SRC          run dir to borrow manifest.json + manifest_complement.json from
#   FWD_WORKERS  worker count of the forward complement box, for the split (default 8)
#   PER_WORKER_MIB  VRAM budget per worker (default 7300)
#   MARGIN_MIB   VRAM left untouched per GPU for the other tenants (default 4096)
#   FORCE=1      launch even if a check fails
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/models/dreamgen/.venv/bin/python"
SRC="${SRC:-$REPO/metrics/WorldScore/results/dense_nr_spec}"
RUN="${RUN:-$REPO/metrics/WorldScore/results/dense_nr_spec_b}"
LIVE="$REPO/metrics/WorldScore/results/dense_nr_full"
FWD_WORKERS="${FWD_WORKERS:-8}"
PER_WORKER_MIB="${PER_WORKER_MIB:-7300}"
MARGIN_MIB="${MARGIN_MIB:-4096}"

[ $# -ge 1 ] || { echo "usage: bash scripts/run_dense_nr_fit.sh '<gpu>:<workers> ...' | auto <gpu>..." >&2; exit 1; }

# ---------------------------------------------------------------- plan
free_mib() { nvidia-smi --id="$1" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits \
               | awk -F', *' '{print $2-$1}'; }

PLAN=""
if [ "$1" = "auto" ]; then
  shift
  for g in "$@"; do
    f=$(free_mib "$g")
    n=$(( (f - MARGIN_MIB) / PER_WORKER_MIB ))
    [ "$n" -gt 8 ] && n=8   # knee: throughput is GPU-bound past ~5, and >8 OOMs
    [ "$n" -lt 0 ] && n=0
    printf 'GPU%s free %s MiB -> %s workers\n' "$g" "$f" "$n"
    [ "$n" -gt 0 ] && PLAN="$PLAN $g:$n"
  done
else
  PLAN="$1"
fi

TOTAL=0
for spec in $PLAN; do
  g="${spec%%:*}"; n="${spec##*:}"
  f=$(free_mib "$g")
  need=$(( n * PER_WORKER_MIB + MARGIN_MIB ))
  printf 'GPU%-2s %2s workers  need %6s MiB (incl %s margin)  free %6s MiB  %s\n' \
    "$g" "$n" "$need" "$MARGIN_MIB" "$f" "$([ "$need" -le "$f" ] && echo OK || echo TIGHT)"
  if [ "$need" -gt "$f" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "ERROR: GPU$g cannot hold $n workers. Lower it, or FORCE=1 to accept OOM retries." >&2
    exit 1
  fi
  TOTAL=$((TOTAL + n))
done
[ "$TOTAL" -gt 0 ] || { echo "ERROR: plan is empty -- no GPU has room." >&2; exit 1; }

CORES=$(nproc); NEED_CORES=$(awk "BEGIN{printf \"%d\", 0.93*$TOTAL + 0.5}")
# nproc is the cpuset SIZE, not what is available. This host's cpuset 0-31 was measured
# 88.7% busy with ~3.6 cores idle: other tenants (other PID namespaces, unkillable from
# here) already hold ~28 cores. The workload needs ~0.93 core/worker, so sizing against
# nproc would oversubscribe 6x and slow the jobs already running -- a zero-sum trade, since
# every clip they lose is a clip we gain. Measure, do not assume.
FREE_CORES=$(python3 - <<'PY'
import time
def snap():
    d={}
    for ln in open('/proc/stat'):
        if ln.startswith('cpu') and ln[3].isdigit():
            p = ln.split(); d[int(p[0][3:])] = [int(x) for x in p[1:]]
    return d
cpus = set()
for f in ('/sys/fs/cgroup/cpuset.cpus.effective', '/sys/fs/cgroup/cpuset/cpuset.cpus'):
    try:
        for part in open(f).read().strip().split(','):
            a, _, b = part.partition('-'); cpus.update(range(int(a), int(b or a) + 1))
        break
    except OSError:
        pass
a = snap(); time.sleep(5); b = snap()
busy = idle = 0
for c in (cpus or a):
    if c not in a: continue
    da = [y - x for x, y in zip(a[c], b[c])]
    busy += sum(da) - da[3] - da[4]; idle += da[3] + da[4]
print(f'{len(cpus or a) * idle / (busy + idle):.2f}')
PY
)
FIT_CORES=$(awk "BEGIN{printf \"%d\", $FREE_CORES/0.93}")
echo "total $TOTAL workers | cores $CORES (cpuset), measured FREE $FREE_CORES -> room for $FIT_CORES workers"
echo "  this plan needs ~$NEED_CORES cores, RAM ~$((TOTAL*3)) GB"
if [ "$TOTAL" -gt "$FIT_CORES" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "ERROR: only $FREE_CORES cores are actually idle in this cpuset ($FIT_CORES workers'" >&2
  echo "       worth). Launching $TOTAL would steal CPU from the waves already running on" >&2
  echo "       this host, which is zero-sum. Shrink the plan or FORCE=1." >&2
  exit 1
fi

# ---------------------------------------------------------------- run dir
mkdir -p "$RUN/logs"
# Hardlink, not copy: the manifest is 78 MB and MUST be byte-identical to the one the
# complement's id list was computed against (the partition is positional).
[ -f "$RUN/manifest.json" ] || ln "$SRC/manifest.json" "$RUN/manifest.json"
# ONE results dir for the whole project: `id` is sha256(method:relative:view), a pure
# content hash, so every wave skips whatever any other wave already wrote.
[ -e "$RUN/results" ] || ln -s ../dense_nr_full/results "$RUN/results"
[ -e "$RUN/errors" ]  || ln -s ../dense_nr_full/errors  "$RUN/errors"

IDS="$RUN/manifest_fit_tail.json"
"$PY" - "$SRC/manifest_complement.json" "$RUN/manifest.json" "$RUN/results" \
       "$LIVE/manifest_grid_target300.json" "$IDS" "$TOTAL" "$FWD_WORKERS" <<'PY'
import json, os, sys
comp_f, man_f, res_d, live_f, out_f, mine, fwd = sys.argv[1:]
mine, fwd = int(mine), int(fwd)
comp = set(json.load(open(comp_f)))
rows = json.load(open(man_f))
# The live subset wave must not be touched; re-assert it rather than trust the note.
if os.path.exists(live_f):
    live = set(json.load(open(live_f)))
    both = comp & live
    if both:
        raise SystemExit(f'ERROR: {len(both)} complement ids are also in the live subset')
    print(f'disjoint from live subset OK (complement {len(comp)}, live {len(live)})')
have = set(os.listdir(res_d))
# Workers walk `rows.sort(key=id)`, so slicing the id-sorted order is what makes this the
# TAIL the forward box reaches last.
ids = sorted(r['id'] for r in rows if r['id'] in comp)
todo = [i for i in ids if f'{i}.json' not in have]
cut = round(len(todo) * fwd / (fwd + mine))   # forward box keeps the head, we take the tail
tail = todo[cut:]
if not tail:
    raise SystemExit('ERROR: nothing left in the complement tail')
json.dump(tail, open(out_f, 'w'))
print(f'complement {len(ids)} ids | unscored {len(todo)} | forward keeps head {cut} '
      f'({fwd}w) | THIS BOX takes tail {len(tail)} ({mine}w)')
PY

if [ "${DRY:-0}" = "1" ]; then echo; echo "DRY=1: plan and id list built, nothing launched."; exit 0; fi

# ---------------------------------------------------------------- wave mode lock
MODE_FILE="$RUN/wave_mode.json"
MODE_NOW="machines=$TOTAL workers=1 list=ids:$(sha256sum "$IDS" | cut -c1-16)"
if [ -f "$MODE_FILE" ] && [ "$(cat "$MODE_FILE")" != "$MODE_NOW" ] && [ "${FORCE_MODE:-0}" != "1" ]; then
  echo "ERROR: $RUN was pinned as '$(cat "$MODE_FILE")' but this launch is '$MODE_NOW'." >&2
  echo "  The split is positional: mixing the two leaves rows owned by nobody, silently." >&2
  echo "  Stop this box's workers, rm $MODE_FILE, relaunch. FORCE_MODE=1 overrides." >&2
  exit 1
fi
printf '%s' "$MODE_NOW" > "$MODE_FILE"

# ---------------------------------------------------------------- launch
# Launch the WORKERS directly (--shard >= 0 is worker mode). The usual parent process only
# fans out W children and would add 22 useless manifest loads here; with W=1 there is
# nothing for it to fan out.
STAMP="$(date +%Y%m%d_%H%M%S)"
echo
k=0
for spec in $PLAN; do
  g="${spec%%:*}"; n="${spec##*:}"
  for _ in $(seq 1 "$n"); do
    LOG="$RUN/worker_fit${k}.log"
    setsid env CUDA_VISIBLE_DEVICES="$g" \
      bash "$REPO/scripts/run_dense_nr.sh" \
        --out "$RUN" --reuse-manifest --only-ids "$IDS" \
        --workers 1 --machines "$TOTAL" --machine "$k" --shard 0 \
      >>"$LOG" 2>&1 &
    echo "  index $k -> GPU $g  (PGID $!)  $LOG"
    printf '%s index %s gpu %s pgid %s\n' "$STAMP" "$k" "$g" "$!" >> "$RUN/logs/launch_$STAMP.txt"
    k=$((k + 1))
    # Stagger: 22 workers loading ~7 GB of weights at once spikes VRAM well above the
    # steady state that was budgeted above, which is exactly when the other tenants' peaks
    # turn into our OOM.
    sleep 6
  done
done

echo
echo "launched $TOTAL workers = machine indices 0..$((TOTAL-1)) of $TOTAL (W=1, stride $TOTAL)"
echo "monitor : tail -f $RUN/worker_fit0.log"
echo "          grep -hc '^DONE' $RUN/worker_fit*.log | paste -sd+ | awk '{print}' | tr '+' ' ' | xargs -n1 | awk '{s+=\$1}END{print s}'"
echo "          ls $RUN/results | wc -l"
echo "stop    : pkill -f 'run_dense_nr.py --out $RUN'"
