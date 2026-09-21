#!/usr/bin/env bash
# Status + resume helper for the multi-machine dense-NR wave.
#
# Why this exists: the Determined shell containers these runs live in are EPHEMERAL.
# Measured on this wave -- machine 0's workers were killed at 22:00 after 4h17m when its
# container ended, and a second machine's workers died at 05:39 while THIS container was
# still up (i.e. its own container ended). A 9-10 day wave will therefore be interrupted
# tens of times. Nothing is lost when that happens (every finished clip is an atomic JSON
# and workers skip existing ids), but somebody has to notice and relaunch.
#
# This script makes both steps one command:
#   bash scripts/dense_nr_status.sh          # report only
#   bash scripts/dense_nr_status.sh --resume # report, then relaunch THIS machine if dead
#
# It also catches the quieter failure: a worker that hits a non-OOM exception does
# `break` and EXITS, so a machine can silently drop from 8 workers to 3 and keep going at
# a third of the rate with nothing in the log to announce it. `--resume` restarts the
# whole machine when the worker count is short, which is safe (skip-if-exists) and is the
# only way to get the missing shards moving again.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUN="${RUN:-$REPO/metrics/WorldScore/results/dense_nr_full}"
# Which machine is THIS box? Defaulting to 0 was a live footgun: running this bare on m1
# or m2 reported machine 0's health, and with --resume it would launch a SECOND set of
# machine-0 shards while this box's own workers (a different --machine index) kept
# running -- duplicating m0's slice and leaving this box's slice orphaned. Infer the
# index from the workers actually running here, and only fall back to 0 when nothing is
# running (fresh box) or the caller states it explicitly.
if [ -z "${MACHINE:-}" ]; then
  MACHINE=$(ps -eo cmd \
    | grep -oE -- '--shard [0-9]+ --machines [0-9]+ --machine [0-9]+' \
    | grep -oE -- '--machine [0-9]+$' | awk '{print $2}' | sort -u | head -1)
  if [ -z "$MACHINE" ]; then
    MACHINE=0
    echo "note: no workers running here; assuming MACHINE=0. Pass MACHINE=k to be sure." >&2
  else
    echo "note: detected MACHINE=$MACHINE from the workers running on this box." >&2
  fi
fi
# 5 since 2026-09-21 (was 3). Override with MACHINES= if you are running fewer boxes --
# it only affects the rate/ETA arithmetic and which per-machine logs are looked for, not
# the partition (each box's own launcher passes its own --machines).
MACHINES="${MACHINES:-5}"
WORKERS="${WORKERS:-8}"
RESUME=0
FORCE=0
for a in "$@"; do
  case "$a" in
    --resume) RESUME=1 ;;
    # Restart this machine even though its worker count is FULL. Needed whenever the fix
    # is in the CODE rather than in the process table: workers exec run_dense_nr.py once
    # at launch, so a box that has been up for a day is still running whatever the file
    # said then. On 2026-09-19 m1/m2 were healthy at 8/8 while running the pre-retry
    # code, and plain --resume is a no-op in that state -- there was no way to roll the
    # fix out without hand-killing the process group.
    --force) FORCE=1; RESUME=1 ;;
  esac
done

TOTAL=$(python3 -c "import json;print(len(json.load(open('$RUN/manifest.json'))))")
# Count only COMPLETE records. `ls | wc -l` also counted the 0-byte `.tmp` scratch files
# a failed write leaves behind (28 of them during the 2026-09-19 quota squeeze), so
# progress read high while clips were actually being lost.
DONE=$(find "$RUN/results" -name '*.json' 2>/dev/null | wc -l)
# Permanent failures. A row is owned by exactly one worker shard, so anything recorded
# here is a clip NOBODY will revisit -- it must be visible, not buried in a log.
ERRS=$(ls "$RUN/errors" 2>/dev/null | wc -l)
STUBS=$(find "$RUN/results" -name '*.tmp' 2>/dev/null | wc -l)
LEFT=$((TOTAL - DONE))

# Workers of THIS machine only: match the exact --machine index so a shared filesystem
# view of another box's processes (there is none, but be explicit) cannot inflate this.
#
# The parent launcher also carries `--shard -1 --machine k` on its command line, so a
# plain `grep -c` counts it too and reports 9/8 "alive" -- which reads as a bug and, worse,
# can never be LESS than WORKERS, so the --resume trigger below would miss a machine that
# lost one worker. Require a non-negative --shard to count only the real workers.
alive_here() {
  ps -eo cmd | grep -cE -- "--shard [0-9]+ --machines [0-9]+ --machine $MACHINE( |\$)" || true
}
ALIVE=$(alive_here)

echo "=== dense-NR wave status  ($(date -u '+%Y-%m-%d %H:%M UTC')) ==="
printf '  container up      : %s   (ephemeral -- expect restarts)\n' "$(ps -o etime= -p 1 | tr -d ' ')"
printf '  manifest          : %s clips\n' "$TOTAL"
printf '  scored            : %s\n' "$DONE"
printf '  remaining         : %s\n' "$LEFT"
printf '  failed (perm)     : %s   %s\n' "$ERRS" \
  "$([ "$ERRS" -gt 0 ] && echo '<- nobody retries these; see errors/' || echo '')"
# Manifest rows whose source video was deleted after the inventory was built. These are
# DEAD rows, not failures: 567 of them point into the four stale trees removed on
# 2026-09-19 before the HF upload. run_dense_nr.py now skips them with SKIP-GONE instead
# of letting them fail on fps<=0, so they must be reported separately -- otherwise
# `remaining` never reaches 0 and it looks like the wave stalled short of the manifest.
GONE=$(grep -ch '^SKIP-GONE' "$RUN"/worker_*.log 2>/dev/null | awk '{s+=$1} END{print s+0}')
if [ "$GONE" -gt 0 ]; then
  printf '  skipped (source gone): %s   <- deleted stale trees; expected, not a failure\n' "$GONE"
fi
if [ "$STUBS" -gt 0 ]; then
  # A non-zero count here means writes are failing RIGHT NOW (quota/ENOSPC) or a worker
  # was killed mid-write. Either way these are not scored clips.
  printf '  !! write stubs    : %s incomplete .tmp in results/ -- writes are failing\n' "$STUBS"
fi
# Fresh quota/ENOSPC failures are the one fault that silently eats clips at a steady rate,
# so surface it explicitly rather than making someone grep 31k error files.
if [ -d "$RUN/errors" ]; then
  RECENT_Q=$(grep -l 'Disk quota\|No space left' \
    $(find "$RUN/errors" -name '*.json' -newermt '60 minutes ago' 2>/dev/null) 2>/dev/null | wc -l)
  if [ "${RECENT_Q:-0}" -gt 0 ]; then
    printf '  !! QUOTA          : %s write failures in the last 60min (shared fileset is full)\n' "$RECENT_Q"
    printf '                      retried 4x w/ backoff; anything in errors/ is still lost\n'
  fi
fi
printf '  this machine      : %s of %s, %s/%s workers alive\n' \
  "$MACHINE" "$MACHINES" "$ALIVE" "$WORKERS"

if [ -d "$RUN/results" ]; then
  # Sustained rate from the last 30 min of file mtimes -- the only honest measure. A
  # short spot-check right after launch reads ~2x too high because the fast clips in each
  # worker's queue finish first (measured: 4.80 clips/min over the first 5 minutes vs
  # 2.02 clips/min over the following 4.3 hours).
  #
  # This counts the SHARED results dir, so it is the AGGREGATE over every machine
  # currently working -- not this machine's own rate. Divide by the number of machines
  # whose logs are still being written to get per-machine throughput. (An earlier version
  # labelled this "ONE machine" and understated the ETA by 2x once a second machine
  # rejoined.)
  N=$(find "$RUN/results" -name '*.json' -newermt '30 minutes ago' 2>/dev/null | wc -l)
  # A machine counts as active if any of its worker logs was written in the last 5 min.
  # `worker_rev_*` is the extra (4th+) box added via run_dense_nr_full_extra.sh, which
  # walks the manifest backwards rather than owning a stride slot -- it must be counted or
  # the per-machine rate is overstated.
  ACTIVE=0
  for m in $(seq 0 $((MACHINES - 1))); do
    if [ -n "$(find "$RUN" -maxdepth 1 -name "worker_m${m}_*.log" -newermt '5 minutes ago' 2>/dev/null | head -1)" ]; then
      ACTIVE=$((ACTIVE + 1))
    fi
  done
  REV_ACTIVE=0
  if [ -n "$(find "$RUN" -maxdepth 1 -name 'worker_rev_*.log' -newermt '5 minutes ago' 2>/dev/null | head -1)" ]; then
    REV_ACTIVE=1
    ACTIVE=$((ACTIVE + 1))
  fi
  python3 - "$LEFT" "$N" "$ACTIVE" "$MACHINES" "$REV_ACTIVE" <<'PY'
import sys
left, n, active, machines, rev = (int(x) for x in sys.argv[1:6])
rate = n / 30
extra = ' (incl. 1 reverse walker)' if rev else ''
print(f'  rate (last 30min) : {rate:.2f} clips/min  TOTAL over {active} active box(es){extra}')
if active:
    print(f'                      = {rate/active:.2f} clips/min per box')
if rate <= 0:
    print('  ETA               : n/a (nothing completing)')
    raise SystemExit
print(f'  ETA as-is         : {left/rate/1440:.1f} days   ({active} box(es))')
if active < machines + 1:
    per = rate / active
    print(f'  ETA at {machines}+1 boxes  : {left/(per*(machines+1))/1440:.1f} days'
          f'   (3 stride shards + 1 reverse walker)')
PY
fi

echo '  per-machine STARTs:'
for m in $(seq 0 $((MACHINES - 1))); do
  t=0
  for i in $(seq 0 $((WORKERS - 1))); do
    f="$RUN/worker_m${m}_${i}.log"
    # NOTE: `grep -c` PRINTS 0 and EXITS 1 on no match, so `|| echo 0` appends a second
    # 0 -> "0\n0" -> `t + 0\n0` syntax error. That happens on every freshly-launched
    # machine (a worker that has not logged its first START yet) and aborted the whole
    # report, hiding the machines after it. Capture, then default only if empty.
    [ -f "$f" ] || continue
    c=$(grep -c '^START' "$f" 2>/dev/null) || true
    t=$((t + ${c:-0}))
  done
  last='-'
  newest=$(ls -t "$RUN"/worker_m${m}_*.log 2>/dev/null | head -1)
  [ -n "$newest" ] && last=$(date -u -r "$newest" '+%m-%d %H:%M')
  printf '    m%s: %6s STARTs   last write %s\n' "$m" "$t" "$last"
done
if ls "$RUN"/worker_rev_*.log >/dev/null 2>&1; then
  t=0
  for f in "$RUN"/worker_rev_*.log; do
    c=$(grep -c '^START' "$f" 2>/dev/null) || true   # same grep -c exit-1 trap as above
    t=$((t + ${c:-0}))
  done
  last=$(date -u -r "$(ls -t "$RUN"/worker_rev_*.log | head -1)" '+%m-%d %H:%M')
  printf '    rev:%6s STARTs   last write %s   (extra box, walks backwards)\n' "$t" "$last"
fi

if [ "$ALIVE" -lt "$WORKERS" ] || [ "$FORCE" = "1" ]; then
  echo
  if [ "$ALIVE" -lt "$WORKERS" ]; then
    echo "  !! machine $MACHINE has $ALIVE/$WORKERS workers"
  else
    echo "  machine $MACHINE is at $ALIVE/$WORKERS but --force was given (code-level restart)"
  fi
  if [ "$RESUME" = "1" ]; then
    if [ "$ALIVE" -gt 0 ]; then
      echo "  stopping the running set first (a second launcher would double the shards)"
      # pkill -f matches the worker command line; the parent launcher is matched too.
      # Kill the whole process group so no worker is orphaned onto the GPU: a survivor
      # keeps ~7GB and the relaunched set then OOMs on multiview.
      for pg in $(ps -eo pgid,cmd | grep -- "--machine $MACHINE" | grep -v grep \
                  | awk '{print $1}' | sort -u); do
        kill -- -"$pg" 2>/dev/null || true
      done
      pkill -f -- "run_dense_nr.py --out $RUN .*--machine $MACHINE" 2>/dev/null || true
      sleep 10
      echo "  workers still alive after kill: $(alive_here)"
    fi
    echo "  relaunching machine $MACHINE ..."
    MACHINE="$MACHINE" MACHINES="$MACHINES" WORKERS="$WORKERS" RUN="$RUN" \
      bash "$REPO/scripts/_run_dense_nr_full.sh"
  else
    echo "  re-run with --resume to restart it, or: bash scripts/run_dense_nr_full_m${MACHINE}.sh"
  fi
else
  echo
  echo "  machine $MACHINE healthy."
  echo "  (workers exec the script at launch -- use --force to reload code changes)"
fi
