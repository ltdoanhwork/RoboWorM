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
MACHINE="${MACHINE:-0}"
MACHINES="${MACHINES:-3}"
WORKERS="${WORKERS:-8}"
RESUME=0
[ "${1:-}" = "--resume" ] && RESUME=1

TOTAL=$(python3 -c "import json;print(len(json.load(open('$RUN/manifest.json'))))")
DONE=$(ls "$RUN/results" 2>/dev/null | wc -l)
LEFT=$((TOTAL - DONE))

# Workers of THIS machine only: match the exact --machine index so a shared filesystem
# view of another box's processes (there is none, but be explicit) cannot inflate this.
alive_here() {
  ps -eo cmd | grep -- '--shard' | grep -c -- "--machine $MACHINE" || true
}
ALIVE=$(alive_here)

echo "=== dense-NR wave status  ($(date -u '+%Y-%m-%d %H:%M UTC')) ==="
printf '  container up      : %s   (ephemeral -- expect restarts)\n' "$(ps -o etime= -p 1 | tr -d ' ')"
printf '  manifest          : %s clips\n' "$TOTAL"
printf '  scored            : %s\n' "$DONE"
printf '  remaining         : %s\n' "$LEFT"
printf '  this machine      : %s of %s, %s/%s workers alive\n' \
  "$MACHINE" "$MACHINES" "$ALIVE" "$WORKERS"

if [ -d "$RUN/results" ]; then
  # Sustained rate from the last 30 min of file mtimes -- the only honest measure. A
  # short spot-check right after launch reads ~2x too high because the fast clips in each
  # worker's queue finish first (measured: 4.80 clips/min over the first 5 minutes vs
  # 2.02 clips/min over the following 4.3 hours).
  RATE=$(find "$RUN/results" -name '*.json' -newermt '30 minutes ago' 2>/dev/null | wc -l)
  RATE=$(python3 -c "print(f'{$RATE/30:.2f}')")
  printf '  rate (last 30min) : %s clips/min\n' "$RATE"
  python3 - "$LEFT" "$RATE" "$MACHINES" <<'PY'
import sys
left, rate, machines = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
if rate <= 0:
    print('  ETA               : n/a (nothing completing)')
else:
    print(f'  ETA at this rate  : {left/rate/1440:.1f} days')
    print(f'  ETA if {machines} machines : {left/(rate*machines)/1440:.1f} days'
          f'   (this rate is ONE machine)')
PY
fi

echo '  per-machine STARTs:'
for m in $(seq 0 $((MACHINES - 1))); do
  t=0
  for i in $(seq 0 $((WORKERS - 1))); do
    f="$RUN/worker_m${m}_${i}.log"
    [ -f "$f" ] && t=$((t + $(grep -c '^START' "$f" 2>/dev/null || echo 0)))
  done
  last='-'
  newest=$(ls -t "$RUN"/worker_m${m}_*.log 2>/dev/null | head -1)
  [ -n "$newest" ] && last=$(date -u -r "$newest" '+%m-%d %H:%M')
  printf '    m%s: %6s STARTs   last write %s\n' "$m" "$t" "$last"
done

if [ "$ALIVE" -lt "$WORKERS" ]; then
  echo
  echo "  !! machine $MACHINE has $ALIVE/$WORKERS workers"
  if [ "$RESUME" = "1" ]; then
    if [ "$ALIVE" -gt 0 ]; then
      echo "  stopping the partial set first (a second launcher would double the shards)"
      pkill -f -- "run_dense_nr.py --out $RUN .*--machine $MACHINE" || true
      sleep 5
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
fi
