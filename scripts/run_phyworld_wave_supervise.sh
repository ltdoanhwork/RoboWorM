#!/usr/bin/env bash
#
# Supervise the PhyWorld wave across container restarts.
#
# This box kills containers without warning (it already happened once during the
# single_arm MV dreamgen re-render). The wave itself is resumable -- every
# episode whose per-episode `phyworld.json` sidecar exists is skipped -- so the
# only thing missing is something to start it again. This loop does that, and
# stops as soon as a pass completes with nothing left to score.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_phyworld_wave_supervise.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MAX_PASSES="${MAX_PASSES:-40}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-30}"
LOG_DIR="$REPO/logs/phyworld"
mkdir -p "$LOG_DIR"
SUP_LOG="$LOG_DIR/supervise.log"

remaining() {
  EMBODIMENT="${EMBODIMENT:-single_arm}" VIEW="${VIEW:-multiview}" \
  MODEL="${MODEL:-dreamgen}" bash scripts/phyworld_wave_progress.sh 2>/dev/null \
    | awk '/TOTAL:/ {split($2,a,"/"); print a[2]-a[1]}'
}

for pass in $(seq 1 "$MAX_PASSES"); do
  left="$(remaining)"
  echo "[$(date '+%F %T')] pass $pass/$MAX_PASSES — remaining=${left:-?}" >>"$SUP_LOG"
  if [[ -n "${left:-}" && "$left" -le 0 ]]; then
    echo "[$(date '+%F %T')] wave COMPLETE" >>"$SUP_LOG"
    break
  fi
  bash scripts/run_phyworld_wave_single_arm_mv_dreamgen.sh >>"$SUP_LOG" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] pass $pass exited rc=$rc" >>"$SUP_LOG"
  sleep "$SLEEP_BETWEEN"
done

echo "[$(date '+%F %T')] supervisor exiting" >>"$SUP_LOG"
bash scripts/phyworld_wave_progress.sh >>"$SUP_LOG" 2>&1
