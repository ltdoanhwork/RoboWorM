#!/usr/bin/env bash
#
# PhyWorld wave driver: score ONE cell across every method, all three tracks,
# sequentially on a single GPU.
#
# Why sequential (one accel method per invocation)
# ------------------------------------------------
# Scoring the full cell is a multi-day job on one GPU, and this box restarts
# containers without warning. Running `--methods dense <accel>` once per method
# keeps each invocation short enough that a kill costs at most one episode, and
# `compute_phyworld.py` skips any episode whose per-episode `phyworld.json`
# sidecar already exists -- so re-running this script simply resumes.
# `dense` is scored first because PhyFid-DR uses the dense pred as reference;
# later invocations reuse those dense sidecars instead of recomputing them.
#
# Usage
# -----
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_phyworld_wave_single_arm_mv_dreamgen.sh
#
#   SPLITS="makovian"          # subset of splits
#   ACCEL="pisa svg1"          # subset of accel methods
#   MAX_EPISODES=32            # cap episodes per split
#   TRACK=all                  # qual | dyn | fid | all
#
set -uo pipefail   # deliberately NOT -e: one failing method must not abort the wave

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

EMBODIMENT="${EMBODIMENT:-single_arm}"
VIEW="${VIEW:-multiview}"
MODEL="${MODEL:-dreamgen}"
SPLITS="${SPLITS:-makovian non_makovian}"
# Ordered cheapest-signal-first so a partial wave still yields a usable table.
ACCEL="${ACCEL:-pisa svg1 svg2 radial sito itm worldcache dicache fastercache}"
MAX_EPISODES="${MAX_EPISODES:-999}"
TRACK="${TRACK:-all}"

LOG_DIR="${LOG_DIR:-$REPO/logs/phyworld}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
WAVE_LOG="$LOG_DIR/wave_${EMBODIMENT}_${VIEW}_${MODEL}_${STAMP}.log"

echo "===============================================================" | tee -a "$WAVE_LOG"
echo "PhyWorld wave: $EMBODIMENT / $VIEW / $MODEL" | tee -a "$WAVE_LOG"
echo "  splits       : $SPLITS" | tee -a "$WAVE_LOG"
echo "  accel        : $ACCEL" | tee -a "$WAVE_LOG"
echo "  track        : $TRACK" | tee -a "$WAVE_LOG"
echo "  max episodes : $MAX_EPISODES" | tee -a "$WAVE_LOG"
echo "  GPU          : ${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$WAVE_LOG"
echo "  log          : $WAVE_LOG" | tee -a "$WAVE_LOG"
echo "===============================================================" | tee -a "$WAVE_LOG"

run_step() {
  local split="$1"; shift
  local label="$1"; shift
  local step_log="$LOG_DIR/${MODEL}_${split}_${label}.log"
  echo "" | tee -a "$WAVE_LOG"
  echo ">>> [$(date +%H:%M:%S)] $split / $label" | tee -a "$WAVE_LOG"
  bash scripts/run_phyworld.sh \
    --track "$TRACK" \
    --embodiment "$EMBODIMENT" --view "$VIEW" --model "$MODEL" \
    --split "$split" \
    --max-episodes "$MAX_EPISODES" \
    --matched-only \
    "$@" >>"$step_log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "    OK   (rc=0)  -> $step_log" | tee -a "$WAVE_LOG"
  else
    echo "    FAIL (rc=$rc) -> $step_log  [continuing]" | tee -a "$WAVE_LOG"
    tail -n 15 "$step_log" | sed 's/^/      /' | tee -a "$WAVE_LOG"
  fi
  return 0
}

for split in $SPLITS; do
  # dense first: it is the PhyFid-DR reference for every accel method below.
  run_step "$split" "dense" --methods dense
  for m in $ACCEL; do
    run_step "$split" "$m" --methods dense "$m"
  done
done

echo "" | tee -a "$WAVE_LOG"
echo "===== wave finished $(date) =====" | tee -a "$WAVE_LOG"
bash scripts/phyworld_wave_progress.sh 2>/dev/null | tee -a "$WAVE_LOG"
