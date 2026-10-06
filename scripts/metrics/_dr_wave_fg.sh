#!/usr/bin/env bash
# Foreground DR-wave worker launcher, for running INSIDE a detached tmux session.
#
# WHY THIS EXISTS (measured, do not "simplify" back):
#   scripts/metrics/run_dr_metrics.sh launches its python with `setsid ... &` and then RETURNS.
#   Inside `det shell open ... tmux new-session -d -s X "<cmd>"` that is fatal: the tmux
#   command finishes immediately, tmux tears the session down, and the cluster's
#   session-scope cleanup kills the orphaned setsid child. Verified on shell
#   e9f7ae15: `tmux ls` -> "no server running", pgrep empty, GPU 0 MiB, machine log
#   0 bytes, results count unchanged after 3 minutes.
#
#   This script instead `exec`s python in the FOREGROUND, so the tmux session lives
#   exactly as long as the job and the job is never orphaned.
#
# Same partition geometry / env / worker count as run_dr_metrics.sh, so a wave may mix
# boxes launched either way as long as MACHINES and WORKERS agree everywhere.
#
#   MACHINES=<N> bash scripts/metrics/_dr_wave_fg.sh <machine 0..N-1>
set -u
MACHINE="${1:?usage: MACHINES=<N> bash scripts/metrics/_dr_wave_fg.sh <machine 0..N-1>}"
MACHINES="${MACHINES:?set MACHINES to the total machine count (same on every box)}"
WORKERS="${WORKERS:-4}"
GPU="${GPU:-0}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
RUN="${RUN:-$REPO/metrics/WorldScore/results/dr_full}"
PY="$REPO/models/dreamgen/.venv/bin/python"

[ -f "$RUN/manifest.json" ] || { echo "ERROR: no manifest at $RUN/manifest.json" >&2; exit 1; }

# Same wave-mode lock as run_dr_metrics.sh: the first box pins the partition and any box
# that would join with a different one refuses rather than silently breaking coverage.
MODE_FILE="$RUN/wave_mode.json"
MODE_NOW="machines=$MACHINES workers=$WORKERS"
if [ -f "$MODE_FILE" ]; then
  WAS="$(cat "$MODE_FILE")"
  if [ "$WAS" != "$MODE_NOW" ] && [ "${FORCE_MODE:-0}" != "1" ]; then
    echo "ERROR: wave pinned as '$WAS' but this box wants '$MODE_NOW'." >&2
    exit 1
  fi
else
  printf '%s' "$MODE_NOW" > "$MODE_FILE.$$.tmp" && mv "$MODE_FILE.$$.tmp" "$MODE_FILE"
fi

mkdir -p "$RUN/results" "$RUN/errors" "$RUN/logs"
echo "=== DR wave (foreground) machine $MACHINE of $MACHINES, $WORKERS workers, GPU $GPU ==="
echo "run dir : $RUN"
echo "host    : $(hostname)"
cd "$REPO"
exec env CUDA_VISIBLE_DEVICES="$GPU" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
  "$PY" -u scripts/metrics/compute_dr_metrics.py \
    --out "$RUN" --reuse-manifest \
    --workers "$WORKERS" --machines "$MACHINES" --machine "$MACHINE"
