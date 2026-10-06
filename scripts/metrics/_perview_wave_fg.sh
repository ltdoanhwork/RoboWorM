#!/usr/bin/env bash
# Foreground per-camera FVD feature-extraction worker, for running INSIDE a detached tmux
# session on a Determined shell.
#
# WHY FOREGROUND (measured, do not "simplify" to nohup ... &):
#   `det shell open <id> -- 'tmux new-session -d -s X "<cmd>"'` tears the session down as
#   soon as <cmd> returns, and the cluster's session-scope cleanup then kills any
#   backgrounded child. Verified on shell e9f7ae15: tmux ls -> "no server running", GPU
#   0 MiB, log 0 bytes. `exec`ing python in the foreground keeps the session alive for
#   exactly the job's lifetime. Same reason scripts/metrics/_dr_wave_fg.sh exists.
#
#   MACHINES=<N> bash scripts/metrics/_perview_wave_fg.sh <machine 0..N-1>
set -u
MACHINE="${1:?usage: MACHINES=<N> bash scripts/metrics/_perview_wave_fg.sh <machine 0..N-1>}"
MACHINES="${MACHINES:?set MACHINES to the total machine count (same on every box)}"
GPU="${GPU:-0}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT="${OUT:-$REPO/metrics/WorldScore/results/fvd_perview}"
PY="$REPO/models/dreamgen/.venv/bin/python"

[ -f "$REPO/metrics/WorldScore/results/dr_full/manifest.json" ] || {
  echo "ERROR: no DR manifest to read rows from" >&2; exit 1; }

mkdir -p "$OUT/feats" "$OUT/errors" "$OUT/logs"
echo "=== per-view FVD extraction: machine $MACHINE of $MACHINES, GPU $GPU ==="
echo "host : $(hostname)"
echo "out  : $OUT"
cd "$REPO"
exec env CUDA_VISIBLE_DEVICES="$GPU" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
  "$PY" -u assets/scripts/extract_perview_fvd_feats.py \
    --out "$OUT" --machines "$MACHINES" --machine "$MACHINE"
