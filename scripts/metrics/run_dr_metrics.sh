#!/usr/bin/env bash
# Launch ONE machine of the distortion-reference (DR) wave: PSNR / SSIM / LPIPS / FVD.
#
#   MACHINES=<N> bash scripts/metrics/run_dr_metrics.sh <machine 0..N-1>
#
# Mirrors the dense-NR wave's geometry exactly (`rows[MACHINE*W + shard :: W*MACHINES]`),
# so the same rules apply and for the same measured reasons:
#
#   * MACHINES and WORKERS must be IDENTICAL on every box. The split is positional, so a
#     disagreement silently stops the union from being a partition -- some rows scored
#     twice, others never -- and nothing errors, because every write is idempotent.
#   * One action per `det shell open`, and launch inside tmux (or setsid where tmux is
#     absent): a bare background job dies when the ssh session closes, and a compound
#     remote command gets truncated before its last step.
#
# WHY WORKERS DEFAULTS TO 4 AND NOT 8
#   DR holds TWO clips in VRAM at once (prediction + reference) plus VGG-LPIPS and I3D,
#   where the NR wave held one clip plus SAM2/GroundingDINO. Measured on a live 80 GB A100:
#   6 workers sat at 80.5 GB -- i.e. ~13.4 GB each, 1.6x an NR worker -- and produced
#   240 OOM retries with 22 clips lost outright in the first 1,115 (a 2% loss rate).
#   4 workers is ~53.6 GB, which leaves real headroom for the largest clips (an 800-frame
#   480x640 episode allocates ~1.9 GB in one step). Do not raise this without watching
#   `RETRY OOM` counts in the worker logs.
#
# ENV
#   MACHINES  total machines in this wave (no default -- must be stated explicitly)
#   WORKERS   workers per GPU (default 6; same on every box)
#   GPU       CUDA_VISIBLE_DEVICES (default 0)
#   RUN       shared run dir (default metrics/WorldScore/results/dr_full)
set -euo pipefail
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
MACHINE="${1:?usage: MACHINES=<N> bash scripts/metrics/run_dr_metrics.sh <machine 0..N-1>}"
MACHINES="${MACHINES:?set MACHINES to the total machine count (same on every box)}"
WORKERS="${WORKERS:-4}"
GPU="${GPU:-0}"
RUN="${RUN:-$REPO/metrics/WorldScore/results/dr_full}"
PY="$REPO/models/dreamgen/.venv/bin/python"

if [ "$MACHINE" -lt 0 ] || [ "$MACHINE" -ge "$MACHINES" ]; then
  echo "ERROR: machine index $MACHINE must be in [0, $MACHINES)" >&2
  exit 1
fi
if [ ! -f "$RUN/manifest.json" ]; then
  echo "ERROR: no manifest at $RUN/manifest.json -- build it ONCE first (CPU only):" >&2
  echo "  $PY scripts/metrics/compute_dr_metrics.py --out $RUN --inventory-only" >&2
  exit 1
fi

# Same wave-mode lock as the NR wave: the FIRST box pins machines/workers and any box that
# would join with a different partition refuses instead of silently corrupting coverage.
MODE_FILE="$RUN/wave_mode.json"
MODE_NOW="machines=$MACHINES workers=$WORKERS"
if [ -f "$MODE_FILE" ]; then
  WAS="$(cat "$MODE_FILE")"
  if [ "$WAS" != "$MODE_NOW" ] && [ "${FORCE_MODE:-0}" != "1" ]; then
    echo "ERROR: wave pinned as '$WAS' but this box wants '$MODE_NOW'." >&2
    echo "Stop every box, delete $MODE_FILE, and relaunch them all." >&2
    exit 1
  fi
else
  printf '%s' "$MODE_NOW" > "$MODE_FILE.$$.tmp" && mv "$MODE_FILE.$$.tmp" "$MODE_FILE"
fi

mkdir -p "$RUN/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$RUN/logs/machine${MACHINE}_${STAMP}.log"
TODO=$("$PY" - "$RUN" <<'PY'
import json, os, sys
run = sys.argv[1]
rows = json.load(open(f'{run}/manifest.json'))
have = set(os.listdir(f'{run}/results')) if os.path.isdir(f'{run}/results') else set()
print(sum(1 for r in rows if f'{r["id"]}.json' not in have))
PY
)

cat <<EOF
=== DR wave (PSNR / SSIM / LPIPS / I3D features) ===
  machine        : $MACHINE of $MACHINES
  workers        : $WORKERS   (GPU $GPU)
  run dir        : $RUN
  pairs          : $("$PY" -c "import json;print(len(json.load(open('$RUN/manifest.json'))))")
  still to score : $TODO
  log            : $LOG
EOF

cd "$REPO"
setsid env CUDA_VISIBLE_DEVICES="$GPU" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
  "$PY" -u scripts/metrics/compute_dr_metrics.py \
    --out "$RUN" --reuse-manifest \
    --workers "$WORKERS" --machines "$MACHINES" --machine "$MACHINE" \
  >"$LOG" 2>&1 &
echo "  launched PGID $!"
echo
echo "monitor : tail -f $LOG"
echo "          grep -c '^DONE' $RUN/worker_m${MACHINE}_*.log | paste -sd' '"
echo "aggregate FVD once ALL machines are done:"
echo "          $PY scripts/metrics/compute_dr_metrics.py --out $RUN --fvd-only"
