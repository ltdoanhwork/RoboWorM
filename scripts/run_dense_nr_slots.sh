#!/usr/bin/env bash
# Claim a RANGE of machine slots for one box of a dense-NR wave, spread over its GPUs.
#
#   MACHINES=<total> WORKERS=<per slot> bash scripts/run_dense_nr_slots.sh <first> <n> [gpus]
#
#   # 80GB single-GPU box taking slots 0-3 (4 slots x 2 workers = 8 workers on GPU 0):
#   MACHINES=28 WORKERS=2 bash scripts/run_dense_nr_slots.sh 0 4
#
#   # weak 4-GPU box taking slots 24-27, one slot per GPU (2 workers per GPU):
#   MACHINES=28 WORKERS=2 bash scripts/run_dense_nr_slots.sh 24 4 0,1,2,3
#
# WHY SLOTS INSTEAD OF "ONE INDEX PER BOX"
#   The partition is `rows[MACHINE*W + shard :: W*MACHINES]`, so W (workers per machine
#   index) is a factor of the global stride. Every box in the wave MUST use the same W or
#   the union stops being a partition -- some rows owned by nobody, others scored twice --
#   and because every write is idempotent NOTHING ERRORS to tell you.
#
#   That is fine for identical boxes, but it breaks on HETEROGENEOUS hardware: an 80GB
#   A100 fits 8 workers while a 20GB card fits 2, and they cannot both be "one index".
#
#   Fix: make W the *smallest* box's capacity and let bigger GPUs claim SEVERAL slots on
#   the SAME card. W stays uniform (partition stays exact) while the worker count per GPU
#   becomes `W x slots_on_that_gpu`, which is free to differ per box.
#
#     W=2  ->  80GB GPU takes 4 slots = 8 workers;  20GB GPU takes 1 slot = 2 workers
#
# MEASURED SIZING (this workload, 2026-09-21, on a live 8-worker box)
#   VRAM  7.66 GB per worker (8 workers = 62.7 GB of 80 GB), SAM2 propagation is the peak
#   CPU   0.93 core per worker  -- GPU-bound: util sits at 99-100% with 8 workers
#   RAM   ~3.0 GB RSS per worker
#   rate  3.84 clips/min per 8-worker 80GB GPU
#   =>  workers a GPU can hold ~= floor(0.88 * VRAM_GB / 7.66)
#       80GB -> 9 (use the measured knee 8), 40GB -> 4, 24GB -> 2, 20GB -> 2
#
# ENV
#   MACHINES  total slot count of the WHOLE wave (sum of every box's slots). Same on all.
#   WORKERS   workers per slot. Same on all boxes. Set it to the weakest GPU's capacity.
#   RUN       shared run dir (default metrics/WorldScore/results/dense_nr_full)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
FIRST="${1:?usage: MACHINES=<N> WORKERS=<W> bash scripts/run_dense_nr_slots.sh <first slot> <n slots> [gpu,list]}"
NSLOT="${2:?usage: MACHINES=<N> WORKERS=<W> bash scripts/run_dense_nr_slots.sh <first slot> <n slots> [gpu,list]}"
GPUS="${3:-0}"
MACHINES="${MACHINES:?set MACHINES to the TOTAL slot count of the wave (same on every box)}"
WORKERS="${WORKERS:?set WORKERS = workers per slot (same on every box; weakest GPU capacity)}"
RUN="${RUN:-$REPO/metrics/WorldScore/results/dense_nr_full}"

GB_PER_WORKER=7.66     # measured
CORE_PER_WORKER=0.93   # measured
HEADROOM=0.88          # SAM2 propagation peaks above the median; keep 12% free

IFS=',' read -r -a GPUARR <<< "$GPUS"
NGPU="${#GPUARR[@]}"
LAST=$((FIRST + NSLOT - 1))

if [ "$FIRST" -lt 0 ] || [ "$LAST" -ge "$MACHINES" ]; then
  echo "ERROR: slots $FIRST..$LAST do not fit in [0, $MACHINES)." >&2
  echo "  MACHINES must equal the SUM of every box's slots across the whole wave." >&2
  exit 1
fi

# --- per-GPU VRAM check -----------------------------------------------------------------
# Slots are dealt round-robin, so GPU i gets ceil/floor of NSLOT/NGPU slots.
fail=0
for i in "${!GPUARR[@]}"; do
  g="${GPUARR[$i]}"
  slots_here=$(( NSLOT / NGPU + (i < NSLOT % NGPU ? 1 : 0) ))
  [ "$slots_here" -eq 0 ] && continue
  w_here=$(( slots_here * WORKERS ))
  mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo 0)
  if [ "$mem" -eq 0 ]; then
    echo "WARNING: cannot read memory of GPU $g -- skipping its VRAM check" >&2
    continue
  fi
  need=$(awk "BEGIN{printf \"%.1f\", $w_here * $GB_PER_WORKER}")
  cap=$(awk "BEGIN{printf \"%.1f\", $mem/1024 * $HEADROOM}")
  ok=$(awk "BEGIN{print ($need <= $cap) ? \"ok\" : \"OVER\"}")
  printf "  GPU %-3s %5s MiB  slots=%d workers=%d  need %sGB / usable %sGB  %s\n" \
         "$g" "$mem" "$slots_here" "$w_here" "$need" "$cap" "$ok"
  [ "$ok" = "OVER" ] && fail=1
done
if [ "$fail" = "1" ]; then
  echo "ERROR: would OOM. Either lower WORKERS on EVERY box (and recompute MACHINES)," >&2
  echo "       or claim fewer slots here. FORCE=1 overrides." >&2
  [ "${FORCE:-0}" = "1" ] || exit 1
fi

# --- box-wide core check ----------------------------------------------------------------
CORES=$(nproc)
TOTW=$(( NSLOT * WORKERS ))
NEEDC=$(awk "BEGIN{printf \"%d\", $CORE_PER_WORKER * $TOTW + 0.5}")
echo "  box: $TOTW workers total, ~$NEEDC of $CORES cores, ~$(awk "BEGIN{printf \"%.0f\", 3.0*$TOTW}")GB RAM"
if [ "$NEEDC" -gt "$CORES" ]; then
  echo "WARNING: cores oversubscribed ($NEEDC > $CORES). FORCE=1 to proceed." >&2
  [ "${FORCE:-0}" = "1" ] || exit 1
fi

echo
for i in $(seq 0 $((NSLOT - 1))); do
  slot=$((FIRST + i))
  g="${GPUARR[$((i % NGPU))]}"
  echo "=== slot $slot of $MACHINES  ->  GPU $g  ($WORKERS workers) ==="
  # ONLY_IDS is deliberately empty: full spec coverage needs the WHOLE manifest.
  MACHINE="$slot" MACHINES="$MACHINES" GPU="$g" WORKERS="$WORKERS" RUN="$RUN" ONLY_IDS="" \
    "$REPO/scripts/_run_dense_nr_full.sh"
done

echo
echo "claimed slots $FIRST..$LAST of $MACHINES on GPUs $GPUS"
echo "verify : MACHINES=$MACHINES bash scripts/dense_nr_status.sh"
echo "stop   : pkill -f 'run_dense_nr.py'"
