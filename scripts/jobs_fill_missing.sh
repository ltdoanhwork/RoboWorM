#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DRY_RUN="${DRY_RUN:-1}"
WAVE="${WAVE:-all}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

run() {
  echo ""
  echo ">>> $*"
  if [[ "$DRY_RUN" != "0" ]]; then
    echo "    (DRY_RUN=1 — not launched)"
  else
    eval "$@"
  fi
}

wave1() {
  run "SPLITS=non_makovian NUM_CHUNKS='4 6 7' ITER_NAME=iter_000090000 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=${GPU} bash scripts/run_single_arm_singleview_dreamgen_static16fps.sh"
  run "SPLITS=non_makovian NUM_CHUNKS='4 6 7' ITER_NAME=iter_000090000 CUDA_VISIBLE_DEVICES=${GPU} bash scripts/run_single_arm_singleview_dreamgen_sito.sh"
  run "SPLITS=non_makovian NUM_CHUNKS='4 6 7' ITER_NAME=iter_000090000 CUDA_VISIBLE_DEVICES=${GPU} bash scripts/run_single_arm_singleview_dreamgen_itm.sh"
  run "SPLITS=non_makovian NUM_CHUNKS='4 6 7' ITER_NAME=iter_000090000 CUDA_VISIBLE_DEVICES=${GPU} bash scripts/run_single_arm_singleview_dreamgen_pisa.sh"
  run "SPLITS=non_makovian NUM_CHUNKS='4 6 7' ITER_NAME=iter_000090000 CUDA_VISIBLE_DEVICES=${GPU} bash scripts/run_single_arm_singleview_dreamgen_radial.sh"
  run "SPLITS=non_makovian NUM_CHUNKS='4 6 7' ITER_NAME=iter_000090000 CUDA_VISIBLE_DEVICES=${GPU} bash scripts/run_single_arm_singleview_dreamgen_svg1.sh"
  run "SPLITS=non_makovian NUM_CHUNKS='4 6 7' ITER_NAME=iter_000090000 CUDA_VISIBLE_DEVICES=${GPU} bash scripts/run_single_arm_singleview_dreamgen_svg2.sh"
}

wave2() {
  run "SPLITS=non_makovian NUM_CHUNKS='3 5 6 18' CUDA_VISIBLE_DEVICES=${GPU} bash scripts/run_bimanual_singleview_dreamgen_fastercache.sh"
}

case "$WAVE" in
  all) wave1; wave2 ;;
  1) wave1 ;;
  2) wave2 ;;
  *) echo "Unknown WAVE=$WAVE (all|1|2)" >&2; exit 1 ;;
esac
echo "Done. DRY_RUN=$DRY_RUN WAVE=$WAVE GPU=$GPU"
