#!/usr/bin/env bash
# Sim2real interaction metric over a WHOLE ctrlworld method tree.
#
# Same env plumbing as run_sim2real_interaction.sh (single episode). Writes
# per-episode sim2real_metrics.json + tracks.npz into
#   <episode dir>/sim2real/
# and the roll-up to
#   sampling_dataset/{METHOD}/single_arm/output/multiview/ctrlworld/sim2real_summary.json
# so it sits next to the existing metrics.json / physics_metrics.json per the
# sampling-dataset-layout rule.
#
# Tracking dominates the cost (~100 s/episode), so tracks.npz is cached and a
# re-run with different thresholds is nearly free. Set FORCE=1 to retrack.
set -euo pipefail

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
PY=$REPO/models/dreamgen/.venv/bin/python
WS=$REPO/metrics/WorldScore
TP=$WS/worldscore/benchmark/metrics/third_party
CKPT=$REPO/checkpoints/metrics/worldscore

METHOD=${METHOD:-dense}
EMB=${EMB:-single_arm}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO/sampling_dataset/$METHOD/$EMB/output/multiview/ctrlworld}
INPUT_ROOT=${INPUT_ROOT:-$REPO/sampling_dataset/dense/$EMB/input/multiview/ctrlworld}
SPLITS=${SPLITS:-makovian non_makovian}
MAX_EPISODES=${MAX_EPISODES:-}
DEPTH=${DEPTH:-none}
WORK_ROOT=${WORK_ROOT:-}

cd "$WS"
export PYTHONPATH="$REPO:$TP:$TP/sam2:$TP/SEA-RAFT/core:$TP/SEA-RAFT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
# SAM2 emits a tqdm bar per view per video; at 6 bars/episode it buries the log
export TQDM_DISABLE=${TQDM_DISABLE:-1}
export REPO

ARGS=(
  --output_root "$OUTPUT_ROOT"
  --input_root  "$INPUT_ROOT"
  --categories  $SPLITS
  --sam2_cfg    "configs/sam2.1/sam2.1_hiera_b+.yaml"
  --sam2_ckpt   "$CKPT/sam2.1_hiera_base_plus.pt"
  --gdino_cfg   "worldscore/benchmark/metrics/third_party/groundingdino/config/GroundingDINO_SwinT_OGC.py"
  --gdino_ckpt  "$CKPT/groundingdino_swint_ogc.pth"
  --depth       "$DEPTH"
)
[[ -n "$MAX_EPISODES" ]] && ARGS+=(--max_episodes "$MAX_EPISODES")
[[ -n "$WORK_ROOT" ]] && ARGS+=(--work_root "$WORK_ROOT")
[[ "${FORCE:-0}" == "1" ]] && ARGS+=(--force)
[[ "${VIZ:-0}" == "1" ]] && ARGS+=(--viz)

"$PY" "$REPO/metrics/sim2real/compute_sim2real_batch.py" "${ARGS[@]}" "$@"
