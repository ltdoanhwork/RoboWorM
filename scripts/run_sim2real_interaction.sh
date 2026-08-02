#!/usr/bin/env bash
# Sim2real interaction probe on ONE ctrlworld episode.
#
# Mirrors scripts/run_motion_accuracy_gdino.sh: runs from metrics/WorldScore so the
# vendored sam2 hydra configs resolve, with the third_party paths on PYTHONPATH and
# the dreamgen venv python. Checkpoints come from checkpoints/worldscore.
set -euo pipefail

REPO=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics
PY=$REPO/models/dreamgen/.venv/bin/python
WS=$REPO/metrics/WorldScore
TP=$WS/worldscore/benchmark/metrics/third_party
CKPT=$REPO/checkpoints/worldscore

METHOD=${METHOD:-dense}
SPLIT=${SPLIT:-makovian}
EPISODE=${EPISODE:-episode_000255}
EP_DIR=${EP_DIR:-$REPO/sampling_dataset/$METHOD/single_arm/output/multiview/ctrlworld/$SPLIT/$EPISODE}
IN_DIR=${IN_DIR:-$REPO/sampling_dataset/dense/single_arm/input/multiview/ctrlworld/$SPLIT/$EPISODE}
OUT=${OUT:-$REPO/metrics/sim2real_out/$METHOD/$SPLIT/$EPISODE}

cd "$WS"
export PYTHONPATH="$REPO:$TP:$TP/sam2:$TP/SEA-RAFT/core:$TP/SEA-RAFT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
# SAM2 emits a tqdm bar per view per video; at 6 bars/episode it buries the log
export TQDM_DISABLE=${TQDM_DISABLE:-1}

"$PY" "$REPO/metrics/sim2real/interaction_probe.py" \
  --gt "$EP_DIR/gt_all_views.mp4" \
  --pred "$EP_DIR/pred_all_views.mp4" \
  --metadata "$IN_DIR/metadata.json" \
  --out "$OUT" \
  --sam2_cfg "configs/sam2.1/sam2.1_hiera_b+.yaml" \
  --sam2_ckpt "$CKPT/sam2.1_hiera_base_plus.pt" \
  --gdino_cfg "worldscore/benchmark/metrics/third_party/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
  --gdino_ckpt "$CKPT/groundingdino_swint_ogc.pth" \
  "$@"
