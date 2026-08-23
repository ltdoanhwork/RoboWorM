#!/usr/bin/env bash
# Run the standalone motion_accuracy probe on the close_cardboard_box ep000026
# GT (real data) vs dreamgen prediction.
#
# Must run from metrics/WorldScore (SAM2 hydra config_path=".") with vendored
# sam2 + SEA-RAFT core on PYTHONPATH. Uses the dreamgen venv (torch+cuda) and the
# checkpoints set up under checkpoints/worldscore.
set -euo pipefail

REPO=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics
PY=$REPO/models/dreamgen/.venv/bin/python
WS=$REPO/metrics/WorldScore
TP=$WS/worldscore/benchmark/metrics/third_party
CKPT=$REPO/checkpoints/worldscore

GT=${GT:-$REPO/datasets/bimanual_dreamgen/videos/makovian__close_cardboard_box__ep000026.mp4}
PRED=${PRED:-$REPO/sampling_dataset/dense/bimanual/output/singleview/dreamgen/makovian/iter_000110000_bimanual16fps/close_cardboard_box__episode_000026.mp4}
OUT=${OUT:-$REPO/metrics/motion_probe_out/close_cardboard_box__ep000026}
BOX=${BOX:-0.23,0.08,0.78,0.68}

cd "$WS"
export PYTHONPATH="$TP:$TP/sam2:$TP/SEA-RAFT/core:$TP/SEA-RAFT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

"$PY" "$REPO/metrics/dynamics/motion_accuracy_probe.py" \
  --gt "$GT" \
  --pred "$PRED" \
  --out "$OUT" \
  --sam2_cfg "configs/sam2.1/sam2.1_hiera_b+.yaml" \
  --sam2_ckpt "$CKPT/sam2.1_hiera_base_plus.pt" \
  --raft_cfg "worldscore/benchmark/metrics/third_party/SEA-RAFT/config/eval/spring-M.json" \
  --raft_ckpt "$CKPT/Tartan-C-T-TSKH-spring540x960-M.pth" \
  --box "$BOX" \
  "$@"
