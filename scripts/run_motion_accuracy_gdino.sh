#!/usr/bin/env bash
# Embodiment-aware motion_accuracy probe: GroundingDINO detects the manipulated
# object AND the robot arms/grippers from the prompt, seeds a SAM2 track for each,
# and measures foreground(arm+object)-vs-background flow. Compares GT vs DreamGen.
#
# Run from metrics/WorldScore with vendored sam2 + SEA-RAFT + groundingdino on
# PYTHONPATH, dreamgen venv, checkpoints under checkpoints/worldscore.
set -euo pipefail

REPO=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics
PY=$REPO/models/dreamgen/.venv/bin/python
WS=$REPO/metrics/WorldScore
TP=$WS/worldscore/benchmark/metrics/third_party
CKPT=$REPO/checkpoints/worldscore

GT=${GT:-$REPO/datasets/bimanual_dreamgen/videos/makovian__close_cardboard_box__ep000026.mp4}
PRED=${PRED:-$REPO/sampling_dataset/dense/bimanual/output/singleview/dreamgen/makovian/iter_000110000_bimanual16fps/close_cardboard_box__episode_000026.mp4}
OUT=${OUT:-$REPO/metrics/motion_probe_out/close_cardboard_box__ep000026}
OBJECT_PROMPT=${OBJECT_PROMPT:-"cardboard box . box ."}
EMBODIMENT_PROMPT=${EMBODIMENT_PROMPT:-"robot arm . robot gripper ."}

cd "$WS"
export PYTHONPATH="$TP:$TP/sam2:$TP/SEA-RAFT/core:$TP/SEA-RAFT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

"$PY" "$REPO/metrics/dynamics/motion_accuracy_gdino.py" \
  --gt "$GT" --pred "$PRED" --out "$OUT" \
  --sam2_cfg "configs/sam2.1/sam2.1_hiera_b+.yaml" \
  --sam2_ckpt "$CKPT/sam2.1_hiera_base_plus.pt" \
  --raft_cfg "worldscore/benchmark/metrics/third_party/SEA-RAFT/config/eval/spring-M.json" \
  --raft_ckpt "$CKPT/Tartan-C-T-TSKH-spring540x960-M.pth" \
  --gdino_cfg "worldscore/benchmark/metrics/third_party/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
  --gdino_ckpt "$CKPT/groundingdino_swint_ogc.pth" \
  --object_prompt "$OBJECT_PROMPT" \
  --embodiment_prompt "$EMBODIMENT_PROMPT" \
  "$@"
