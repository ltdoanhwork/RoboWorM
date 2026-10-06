#!/usr/bin/env bash
# Transform GT to each model's (resolution, fps) grid, then run the SAME
# GroundingDINO->SAM2->SEA-RAFT motion_accuracy probe on the matched pair so
# px/frame flow is directly comparable (no res/fps confound, no formula-normalize).
#
# Mirrors run_motion_accuracy_gdino.sh (same cwd/env/ckpts); only adds an ffmpeg
# pre-step that resamples GT to each model's grid and runs two matched pairs.
set -euo pipefail

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
PY=$REPO/models/dreamgen/.venv/bin/python
WS=$REPO/metrics/WorldScore
TP=$WS/worldscore/benchmark/metrics/third_party
CKPT=$REPO/checkpoints/metrics/worldscore

GT=${GT:-$REPO/datasets/bimanual_dreamgen/videos/makovian__close_cardboard_box__ep000026.mp4}
DG=${DG:-$REPO/sampling_dataset/dense/bimanual/output/singleview/dreamgen/makovian/iter_000110000_bimanual16fps/close_cardboard_box__episode_000026.mp4}
DD=${DD:-$REPO/sampling_dataset/dense/bimanual/output/singleview/dreamdojo/makovian/iter_000100000/close_cardboard_box__episode_000026/full_pred.mp4}
OUT=${OUT:-$REPO/metrics/motion_probe_out/close_cardboard_box__ep000026_matched}
OBJECT_PROMPT=${OBJECT_PROMPT:-"cardboard box . box ."}
EMBODIMENT_PROMPT=${EMBODIMENT_PROMPT:-"robot arm . robot gripper ."}

mkdir -p "$OUT"

# --- Build GT variants matched to each model's grid (scale WxH + resample fps) ---
GT_AS_DG="$OUT/gt_as_dreamgen_768x432_16fps.mp4"
GT_AS_DD="$OUT/gt_as_dreamdojo_640x480_10fps.mp4"
echo "[ffmpeg] GT -> DreamGen grid (768x432 @16fps)"
ffmpeg -y -loglevel error -i "$GT" -vf "scale=768:432:flags=bicubic,fps=16" -an "$GT_AS_DG"
echo "[ffmpeg] GT -> DreamDojo grid (640x480 @10fps)"
ffmpeg -y -loglevel error -i "$GT" -vf "scale=640:480:flags=bicubic,fps=10" -an "$GT_AS_DD"

cd "$WS"
export PYTHONPATH="$TP:$TP/sam2:$TP/SEA-RAFT/core:$TP/SEA-RAFT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

run_pair() {  # $1=gt_matched  $2=pred  $3=subout
  "$PY" "$REPO/metrics/dynamics/motion_accuracy_gdino.py" \
    --gt "$1" --pred "$2" --out "$3" \
    --sam2_cfg "configs/sam2.1/sam2.1_hiera_b+.yaml" \
    --sam2_ckpt "$CKPT/sam2.1_hiera_base_plus.pt" \
    --raft_cfg "worldscore/benchmark/metrics/third_party/SEA-RAFT/config/eval/spring-M.json" \
    --raft_ckpt "$CKPT/Tartan-C-T-TSKH-spring540x960-M.pth" \
    --gdino_cfg "worldscore/benchmark/metrics/third_party/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
    --gdino_ckpt "$CKPT/groundingdino_swint_ogc.pth" \
    --object_prompt "$OBJECT_PROMPT" \
    --embodiment_prompt "$EMBODIMENT_PROMPT" \
    ${MAX_FRAMES:+--max_frames "$MAX_FRAMES"}
}

echo "=== PAIR: GT(matched to DreamGen grid) vs DreamGen  [768x432 @16fps] ==="
run_pair "$GT_AS_DG" "$DG" "$OUT/dreamgen"

echo "=== PAIR: GT(matched to DreamDojo grid) vs DreamDojo [640x480 @10fps] ==="
run_pair "$GT_AS_DD" "$DD" "$OUT/dreamdojo"

echo "Done. Outputs in $OUT/{dreamgen,dreamdojo}/summary_gdino.json"
