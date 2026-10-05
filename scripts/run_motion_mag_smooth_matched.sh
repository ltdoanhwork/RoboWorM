#!/usr/bin/env bash
# motion_magnitude + motion_smoothness on the SAME matched-grid pairs used by
# run_motion_accuracy_matched.sh (GT resampled to each model's own res+fps).
# Reuses the resampled GT clips if present, else rebuilds them with ffmpeg.
set -euo pipefail

REPO=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics
PY=$REPO/models/dreamgen/.venv/bin/python
WS=$REPO/metrics/WorldScore
TP=$WS/worldscore/benchmark/metrics/third_party
CKPT=$REPO/checkpoints/metrics/worldscore

GT=${GT:-$REPO/datasets/bimanual_dreamgen/videos/makovian__close_cardboard_box__ep000026.mp4}
DG=${DG:-$REPO/sampling_dataset/dense/bimanual/output/singleview/dreamgen/makovian/iter_000110000_bimanual16fps/close_cardboard_box__episode_000026.mp4}
DD=${DD:-$REPO/sampling_dataset/dense/bimanual/output/singleview/dreamdojo/makovian/iter_000100000/close_cardboard_box__episode_000026/full_pred.mp4}
OUT=${OUT:-$REPO/metrics/motion_probe_out/close_cardboard_box__ep000026_matched}
mkdir -p "$OUT"

GT_AS_DG="$OUT/gt_as_dreamgen_768x432_16fps.mp4"
GT_AS_DD="$OUT/gt_as_dreamdojo_640x480_10fps.mp4"
[ -f "$GT_AS_DG" ] || { echo "[ffmpeg] GT -> DreamGen grid"; ffmpeg -y -loglevel error -i "$GT" -vf "scale=768:432:flags=bicubic,fps=16" -an "$GT_AS_DG"; }
[ -f "$GT_AS_DD" ] || { echo "[ffmpeg] GT -> DreamDojo grid"; ffmpeg -y -loglevel error -i "$GT" -vf "scale=640:480:flags=bicubic,fps=10" -an "$GT_AS_DD"; }

cd "$WS"
export PYTHONPATH="$TP:$TP/SEA-RAFT/core:$TP/SEA-RAFT:$TP/VFIMamba:${PYTHONPATH:-}"

run() {  # $1=gt_matched $2=pred $3=subout $4=label
  "$PY" "$REPO/metrics/dynamics/motion_mag_smooth_probe.py" \
    --gt "$1" --pred "$2" --out "$3" --label "$4" \
    --raft_cfg "worldscore/benchmark/metrics/third_party/SEA-RAFT/config/eval/spring-M.json" \
    --raft_ckpt "$CKPT/Tartan-C-T-TSKH-spring540x960-M.pth" \
    --smooth_stride "${SMOOTH_STRIDE:-8}" \
    ${TTA:+--tta} \
    ${NO_SMOOTHNESS:+--no_smoothness}
}

echo "=== PAIR: GT(matched) vs DreamGen  [768x432 @16fps] ==="
run "$GT_AS_DG" "$DG" "$OUT/dreamgen" "dreamgen_768x432_16fps"

echo "=== PAIR: GT(matched) vs DreamDojo [640x480 @10fps] ==="
run "$GT_AS_DD" "$DD" "$OUT/dreamdojo" "dreamdojo_640x480_10fps"

echo "Done. mag_smooth_summary.json in $OUT/{dreamgen,dreamdojo}/"
