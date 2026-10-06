#!/usr/bin/env bash
# WorldScore Quality + Controllability probe on the paired robot sample:
#   subjective_quality (clip_iqa+ , clip_aesthetic) | content_alignment (clip_score)
#   | object_control (GroundingDINO detection rate)
#
# Same env recipe as run_motion_accuracy_gdino.sh (cwd = metrics/WorldScore so the
# vendored groundingdino + relative config paths resolve).
set -euo pipefail

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
PY=$REPO/models/dreamgen/.venv/bin/python
WS=$REPO/metrics/WorldScore
TP=$WS/worldscore/benchmark/metrics/third_party
CKPT=$REPO/checkpoints/metrics/worldscore

GT=${GT:-$REPO/datasets/bimanual_dreamgen/videos/makovian__close_cardboard_box__ep000026.mp4}
DG=${DG:-$REPO/sampling_dataset/dense/bimanual/output/singleview/dreamgen/makovian/iter_000110000_bimanual16fps/close_cardboard_box__episode_000026.mp4}
DD=${DD:-$REPO/sampling_dataset/dense/bimanual/output/singleview/dreamdojo/makovian/iter_000100000/close_cardboard_box__episode_000026/full_pred.mp4}
OUT=${OUT:-$REPO/metrics/motion_probe_out/close_cardboard_box__ep000026_quality}

# Task prompt (from the dreamgen sidecar .txt) and the objects we expect to see.
CAPTION=${CAPTION:-"A bimanual robot performing the task: Close cardboard box."}
OBJECTS=${OBJECTS:-"robot arm, cardboard box"}
STRIDE=${STRIDE:-10}

mkdir -p "$OUT"
cd "$WS"
export PYTHONPATH="$TP:$TP/groundingdino:$TP/sam2:${PYTHONPATH:-}"

"$PY" "$REPO/metrics/quality/quality_control_probe.py" \
  --clips "gt=$GT" "dreamgen=$DG" "dreamdojo=$DD" \
  --out "$OUT" \
  --caption "$CAPTION" \
  --objects "$OBJECTS" \
  --stride "$STRIDE" \
  --gdino_cfg "worldscore/benchmark/metrics/third_party/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
  --gdino_ckpt "$CKPT/groundingdino_swint_ogc.pth" \
  "$@"

echo "Done -> $OUT/quality_control_summary.json"
