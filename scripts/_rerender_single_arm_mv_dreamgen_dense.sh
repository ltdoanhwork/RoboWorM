#!/usr/bin/env bash
# =============================================================================
# _rerender_single_arm_mv_dreamgen_dense.sh
#
# Re-render the dense single_arm/multiview/dreamgen cell with the SAME rollout
# config as the 9 acceleration trees.
#
# WHY: the old dense tree (2026-05-15) was produced through the LVG branch
# (USE_LVG=1, 3 chunks) whose writer hardcodes fps=16 and whose chunk chaining
# yields 277 frames, while every accel script mirrors the single-shot
# `examples.video2world_gr00t` flow at --fps 10 (93 frames). That made all
# 216 accel-vs-dense pairs in this cell incomparable (see
# comparable_pair() in assets/scripts/table_dense_nr.py). Archived copy:
#   archive/dense_single_arm_mv_dreamgen_LVG16fps_pre20260917/
#
# The unified target is the ACCEL config (10 fps, 93 frames, single shot),
# because 9/9 accel trees already agree on it and there is no GT clip in this
# cell to anchor to a different fps.
#
# Verified on episode_000255 before launching the full cell:
#   this flow  -> 768x432, 10/1, 93 frames   (identical to pisa/dicache/...)
#   old dense  -> 768x432, 16/1, 277 frames
#
# Env:
#   CATEGORIES  default "makovian non_makovian"
#   GPU         default 0
#   FORCE       set 1 to re-render episodes whose mp4+txt already exist
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CATEGORIES="${CATEGORIES:-makovian non_makovian}"
GPU="${GPU:-0}"
FORCE="${FORCE:-0}"

# Never let a stale shard env silently drop most episodes.
unset NUM_SHARDS SHARD_ID || true

OUT_ROOT="$ROOT/sampling_dataset/dense/single_arm/output/multiview/dreamgen"
IN_ROOT="$ROOT/sampling_dataset/dense/single_arm/input/multiview/dreamgen"

echo "============================================================"
echo " Re-render dense single_arm/multiview/dreamgen (single-shot 10fps)"
echo " Categories : ${CATEGORIES}"
echo " GPU        : ${GPU}"
echo " Force      : ${FORCE}"
echo " Output     : ${OUT_ROOT}/<category>/"
echo "============================================================"

for CAT in ${CATEGORIES}; do
  echo ""
  echo "===== ${CAT} ====="
  # OUTPUT_DIR is used verbatim by the prepare script, so it must already
  # include the category (the batch script does NOT append it).
  CUDA_VISIBLE_DEVICES="${GPU}" \
  DATASET_PATH="./datasets/single_arm/multiview/${CAT}" \
  CATEGORY="${CAT}" \
  INPUT_DIR="${IN_ROOT}" \
  OUTPUT_DIR="${OUT_ROOT}/${CAT}" \
  NUM_GPUS=1 \
  FPS=10 \
  USE_LVG=0 \
  FORCE_REGENERATE_OUTPUTS="${FORCE}" \
  bash scripts/run_single_arm_multiview_dreamgen_batch.sh

  n=$(find "${OUT_ROOT}/${CAT}" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)
  echo "  -> ${CAT}: ${n} mp4 in tree"
done

echo ""
echo "============================================================"
echo " Done. Verifying render config (expect 10/1 and 93 frames):"
for CAT in ${CATEGORIES}; do
  f=$(find "${OUT_ROOT}/${CAT}" -maxdepth 1 -name '*.mp4' 2>/dev/null | head -1)
  [ -n "$f" ] || continue
  printf '  %-14s %s\n' "${CAT}" \
    "$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=width,height,r_frame_rate,nb_frames \
        -of csv=p=0 "$f")"
done
echo "============================================================"
