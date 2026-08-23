#!/usr/bin/env bash
# =============================================================================
# bench_itm_single_episode.sh
# Quick ITM quality/speed bench on ONE episode (default: makovian episode_000397).
#
# Builds a single-item batch_input.json from the existing dense input grid PNG,
# runs DreamGen multiview single_arm with ITM, and writes the result next to the
# other itm/ outputs as `_bench_<TAG>.mp4`. Use it to sweep ITM configs without
# retyping the long python command.
#
# All ITM knobs are env-overridable. Defaults below are the "mid" config that
# trades a little speed for clean output (start late, light compress, no
# prune-only phase). Compare against the dense baseline of the same episode.
#
# Usage (repo root):
#   bash scripts/bench_itm_single_episode.sh
#   GPU=1 TAG=mid bash scripts/bench_itm_single_episode.sh
#   TAG=cons START_LAYER=24 KEEP_LAST=4 COMPRESS=0.15 bash scripts/bench_itm_single_episode.sh
#   TAG=aggr START_LAYER=6 KEEP_LAST=2 COMPRESS=0.35 PRUNE_FROM=1 MERGE_FROM=2 bash scripts/bench_itm_single_episode.sh
#
# After it finishes it prints the output path; use the COMPARE block at the
# bottom (auto-runs if MAKE_COMPARE=1) to dump side-by-side frames vs dense.
#
# Environment:
#   EPISODE      default 000397
#   CATEGORY     default makovian
#   GPU          default 0          (CUDA_VISIBLE_DEVICES)
#   TAG          default mid         (output suffix _bench_<TAG>.mp4)
#   MODEL_SIZE   default 14B
#   FPS          default 10
#   START_LAYER  default 14         (--itm_start_layer_idx)
#   KEEP_LAST    default 4          (--itm_keep_last_n_dense)
#   COMPRESS     default 0.2        (--itm_compress_ratio; empty = auto schedule)
#   PRUNE_FROM   default 0          (--itm_prune_from_step)
#   MERGE_FROM   default 0          (--itm_merge_from_step)
#   MAKE_COMPARE default 1          (1 = extract side-by-side frames vs dense)
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

EPISODE="${EPISODE:-000397}"
CATEGORY="${CATEGORY:-makovian}"
GPU="${GPU:-0}"
TAG="${TAG:-mid}"
MODEL_SIZE="${MODEL_SIZE:-14B}"
FPS="${FPS:-10}"
START_LAYER="${START_LAYER:-14}"
KEEP_LAST="${KEEP_LAST:-4}"
COMPRESS="${COMPRESS:-0.2}"
PRUNE_FROM="${PRUNE_FROM:-0}"
MERGE_FROM="${MERGE_FROM:-0}"
MAKE_COMPARE="${MAKE_COMPARE:-1}"

DENSE_DIR="./sampling_dataset/dense/single_arm/input/multiview/dreamgen/${CATEGORY}"
ITM_OUT_DIR="./sampling_dataset/itm/single_arm/output/multiview/dreamgen/${CATEGORY}"
GRID_PNG="${DENSE_DIR}/episode_${EPISODE}.png"
DENSE_MP4="./sampling_dataset/dense/single_arm/output/multiview/dreamgen/${CATEGORY}/episode_${EPISODE}.mp4"
PROMPT_TXT="${ITM_OUT_DIR}/episode_${EPISODE}.txt"
OUT_MP4="${ITM_OUT_DIR}/_bench_${TAG}.mp4"
BATCH_JSON="/tmp/bench_itm_${TAG}.json"

if [[ ! -f "${GRID_PNG}" ]]; then
  echo "[ERROR] Missing input grid PNG: ${GRID_PNG}" >&2
  echo "        (run the dense prepare for this episode first)" >&2
  exit 1
fi
if [[ ! -f "${PROMPT_TXT}" ]]; then
  echo "[ERROR] Missing prompt sidecar: ${PROMPT_TXT}" >&2
  echo "        (run dense or a prior itm pass to produce the .txt prompt)" >&2
  exit 1
fi

mkdir -p "${ITM_OUT_DIR}"

VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}:${ROOT}/models/dreamgen:${PYTHONPATH:-}"

python - "$GRID_PNG" "$PROMPT_TXT" "$OUT_MP4" "$BATCH_JSON" <<'PY'
import json, sys
grid, prompt_txt, out_mp4, batch_json = sys.argv[1:5]
prompt = open(prompt_txt).read().strip()
json.dump([{"input_video": grid, "prompt": prompt, "output_video": out_mp4}],
          open(batch_json, "w"), indent=2)
print(f"[batch] {batch_json} -> {out_mp4}")
PY

echo "============================================================"
echo " ITM bench — ${CATEGORY}/episode_${EPISODE}  TAG=${TAG}  GPU=${GPU}"
echo " start_layer=${START_LAYER} keep_last=${KEEP_LAST} compress=${COMPRESS:-auto} prune_from=${PRUNE_FROM} merge_from=${MERGE_FROM}"
echo " out: ${OUT_MP4}"
echo "============================================================"

ITM_FLAGS=(
  --use_itm
  --itm_start_layer_idx   "${START_LAYER}"
  --itm_keep_last_n_dense "${KEEP_LAST}"
  --itm_prune_from_step   "${PRUNE_FROM}"
  --itm_merge_from_step   "${MERGE_FROM}"
)
if [[ -n "${COMPRESS}" ]]; then
  ITM_FLAGS+=(--itm_compress_ratio "${COMPRESS}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" python -m examples.video2world_gr00t \
  --model_size       "${MODEL_SIZE}" \
  --gr00t_variant    droid \
  --batch_input_json "${BATCH_JSON}" \
  --prompt_prefix    "" \
  --num_gpus         1 \
  --disable_guardrail \
  --fps              "${FPS}" \
  "${ITM_FLAGS[@]}"

echo ""
echo "[Done] ${OUT_MP4}"

if [[ "${MAKE_COMPARE}" == "1" && -f "${OUT_MP4}" && -f "${DENSE_MP4}" ]]; then
  CMP_DIR="/tmp/qcmp"
  mkdir -p "${CMP_DIR}"
  for fidx in 13 46; do
    ffmpeg -y -loglevel error -i "${DENSE_MP4}" -vf "select=eq(n\,${fidx})" -vframes 1 "${CMP_DIR}/${TAG}_${fidx}_dense.png"
    ffmpeg -y -loglevel error -i "${OUT_MP4}"   -vf "select=eq(n\,${fidx})" -vframes 1 "${CMP_DIR}/${TAG}_${fidx}_itm.png"
    ffmpeg -y -loglevel error -i "${CMP_DIR}/${TAG}_${fidx}_dense.png" -i "${CMP_DIR}/${TAG}_${fidx}_itm.png" \
      -filter_complex "[0:v]drawtext=text='DENSE':x=10:y=10:fontsize=28:fontcolor=yellow[a];[1:v]drawtext=text='ITM-${TAG}':x=10:y=10:fontsize=28:fontcolor=yellow[b];[a][b]hstack" \
      "${CMP_DIR}/cmp_${TAG}_${fidx}.png"
  done
  echo "[Compare] ${CMP_DIR}/cmp_${TAG}_13.png  ${CMP_DIR}/cmp_${TAG}_46.png"
fi
