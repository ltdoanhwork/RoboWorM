#!/usr/bin/env bash
# =============================================================================
# run_humanoid_singleview_dreamgen_sito.sh
# DreamGen (video2world_gr00t_lvg, gr1 variant) humanoid SINGLEVIEW with SiTo
# self-attention token pruning.
#
# Mirrors the dense humanoid singleview pipeline (run_humanoid_singleview_dreamgen_static16fps.sh
# / the humanoid cache scripts): SAME checkpoint (humanoid_singleview_16fps),
# SAME resolution / fps / guidance / aspect / cond frames, SAME shared dense
# batch inputs. Only differences:
#   1. forces NUM_GPUS=1 (SiTo has no context parallelism -> single-GPU only)
#   2. adds the SiTo flags
#   3. rewrites each episode's output_video from the shared dense batch json into
#      sampling_dataset/sito/ (per sampling-dataset-layout rule)
#
# Output tag mirrors the dense baseline sub-path (dense uses ${ITER_NAME}_static16fps)
# so SiTo outputs sit at the same relative path as dense for trivial comparison:
#   sampling_dataset/sito/humanoid/output/singleview/dreamgen/{split}/{iter}_static16fps/
#
# Usage (repo root):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_singleview_dreamgen_sito.sh
#   CUDA_VISIBLE_DEVICES=0 SPLITS="makovian" NUM_CHUNKS="2 3" bash scripts/run_humanoid_singleview_dreamgen_sito.sh
#   SITO_PRUNE_RATIO=0.3 SITO_START_LAYER_IDX=6 bash scripts/run_humanoid_singleview_dreamgen_sito.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- Environment (DreamGen) ---
export VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/models/dreamgen:${ROOT}:${PYTHONPATH:-}"

# --- Checkpoint (posttrained humanoid_singleview_16fps; matches dense baseline) ---
CKPT_BASE="${CKPT_BASE:-${ROOT}/exp_logs/dreamgen_logs/posttraining/singleview/humanoid_singleview_16fps}"
ITER_NAME="${ITER_NAME:-iter_000090000}"
DIT_PATH="${DIT_PATH:-${CKPT_BASE}/checkpoints_40_per/model/${ITER_NAME}.pt}"
# Output tag mirrors the dense baseline sub-path (dense uses ${ITER_NAME}_static16fps).
OUT_ITER="${OUT_ITER:-${ITER_NAME}_static16fps}"

if [[ ! -f "${DIT_PATH}" ]]; then
  echo "ERROR: Checkpoint not found at ${DIT_PATH}" >&2
  ls "${CKPT_BASE}/checkpoints_40_per/model/"*.pt 2>/dev/null >&2 || true
  exit 1
fi

# --- Inference settings (identical to dense baseline) ---
# SiTo is single-GPU only (no context parallelism).
NUM_GPUS=1
MODEL_SIZE="${MODEL_SIZE:-2B}"
RESOLUTION="${RESOLUTION:-480}"
FPS="${FPS:-16}"
GUIDANCE="${GUIDANCE:-7}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"
NUM_CONDITIONAL_FRAMES="${NUM_CONDITIONAL_FRAMES:-1}"
PROMPT_PREFIX="${PROMPT_PREFIX:-}"

# --- SiTo params (seeded from the single_arm singleview optimal: temporal group,
#     start_layer 6, keep_last 2, prune_ratio 0.30 = quality/speed knee) ---
SITO_START_LAYER_IDX="${SITO_START_LAYER_IDX:-6}"
SITO_KEEP_LAST_N_DENSE="${SITO_KEEP_LAST_N_DENSE:-2}"
SITO_PRUNE_RATIO="${SITO_PRUNE_RATIO:-0.30}"
SITO_GROUP_MODE="${SITO_GROUP_MODE:-temporal}"
SITO_NOISE_ALPHA="${SITO_NOISE_ALPHA:-0}"

# --- Dataset / output ---
# Inputs are the SHARED dense batch inputs; only output_video is rewritten to sito/.
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/dense/humanoid/input/singleview/dreamgen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/sito/humanoid/output/singleview/dreamgen}"
SPLITS="${SPLITS:-makovian non_makovian}"
NUM_CHUNKS="${NUM_CHUNKS:-2 3}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
MAX_EPISODES="${MAX_EPISODES:-}"

echo "============================================"
echo "DreamGen Humanoid Singleview — SiTo (single-GPU)"
echo "  Checkpoint: ${DIT_PATH}"
echo "  Iter tag:   ${OUT_ITER}"
echo "  Model:      ${MODEL_SIZE} | Resolution: ${RESOLUTION} | FPS: ${FPS}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Guidance:   ${GUIDANCE} | Cond frames: ${NUM_CONDITIONAL_FRAMES} | Aspect: ${ASPECT_RATIO}"
echo "  SiTo:       group=${SITO_GROUP_MODE} start_layer=${SITO_START_LAYER_IDX} keep_last=${SITO_KEEP_LAST_N_DENSE} prune=${SITO_PRUNE_RATIO} noise=${SITO_NOISE_ALPHA}"
echo "  Splits:     ${SPLITS} | Chunks: ${NUM_CHUNKS} | MaxEps: ${MAX_EPISODES:-all}"
echo "  Output:     ${OUTPUT_ROOT}/{split}/${OUT_ITER}/"
echo "============================================"

run_inference() {
  local src_batch_json="$1"
  local label="$2"
  local num_chunks="$3"
  local split="$4"

  if [[ ! -f "${src_batch_json}" ]]; then
    echo "  [SKIP] ${label}: batch file not found"
    return
  fi

  # Rewrite output paths dense/ -> sito/ (mirror the dense OUT_ITER sub-path) and
  # filter to pending (missing output), optionally capping at MAX_EPISODES.
  local sito_batch_json
  sito_batch_json="$(mktemp --suffix=_dreamgen_sito.json)"
  local pending
  pending="$(python3 - "${src_batch_json}" "${sito_batch_json}" "${OUTPUT_ROOT}/${split}/${OUT_ITER}" "${FORCE_REGENERATE}" "${MAX_EPISODES}" <<'PY'
import json, sys
from pathlib import Path

src, dst_json, out_dir, force, max_eps = sys.argv[1:6]
force = force == "1"
max_eps = int(max_eps) if max_eps else None
items = json.loads(Path(src).read_text())
for item in items:
    episode_name = Path(item["output_video"]).name
    item["output_video"] = f"{out_dir}/{episode_name}"
if force:
    pending = list(items)
else:
    pending = [x for x in items if not Path(x["output_video"]).is_file()]
if max_eps is not None:
    pending = pending[:max_eps]
Path(dst_json).write_text(json.dumps(pending, indent=2))
print(len(pending))
PY
)"

  if [[ "${pending}" -eq 0 ]]; then
    echo "  [SKIP] ${label}: all episodes already generated"
    rm -f "${sito_batch_json}"
    return
  fi

  mkdir -p "${OUTPUT_ROOT}/${split}/${OUT_ITER}"
  echo "  === Running: ${label} (${pending} pending, chunks=${num_chunks}) ==="

  SITO_FLAGS=(
    --use_sito
    --sito_start_layer_idx   "${SITO_START_LAYER_IDX}"
    --sito_keep_last_n_dense "${SITO_KEEP_LAST_N_DENSE}"
    --sito_group_mode        "${SITO_GROUP_MODE}"
  )
  if [[ -n "${SITO_PRUNE_RATIO}" ]]; then
    SITO_FLAGS+=(--sito_prune_ratio "${SITO_PRUNE_RATIO}")
  fi
  if [[ -n "${SITO_NOISE_ALPHA}" ]]; then
    SITO_FLAGS+=(--sito_noise_alpha "${SITO_NOISE_ALPHA}")
  fi

  python -m models.dreamgen.examples.video2world_gr00t_lvg \
    --model_size "${MODEL_SIZE}" \
    --gr00t_variant gr1 \
    --dit_path "${DIT_PATH}" \
    --batch_input_json "${sito_batch_json}" \
    --num_chunks "${num_chunks}" \
    --guidance "${GUIDANCE}" \
    --num_gpus "${NUM_GPUS}" \
    --aspect_ratio "${ASPECT_RATIO}" \
    --num_conditional_frames "${NUM_CONDITIONAL_FRAMES}" \
    --prompt_prefix "${PROMPT_PREFIX}" \
    --disable_guardrail \
    --no_skip_existing \
    "${SITO_FLAGS[@]}"

  rm -f "${sito_batch_json}"
  echo "  === Done: ${label} ==="
}

for split in ${SPLITS}; do
  input_dir="${INPUT_ROOT}/${split}"
  if [[ ! -d "${input_dir}" ]]; then
    echo "  [SKIP] ${split}: input dir not found at ${input_dir}"
    continue
  fi

  for nc in ${NUM_CHUNKS}; do
    batch_json="${input_dir}/batch_input_short_${nc}chunks.json"
    run_inference "${batch_json}" "humanoid/${split}/short_${nc}chunks" "${nc}" "${split}"
  done
done

echo ""
echo "============================================"
echo "Done! SiTo results: ${OUTPUT_ROOT}/{split}/${OUT_ITER}/"
echo "============================================"
