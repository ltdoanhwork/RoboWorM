#!/usr/bin/env bash
# =============================================================================
# Run DreamGen (video2world_lvg) singleview DENSE inference on all 3 embodiments
# using the all_singleview post-training checkpoint.
#
# Checkpoint dir:
#   exp_logs/dreamgen_logs/posttraining/singleview/all_singleview/
#   By default uses checkpoints_40_per/model/iter_000090000.pt (latest)
#
# Training config: 2B model, 480p (432×768), per-embodiment fps, 93 frames
#
# Uses LVG (Long Video Generation) to chain multiple chunks autoregressively:
#   - Each chunk generates 93 frames; subsequent chunks use last predicted frame
#   - No GT anchoring: only the first frame is from ground truth
#   - FPS matches training data: bimanual=30, single_arm=15, humanoid=20
#   - num_chunks per embodiment sized to match full episode at native fps
#
# Datasets:
#   datasets/bimanual/singleview/{makovian,non_makovian}
#   datasets/humanoid/singleview/{makovian,non_makovian}
#   datasets/single_arm/singleview/{makovian,non_makovian}
#
# Output layout:
#   sampling_dataset/dense/{embodiment}/output/singleview/dreamgen/{split}/{iter_name}/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_all_singleview_dreamgen_dense.sh
#   MAX_EPISODES=5 bash scripts/run_all_singleview_dreamgen_dense.sh
#   EMBODIMENTS="bimanual" bash scripts/run_all_singleview_dreamgen_dense.sh
#   ITER_NAME="iter_000080000" bash scripts/run_all_singleview_dreamgen_dense.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- Environment ---
export VENV="${ROOT}/models/DreamDojo/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.10/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.10/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/models/dreamgen:${PYTHONPATH:-}"

# --- Checkpoint ---
CKPT_BASE="${ROOT}/exp_logs/dreamgen_logs/posttraining/singleview/all_singleview"
ITER_NAME="${ITER_NAME:-iter_000090000}"
DIT_PATH="${DIT_PATH:-${CKPT_BASE}/checkpoints_40_per/model/${ITER_NAME}.pt}"

# Fallback: if the specified iter doesn't exist, try backup_60k_70k
if [[ ! -f "${DIT_PATH}" ]]; then
  ALT_PATH="${CKPT_BASE}/checkpoints_40_per/backup_60k_70k/model/${ITER_NAME}.pt"
  if [[ -f "${ALT_PATH}" ]]; then
    DIT_PATH="${ALT_PATH}"
  else
    echo "ERROR: Checkpoint not found at ${DIT_PATH}" >&2
    echo "Available checkpoints:" >&2
    ls "${CKPT_BASE}/checkpoints_40_per/model/"*.pt 2>/dev/null >&2 || true
    exit 1
  fi
fi

# --- Inference settings (match training) ---
NUM_GPUS="${NUM_GPUS:-2}"
MODEL_SIZE="${MODEL_SIZE:-2B}"
RESOLUTION="${RESOLUTION:-480}"
GUIDANCE="${GUIDANCE:-7}"
MASTER_PORT="${MASTER_PORT:-29503}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"

# Per-embodiment FPS matching training data (no resampling during training)
declare -A EMBODIMENT_FPS=(
  [bimanual]=30
  [single_arm]=15
  [humanoid]=20
)

# --- Dataset / output ---
DATASETS_ROOT="${DATASETS_ROOT:-${ROOT}/datasets}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/dense}"
EMBODIMENTS="${EMBODIMENTS:-bimanual humanoid single_arm}"
SPLITS="${SPLITS:-makovian non_makovian}"
MAX_EPISODES="${MAX_EPISODES:-}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"

echo "============================================"
echo "DreamGen Singleview Dense — All Embodiments"
echo "  Checkpoint: ${DIT_PATH}"
echo "  Iter name:  ${ITER_NAME}"
echo "  Model:      ${MODEL_SIZE} | Resolution: ${RESOLUTION}"
echo "  FPS:        per-embodiment (bimanual=30, single_arm=15, humanoid=20)"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Guidance:   ${GUIDANCE}"
echo "  Embodiments: ${EMBODIMENTS}"
echo "  Splits:     ${SPLITS}"
echo "  Chunks:     per-episode (computed by prepare script)"
echo "  Save root:  ${SAVE_ROOT}"
echo "  Episodes:   ${MAX_EPISODES:-all}"
echo "============================================"

# --- Step 1: Prepare batch_input.json for all datasets ---
if [[ "${SKIP_PREPARE}" != "1" ]]; then
  echo ""
  echo "[STEP 1] Preparing batch_input.json for all embodiments..."
  PREPARE_ARGS=(
    --datasets-root "${DATASETS_ROOT}"
    --save-root "${SAVE_ROOT}"
    --embodiments ${EMBODIMENTS}
    --splits ${SPLITS}
    --iter-name "${ITER_NAME}"
    --skip-existing
  )
  if [[ -n "${MAX_EPISODES}" ]]; then
    PREPARE_ARGS+=(--max-episodes "${MAX_EPISODES}")
  fi
  python "${ROOT}/scripts/prepare_singleview_dreamgen_batch.py" "${PREPARE_ARGS[@]}"
fi

# --- Step 2: Run inference on each batch_input_*chunks.json ---
echo ""
echo "[STEP 2] Running DreamGen LVG inference..."

run_inference() {
  local batch_json="$1"
  local label="$2"
  local num_chunks="$3"
  local fps="$4"

  if [[ ! -f "${batch_json}" ]]; then
    echo "  [SKIP] ${label}: not found"
    return
  fi

  local item_count
  item_count="$(python3 -c "import json; print(len(json.load(open('${batch_json}'))))")"
  if [[ "${item_count}" -eq 0 ]]; then
    echo "  [SKIP] ${label}: empty batch"
    return
  fi

  # Filter already-done episodes unless FORCE_REGENERATE
  local batch_for_infer="${batch_json}"
  if [[ "${FORCE_REGENERATE}" != "1" ]]; then
    local batch_tmp
    batch_tmp="$(mktemp --suffix=_dreamgen_sv.json)"
    local pending
    pending="$(python3 - "${batch_json}" "${batch_tmp}" <<'PY'
import json, sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
items = json.loads(src.read_text(encoding="utf-8"))
pending = [x for x in items if not Path(x["output_video"]).is_file()]
dst.write_text(json.dumps(pending, indent=2, ensure_ascii=False), encoding="utf-8")
print(len(pending))
PY
    )"
    if [[ "${pending}" -eq 0 ]]; then
      echo "  [SKIP] ${label}: all ${item_count} episodes already generated"
      rm -f "${batch_tmp}"
      return
    fi
    echo "  [INFO] ${label}: ${pending}/${item_count} episodes pending (num_chunks=${num_chunks})"
    batch_for_infer="${batch_tmp}"
  else
    echo "  [INFO] ${label}: ${item_count} episodes (num_chunks=${num_chunks})"
  fi

  echo "  === Running: ${label} (fps=${fps}) ==="
  if [[ "${NUM_GPUS}" -gt 1 ]]; then
    torchrun \
      --nproc_per_node="${NUM_GPUS}" \
      --master_port="${MASTER_PORT}" \
      -m models.dreamgen.examples.video2world_lvg \
      --model_size "${MODEL_SIZE}" \
      --resolution "${RESOLUTION}" \
      --fps "${fps}" \
      --dit_path "${DIT_PATH}" \
      --batch_input_json "${batch_for_infer}" \
      --guidance "${GUIDANCE}" \
      --num_gpus "${NUM_GPUS}" \
      --num_chunks "${num_chunks}" \
      --num_conditional_frames 1 \
      --aspect_ratio "${ASPECT_RATIO}" \
      --disable_guardrail \
      --disable_prompt_refiner
  else
    python -m models.dreamgen.examples.video2world_lvg \
      --model_size "${MODEL_SIZE}" \
      --resolution "${RESOLUTION}" \
      --fps "${fps}" \
      --dit_path "${DIT_PATH}" \
      --batch_input_json "${batch_for_infer}" \
      --guidance "${GUIDANCE}" \
      --num_gpus "${NUM_GPUS}" \
      --num_chunks "${num_chunks}" \
      --num_conditional_frames 1 \
      --aspect_ratio "${ASPECT_RATIO}" \
      --disable_guardrail \
      --disable_prompt_refiner
  fi

  MASTER_PORT=$((MASTER_PORT + 1))

  if [[ "${batch_for_infer}" != "${batch_json}" ]]; then
    rm -f "${batch_for_infer}"
  fi
  echo "  === Done: ${label} ==="
}

for embodiment in ${EMBODIMENTS}; do
  fps="${EMBODIMENT_FPS[${embodiment}]:-16}"
  echo ""
  echo ">>> ${embodiment} (fps=${fps})"
  for split in ${SPLITS}; do
    input_dir="${SAVE_ROOT}/${embodiment}/input/singleview/dreamgen/${split}"
    if [[ -d "${input_dir}" ]]; then
      for batch_json in $(ls "${input_dir}"/batch_input_*chunks.json 2>/dev/null | sort -t_ -k3 -n); do
        fname="$(basename "${batch_json}")"
        num_chunks="${fname#batch_input_}"
        num_chunks="${num_chunks%chunks.json}"
        run_inference "${batch_json}" "${embodiment}/${split} (${num_chunks} chunks)" "${num_chunks}" "${fps}"
      done
    else
      echo "  [SKIP] ${embodiment}/${split}: input dir not found"
    fi
  done
done

echo ""
echo "All done. Results saved under: ${SAVE_ROOT}/{embodiment}/output/singleview/dreamgen/{split}/${ITER_NAME}/"
