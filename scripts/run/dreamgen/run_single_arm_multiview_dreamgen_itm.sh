#!/usr/bin/env bash
# =============================================================================
# run_single_arm_multiview_dreamgen_itm.sh
# DreamGen multiview (single_arm, droid layout) with ITM token-merge acceleration.
#
# Mirrors the dense run (run_single_arm_multiview_dreamgen_batch.sh) but routes
# outputs to the itm/ method tree and enables Importance Token Merge.
#
# ITM is SINGLE-GPU only (no context parallelism), so this script forces
# NUM_GPUS=1 and calls `python -m examples.video2world_gr00t` directly.
#
# ITM needs classifier-free guidance — the gr00t pipeline already runs with
# guidance > 0, so no extra flag is required here.
#
# Runs both categories (makovian, non_makovian). Outputs (method = top-level dir):
#   sampling_dataset/itm/single_arm/output/multiview/dreamgen/<cat>/
#
# Usage (repo root):
#   bash scripts/run/dreamgen/run_single_arm_multiview_dreamgen_itm.sh
#   CUDA_VISIBLE_DEVICES=1 MAX_EPISODES=2 bash scripts/run/dreamgen/run_single_arm_multiview_dreamgen_itm.sh
#   CATEGORY=makovian bash scripts/run/dreamgen/run_single_arm_multiview_dreamgen_itm.sh
#
# Environment:
#   DATASET_ROOT        default ./datasets/single_arm/multiview
#   INPUT_DIR           default ./sampling_dataset/dense/single_arm/input/multiview/dreamgen
#   DENSE_OUTPUT_ROOT   default ./sampling_dataset/dense/single_arm/output/multiview/dreamgen
#   OUTPUT_ROOT         default ./sampling_dataset/itm/single_arm/output/multiview/dreamgen
#   CATEGORY            empty = both; or makovian | non_makovian only
#   MATCH_DENSE         default 1 (build batch from dense episode set; 0 = use prepare discovery)
#   MODEL_SIZE, FPS, MAX_EPISODES, MIN_EPISODE_INDEX, EPISODE_SOURCE, FORCE_REGENERATE_OUTPUTS=1
#   ITM_* overrides (see defaults below; "mid" quality-preserving config for 14B)
# =============================================================================
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "${ROOT}"

VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}:${ROOT}/models/dreamgen:${PYTHONPATH:-}"

DATASET_ROOT="${DATASET_ROOT:-./datasets/single_arm/multiview}"
INPUT_DIR="${INPUT_DIR:-./sampling_dataset/dense/single_arm/input/multiview/dreamgen}"
DENSE_OUTPUT_ROOT="${DENSE_OUTPUT_ROOT:-./sampling_dataset/dense/single_arm/output/multiview/dreamgen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./sampling_dataset/itm/single_arm/output/multiview/dreamgen}"
CATEGORY_FILTER="${CATEGORY:-}"
MODEL_SIZE="${MODEL_SIZE:-14B}"
FPS="${FPS:-10}"
FORCE_REGENERATE_OUTPUTS="${FORCE_REGENERATE_OUTPUTS:-0}"
MAX_EPISODES="${MAX_EPISODES:-}"
MIN_EPISODE_INDEX="${MIN_EPISODE_INDEX:-}"
EPISODE_SOURCE="${EPISODE_SOURCE:-}"
# Restrict ITM to exactly the episode set that dense already benchmarked
# (dense/.../output has ~128 episodes, while the dataset has ~95k). With
# MATCH_DENSE=1 we build the batch straight from the dense grid PNGs + prompt
# sidecars instead of letting prepare discover all dataset episodes.
MATCH_DENSE="${MATCH_DENSE:-1}"

# ITM is single-GPU only.
NUM_GPUS=1

# ITM defaults — "mid" config (quality-preserving) for the 14B 36-block model.
# The README defaults (start_layer 6, keep_last 2, auto compress 0.35, prune-only
# phase) corrupt multiview single_arm output into block-noise: merging too many
# tokens too early destroys spatial structure. Verified on episode_000397:
#   start=24/keep=4/compress=0.15  -> clean (≈ dense)
#   start=6 /keep=2/compress=0.35  -> corrupt
# This "mid" point (start=14, keep=4, compress=0.2, no prune-only phase) keeps
# quality close to dense while still skipping dense self-attn in the late blocks.
ITM_START_LAYER_IDX="${ITM_START_LAYER_IDX:-14}"
ITM_KEEP_LAST_N_DENSE="${ITM_KEEP_LAST_N_DENSE:-4}"
ITM_PRUNE_FROM_STEP="${ITM_PRUNE_FROM_STEP:-0}"
ITM_MERGE_FROM_STEP="${ITM_MERGE_FROM_STEP:-0}"
ITM_COMPRESS_RATIO="${ITM_COMPRESS_RATIO:-0.2}"

CATEGORIES=()
if [[ -n "${CATEGORY_FILTER}" ]]; then
  CATEGORIES+=("${CATEGORY_FILTER}")
else
  CATEGORIES=(makovian non_makovian)
fi

echo "============================================================"
echo " DreamGen multiview — single_arm — ITM (single-GPU)"
echo "============================================================"
echo " Data root  : ${DATASET_ROOT}"
echo " Categories : ${CATEGORIES[*]}"
echo " Output root: ${OUTPUT_ROOT}"
echo " Model      : ${MODEL_SIZE}  GPUs: ${NUM_GPUS}  FPS: ${FPS}"
echo " Match dense: ${MATCH_DENSE} (1 = same episode set as dense benchmark)"
echo " ITM        : start_layer=${ITM_START_LAYER_IDX} keep_last=${ITM_KEEP_LAST_N_DENSE} prune_from=${ITM_PRUNE_FROM_STEP} merge_from=${ITM_MERGE_FROM_STEP} compress=${ITM_COMPRESS_RATIO:-auto}"
echo "============================================================"
echo ""

AVAILABLE_GPUS=$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)
if [[ "${AVAILABLE_GPUS}" -lt 1 ]]; then
  echo "[ERROR] No CUDA GPUs visible to PyTorch." >&2
  exit 1
fi

for CAT in "${CATEGORIES[@]}"; do
  DS_PATH="${DATASET_ROOT}/${CAT}"
  if [[ ! -d "${DS_PATH}" ]]; then
    echo "[WARN] Skip missing: ${DS_PATH}"
    continue
  fi
  if [[ ! -f "${DS_PATH}/meta/episodes.jsonl" ]]; then
    echo "[WARN] No meta/episodes.jsonl in ${DS_PATH}, skip"
    continue
  fi

  BATCH_JSON="${INPUT_DIR}/${CAT}/batch_input.json"
  OUTPUT_DIR="${OUTPUT_ROOT}/${CAT}"

  echo "========== ${CAT} =========="

  if [[ "${MATCH_DENSE}" == "1" ]]; then
    # Build the ITM batch from the dense episode set (grid PNG + prompt sidecar),
    # skipping episodes whose ITM output already exists. Keeps ITM on the exact
    # same episodes that dense benchmarked.
    DENSE_OUT_CAT="${DENSE_OUTPUT_ROOT}/${CAT}"
    GRID_DIR="${INPUT_DIR}/${CAT}"
    python - "$GRID_DIR" "$DENSE_OUT_CAT" "$OUTPUT_DIR" "$BATCH_JSON" "${FORCE_REGENERATE_OUTPUTS}" "${MAX_EPISODES}" <<'PY'
import json, sys
from pathlib import Path

grid_dir, dense_out, itm_out, batch_json, force, max_eps = sys.argv[1:7]
grid_dir, dense_out, itm_out, batch_json = map(Path, (grid_dir, dense_out, itm_out, batch_json))
force = force == "1"
max_eps = int(max_eps) if max_eps else None
itm_out.mkdir(parents=True, exist_ok=True)

items, skipped = [], 0
for grid in sorted(grid_dir.glob("episode_*.png")):
    ep = grid.stem  # episode_000397
    prompt_txt = dense_out / f"{ep}.txt"
    if not prompt_txt.exists():
        continue
    out_mp4 = itm_out / f"{ep}.mp4"
    if not force and out_mp4.exists() and out_mp4.stat().st_size > 0:
        skipped += 1
        continue
    prompt = prompt_txt.read_text().strip()
    items.append({"input_video": str(grid), "prompt": prompt, "output_video": str(out_mp4)})
    if max_eps is not None and len(items) >= max_eps:
        break

batch_json.parent.mkdir(parents=True, exist_ok=True)
json.dump(items, open(batch_json, "w"), indent=2)
print(f"[match_dense] {len(items)} to generate, {skipped} already done -> {batch_json}")
PY
  else
    PREPARE_ARGS=(
      --dataset_path "${DS_PATH}"
      --input_dir    "${INPUT_DIR}"
      --category     "${CAT}"
      --batch_json   "${BATCH_JSON}"
      --output_dir   "${OUTPUT_DIR}"
      --skip_existing
    )
    if [[ -n "${MAX_EPISODES}" ]]; then
      PREPARE_ARGS+=(--max_episodes "${MAX_EPISODES}")
    fi
    if [[ -n "${MIN_EPISODE_INDEX}" ]]; then
      PREPARE_ARGS+=(--min_episode_index "${MIN_EPISODE_INDEX}")
    fi
    if [[ -n "${EPISODE_SOURCE}" ]]; then
      PREPARE_ARGS+=(--episode_source "${EPISODE_SOURCE}")
    fi
    if [[ "${FORCE_REGENERATE_OUTPUTS}" == "1" ]]; then
      PREPARE_ARGS+=(--force_regenerate_outputs)
    fi

    python "${ROOT}/scripts/prepare/prepare_single_arm_multiview_dreamgen_batch.py" "${PREPARE_ARGS[@]}"
  fi

  if [[ ! -f "${BATCH_JSON}" ]]; then
    echo "[WARN] No batch at ${BATCH_JSON}, skip inference"
    continue
  fi

  ITM_FLAGS=(
    --use_itm
    --itm_start_layer_idx   "${ITM_START_LAYER_IDX}"
    --itm_keep_last_n_dense "${ITM_KEEP_LAST_N_DENSE}"
    --itm_prune_from_step   "${ITM_PRUNE_FROM_STEP}"
    --itm_merge_from_step   "${ITM_MERGE_FROM_STEP}"
  )
  if [[ -n "${ITM_COMPRESS_RATIO}" ]]; then
    ITM_FLAGS+=(--itm_compress_ratio "${ITM_COMPRESS_RATIO}")
  fi

  python -m examples.video2world_gr00t \
    --model_size       "${MODEL_SIZE}" \
    --gr00t_variant    droid \
    --batch_input_json "${BATCH_JSON}" \
    --prompt_prefix    "" \
    --num_gpus         "${NUM_GPUS}" \
    --disable_guardrail \
    --fps              "${FPS}" \
    "${ITM_FLAGS[@]}"

  echo ""
done

echo "[Done] ITM single_arm multiview — ${CATEGORIES[*]}"
echo "       Results: ${OUTPUT_ROOT}/<cat>/"
