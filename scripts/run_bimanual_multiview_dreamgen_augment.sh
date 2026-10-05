#!/bin/bash
# DreamGen bimanual MULTIVIEW AUGMENT run — 3 branches x 5 videos per category.
#
# Mirrors run_bimanual_multiview_dreamgen_full.sh (same seg16 checkpoint, 2x2 grid,
# 480p @ 16fps, guidance 7, cond_frames 5, autoregressive segment rollout) and calls
# the augment copy of the rollout script. Only the input is perturbed:
#
#   edit_only    the object is RECOLORED in the 2x2 grid conditioning frame,
#                prompt stays byte-identical to dense (refiner off).
#   prompt_only  grid frame untouched, the perturbed behaviour is injected into the
#                grid caption's task slot (the 2x2-layout sentences MUST stay, or the
#                model stops emitting a grid), refiner on.
#   combo        recolored grid + perturbed task phrase naming the new color, refiner on.
#   object_relocation  grid frame untouched, hand-curated task phrase naming a REAL
#                loose object AND a REAL container both visible in the grid's frame 0
#                (injected into the same task slot as prompt_only), refiner on.
#
# This config differs from the other three DreamGen configs: it is a segment
# rollout, so the manifest is read by the inference script itself (there is no
# batch_input json) and only the edit step is separate.
#
# Output (leaf gains a __{branch} tag; drop it to get the dense path):
#   sampling_dataset/augment/bimanual/output/multiview/dreamgen/{split}/{iter}/
#       {task}/episode_XXXXXX__{branch}/{full_pred.mp4, metrics.json, augment_info.json}
#
# --no-gt is passed: the recolored/reprompted episode has no ground truth, so no
# full_gt.mp4 / full_merged.mp4 and no PSNR/SSIM/LPIPS. Reference-free metrics come
# from scripts/augment/compute_augment_metrics.py.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_multiview_dreamgen_augment.sh
#   BRANCHES=edit_only MAX_EPISODES=1 MAX_SEGMENTS=2 bash scripts/run_bimanual_multiview_dreamgen_augment.sh
#
# Env overrides: BRANCHES, SPLITS, MANIFEST, ITER_NAME, DIT_PATH, CHECKPOINTS_DIR,
#   NUM_GPUS, GUIDANCE, FPS, SAVE_FPS, NUM_CONDITIONAL_FRAMES, MAX_EPISODES,
#   MAX_SEGMENTS, MULTIVIEW_ROOT, SAVE_ROOT, EDIT_MODEL, EDIT_SEED, SKIP_EDIT=1,
#   SKIP_INFER=1.
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG_KEY="dreamgen_bimanual_multiview"
PATHS="${ROOT}/scripts/augment/augment_paths.py"

SPLITS="${SPLITS:-makovian non_makovian}"
BRANCHES="${BRANCHES:-edit_only prompt_only combo object_relocation}"
MANIFEST="${MANIFEST:-${ROOT}/augment_manifest.json}"
SKIP_EDIT="${SKIP_EDIT:-0}"
SKIP_INFER="${SKIP_INFER:-0}"
LIMIT="${LIMIT:-}"

EDIT_MODEL="${EDIT_MODEL:-qwen}"
EDIT_SEED="${EDIT_SEED:-0}"
EDIT_IMAGE_GUIDANCE="${EDIT_IMAGE_GUIDANCE:-1.8}"
EDIT_VENV="${EDIT_VENV:-${ROOT}/.venv_edit}"

# --- checkpoint (identical default to the dense full rollout) ---
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/finetuned/dreamgen/posttraining/multiview/bimanual_multiview_seg16/checkpoints_40_per}"
if [ -n "${ITER_NAME:-}" ]; then
  _LATEST_FNAME="${ITER_NAME%.pt}.pt"
elif [ -f "${CHECKPOINTS_DIR}/latest_checkpoint.txt" ]; then
  _LATEST_FNAME="$(cat "${CHECKPOINTS_DIR}/latest_checkpoint.txt")"
else
  _LATEST_FNAME="iter_000120000.pt"
fi
ITER_NAME="${_LATEST_FNAME%.pt}"
DIT_PATH="${DIT_PATH:-${CHECKPOINTS_DIR}/model/${_LATEST_FNAME}}"
if [ ! -f "${DIT_PATH}" ]; then
  echo "[ERROR] Checkpoint not found: ${DIT_PATH}" >&2
  exit 1
fi

MODEL_SIZE="${MODEL_SIZE:-2B}"
RESOLUTION="${RESOLUTION:-480}"
FPS="${FPS:-16}"
SAVE_FPS="${SAVE_FPS:-16}"
GUIDANCE="${GUIDANCE:-7}"
NUM_CONDITIONAL_FRAMES="${NUM_CONDITIONAL_FRAMES:-5}"
NUM_GPUS="${NUM_GPUS:-1}"
SEED="${SEED:-0}"
MASTER_PORT="${MASTER_PORT:-29514}"

MULTIVIEW_ROOT="${MULTIVIEW_ROOT:-${ROOT}/datasets/bimanual/multiview}"
SAVE_ROOT="${SAVE_ROOT:-$(python3 "${PATHS}" --config "${CONFIG_KEY}" --io output --base)}"
EDITED_ROOT="${EDITED_ROOT:-$(python3 "${PATHS}" --config "${CONFIG_KEY}" --io input --base)}"

# Only the task dirs the manifest actually references — the rollout script would
# otherwise scan all ~100 tasks just to find nothing to do in most of them.
DATASET_PATH_CSV="$(python3 - "${MANIFEST}" "${MULTIVIEW_ROOT}" "${SPLITS}" "${BRANCHES}" <<'PY'
import json, sys
from pathlib import Path
manifest, root, splits, branches = sys.argv[1], Path(sys.argv[2]), sys.argv[3].split(), sys.argv[4].split()
section = json.loads(Path(manifest).read_text()).get("dreamgen_bimanual_multiview", {})
paths = []
for split in splits:
    for branch in branches:
        for e in section.get(split, {}).get(branch, []):
            d = root / split / e["task"]
            if (d / "meta" / "episodes.jsonl").exists() and str(d) not in paths:
                paths.append(str(d))
print(",".join(paths))
PY
)"
if [ -z "${DATASET_PATH_CSV}" ]; then
  echo "[ERROR] no manifest task dirs found under ${MULTIVIEW_ROOT}" >&2
  exit 1
fi

echo "=========================================================="
echo "DreamGen AUGMENT | ${CONFIG_KEY} (segment rollout, no-gt)"
echo "  categories : ${SPLITS}"
echo "  branches   : ${BRANCHES}"
echo "  ckpt       : ${DIT_PATH} (iter ${ITER_NAME})"
echo "  tasks      : $(echo "${DATASET_PATH_CSV}" | tr ',' '\n' | wc -l)"
echo "  output     : ${SAVE_ROOT}"
echo "=========================================================="

# ---- Step 1: recolor grid first frames (edit_only / combo), editor venv ----
if [ "${SKIP_EDIT}" != "1" ]; then
  for split in ${SPLITS}; do
    for branch in ${BRANCHES}; do
      [ "${branch}" = "prompt_only" ] && continue   # frame deliberately untouched
    [ "${branch}" = "object_relocation" ] && continue   # frame deliberately untouched
      echo ""
      echo "[edit] ${split}/${branch}: recoloring 2x2 grid first frames..."
      (
        VENV="${EDIT_VENV}"
        export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
        # shellcheck disable=SC1091
        source "${VENV}/bin/activate"
        export HF_HOME="${HF_HOME:-/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.cache/huggingface}"
        export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
        limit_flag=()
        [ -n "${LIMIT}" ] && limit_flag=(--limit "${LIMIT}")
        python "${ROOT}/scripts/augment/edit_dreamgen_first_frames.py" \
          --config "${CONFIG_KEY}" --split "${split}" --branch "${branch}" \
          --manifest "${MANIFEST}" --model "${EDIT_MODEL}" --seed "${EDIT_SEED}" \
          --image-guidance "${EDIT_IMAGE_GUIDANCE}" "${limit_flag[@]}"
      )
    done
  done
fi

[ "${SKIP_INFER}" = "1" ] && { echo "[augment] SKIP_INFER=1, stopping after edit."; exit 0; }

# ---- Step 2: segment rollout per branch, dreamgen venv ----
export VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/models/dreamgen:${ROOT}:${PYTHONPATH:-}"

for branch in ${BRANCHES}; do
  echo ""
  echo "---------- rollout: ${branch} ----------"
  PY_ARGS=(
    --dataset-path "${DATASET_PATH_CSV}"
    --save-root    "${SAVE_ROOT}"
    --dit-path     "${DIT_PATH}"
    --iter-name    "${ITER_NAME}"
    --model-size   "${MODEL_SIZE}"
    --resolution   "${RESOLUTION}"
    --fps          "${FPS}"
    --save-fps     "${SAVE_FPS}"
    --guidance     "${GUIDANCE}"
    --num-gpus     "${NUM_GPUS}"
    --num-conditional-frames "${NUM_CONDITIONAL_FRAMES}"
    --seed         "${SEED}"
    --disable-guardrail
    --manifest     "${MANIFEST}"
    --manifest-config "${CONFIG_KEY}"
    --branch       "${branch}"
    --no-gt
  )
  # The refiner rewrites the prompt, so enable it only where a rewritten prompt IS
  # the perturbation; edit_only must keep the dense text path exactly.
  [ "${branch}" != "edit_only" ] && PY_ARGS+=(--prompt-refine)
  # prompt_only / object_relocation keep the original GT first frame -> no
  # edited-frames lookup.
  [ "${branch}" != "prompt_only" ] && [ "${branch}" != "object_relocation" ] && PY_ARGS+=(--edited-frames-root "${EDITED_ROOT}")
  [ -n "${MAX_EPISODES:-}" ] && PY_ARGS+=(--max-episodes "${MAX_EPISODES}")
  [ -n "${MAX_SEGMENTS:-}" ] && PY_ARGS+=(--max-segments "${MAX_SEGMENTS}")

  if [ "${NUM_GPUS}" -gt 1 ]; then
    torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
      "${ROOT}/scripts/augment/infer_bimanual_multiview_rollout_augment.py" "${PY_ARGS[@]}"
    MASTER_PORT=$((MASTER_PORT + 1))
  else
    python "${ROOT}/scripts/augment/infer_bimanual_multiview_rollout_augment.py" "${PY_ARGS[@]}"
  fi
done

echo ""
echo "=========================================================="
echo "Done. Augment output: ${SAVE_ROOT}"
echo "Compare against:      $(python3 "${PATHS}" --config "${CONFIG_KEY}" --io output --base --dense)"
echo "=========================================================="
