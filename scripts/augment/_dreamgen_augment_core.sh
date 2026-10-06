#!/bin/bash
# Shared body for the DreamGen AUGMENT run scripts (batch-json configs).
#
# NOT meant to be run directly — the wrappers
#   scripts/run/dreamgen/run_{single_arm,bimanual}_{singleview,multiview}_dreamgen_augment.sh
# set CONFIG_KEY and source this. bimanual MULTIVIEW is NOT handled here: that
# config uses the segment-rollout script, which reads the manifest itself, so it
# has its own standalone runner.
#
# Contract with the caller: ROOT (repo root), CONFIG_KEY (an augment_paths.py
# DreamGen config) are set.
#
# For each category x branch:
#   Step 1 (edit_only/combo only): recolor the conditioning first frame
#           -> augment input tree (.venv_edit / Qwen-Image-Edit).
#   Step 2: build the augment batch json (frames + branch prompt + tagged output
#           path) via scripts/augment/prepare_dreamgen_augment_batch.py.
#   Step 3: run the SAME dense inference entrypoint with the SAME flags, only
#           pointed at the augment batch json. prompt_only/combo additionally
#           enable DreamGen's prompt refiner (that is part of the perturbation).
#
# No GT is produced or compared: DreamGen writes only the generated mp4 + its
# prompt sidecar, so "--no-gt" needs no flag here — there is simply nothing to
# strip. Reference-free metrics come from scripts/augment/compute_augment_metrics.py.

: "${ROOT:?ROOT must be set by the wrapper}"
: "${CONFIG_KEY:?CONFIG_KEY must be set by the wrapper}"

PATHS="${ROOT}/scripts/augment/augment_paths.py"
PREP="${ROOT}/scripts/augment/prepare_dreamgen_augment_batch.py"

SPLITS="${SPLITS:-makovian non_makovian}"
BRANCHES="${BRANCHES:-edit_only prompt_only combo}"
MANIFEST="${MANIFEST:-${ROOT}/augment_manifest.json}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
SKIP_EDIT="${SKIP_EDIT:-0}"
SKIP_INFER="${SKIP_INFER:-0}"
LIMIT="${LIMIT:-}"

# --- editor (Step 1) --------------------------------------------------------
EDIT_MODEL="${EDIT_MODEL:-qwen}"
EDIT_SEED="${EDIT_SEED:-0}"
EDIT_IMAGE_GUIDANCE="${EDIT_IMAGE_GUIDANCE:-1.8}"
EDIT_VENV="${EDIT_VENV:-${ROOT}/.venv_edit}"

# --- per-config settings ----------------------------------------------------
# Every value below mirrors the DENSE run script for that config, so the only
# difference between augment and dense output is the perturbation itself.
case "${CONFIG_KEY}" in
  dreamgen_single_arm_singleview)
    # mirrors run_single_arm_singleview_dreamgen_static16fps.sh
    ENTRYPOINT=lvg
    CKPT_BASE="${CKPT_BASE:-${ROOT}/checkpoints/finetuned/dreamgen/posttraining/singleview/single_arm_static_singleview_16fps_v2}"
    CKPT_MODEL_DIR="${CKPT_BASE}/checkpoints_40_per/model"
    DEFAULT_ITER=iter_000090000
    ITER_SUFFIX=_static16fps
    MODEL_SIZE="${MODEL_SIZE:-2B}"
    ASPECT_RATIO="${ASPECT_RATIO:-4:3}"
    NUM_CONDITIONAL_FRAMES="${NUM_CONDITIONAL_FRAMES:-1}"
    GUIDANCE="${GUIDANCE:-7}"
    NUM_GPUS="${NUM_GPUS:-1}"
    ;;
  dreamgen_bimanual_singleview)
    # mirrors run_bimanual_singleview_dreamgen_16fps.sh
    ENTRYPOINT=lvg
    CKPT_BASE="${CKPT_BASE:-${ROOT}/checkpoints/finetuned/dreamgen/posttraining/singleview/bimanual_singleview_16fps}"
    CKPT_MODEL_DIR="${CKPT_BASE}/checkpoints_40_per/model"
    DEFAULT_ITER=iter_000110000
    ITER_SUFFIX=_bimanual16fps
    MODEL_SIZE="${MODEL_SIZE:-2B}"
    ASPECT_RATIO="${ASPECT_RATIO:-16:9}"
    NUM_CONDITIONAL_FRAMES=""     # dense bimanual singleview leaves this at default
    GUIDANCE="${GUIDANCE:-7}"
    NUM_GPUS="${NUM_GPUS:-1}"
    ;;
  dreamgen_single_arm_multiview)
    # mirrors run_single_arm_multiview_dreamgen_batch.sh (single-shot gr00t, 14B)
    ENTRYPOINT=gr00t
    MODEL_SIZE="${MODEL_SIZE:-14B}"
    FPS="${FPS:-10}"
    NUM_GPUS="${NUM_GPUS:-1}"
    DEFAULT_ITER=""               # base GR00T DROID ckpt: no iter level in the path
    ITER_SUFFIX=""
    ;;
  *)
    echo "[dreamgen-augment-core] unsupported CONFIG_KEY '${CONFIG_KEY}'" >&2
    exit 1
    ;;
esac

# Resolve the checkpoint / iter tag for the fine-tuned (LVG) configs.
ITER_ARGS=()
OUT_ITER=""
if [ "${ENTRYPOINT}" = "lvg" ]; then
  if [ -z "${ITER_NAME:-}" ]; then
    latest_file="${CKPT_BASE}/checkpoints_40_per/latest_checkpoint.txt"
    if [ -f "${latest_file}" ]; then
      ITER_NAME="$(basename "$(cat "${latest_file}")" .pt)"
    else
      ITER_NAME="${DEFAULT_ITER}"
    fi
  fi
  DIT_PATH="${DIT_PATH:-${CKPT_MODEL_DIR}/${ITER_NAME}.pt}"
  if [ ! -f "${DIT_PATH}" ]; then
    echo "ERROR: checkpoint not found: ${DIT_PATH}" >&2
    ls "${CKPT_MODEL_DIR}/"*.pt 2>/dev/null >&2 || true
    exit 1
  fi
  OUT_ITER="${ITER_NAME}${ITER_SUFFIX}"
  ITER_ARGS=(--iter "${OUT_ITER}")
fi

echo "=========================================================="
echo "DreamGen AUGMENT | ${CONFIG_KEY}"
echo "  categories : ${SPLITS}"
echo "  branches   : ${BRANCHES}"
echo "  manifest   : ${MANIFEST}"
echo "  entrypoint : ${ENTRYPOINT}  model=${MODEL_SIZE}  gpus=${NUM_GPUS}"
[ -n "${OUT_ITER}" ] && echo "  ckpt       : ${DIT_PATH} (iter tag ${OUT_ITER})"
echo "  output     : $(python3 "${PATHS}" --config "${CONFIG_KEY}" --io output --base)"
echo "=========================================================="

# --------------------------------------------------------------- Step 1: edit
edit_frames() {
  local split="$1" branch="$2"
  case "${branch}" in
    prompt_only|task_swap|object_relocation) return 0 ;;   # frame is deliberately untouched
  esac
  [ "${SKIP_EDIT}" = "1" ] && return 0
  echo "  [edit] ${split}/${branch}: recoloring conditioning first frames..."
  (
    VENV="${EDIT_VENV}"
    export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    export HF_HOME="${HF_HOME:-/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.cache/huggingface}"
    export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
    local limit_flag=()
    [ -n "${LIMIT}" ] && limit_flag=(--limit "${LIMIT}")
    python "${ROOT}/scripts/augment/edit_dreamgen_first_frames.py" \
      --config "${CONFIG_KEY}" --split "${split}" --branch "${branch}" \
      --manifest "${MANIFEST}" --model "${EDIT_MODEL}" --seed "${EDIT_SEED}" \
      --image-guidance "${EDIT_IMAGE_GUIDANCE}" "${limit_flag[@]}"
  )
}

# ------------------------------------------------------ Step 3: run inference
# `pending_json` drops already-generated episodes so a re-run resumes instead of
# regenerating (FORCE_REGENERATE=1 disables that).
pending_json() {
  local src="$1" dst="$2"
  python3 - "${src}" "${dst}" "${FORCE_REGENERATE}" <<'PY'
import json, sys
from pathlib import Path
src, dst, force = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "1"
items = json.loads(src.read_text())
if not force:
    items = [x for x in items if not Path(x["output_video"]).is_file()]
dst.write_text(json.dumps(items, indent=2, ensure_ascii=False))
print(len(items))
PY
}

run_batch() {
  local batch_json="$1" label="$2" num_chunks="$3" branch="$4"
  [ -f "${batch_json}" ] || { echo "  [SKIP] ${label}: no batch json"; return 0; }

  local tmp count
  tmp="$(mktemp --suffix=_dreamgen_augment.json)"
  count="$(pending_json "${batch_json}" "${tmp}")"
  if [ "${count}" -eq 0 ]; then
    echo "  [SKIP] ${label}: all episodes already generated"
    rm -f "${tmp}"
    return 0
  fi
  echo "  [run] ${label}: ${count} episode(s)"

  # The refiner rewrites the prompt, so it is only enabled where a rewritten
  # prompt IS the perturbation. edit_only must keep the dense text path exactly.
  local pr_flag=()
  if [ "${branch}" != "edit_only" ]; then pr_flag=(--enable_prompt_refiner_per_chunk); fi

  if [ "${ENTRYPOINT}" = "lvg" ]; then
    local cond_args=()
    [ -n "${NUM_CONDITIONAL_FRAMES}" ] && cond_args=(--num_conditional_frames "${NUM_CONDITIONAL_FRAMES}")
    python -m models.dreamgen.examples.video2world_gr00t_lvg \
      --model_size "${MODEL_SIZE}" \
      --gr00t_variant droid \
      --dit_path "${DIT_PATH}" \
      --batch_input_json "${tmp}" \
      --num_chunks "${num_chunks}" \
      --guidance "${GUIDANCE}" \
      --num_gpus "${NUM_GPUS}" \
      --aspect_ratio "${ASPECT_RATIO}" \
      "${cond_args[@]}" \
      --prompt_prefix "" \
      --disable_guardrail \
      --no_skip_existing \
      "${pr_flag[@]}"
  else
    python -m examples.video2world_gr00t \
      --model_size "${MODEL_SIZE}" \
      --gr00t_variant droid \
      --batch_input_json "${tmp}" \
      --prompt_prefix "" \
      --num_gpus "${NUM_GPUS}" \
      --disable_guardrail \
      --fps "${FPS}"
  fi

  rm -f "${tmp}"
}

# ------------------------------------------------------------------ main loop
for split in ${SPLITS}; do
  for branch in ${BRANCHES}; do
    echo ""
    echo "---------- ${CONFIG_KEY} / ${split} / ${branch} ----------"
    edit_frames "${split}" "${branch}"

    # Step 2 runs in the dreamgen venv-independent python3 (pure stdlib).
    prep_limit=()
    [ -n "${LIMIT}" ] && prep_limit=(--limit "${LIMIT}")
    python3 "${PREP}" --config "${CONFIG_KEY}" --split "${split}" \
      --branch "${branch}" --manifest "${MANIFEST}" "${ITER_ARGS[@]}" "${prep_limit[@]}"
  done
done

if [ "${SKIP_INFER}" = "1" ]; then
  echo "[dreamgen-augment-core] SKIP_INFER=1, stopping after batch preparation."
  return 0 2>/dev/null || exit 0
fi

# Inference needs the dreamgen venv. Do it AFTER all prep so the editor venv and
# the dreamgen venv are never active at the same time.
export VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
# Repo root first: methods.* must be importable before models/dreamgen loads config.
export PYTHONPATH="${ROOT}:${ROOT}/models/dreamgen:${PYTHONPATH:-}"

for split in ${SPLITS}; do
  for branch in ${BRANCHES}; do
    input_dir="$(python3 -c "
import sys; sys.path.insert(0, '${ROOT}/scripts/augment')
import augment_paths as AP
print(AP.dreamgen_input_dir('${CONFIG_KEY}', '${split}'))")"
    [ -d "${input_dir}" ] || { echo "  [SKIP] ${split}/${branch}: no input dir"; continue; }

    if [ "${ENTRYPOINT}" = "lvg" ]; then
      # One batch json per required chunk count (chunk count == GT length class).
      for bj in "${input_dir}/batch_input_${branch}_"*chunks.json; do
        [ -f "${bj}" ] || continue
        nc="$(basename "${bj}" | sed -E 's/.*_([0-9]+)chunks\.json/\1/')"
        run_batch "${bj}" "${split}/${branch}/${nc}chunks" "${nc}" "${branch}"
      done
    else
      run_batch "${input_dir}/batch_input_${branch}.json" \
        "${split}/${branch}" 1 "${branch}"
    fi

    # Attach provenance to the videos that now exist (and drop sidecars for any
    # episode that failed), so the output dir never lists a phantom episode.
    prep_limit=()
    [ -n "${LIMIT}" ] && prep_limit=(--limit "${LIMIT}")
    python3 "${PREP}" --config "${CONFIG_KEY}" --split "${split}" \
      --branch "${branch}" --manifest "${MANIFEST}" "${ITER_ARGS[@]}" \
      "${prep_limit[@]}" --finalize
  done
done

echo ""
echo "=========================================================="
echo "Done. Augment output: $(python3 "${PATHS}" --config "${CONFIG_KEY}" --io output --base)"
echo "Compare against:      $(python3 "${PATHS}" --config "${CONFIG_KEY}" --io output --base --dense)"
echo "=========================================================="
