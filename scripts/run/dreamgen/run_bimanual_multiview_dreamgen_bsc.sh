#!/usr/bin/env bash
# DreamGen bimanual MULTIVIEW (2x2 grid) FULL roll-out inference with BSC.
#
# Exact mirror of scripts/run/dreamgen/run_bimanual_multiview_dreamgen_worldcache.sh in logic,
# checkpoint, prompt, rollout convention and paths -- only:
#   * routes output to sampling_dataset/bsc/... (method = top-level dir),
#   * adds the BSC flags,
#   * puts AM_DiT/src on PYTHONPATH, which is how `bsc.backend` is importable.
#
# BSC is OURS: algorithm core in AM_DiT/src/bsc, DiT adaptation in
# AM_DiT/methods/cache_strategy/BSC. See AM_DiT/docs/ALGORITHM.md.
#
# Outputs (canonical layout, {model}=dreamgen):
#   sampling_dataset/bsc/bimanual/output/multiview/dreamgen/<cat>/<iter>/<task>/episode_XXXXXX/
#
# Usage (repo root):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamgen/run_bimanual_multiview_dreamgen_bsc.sh
#
# Overrideable env: ITER_NAME, CHECKPOINTS_DIR, DIT_PATH, GUIDANCE, FPS,
#   NUM_CONDITIONAL_FRAMES, MAX_EPISODES, MAX_SEGMENTS, MULTIVIEW_ROOT, SAVE_ROOT,
#   AM_DIT_ROOT, BSC_{TAU,MFI,GROUP_SIZE,GROUP_PREDICT,GATE_MODE,PREDICTOR,WARMUP,
#                     GROUP_BOUNDS,PROTECT_FIRST_N,SKIP_ELIGIBILITY,SNR_SKIP_T}
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

export VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"

# AM_DiT/src is what makes `from bsc.backend import apply_bsc` resolve. AM_DiT's ROOT is
# deliberately NOT added: its methods/ is a regular package and would shadow this repo's
# namespace package. The shim appends what it needs to __path__ instead (AM_DiT/src/README.md).
AM_DIT_ROOT="${AM_DIT_ROOT:-$(cd "${ROOT}/../../AM_DiT" 2>/dev/null && pwd || true)}"
if [[ -z "${AM_DIT_ROOT}" || ! -d "${AM_DIT_ROOT}/src/bsc" ]]; then
  echo "[ERROR] AM_DiT not found. Set AM_DIT_ROOT to the AM_DiT checkout (needs src/bsc/)." >&2
  exit 1
fi
# Repo root must precede models/dreamgen so `methods.cache_strategy` is importable.
export PYTHONPATH="${ROOT}:${ROOT}/models/dreamgen:${AM_DIT_ROOT}/src:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/finetuned/dreamgen/posttraining/multiview/bimanual_multiview_seg16/checkpoints_40_per}"
if [[ -n "${ITER_NAME:-}" ]]; then
  _LATEST_FNAME="${ITER_NAME%.pt}.pt"
else
  _LATEST_FILE="${CHECKPOINTS_DIR}/latest_checkpoint.txt"
  if [[ -f "${_LATEST_FILE}" ]]; then
    _LATEST_FNAME="$(cat "${_LATEST_FILE}")"
  else
    _LATEST_FNAME="iter_000120000.pt"
  fi
fi
ITER_NAME="${_LATEST_FNAME%.pt}"
DIT_PATH="${DIT_PATH:-${CHECKPOINTS_DIR}/model/${_LATEST_FNAME}}"

# Cache backends are single-GPU only.
NUM_GPUS=1
MODEL_SIZE="${MODEL_SIZE:-2B}"
RESOLUTION="${RESOLUTION:-480}"
FPS="${FPS:-16}"
SAVE_FPS="${SAVE_FPS:-16}"
GUIDANCE="${GUIDANCE:-7}"
NUM_CONDITIONAL_FRAMES="${NUM_CONDITIONAL_FRAMES:-5}"
MAX_EPISODES="${MAX_EPISODES:-}"
MAX_SEGMENTS="${MAX_SEGMENTS:-}"
SEED="${SEED:-0}"
ITER_TAG="${ITER_TAG:-${ITER_NAME}}"

# BSC parameters. These defaults are the Ctrl-World reference point
# (AM_DiT/docs/ALGORITHM.md section 9) and are NOT tuned for DreamGen -- this backbone has
# 28 DiT blocks, not 16 U-Net attention units, and no DreamGen sweep has been run yet.
# Treat the first run as a guard (rc=0 and skip% > 0), then sweep BSC_TAU x BSC_MFI.
#
# BSC_MFI / BSC_WARMUP ARE IN NET-FORWARD UNITS, NOT DENOISE STEPS: DreamGen runs CFG
# sequentially, so net.forward fires twice per denoise step. MFI=6 means "re-anchor every
# 3 denoise steps". Same doubling the WorldCache reset applies to its own num_steps.
BSC_GATE_MODE="${BSC_GATE_MODE:-group}"
BSC_TAU="${BSC_TAU:-0.1}"
BSC_MFI="${BSC_MFI:-6}"
BSC_WARMUP="${BSC_WARMUP:-4}"
BSC_GROUP_SIZE="${BSC_GROUP_SIZE:-4}"
BSC_GROUP_PREDICT="${BSC_GROUP_PREDICT:-io}"
BSC_PREDICTOR="${BSC_PREDICTOR:-linear_s1}"
BSC_GROUP_BOUNDS="${BSC_GROUP_BOUNDS:-}"
BSC_PROTECT_FIRST_N="${BSC_PROTECT_FIRST_N:-0}"
BSC_SKIP_ELIGIBILITY="${BSC_SKIP_ELIGIBILITY:-1}"
BSC_SNR_SKIP_T="${BSC_SNR_SKIP_T:-0.0}"

MULTIVIEW_ROOT="${MULTIVIEW_ROOT:-${ROOT}/datasets/bimanual/multiview}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/bsc/bimanual/output/multiview/dreamgen}"

# WHERE THE SKIP ACCOUNTING LANDS. The rollout harness reports PSNR/SSIM/LPIPS and
# nothing about cache behaviour, so without this a run that produced a perfectly good
# video and a run where the gate never fired look IDENTICAL -- and every sweep row would
# be a dense run wearing a BSC label. bsc.backend writes this at exit; the schema is the
# one AM_DiT/analysis/flops_profile.py::score_skip_json reads.
export BSC_STATS_JSON="${BSC_STATS_JSON:-${SAVE_ROOT}/${ITER_TAG}_bsc_skip.json}"

if [[ ! -f "${DIT_PATH}" ]]; then
  echo "[ERROR] Checkpoint not found: ${DIT_PATH}" >&2
  exit 1
fi

DATASET_PATHS=()
for SPLIT in makovian non_makovian; do
  SPLIT_DIR="${MULTIVIEW_ROOT}/${SPLIT}"
  [[ -d "${SPLIT_DIR}" ]] || continue
  while IFS= read -r -d '' TASK_DIR; do
    [[ -f "${TASK_DIR}/meta/episodes.jsonl" ]] || continue
    DATASET_PATHS+=("${TASK_DIR}")
  done < <(find "${SPLIT_DIR}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
done
if [[ "${#DATASET_PATHS[@]}" -eq 0 ]]; then
  echo "[ERROR] No task dirs with meta/episodes.jsonl under ${MULTIVIEW_ROOT}" >&2
  exit 1
fi
DATASET_PATH_CSV="$(IFS=,; echo "${DATASET_PATHS[*]}")"

echo "============================================"
echo "DreamGen Bimanual Multiview FULL ROLL-OUT + BSC"
echo "  Checkpoint  : ${DIT_PATH}"
echo "  Iter dir    : ${ITER_TAG}"
echo "  Model       : ${MODEL_SIZE} | Res: ${RESOLUTION} | FPS: ${FPS}"
echo "  GPUs        : ${NUM_GPUS} (cache = single-GPU)"
echo "  Guidance    : ${GUIDANCE} | cond_frames: ${NUM_CONDITIONAL_FRAMES}"
echo "  AM_DiT      : ${AM_DIT_ROOT}"
echo "  BSC         : gate=${BSC_GATE_MODE} tau=${BSC_TAU} mfi=${BSC_MFI} (net fwds) warmup=${BSC_WARMUP}"
echo "                group=${BSC_GROUP_SIZE}/${BSC_GROUP_PREDICT} predictor=${BSC_PREDICTOR}"
echo "                bounds=${BSC_GROUP_BOUNDS:-uniform} protect_first_n=${BSC_PROTECT_FIRST_N} elig=${BSC_SKIP_ELIGIBILITY} snr_t=${BSC_SNR_SKIP_T}"
echo "  Save root   : ${SAVE_ROOT}"
echo "  Skip json   : ${BSC_STATS_JSON}"
echo "  Episodes    : ${MAX_EPISODES:-all} (per task)"
echo "============================================"

BSC_FLAGS=(
  --bsc-enabled
  --bsc-gate-mode         "${BSC_GATE_MODE}"
  --bsc-rcs-threshold     "${BSC_TAU}"
  --bsc-min-full-interval "${BSC_MFI}"
  --bsc-warmup-steps      "${BSC_WARMUP}"
  --bsc-group-size        "${BSC_GROUP_SIZE}"
  --bsc-group-predict     "${BSC_GROUP_PREDICT}"
  --bsc-predictor-mode    "${BSC_PREDICTOR}"
  --bsc-protect-first-n   "${BSC_PROTECT_FIRST_N}"
  --bsc-snr-skip-t        "${BSC_SNR_SKIP_T}"
)
[[ -n "${BSC_GROUP_BOUNDS}" ]] && BSC_FLAGS+=(--bsc-group-bounds "${BSC_GROUP_BOUNDS}")
[[ "${BSC_SKIP_ELIGIBILITY}" == "1" ]] && BSC_FLAGS+=(--bsc-skip-eligibility)

PY_ARGS=(
  --dataset-path "${DATASET_PATH_CSV}"
  --save-root    "${SAVE_ROOT}"
  --dit-path     "${DIT_PATH}"
  --iter-name    "${ITER_TAG}"
  --model-size   "${MODEL_SIZE}"
  --resolution   "${RESOLUTION}"
  --fps          "${FPS}"
  --save-fps     "${SAVE_FPS}"
  --guidance     "${GUIDANCE}"
  --num-gpus     "${NUM_GPUS}"
  --num-conditional-frames "${NUM_CONDITIONAL_FRAMES}"
  --seed         "${SEED}"
  --disable-guardrail
  "${BSC_FLAGS[@]}"
)
[[ -n "${MAX_EPISODES}" ]] && PY_ARGS+=(--max-episodes "${MAX_EPISODES}")
[[ -n "${MAX_SEGMENTS}" ]] && PY_ARGS+=(--max-segments "${MAX_SEGMENTS}")

python "${ROOT}/scripts/infer/infer_bimanual_multiview_rollout.py" "${PY_ARGS[@]}"

echo ""
echo "All done. Results under: ${SAVE_ROOT}/<category>/${ITER_TAG}/<task>/episode_*/"
echo "Summary: ${SAVE_ROOT}/${ITER_TAG}_summary.json"
