#!/bin/bash
# KIVA -- hallucination metric (kinescore rigidity + jerk violation rate).
#
# Scores one or more method variants with the kinescore physics-violation
# detectors vendored in metrics/kiva/ from branch `quang/hallucination-official`
# @5225ce7 (see metrics/kiva/VENDORED.md and docs/metrics/kiva.md). Emits one output dir per (variant, embodiment),
# plus a cross-variant comparison via metrics/kiva/aggregate.py.
#
# The metric consumes MP4s DIRECTLY (no frame extraction), one clip per file, at
# the reader's exact input resolution:
#   single_arm 640x480   bimanual 768x432   humanoid 768x432
# A clip at any other size is rejected by `view.check_frame_size`.
#
# Usage:
#   # score the baseline + 4 accelerated variants in ONE invocation
#   VARIANTS="dense pisa radial svg1 svg2" bash metrics/kiva/run.sh
#
#   # smoke test: 2 clips, one embodiment
#   MAX_CLIPS=2 VARIANTS=dense EMBODIMENTS=single_arm \
#     bash metrics/kiva/run.sh
#
#   # the real-video anchor: score the PAIRED *_gt.mp4 next to every prediction
#   ROLE=gt VARIANTS=dense bash metrics/kiva/run.sh
#
#   # long sweep in tmux
#   VARIANTS="dense pisa radial svg1 svg2" bash metrics/kiva/run.sh --tmux
#
# Env:
#   HAL_ROOT          kinescore source tree (default: metrics/kiva, the vendored copy;
#                     imported via PYTHONPATH=$HAL_ROOT/src, nothing is pip-installed)
#   HAL_STORE         kinescore store holding assets/ data/ cache/ (and ckpt/)
#                     (default: $KINESCORE_STORE, else /pfss/.../triquang/kinescore-store)
#   HAL_CKPT_DIR      reader checkpoints (default: checkpoints/metrics/kiva if it
#                     exists, else $HAL_STORE/ckpt)
#   HAL_PYTHON        python to use (default: the `hdmi` conda env, which already
#                     has torch+transformers+pytorch_kinematics; see KNOWN GOTCHAS
#                     in the doc for why `uv sync` is not used here)
#   HAL_DATA_ROOT     root of generated-video variants (default: $ROOT/sampling_dataset)
#   HAL_OUT_ROOT      where scored dirs land (default: $ROOT/results/hallucination)
#   VARIANTS          space-separated variant dir names under HAL_DATA_ROOT (default: dense)
#   EMBODIMENTS       space-separated subset of "single_arm bimanual humanoid" (default: all three)
#   SPLITS            space-separated subset of "makovian non_makovian" (default: both)
#   ROLE              pred | gt  -- `gt` scores the paired *_gt.mp4 real-video
#                     anchor instead of the prediction (default: pred)
#   MAX_CLIPS         cap clips per (variant, embodiment, split) (default: 0 = all)
#   FRAME_CHUNK       frames per DINOv3 forward (default: 16; lower = less VRAM)
#   MAX_FRAMES        cap frames per clip (default: 0 = all)
#   VIDEOS_DIR       score exactly one explicit directory and exit (needs EMBODIMENTS=<one>)
#   OUT_DIR           output dir for VIDEOS_DIR mode
#
# Reader/threshold overrides (per embodiment; defaults are the three
# benchmark readers and their thresholds in metrics/kiva/thresholds/):
#   HAL_CKPT_SINGLE_ARM, HAL_CKPT_BIMANUAL, HAL_CKPT_HUMANOID
#   HAL_THR_SINGLE_ARM,  HAL_THR_BIMANUAL,  HAL_THR_HUMANOID

set -uo pipefail
START_TIME=$SECONDS

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

HAL_ROOT="${HAL_ROOT:-${ROOT}/metrics/kiva}"
HAL_STORE="${HAL_STORE:-${KINESCORE_STORE:-/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/triquang/kinescore-store}}"
if [ -z "${HAL_CKPT_DIR:-}" ]; then
  HAL_CKPT_DIR="${ROOT}/checkpoints/metrics/kiva"
  [ -d "${HAL_CKPT_DIR}" ] || HAL_CKPT_DIR="${HAL_STORE}/ckpt"
fi
HAL_DATA_ROOT="${HAL_DATA_ROOT:-${ROOT}/sampling_dataset}"
HAL_OUT_ROOT="${HAL_OUT_ROOT:-${ROOT}/results/hallucination}"
HAL_FFMPEG_BIN="${HAL_FFMPEG_BIN:-/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/ffmpeg/ffmpeg-master-latest-linux64-gpl-shared}"

VARIANTS=(${VARIANTS:-dense})
EMBODIMENTS=(${EMBODIMENTS:-single_arm bimanual humanoid})
SPLITS=(${SPLITS:-makovian non_makovian})
ROLE="${ROLE:-pred}"
MAX_CLIPS="${MAX_CLIPS:-0}"
FRAME_CHUNK="${FRAME_CHUNK:-16}"
MAX_FRAMES="${MAX_FRAMES:-0}"
VIDEOS_DIR="${VIDEOS_DIR:-}"

# ----------------------------------------------------------------------
# tmux fan-out (one session; long sweeps must not block the caller)
# ----------------------------------------------------------------------
if [ "${1:-}" = "--tmux" ]; then
  SESSION="${SESSION:-hal_$(date +%H%M%S)}"
  mkdir -p "${ROOT}/exp_logs/tmux_logs"
  LOG="${ROOT}/exp_logs/tmux_logs/${SESSION}.log"
  tmux new-session -d -s "${SESSION}" \
    "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-0}' HAL_ROOT='${HAL_ROOT}' \
     HAL_CKPT_DIR='${HAL_CKPT_DIR}' HAL_STORE='${HAL_STORE}' HAL_PYTHON='${HAL_PYTHON:-}' \
     HAL_DATA_ROOT='${HAL_DATA_ROOT}' HAL_OUT_ROOT='${HAL_OUT_ROOT}' \
     VARIANTS='${VARIANTS[*]}' EMBODIMENTS='${EMBODIMENTS[*]}' SPLITS='${SPLITS[*]}' \
     ROLE='${ROLE}' MAX_CLIPS='${MAX_CLIPS}' FRAME_CHUNK='${FRAME_CHUNK}' \
     MAX_FRAMES='${MAX_FRAMES}' \
     bash metrics/kiva/run.sh 2>&1 | tee '${LOG}'; exec bash"
  echo "launched tmux session ${SESSION} (log ${LOG})"
  echo "  attach:  tmux attach -t ${SESSION}"
  exit 0
fi

# ----------------------------------------------------------------------
# 0. Metric code
# ----------------------------------------------------------------------
[ -f "${HAL_ROOT}/src/kinescore/cli/main.py" ] || {
  echo "ERROR: kinescore source not found under HAL_ROOT=${HAL_ROOT}" >&2
  echo "       (expected the vendored copy in metrics/kiva/, see VENDORED.md)" >&2
  exit 1; }

# ----------------------------------------------------------------------
# 1. Interpreter.
#
# `uv sync` on the branch's pyproject.toml does NOT work on this cluster: it
# resolves torch 2.14.1 -> triton 3.8.0, whose unpack blows the shared
# ~/.cache/uv disk quota ("Disk quota exceeded (os error 122)"). The `hdmi`
# conda env already carries torch 2.7.0+cu128 / transformers 5.2.0 / numpy
# 1.26.4, and PYTHONUSERBASE supplies pytorch_kinematics. See the doc's
# KNOWN GOTCHAS for the full list of undeclared/unresolvable deps.
# ----------------------------------------------------------------------
HAL_PYTHON="${HAL_PYTHON:-/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/miniconda3/envs/hdmi/bin/python}"
HAL_PYTHONUSERBASE="${HAL_PYTHONUSERBASE:-/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local}"
[ -x "${HAL_PYTHON}" ] || { echo "ERROR: HAL_PYTHON not executable: ${HAL_PYTHON}" >&2; exit 1; }
echo "[hal] interpreter: ${HAL_PYTHON}"

# ----------------------------------------------------------------------
# 2. Store: checkpoints, URDF assets, HF cache.
# ----------------------------------------------------------------------
for sub in assets cache; do
  [ -d "${HAL_STORE}/${sub}" ] || {
    echo "ERROR: ${HAL_STORE}/${sub} missing." >&2
    echo "       Set HAL_STORE (or KINESCORE_STORE) to a kinescore store, or" >&2
    echo "       populate one per docs/metrics/kiva.md#checkpoints" >&2
    exit 1; }
done

# `ffmpeg`/`ffprobe` from /usr/bin die once torch is imported in the same
# process tree (liblapack/gotoblas symbol clash). The pfss build is immune, but
# only if its lib dir is on LD_LIBRARY_PATH too -- otherwise the binary cannot
# find libavdevice.so.62.
if [ -d "${HAL_FFMPEG_BIN}/bin" ]; then
  export PATH="${HAL_FFMPEG_BIN}/bin:${PATH}"
  export LD_LIBRARY_PATH="${HAL_FFMPEG_BIN}/lib:${LD_LIBRARY_PATH:-}"
fi
command -v ffprobe >/dev/null || { echo "ERROR: ffprobe not on PATH" >&2; exit 1; }

export PYTHONPATH="${HAL_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUSERBASE="${HAL_PYTHONUSERBASE}"
export KINESCORE_ASSETS="${HAL_STORE}/assets"
export KINESCORE_CKPT_DIR="${HAL_CKPT_DIR}"
export KINESCORE_CACHE_DIR="${HAL_STORE}/cache"
export KINESCORE_DATA_ROOT="${HAL_STORE}/data"
export KINESCORE_OUTPUT_DIR="${HAL_OUT_ROOT}"
export HF_HOME="${HAL_STORE}/cache/hf"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ----------------------------------------------------------------------
# 3. Per-embodiment checkpoint + thresholds.
#
# These three are the BENCHMARK readers -- the ones whose sha256 appears in the
# published summary.json files. In particular the humanoid default is
# `*.huber_old.pt` (sha c40fc006..), which is the K=12 arms-only reader the
# tables were scored with; the four K=22 `*_newsampler`/`mse` checkpoints in the
# same dir are DIFFERENT trainings and are not benchmark-consistent.
# ----------------------------------------------------------------------
THR_SINGLE_ARM="${HAL_THR_SINGLE_ARM:-${HAL_ROOT}/thresholds/a1x_ee.a1x_sv.sv1_4x3.json}"
THR_BIMANUAL="${HAL_THR_BIMANUAL:-${HAL_ROOT}/thresholds/aloha_bimanual.bimanual_sv.sv1_16x9.json}"
THR_HUMANOID="${HAL_THR_HUMANOID:-${HAL_ROOT}/thresholds/fourier_gr1.humanoid_sv.sv1_16x9.json}"

CKPT_SINGLE_ARM="${HAL_CKPT_SINGLE_ARM:-${HAL_CKPT_DIR}/a1x_ee.a1x_sv.sv1_4x3.pt}"
CKPT_BIMANUAL="${HAL_CKPT_BIMANUAL:-${HAL_CKPT_DIR}/aloha_bimanual.bimanual_sv.sv1_16x9.pt}"
CKPT_HUMANOID="${HAL_CKPT_HUMANOID:-${HAL_CKPT_DIR}/fourier_gr1.humanoid_sv.sv1_16x9.huber_old.pt}"

# short tag used in the output dir name, mirroring the existing scored dirs
TAG_SINGLE_ARM="sv1_4x3.a1x"
TAG_BIMANUAL="bimanual_sv16.aloha"
TAG_HUMANOID="humanoid_sv16.gr1"

thr_for()   { case "$1" in single_arm) echo "${THR_SINGLE_ARM}";;  bimanual) echo "${THR_BIMANUAL}";;  humanoid) echo "${THR_HUMANOID}";;  *) echo "";; esac; }
ckpt_for()  { case "$1" in single_arm) echo "${CKPT_SINGLE_ARM}";; bimanual) echo "${CKPT_BIMANUAL}";; humanoid) echo "${CKPT_HUMANOID}";; *) echo "";; esac; }
tag_for()   { case "$1" in single_arm) echo "${TAG_SINGLE_ARM}";;  bimanual) echo "${TAG_BIMANUAL}";;  humanoid) echo "${TAG_HUMANOID}";;  *) echo "";; esac; }

# `*_gt.mp4` sits next to every prediction in the same directory at the same
# resolution, so the real-video anchor is a --pattern switch, not a second tree.
case "${ROLE}" in
  pred) PATTERN='*[0-9].mp4' ;;
  gt)   PATTERN='*_gt.mp4' ;;
  *)    echo "ERROR: ROLE must be 'pred' or 'gt', got '${ROLE}'" >&2; exit 1 ;;
esac
PATTERN="${HAL_PATTERN:-${PATTERN}}"

cleanup() {
  local d=$(( SECONDS - START_TIME ))
  printf "[hal] total wall time: %02dh:%02dm:%02ds\n" $((d/3600)) $(((d%3600)/60)) $((d%60))
}
trap cleanup EXIT INT TERM

score_dir() {
  local embodiment="$1" videos="$2" out_dir="$3"
  local thr ckpt
  thr="$(thr_for "${embodiment}")"
  ckpt="$(ckpt_for "${embodiment}")"
  [ -n "${thr}" ] || { echo "[hal] SKIP unknown embodiment '${embodiment}'" >&2; return 1; }
  [ -f "${ckpt}" ] || {
    echo "[hal] SKIP ${embodiment}: no reader checkpoint at ${ckpt}" >&2
    echo "      see metrics/kiva/README.md#install" >&2; return 1; }
  [ -f "${thr}" ] || { echo "[hal] SKIP ${embodiment}: no thresholds at ${thr}" >&2; return 1; }

  local n
  n="$(find "${videos}" -name "${PATTERN}" 2>/dev/null | wc -l)"
  [ "${n}" -gt 0 ] || { echo "[hal] SKIP ${out_dir##*/}: no ${PATTERN} under ${videos}" >&2; return 1; }

  echo "----------------------------------------------------------------------"
  echo "[hal] ckpt:    ${ckpt}"
  echo "[hal] thr:     ${thr}"
  echo "[hal] videos:  ${videos}  (${n} x ${PATTERN})"
  echo "[hal] out:     ${out_dir}"
  mkdir -p "${out_dir}"

  local args=(
    -m kinescore.cli.main score
    --model "${ckpt}" --threshold "${thr}" --device cuda
    --videos "${videos}" --pattern "${PATTERN}"
    --out "${out_dir}"
    --frame-chunk "${FRAME_CHUNK}"
  )
  [ "${MAX_CLIPS}" -gt 0 ] && args+=(--limit "${MAX_CLIPS}")
  [ "${MAX_FRAMES}" -gt 0 ] && args+=(--max-frames "${MAX_FRAMES}")

  ( cd "${HAL_ROOT}" && "${HAL_PYTHON}" "${args[@]}" ) \
    || echo "[hal] WARN failed: ${videos}" >&2
}

# ----------------------------------------------------------------------
# 4a. Single explicit directory
# ----------------------------------------------------------------------
if [ -n "${VIDEOS_DIR}" ]; then
  [ "${#EMBODIMENTS[@]}" -eq 1 ] || {
    echo "ERROR: VIDEOS_DIR mode needs exactly one EMBODIMENTS value" >&2; exit 1; }
  OUT_DIR="${OUT_DIR:-${HAL_OUT_ROOT}/single/$(basename "${VIDEOS_DIR}").${ROLE}}"
  score_dir "${EMBODIMENTS[0]}" "${VIDEOS_DIR}" "${OUT_DIR}"
  echo "[hal] wrote ${OUT_DIR}"
  exit 0
fi

# ----------------------------------------------------------------------
# 4b. Multi-variant sweep
# ----------------------------------------------------------------------
TOTAL=0
for variant in "${VARIANTS[@]}"; do
  echo "======================================================================"
  echo "[hal] variant: ${variant}   role: ${ROLE}"
  echo "======================================================================"
  for embodiment in "${EMBODIMENTS[@]}"; do
    for split in "${SPLITS[@]}"; do
      videos="${HAL_DATA_ROOT}/${variant}/${embodiment}/output/singleview/dreamgen/${split}"
      [ -d "${videos}" ] || { echo "[hal] skip (no dir): ${videos}" >&2; continue; }
      out_dir="${HAL_OUT_ROOT}/${ROLE}/${variant}.${split}.$(tag_for "${embodiment}")"
      score_dir "${embodiment}" "${videos}" "${out_dir}" && TOTAL=$((TOTAL + 1))
    done
  done
done

echo ""
echo "[hal] scored ${TOTAL} (variant, embodiment, split) cells -> ${HAL_OUT_ROOT}/${ROLE}"
echo "[hal] aggregate with:"
echo "  python metrics/kiva/aggregate.py \\"
echo "    --results-root ${HAL_OUT_ROOT} --variants ${VARIANTS[*]}"
