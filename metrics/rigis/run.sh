#!/bin/bash
# RIGIS -- multi-view consistency metric (VGGT-Omega "rigid geometry" variant).
#
# Scores one or more method variants with the VGGT-Omega multiview-consistency
# metric vendored in metrics/rigis/ from branch
# `thanhnv/multiview-consistency-indep` @8015d97 (see metrics/rigis/VENDORED.md
# and docs/metrics/rigis.md). Emits one CSV per episodes-root, plus a
# cross-variant comparison via metrics/rigis/aggregate.py.
#
# The metric consumes EXTRACTED FRAMES, not mp4s:
#   <episodes_root>/<episode_id>/view1/frame_0000.png
#   <episodes_root>/<episode_id>/view2/frame_0000.png
# Use metrics/rigis/extract_frames.py to go from mosaic mp4s to
# that layout.
#
# Usage:
#   # score the baseline + 3 accelerated variants in ONE invocation
#   VARIANTS="dense pisa radial svg1" bash metrics/rigis/run.sh
#
#   # smoke test: 1 episode, one subtree
#   MAX_EPISODES=1 VARIANTS=dense SUBTREE=bimanual/ctrlworld/non_makovian \
#     bash metrics/rigis/run.sh
#
#   # single explicit episode folder
#   EPISODE_DIR=/path/to/episode_000025 bash metrics/rigis/run.sh
#
#   # long sweep in tmux
#   VARIANTS="dense pisa radial svg1 svg2" bash metrics/rigis/run.sh --tmux
#
# Env:
#   MVC_ROOT          uv project holding the metric code (default: metrics/rigis, the
#                     vendored copy; its .venv lives at metrics/rigis/.venv)
#   MVC_CKPT          VGGT-Omega weights (default: checkpoints/metrics/vggt_omega/vggt_omega_1b_512.pt,
#                     falling back to the legacy checkpoints/vggt_omega/ path)
#   MVC_PYTHON        python to use (default: `uv run --project $MVC_ROOT --with numba python`)
#   MVC_DATA_ROOT     root of frame-extracted episodes (default: $ROOT/sampling_dataset/multiview_consistency/frames)
#   MVC_OUT_ROOT      where CSVs land (default: $ROOT/results/multiview_consistency)
#   VARIANTS          space-separated variant dir names under MVC_DATA_ROOT (default: dense)
#   SUBTREE           optional relative subtree inside each variant dir (e.g. bimanual/ctrlworld/non_makovian)
#   VIEWS             comma-separated view folder names (default: view1,view2)
#   FRAME_BATCH_SIZE  VGGT timesteps per forward (default: 4)
#   MAX_EPISODES      cap episodes per episodes-root via a symlink staging dir (default: all)
#   MVC_PORT          model-service port (default: 6111)
#   MVC_DTYPE         float16 | bfloat16 | float32 (default: float16)
#   MVC_COMPILE       1 to enable torch.compile on the aggregator (default: 0)
#   EPISODE_DIR       score exactly one episode folder and exit

set -euo pipefail
START_TIME=$SECONDS

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

MVC_ROOT="${MVC_ROOT:-${ROOT}/metrics/rigis}"
if [ -z "${MVC_CKPT:-}" ]; then
  MVC_CKPT="${ROOT}/checkpoints/metrics/vggt_omega/vggt_omega_1b_512.pt"
  # legacy location (pre-2026-10-05 refactor), kept as a fallback
  [ -f "${MVC_CKPT}" ] || [ ! -f "${ROOT}/checkpoints/vggt_omega/vggt_omega_1b_512.pt" ] \
    || MVC_CKPT="${ROOT}/checkpoints/vggt_omega/vggt_omega_1b_512.pt"
fi
MVC_DATA_ROOT="${MVC_DATA_ROOT:-${ROOT}/sampling_dataset/multiview_consistency/frames}"
MVC_OUT_ROOT="${MVC_OUT_ROOT:-${ROOT}/results/multiview_consistency}"
MVC_HOST="${MVC_HOST:-127.0.0.1}"
MVC_PORT="${MVC_PORT:-6111}"
MVC_AUTHKEY="${MVC_AUTHKEY:-vggt_omega}"
MVC_DTYPE="${MVC_DTYPE:-float16}"
MVC_COMPILE="${MVC_COMPILE:-0}"
VIEWS="${VIEWS:-view1,view2}"
FRAME_BATCH_SIZE="${FRAME_BATCH_SIZE:-4}"
MAX_EPISODES="${MAX_EPISODES:-0}"
SUBTREE="${SUBTREE:-}"
EPISODE_DIR="${EPISODE_DIR:-}"
VARIANTS=(${VARIANTS:-dense})

# ----------------------------------------------------------------------
# tmux fan-out (one session, long sweeps must not block the caller)
# ----------------------------------------------------------------------
if [ "${1:-}" = "--tmux" ]; then
  SESSION="${SESSION:-mvc_$(date +%H%M%S)}"
  mkdir -p "${ROOT}/exp_logs/tmux_logs"
  LOG="${ROOT}/exp_logs/tmux_logs/${SESSION}.log"
  tmux new-session -d -s "${SESSION}" \
    "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-0}' MVC_ROOT='${MVC_ROOT}' MVC_CKPT='${MVC_CKPT}' \
     MVC_PYTHON='${MVC_PYTHON:-}' MVC_DATA_ROOT='${MVC_DATA_ROOT}' MVC_OUT_ROOT='${MVC_OUT_ROOT}' \
     VARIANTS='${VARIANTS[*]}' SUBTREE='${SUBTREE}' VIEWS='${VIEWS}' \
     FRAME_BATCH_SIZE='${FRAME_BATCH_SIZE}' MAX_EPISODES='${MAX_EPISODES}' MVC_PORT='${MVC_PORT}' \
     MVC_DTYPE='${MVC_DTYPE}' MVC_COMPILE='${MVC_COMPILE}' \
     bash metrics/rigis/run.sh 2>&1 | tee '${LOG}'; exec bash"
  echo "launched tmux session ${SESSION} (log ${LOG})"
  echo "  attach:  tmux attach -t ${SESSION}"
  exit 0
fi

# ----------------------------------------------------------------------
# 0. Metric code + checkpoint + interpreter
# ----------------------------------------------------------------------
[ -f "${MVC_ROOT}/multiview_consistency/run_video_metrics.py" ] || {
  echo "ERROR: metric code not found under MVC_ROOT=${MVC_ROOT}" >&2
  echo "       (expected the vendored copy in metrics/rigis/, see VENDORED.md)" >&2
  exit 1
}

[ -f "${MVC_CKPT}" ] || {
  echo "ERROR: VGGT-Omega checkpoint not found: ${MVC_CKPT}" >&2
  echo "       Set MVC_CKPT, or place/symlink vggt_omega_1b_512.pt there." >&2
  echo "       See docs/metrics/rigis.md#checkpoints" >&2
  exit 1
}

if [ -n "${MVC_PYTHON:-}" ]; then
  # shellcheck disable=SC2206
  PY=(${MVC_PYTHON})
else
  command -v uv >/dev/null || { echo "ERROR: uv not found and MVC_PYTHON unset" >&2; exit 1; }
  # `numba` is imported by multiview_consistency/geometric_consistency.py but is
  # NOT declared in the branch's pyproject.toml/uv.lock -> inject it here.
  # See docs/metrics/rigis.md#known-gotchas
  PY=(uv run --project "${MVC_ROOT}")
  for dep in ${MVC_UV_EXTRA_DEPS:-numba}; do PY+=(--with "${dep}"); done
  PY+=(python)
fi
echo "[mvc] interpreter: ${PY[*]}"

# ----------------------------------------------------------------------
# Helpers: episode discovery (mirrors the upstream run.sh contract)
# ----------------------------------------------------------------------
PRIMARY_VIEW="${VIEWS%%,*}"
SECOND_VIEW="${VIEWS#*,}"; SECOND_VIEW="${SECOND_VIEW%%,*}"

is_episode_dir() {
  [ -d "$1/${PRIMARY_VIEW}" ] && [ -d "$1/${SECOND_VIEW}" ]
}
has_episode_children() {
  local child
  for child in "$1"/*/; do
    [ -d "$child" ] || continue
    is_episode_dir "${child%/}" && return 0
  done
  return 1
}
# Print every directory whose immediate children are episode folders.
find_episode_roots() {
  local dir="$1" child
  [ -d "$dir" ] || return 0
  if has_episode_children "$dir"; then echo "$dir"; return 0; fi
  for child in "$dir"/*/; do
    [ -d "$child" ] || continue
    find_episode_roots "${child%/}"
  done
}

STAGE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mvc_stage.XXXXXX")"
cleanup_stage() { rm -rf "${STAGE_ROOT}"; }

# When MAX_EPISODES>0, stage a symlink-only copy of the first N episodes so the
# upstream batch client (which has no --max_episodes) sees a smaller root.
stage_root() {
  local src="$1" n="$2" out i=0 child
  out="${STAGE_ROOT}/$(echo "$src" | md5sum | cut -c1-12)"
  mkdir -p "$out"
  for child in "$src"/*/; do
    [ -d "$child" ] || continue
    is_episode_dir "${child%/}" || continue
    ln -sfn "$(cd "${child%/}" && pwd)" "${out}/$(basename "${child%/}")"
    i=$((i + 1))
    [ "$i" -ge "$n" ] && break
  done
  echo "$out"
}

# ----------------------------------------------------------------------
# 1. Model service (loads the 4.6 GB checkpoint once for the whole sweep)
# ----------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
COMPILE_FLAG="--no-compile"
[ "${MVC_COMPILE}" = "1" ] && COMPILE_FLAG="--compile"

echo "======================================================================"
echo "[mvc] starting VGGT-Omega model service on ${MVC_HOST}:${MVC_PORT} (${COMPILE_FLAG})"
echo "      ckpt: ${MVC_CKPT}"
echo "======================================================================"
# `exec` so SERVICE_PID is the launcher itself, not a subshell whose death would orphan
# `uv run` + the 4.9 GB service on the GPU (seen 2026-10-06: the orphan made every later
# leased row on that GPU refuse to start).
( cd "${MVC_ROOT}" && exec "${PY[@]}" -m multiview_consistency.model_service \
    --host "${MVC_HOST}" --port "${MVC_PORT}" --authkey "${MVC_AUTHKEY}" \
    --dtype "${MVC_DTYPE}" \
    --omega_checkpoint "${MVC_CKPT}" \
    --omega_image_resolution "${MVC_IMAGE_RESOLUTION:-512}" \
    --omega_patch_size "${MVC_PATCH_SIZE:-16}" \
    ${COMPILE_FLAG} ) &
SERVICE_PID=$!

cleanup() {
  kill "${SERVICE_PID}" 2>/dev/null || true
  # uv run may not forward the signal to its python child: kill this run's service by port.
  pkill -TERM -f "multiview_consistency.model_service --host ${MVC_HOST} --port ${MVC_PORT} " 2>/dev/null || true
  wait "${SERVICE_PID}" 2>/dev/null || true
  cleanup_stage
  local d=$(( SECONDS - START_TIME ))
  printf "[mvc] total wall time: %02dh:%02dm:%02ds\n" $((d/3600)) $(((d%3600)/60)) $((d%60))
}
trap cleanup EXIT INT TERM

echo "[mvc] waiting for model service ..."
until "${PY[@]}" - <<EOF 2>/dev/null
from multiprocessing.connection import Client
c = Client(("${MVC_HOST}", ${MVC_PORT}), authkey=b"${MVC_AUTHKEY}")
c.send({"cmd": "ping"}); print(c.recv()); c.close()
EOF
do
  kill -0 "${SERVICE_PID}" 2>/dev/null || { echo "ERROR: model service died before becoming ready" >&2; exit 1; }
  sleep 5
done
echo "[mvc] model service ready"
echo ""

run_metrics() {
  local episodes_root="$1" out_csv="$2"
  mkdir -p "$(dirname "${out_csv}")"
  echo "----------------------------------------------------------------------"
  echo "[mvc] episodes_root: ${episodes_root}"
  echo "[mvc] out_csv:       ${out_csv}"
  ( cd "${MVC_ROOT}" && "${PY[@]}" -m multiview_consistency.run_video_metrics \
      --episodes_root "${episodes_root}" \
      --out_csv "${out_csv}" \
      --views "${VIEWS}" \
      --frame_batch_size "${FRAME_BATCH_SIZE}" \
      --host "${MVC_HOST}" --port "${MVC_PORT}" --authkey "${MVC_AUTHKEY}" ) \
    || echo "[mvc] WARN failed: ${episodes_root}" >&2
}

# ----------------------------------------------------------------------
# 2a. Single-episode mode
# ----------------------------------------------------------------------
if [ -n "${EPISODE_DIR}" ]; then
  OUT_CSV="${OUT_CSV:-${MVC_OUT_ROOT}/single/$(basename "${EPISODE_DIR}").csv}"
  mkdir -p "$(dirname "${OUT_CSV}")"
  ( cd "${MVC_ROOT}" && "${PY[@]}" -m multiview_consistency.run_video_metrics \
      --episode_dir "${EPISODE_DIR}" \
      --out_csv "${OUT_CSV}" \
      --views "${VIEWS}" \
      --frame_batch_size "${FRAME_BATCH_SIZE}" \
      --host "${MVC_HOST}" --port "${MVC_PORT}" --authkey "${MVC_AUTHKEY}" )
  echo "[mvc] wrote ${OUT_CSV}"
  exit 0
fi

# ----------------------------------------------------------------------
# 2b. Multi-variant sweep
# ----------------------------------------------------------------------
TOTAL_ROOTS=0
for variant in "${VARIANTS[@]}"; do
  variant_root="${MVC_DATA_ROOT}/${variant}"
  [ -n "${SUBTREE}" ] && variant_root="${variant_root}/${SUBTREE}"
  if [ ! -d "${variant_root}" ]; then
    echo "[mvc] SKIP variant '${variant}': no such dir ${variant_root}" >&2
    continue
  fi
  echo "======================================================================"
  echo "[mvc] variant: ${variant}   (${variant_root})"
  echo "======================================================================"
  found=0
  while IFS= read -r root; do
    [ -n "${root}" ] || continue
    found=1; TOTAL_ROOTS=$((TOTAL_ROOTS + 1))
    rel="${root#"${MVC_DATA_ROOT}/"}"
    out_csv="${MVC_OUT_ROOT}/${rel%/}.csv"
    scored_root="${root}"
    if [ "${MAX_EPISODES}" -gt 0 ]; then
      scored_root="$(stage_root "${root}" "${MAX_EPISODES}")"
      echo "[mvc] MAX_EPISODES=${MAX_EPISODES} -> staged ${scored_root}"
    fi
    run_metrics "${scored_root}" "${out_csv}"
  done < <(find_episode_roots "${variant_root}")
  [ "${found}" -eq 0 ] && echo "[mvc] WARN no ${PRIMARY_VIEW}+${SECOND_VIEW} episodes under ${variant_root}" >&2
done

echo ""
echo "[mvc] scored ${TOTAL_ROOTS} episodes-roots -> ${MVC_OUT_ROOT}"
echo "[mvc] aggregate with:"
echo "  python metrics/rigis/aggregate.py \\"
echo "    --results-root ${MVC_OUT_ROOT} --variants ${VARIANTS[*]}"
