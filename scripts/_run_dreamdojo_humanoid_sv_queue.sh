#!/usr/bin/env bash
# Run a queue of DreamDojo humanoid-singleview sparse-attention jobs on ONE GPU.
#
# Each job is "<method>:<split>", e.g. "radial:makovian". Jobs run sequentially so
# per-episode timings stay clean; the underlying run scripts are resumable (they
# skip episodes whose full_pred.mp4 already exists), so this is safe to re-launch.
#
# Usage:
#   bash scripts/_run_dreamdojo_humanoid_sv_queue.sh <gpu> <job> [job...]
#
# Example:
#   bash scripts/_run_dreamdojo_humanoid_sv_queue.sh 0 radial:makovian svg2:makovian
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU="$1"; shift
if [[ $# -eq 0 ]]; then
  echo "usage: $0 <gpu> <method:split> [method:split...]" >&2
  exit 2
fi

# These leak in via the shell/tmux global env and would silently make a run skip
# most episodes. Clear them defensively.
unset NUM_SHARDS SHARD_ID NUM_EPISODES SPLITS SAVE_ROOT OUTPUT_ROOT

LOG_DIR="${LOG_DIR:-${ROOT}/logs/exp_logs/dreamdojo_sparse_sv}"

for job in "$@"; do
  method="${job%%:*}"
  split="${job##*:}"
  script="${ROOT}/scripts/run_humanoid_singleview_dreamdojo_${method}.sh"

  if [[ ! -f "$script" ]]; then
    echo "[gpu${GPU}] MISSING SCRIPT for method='${method}' ($script) -- skipping" >&2
    continue
  fi

  log="${LOG_DIR}/${method}_${split}.log"
  echo "[gpu${GPU}] START ${method}/${split} -> ${log} ($(date '+%F %T'))"

  CUDA_VISIBLE_DEVICES="${GPU}" SPLITS="${split}" bash "$script" > "$log" 2>&1
  rc=$?

  n_done=$(find "${ROOT}/sampling_dataset/${method}/humanoid/output/singleview/dreamdojo/${split}" \
             -name full_pred.mp4 2>/dev/null | wc -l)
  if [[ $rc -eq 0 ]]; then
    echo "[gpu${GPU}] OK    ${method}/${split} rc=0 episodes_with_output=${n_done} ($(date '+%F %T'))"
  else
    echo "[gpu${GPU}] FAIL  ${method}/${split} rc=${rc} episodes_with_output=${n_done} -- see ${log}" >&2
    tail -15 "$log" >&2
  fi
done

echo "[gpu${GPU}] QUEUE_COMPLETE ($(date '+%F %T'))"
