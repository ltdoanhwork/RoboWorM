#!/usr/bin/env bash
# Run a queue of DreamDojo bimanual-singleview sparse-attention jobs on ONE GPU.
#
# Each job is "<method>:<split>", e.g. "radial:makovian". Jobs run sequentially so
# per-episode timings stay clean; the underlying run scripts are resumable (they
# skip episodes whose full_pred.mp4 already exists), so this is safe to re-launch.
#
# Usage:
#   bash scripts/_run_dreamdojo_bimanual_sv_queue.sh <gpu> <job> [job...]
#
# Example:
#   bash scripts/_run_dreamdojo_bimanual_sv_queue.sh 0 radial:makovian svg2:makovian
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

LOG_DIR="${LOG_DIR:-${ROOT}/logs/exp_logs/dreamdojo_sparse_bm_sv}"
mkdir -p "${LOG_DIR}"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-6}"

# Expected episode count for a split = number of episode parquet files across its tasks.
expected_episodes() {
  local split="$1"
  find "${ROOT}/datasets/bimanual/singleview/${split}" -mindepth 3 -maxdepth 4 \
       -path '*/data/*' -name '*.parquet' 2>/dev/null | wc -l
}

done_episodes() {
  local method="$1" split="$2"
  find "${ROOT}/sampling_dataset/${method}/bimanual/output/singleview/dreamdojo/${split}" \
       -name full_pred.mp4 2>/dev/null | wc -l
}

for job in "$@"; do
  method="${job%%:*}"
  split="${job##*:}"
  script="${ROOT}/scripts/run_bimanual_singleview_dreamdojo_${method}.sh"

  if [[ ! -f "$script" ]]; then
    echo "[gpu${GPU}] MISSING SCRIPT for method='${method}' ($script) -- skipping" >&2
    continue
  fi

  log="${LOG_DIR}/${method}_${split}.log"
  want="$(expected_episodes "${split}")"

  # Retry loop: these runs get killed externally (session reaper) roughly once a day.
  # The run script is resumable, so re-running continues where it stopped.
  for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    have="$(done_episodes "${method}" "${split}")"
    if [[ "${want}" -gt 0 && "${have}" -ge "${want}" ]]; then
      echo "[gpu${GPU}] SKIP  ${method}/${split} already complete ${have}/${want}"
      break
    fi

    echo "[gpu${GPU}] START ${method}/${split} attempt=${attempt}/${MAX_ATTEMPTS} at ${have}/${want} -> ${log} ($(date '+%F %T'))"
    CUDA_VISIBLE_DEVICES="${GPU}" SPLITS="${split}" bash "$script" >> "$log" 2>&1
    rc=$?
    n_done="$(done_episodes "${method}" "${split}")"

    if [[ $rc -eq 0 && ( "${want}" -eq 0 || "${n_done}" -ge "${want}" ) ]]; then
      echo "[gpu${GPU}] OK    ${method}/${split} rc=0 episodes_with_output=${n_done}/${want} ($(date '+%F %T'))"
      break
    fi

    echo "[gpu${GPU}] RETRY ${method}/${split} rc=${rc} episodes_with_output=${n_done}/${want} ($(date '+%F %T'))" >&2
    tail -5 "$log" >&2
    if [[ "${attempt}" -eq "${MAX_ATTEMPTS}" ]]; then
      echo "[gpu${GPU}] FAIL  ${method}/${split} gave up after ${MAX_ATTEMPTS} attempts at ${n_done}/${want} -- see ${log}" >&2
    fi
    sleep 10
  done
done

echo "[gpu${GPU}] QUEUE_COMPLETE ($(date '+%F %T'))"
