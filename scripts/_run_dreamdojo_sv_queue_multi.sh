#!/usr/bin/env bash
# Generic queue runner for DreamDojo SINGLEVIEW sparse-attention full runs, on ONE GPU.
#
# Jobs are "<embodiment>:<method>:<split>", e.g. "bimanual:radial:non_makovian" or
# "single_arm:pisa:makovian". Jobs run sequentially so per-episode timings stay clean;
# the underlying run scripts are resumable (they skip episodes whose full_pred.mp4
# already exists), so this is safe to re-launch at any time.
#
# Usage:
#   bash scripts/_run_dreamdojo_sv_queue_multi.sh <gpu> <emb:method:split> [...]
#
# Example:
#   bash scripts/_run_dreamdojo_sv_queue_multi.sh 0 single_arm:pisa:makovian
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU="$1"; shift
if [[ $# -eq 0 ]]; then
  echo "usage: $0 <gpu> <emb:method:split> [emb:method:split...]" >&2
  exit 2
fi

# These leak in via the shell/tmux global env and would silently make a run skip
# most episodes. Clear them defensively.
unset NUM_SHARDS SHARD_ID NUM_EPISODES SPLITS SAVE_ROOT OUTPUT_ROOT

LOG_DIR="${LOG_DIR:-${ROOT}/exp_logs/dreamdojo_sparse_sv_multi}"
mkdir -p "${LOG_DIR}"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"

# Expected episode count = number of episode parquet files for that split.
# bimanual nests one level deeper (per-task dirs); single_arm has data/ directly.
expected_episodes() {
  local emb="$1" split="$2" dir="${ROOT}/datasets/$1/singleview/$2"
  if [[ "${emb}" == "bimanual" ]]; then
    find "${dir}" -mindepth 3 -maxdepth 4 -path '*/data/*' -name '*.parquet' 2>/dev/null | wc -l
  else
    find "${dir}/data" -name '*.parquet' 2>/dev/null | wc -l
  fi
}

done_episodes() {
  find "${ROOT}/sampling_dataset/$2/$1/output/singleview/dreamdojo/$3" \
       -name full_pred.mp4 2>/dev/null | wc -l
}

for job in "$@"; do
  IFS=':' read -r emb method split <<< "${job}"
  script="${ROOT}/scripts/run_${emb}_singleview_dreamdojo_${method}.sh"

  if [[ ! -f "$script" ]]; then
    echo "[gpu${GPU}] MISSING SCRIPT for '${job}' ($script) -- skipping" >&2
    continue
  fi

  log="${LOG_DIR}/${emb}_${method}_${split}.log"
  want="$(expected_episodes "${emb}" "${split}")"

  # Retry loop: these runs get killed externally (~once a day). The run script is
  # resumable, so re-running continues where it stopped.
  for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    have="$(done_episodes "${emb}" "${method}" "${split}")"
    if [[ "${want}" -gt 0 && "${have}" -ge "${want}" ]]; then
      echo "[gpu${GPU}] SKIP  ${job} already complete ${have}/${want}"
      break
    fi

    echo "[gpu${GPU}] START ${job} attempt=${attempt}/${MAX_ATTEMPTS} at ${have}/${want} -> ${log} ($(date -u '+%F %T'))"
    CUDA_VISIBLE_DEVICES="${GPU}" SPLITS="${split}" bash "$script" >> "$log" 2>&1
    rc=$?
    n_done="$(done_episodes "${emb}" "${method}" "${split}")"

    if [[ $rc -eq 0 && ( "${want}" -eq 0 || "${n_done}" -ge "${want}" ) ]]; then
      echo "[gpu${GPU}] OK    ${job} rc=0 episodes_with_output=${n_done}/${want} ($(date -u '+%F %T'))"
      break
    fi

    echo "[gpu${GPU}] RETRY ${job} rc=${rc} episodes_with_output=${n_done}/${want} ($(date -u '+%F %T'))" >&2
    tail -5 "$log" >&2
    if [[ "${attempt}" -eq "${MAX_ATTEMPTS}" ]]; then
      echo "[gpu${GPU}] FAIL  ${job} gave up after ${MAX_ATTEMPTS} attempts at ${n_done}/${want} -- see ${log}" >&2
    fi
    sleep 10
  done
done

echo "[gpu${GPU}] QUEUE_COMPLETE ($(date -u '+%F %T'))"
