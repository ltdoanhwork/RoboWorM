#!/usr/bin/env bash
# Watchdog + chainer for the DreamDojo SINGLEVIEW sparse-attention full runs
# (bimanual first, then single_arm) across both GPUs.
#
# WHY: the per-GPU queue processes get killed externally (~once a day; on
# 2026-08-22 22:02 UTC that cost 5.2h of idle GPU before a manual relaunch). The
# queue scripts retry internally but cannot recover from being killed themselves.
# This watchdog re-launches any dead queue AND chains the next phase (single_arm)
# as soon as a GPU finishes its bimanual jobs, so no GPU ever sits idle.
#
# There is no cron on this box, so run it detached:
#   setsid nohup bash scripts/_watchdog_dreamdojo_sv.sh \
#     > exp_logs/dreamdojo_sparse_sv_multi/watchdog.log 2>&1 < /dev/null & disown
#
# Single-instance (flock). Exits once every job on both GPUs is complete.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

unset NUM_SHARDS SHARD_ID NUM_EPISODES SPLITS SAVE_ROOT OUTPUT_ROOT

LOG_DIR="${LOG_DIR:-${ROOT}/exp_logs/dreamdojo_sparse_sv_multi}"
mkdir -p "${LOG_DIR}"

LOCK="${LOCK:-${LOG_DIR}/watchdog.lock}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "another watchdog already holds ${LOCK} -- exiting"
  exit 0
fi

INTERVAL="${INTERVAL:-300}"
QUEUE="${ROOT}/scripts/_run_dreamdojo_sv_queue_multi.sh"

# Ordered job list per GPU: bimanual leftovers first (already in flight), then the
# single_arm canonical run. Winners (radial/pisa) before the slow negatives (svg1/svg2).
GPU0_JOBS="${GPU0_JOBS:-\
bimanual:radial:non_makovian bimanual:svg1:makovian bimanual:svg1:non_makovian \
single_arm:pisa:makovian single_arm:pisa:non_makovian \
single_arm:svg1:makovian single_arm:svg1:non_makovian}"

GPU1_JOBS="${GPU1_JOBS:-\
bimanual:pisa:non_makovian bimanual:svg2:makovian bimanual:svg2:non_makovian \
single_arm:radial:makovian single_arm:radial:non_makovian \
single_arm:svg2:makovian single_arm:svg2:non_makovian}"

log() { echo "[watchdog $(date -u '+%F %T')] $*"; }

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

# A GPU is "busy" if either the legacy bimanual-only queue or the multi queue is
# running for it (the legacy ones were launched before this watchdog existed).
queue_alive() {
  pgrep -f "_run_dreamdojo_sv_queue_multi.sh $1 " >/dev/null 2>&1 && return 0
  pgrep -f "_run_dreamdojo_bimanual_sv_queue.sh $1 " >/dev/null 2>&1 && return 0
  return 1
}

jobs_complete() {
  local job emb method split want have
  for job in $1; do
    IFS=':' read -r emb method split <<< "${job}"
    want="$(expected_episodes "${emb}" "${split}")"
    have="$(done_episodes "${emb}" "${method}" "${split}")"
    [[ "${want}" -gt 0 && "${have}" -ge "${want}" ]] || return 1
  done
  return 0
}

launch() {
  local gpu="$1" jobs="$2"
  log "launching gpu${gpu} queue"
  # shellcheck disable=SC2086
  setsid nohup bash "${QUEUE}" "${gpu}" ${jobs} \
    >> "${LOG_DIR}/queue_gpu${gpu}.log" 2>&1 < /dev/null &
  disown
}

report() {
  local gpu="$1" jobs="$2" job emb method split
  for job in ${jobs}; do
    IFS=':' read -r emb method split <<< "${job}"
    log "  gpu${gpu} ${job} = $(done_episodes "${emb}" "${method}" "${split}")/$(expected_episodes "${emb}" "${split}")"
  done
}

log "start (interval=${INTERVAL}s)"
log "gpu0: ${GPU0_JOBS}"
log "gpu1: ${GPU1_JOBS}"

while true; do
  all_done=1

  for gpu in 0 1; do
    if [[ "${gpu}" == "0" ]]; then jobs="${GPU0_JOBS}"; else jobs="${GPU1_JOBS}"; fi

    if jobs_complete "${jobs}"; then
      continue
    fi
    all_done=0

    if ! queue_alive "${gpu}"; then
      log "gpu${gpu} queue NOT RUNNING -- (re)starting"
      report "${gpu}" "${jobs}"
      launch "${gpu}" "${jobs}"
    fi
  done

  if [[ "${all_done}" -eq 1 ]]; then
    log "ALL JOBS COMPLETE -- exiting"
    exit 0
  fi

  sleep "${INTERVAL}"
done
