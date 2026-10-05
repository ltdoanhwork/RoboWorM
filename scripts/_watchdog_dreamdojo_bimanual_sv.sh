#!/usr/bin/env bash
# Watchdog for the DreamDojo bimanual-singleview sparse-attention full run.
#
# WHY: the per-GPU queue processes get killed externally (~once a day; happened
# 2026-08-22 22:02 UTC and cost 5.2h of idle GPU before a manual relaunch). The
# queue script retries internally, but it cannot recover from being killed itself.
# This watchdog re-launches any dead queue, so the full run finishes unattended.
#
# There is no cron on this box, so run it detached:
#   setsid nohup bash scripts/_watchdog_dreamdojo_bimanual_sv.sh \
#     > logs/exp_logs/dreamdojo_sparse_bm_sv/watchdog.log 2>&1 < /dev/null & disown
#
# Single-instance (flock). Exits once every job in both queues is complete.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

unset NUM_SHARDS SHARD_ID NUM_EPISODES SPLITS SAVE_ROOT OUTPUT_ROOT

LOG_DIR="${LOG_DIR:-${ROOT}/logs/exp_logs/dreamdojo_sparse_bm_sv}"
mkdir -p "${LOG_DIR}"

LOCK="${LOG_DIR}/watchdog.lock"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "another watchdog already holds ${LOCK} -- exiting"
  exit 0
fi

INTERVAL="${INTERVAL:-300}"
QUEUE="${ROOT}/scripts/_run_dreamdojo_bimanual_sv_queue.sh"

# Job lists per GPU (same split as the live run: radial+svg1 on gpu0, pisa+svg2 on gpu1).
GPU0_JOBS="${GPU0_JOBS:-radial:non_makovian svg1:makovian svg1:non_makovian}"
GPU1_JOBS="${GPU1_JOBS:-pisa:non_makovian svg2:makovian svg2:non_makovian}"

log() { echo "[watchdog $(date -u '+%F %T')] $*"; }

expected_episodes() {
  find "${ROOT}/datasets/bimanual/singleview/$1" -mindepth 3 -maxdepth 4 \
       -path '*/data/*' -name '*.parquet' 2>/dev/null | wc -l
}

done_episodes() {
  find "${ROOT}/sampling_dataset/$1/bimanual/output/singleview/dreamdojo/$2" \
       -name full_pred.mp4 2>/dev/null | wc -l
}

queue_alive() {
  pgrep -f "_run_dreamdojo_bimanual_sv_queue.sh $1 " >/dev/null 2>&1
}

# True when every job in the list has done >= expected.
jobs_complete() {
  local job method split want have
  for job in $1; do
    method="${job%%:*}"; split="${job##*:}"
    want="$(expected_episodes "${split}")"
    have="$(done_episodes "${method}" "${split}")"
    [[ "${want}" -gt 0 && "${have}" -ge "${want}" ]] || return 1
  done
  return 0
}

launch() {
  local gpu="$1" jobs="$2"
  log "relaunching gpu${gpu} queue: ${jobs}"
  # shellcheck disable=SC2086
  setsid nohup bash "${QUEUE}" "${gpu}" ${jobs} \
    >> "${LOG_DIR}/queue_gpu${gpu}.log" 2>&1 < /dev/null &
  disown
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
      log "gpu${gpu} queue DEAD"
      for job in ${jobs}; do
        m="${job%%:*}"; s="${job##*:}"
        log "  gpu${gpu} ${m}/${s} = $(done_episodes "$m" "$s")/$(expected_episodes "$s")"
      done
      launch "${gpu}" "${jobs}"
    fi
  done

  if [[ "${all_done}" -eq 1 ]]; then
    log "ALL JOBS COMPLETE -- exiting"
    exit 0
  fi

  sleep "${INTERVAL}"
done
