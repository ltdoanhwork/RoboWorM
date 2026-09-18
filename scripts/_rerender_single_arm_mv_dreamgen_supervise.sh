#!/usr/bin/env bash
# =============================================================================
# _rerender_single_arm_mv_dreamgen_supervise.sh
#
# Supervisor for _rerender_single_arm_mv_dreamgen_dense.sh.
#
# WHY THIS EXISTS: the first full run (2026-09-17 07:17 -> 22:00, 92/257 done)
# was SIGKILLed with no traceback when the Determined shell container was
# restarted (pid 1 age reset to 5h vs 8d host uptime; sshd port 3355 -> 3376;
# /tmp batch json vanished). Nothing in userspace survives a container kill, so
# instead of trying to prevent it we make recovery free:
#   - the underlying batch script already skips episodes with a complete
#     mp4 + .txt, so every retry resumes exactly where it stopped;
#   - all 92 files written before the kill verified clean (768x432, 10/1, 93
#     frames, no 0-byte, no missing sidecar) because the pipeline only writes
#     after a full generation -- there is no partial-file risk;
#   - this loop retries automatically so a transient death costs at most the
#     episode in flight, not the run.
#
# Run it under tmux so it is detached from any tool/ssh shell:
#   tmux new-session -d -s samv 'bash scripts/_rerender_single_arm_mv_dreamgen_supervise.sh'
#
# Env: MAX_ATTEMPTS (default 40), GPU (default 0)
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-40}"
GPU="${GPU:-0}"
OUT_ROOT="$ROOT/sampling_dataset/dense/single_arm/output/multiview/dreamgen"
LOG_DIR="$ROOT/logs/samv_refix"
mkdir -p "$LOG_DIR"

# Expected episode counts for this cell (excel scope: 128 + 129 = 257).
declare -A WANT=( [makovian]=128 [non_makovian]=129 )

done_count() {  # $1 = category
  find "${OUT_ROOT}/$1" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l
}

all_done() {
  for c in "${!WANT[@]}"; do
    [ "$(done_count "$c")" -ge "${WANT[$c]}" ] || return 1
  done
  return 0
}

echo "############################################################"
echo "# Supervisor start $(date -u '+%F %T UTC')"
echo "# makovian     $(done_count makovian)/${WANT[makovian]}"
echo "# non_makovian $(done_count non_makovian)/${WANT[non_makovian]}"
echo "############################################################"

for (( attempt=1; attempt<=MAX_ATTEMPTS; attempt++ )); do
  if all_done; then
    echo "[supervisor] all categories complete."
    break
  fi

  ts="$(date -u '+%Y%m%d_%H%M%S')"
  log="${LOG_DIR}/attempt_${attempt}_${ts}.log"
  echo ""
  echo "[supervisor] attempt ${attempt}/${MAX_ATTEMPTS} at $(date -u '+%F %T UTC')"
  echo "[supervisor]   makovian $(done_count makovian)/${WANT[makovian]}"
  echo "[supervisor]   non_makovian $(done_count non_makovian)/${WANT[non_makovian]}"
  echo "[supervisor]   log -> ${log}"

  before_m=$(done_count makovian); before_n=$(done_count non_makovian)

  GPU="${GPU}" bash scripts/_rerender_single_arm_mv_dreamgen_dense.sh > "${log}" 2>&1
  rc=$?

  after_m=$(done_count makovian); after_n=$(done_count non_makovian)
  made=$(( (after_m - before_m) + (after_n - before_n) ))
  echo "[supervisor] attempt ${attempt} exited rc=${rc}, produced ${made} episode(s)"

  if all_done; then
    echo "[supervisor] all categories complete."
    break
  fi

  # rc==0 but nothing produced => a real blocker (bad path/env), not a kill.
  # Retrying would spin, so stop and leave the log for inspection.
  if [ "$rc" -eq 0 ] && [ "$made" -eq 0 ]; then
    echo "[supervisor] STOP: script succeeded but produced nothing. See ${log}"
    exit 1
  fi

  sleep 20
done

echo ""
echo "############################################################"
echo "# Supervisor end $(date -u '+%F %T UTC')"
echo "# makovian     $(done_count makovian)/${WANT[makovian]}"
echo "# non_makovian $(done_count non_makovian)/${WANT[non_makovian]}"
echo "############################################################"
