#!/usr/bin/env bash
# One-off orchestrator: run FULL bimanual 4-view-grid Ctrl-World generation for all
# 6 acceleration methods, each sharded across 4 GPUs, sequentially (method-by-method
# so the 4 GPUs are fully dedicated to one method at a time). Each method uses its
# TUNED default config baked into its wrapper (SiTo k0.8/d4, ITM k0.85/d4, sparse at
# measured configs). Waits for a method's 4 shard tmux sessions to finish, merges its
# shard summaries, then starts the next method.
#
# Usage: tmux new-session -d -s cwb4_full "bash scripts/run/ctrlworld/run_bimanual_4view_all_methods_full.sh 2>&1 | tee logs/cwb4_full.log"
set -uo pipefail
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

METHODS=(${METHODS_OVERRIDE:-sito itm pisa radial svg1 svg2})
GPUS="${GPUS:-0,1,2,3}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SPLIT="${SPLIT:-all}"

for m in "${METHODS[@]}"; do
  echo "############################################################"
  echo "### METHOD: $m  ($(date '+%F %T'))"
  echo "############################################################"
  wrapper="scripts/run/ctrlworld/run_bimanual_multiview_ctrlworld_4view_grid_${m}.sh"
  out="${ROOT}/sampling_dataset/${m}/bimanual/output/multiview/ctrlworld_4view_grid"

  # launch 4 shards (creates tmux sessions cw_b4*_s0..s3). NOTE: the wrapper's
  # shard launcher ends each session with `exec bash`, so the tmux session STAYS
  # ALIVE after the job finishes. We therefore wait on the shard LOG files printing
  # "Done!" (one per shard), NOT on the tmux sessions disappearing, then kill the
  # now-idle sessions ourselves before moving to the next method.
  GPUS="$GPUS" NUM_SHARDS="$NUM_SHARDS" SPLIT="$SPLIT" bash "$wrapper" --launch-shards

  case "$m" in
    sito)   pfx=cw_b4st;   lg=sito ;;
    itm)    pfx=cw_b4itm;  lg=itm ;;
    pisa)   pfx=cw_b4pisa; lg=pisa ;;
    radial) pfx=cw_b4rad;  lg=radial ;;
    svg1)   pfx=cw_b4svg1; lg=svg1 ;;
    svg2)   pfx=cw_b4svg2; lg=svg2 ;;
  esac
  sleep 20
  while true; do
    done_n=$(grep -lE 'Done! Results saved to' \
      "${ROOT}"/logs/exp_logs/tmux_logs/cw_bim_grid_4view_${lg}_shard*.log 2>/dev/null | wc -l)
    [ "$done_n" -ge "$NUM_SHARDS" ] && break
    sleep 60
  done
  for s in $(tmux ls 2>/dev/null | grep -oE "${pfx}_s[0-9]+"); do tmux kill-session -t "$s" 2>/dev/null; done

  echo "### $m shards done -> merging summaries"
  python scripts/tools/ops/merge_ctrlworld_shard_summaries.py --output-root "$out" || true
  echo "### $m FULL DONE ($(date '+%F %T')) -> $(find "$out" -name full_pred.mp4 2>/dev/null | wc -l) episodes"
done

echo "ALL_METHODS_FULL_DONE ($(date '+%F %T'))"
