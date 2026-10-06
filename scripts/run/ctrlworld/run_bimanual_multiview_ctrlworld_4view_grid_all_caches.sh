#!/bin/bash
# Run the 3 TUNED cache methods on the bimanual 4-VIEW 2x2 GRID Ctrl-World model
# back-to-back on a single GPU (WorldCache -> DiCache -> FasterCache).
#
# Each method uses its own tuned defaults (see
# .cursor/rules/ctrlworld-bimanual-multiview-4view-grid-caching-insights.mdc):
#   WorldCache  thresh 0.04 / ret 0.0 / probe 1  -> 1.81x, PSNR 21.67 (dense 22.12)
#   DiCache     thresh 0.03 / ret 0.0 / probe 1  -> 1.72x, PSNR 21.58
#   FasterCache m5 / b3                          -> 1.02x (known no-op, kept as
#                                                   a valid benchmark point)
#
# Methods run with `;` semantics: a failure in one does NOT abort the rest, so a
# single bad method can't cost you the whole queue. Every underlying run script
# is resumable (episodes whose full_pred.mp4 exists are skipped), so re-running
# this driver after an interruption picks up where it left off.
#
# Usage (single GPU, all 300 episodes = val + train, ~64h total):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/ctrlworld/run_bimanual_multiview_ctrlworld_4view_grid_all_caches.sh
#
# Env overrides:
#   METHODS  (default "worldcache dicache fastercache")
#   SPLIT    (default all), NUM_EPISODES, plus any per-method knob
#            (WORLDCACHE_*, DICACHE_*, FASTERCACHE_*) which is passed through.

set -u

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "${ROOT}"

# Drop any per-episode / scratch overrides inherited from the launching shell.
# A stale DATASET_DIR or OUTPUT_ROOT (e.g. a deleted /tmp sweep dir) would make
# every method silently process 0 episodes. The run scripts re-derive these.
unset DATASET_DIR DATASET_NAME OUTPUT_ROOT INPUT_ROOT NUM_EPISODES CKPT
unset NUM_SHARDS SHARD_ID

METHODS="${METHODS:-worldcache dicache fastercache}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
LOGDIR="${ROOT}/logs/exp_logs/tmux_logs"
mkdir -p "${LOGDIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "############################################################"
echo "# Ctrl-World 4-view grid | cache benchmark queue"
echo "#   GPU=${GPU}  split=${SPLIT:-all}  methods: ${METHODS}"
echo "#   started $(date '+%F %T')"
echo "############################################################"

for m in ${METHODS}; do
  script="scripts/run/ctrlworld/run_bimanual_multiview_ctrlworld_4view_grid_${m}.sh"
  log="${LOGDIR}/cw4v_${m}_${STAMP}.log"
  if [ ! -f "${script}" ]; then
    echo "=== [$(date '+%F %T')] SKIP ${m}: ${script} not found"
    continue
  fi
  echo ""
  echo "=== [$(date '+%F %T')] START ${m}"
  echo "===   script: ${script}"
  echo "===   log:    ${log}"
  _t0=$(date +%s)
  CUDA_VISIBLE_DEVICES="${GPU}" bash "${script}" > "${log}" 2>&1
  code=$?
  _dt=$(( $(date +%s) - _t0 ))
  echo "=== [$(date '+%F %T')] DONE ${m} exit=${code} elapsed=$((_dt/3600))h$((_dt%3600/60))m"
  # Surface the per-category means so progress is visible without opening logs.
  find "sampling_dataset/${m}/bimanual/output/multiview/ctrlworld_4view_grid" \
    -name all_summary.json 2>/dev/null | while read -r f; do
      echo "===   $(dirname "${f}" | xargs basename): $(python -c "
import json,sys
d=json.load(open('${f}'))
print(f\"n={d.get('num_episodes')} PSNR={d.get('mean_psnr',0):.3f} SSIM={d.get('mean_ssim',0):.4f} LPIPS={d.get('mean_lpips',0):.4f}\")
" 2>/dev/null)"
    done
done

echo ""
echo "############################################################"
echo "# ALL DONE $(date '+%F %T')"
echo "############################################################"
