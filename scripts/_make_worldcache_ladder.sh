#!/usr/bin/env bash
# =============================================================================
# _make_worldcache_ladder.sh
# Generate the MISSING rungs of the WorldCache quality ladder for the two
# humanoid DreamGen cells, so a qualitative panel can show
#   dense -> ok (parity) -> degraded (visibly soft) -> broken (collapsed).
#
# The `dense` and `ok` rungs are already published under
# sampling_dataset/data_for_web/ from the two tuning passes; this script only
# produces `degraded` and `broken`, using the SAME episodes and the SAME run
# scripts so every rung is directly comparable.
#
# Ladder knobs (WorldCache): caching begins at step int(num_steps * ret_ratio)
# and a step is skipped when the relative-L1 probe is under rel_l1_thresh. So
# raising thresh and dropping ret_ratio monotonically increases skipping:
#   ret 0.0 + thresh 1.0 = no dense warm-up and skip essentially everything.
#
# Usage (repo root):
#   bash scripts/_make_worldcache_ladder.sh --estimate
#   bash scripts/_make_worldcache_ladder.sh [gpu]
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-${1:-0}}"
[[ "${GPU}" == "--estimate" ]] && GPU=0
# Scratch lives on the workspace filesystem, NOT /tmp: /tmp is container-local and
# gets wiped when the session restarts, which already destroyed one full sweep.
# `tmp/*` is gitignored, so this stays out of version control.
SCRATCH="${SCRATCH:-${ROOT}/tmp/wc_ladder}"

# ---------------------------------------------------------------- SV config --
SV_SCRIPT="scripts/run_humanoid_singleview_dreamgen_worldcache.sh"
SV_EPISODES="episode_000200,episode_000829,episode_001055"
SV_SPLIT="makovian"
# tag|thresh|ret|probe
# `near_ok` sits deliberately between `ok` (0.12/0.20) and collapse: measured
# 0.30/0.10 already saturates the metrics at the same level as full collapse
# (SSIM 0.377 vs 0.395), so it is NOT a usable intermediate rung -- the ladder
# needs a gentler point to show "almost fine" before "broken".
SV_RUNGS=(
  "near_ok|0.20|0.15|4"
  "degraded|0.30|0.10|4"
  "broken|1.00|0.00|4"
)

# ---------------------------------------------------------------- MV config --
MV_SCRIPT="scripts/run_humanoid_multiview_dreamgen_worldcache.sh"
MV_DENSE_SCRIPT="scripts/run_humanoid_multiview_dreamgen_full.sh"
# task-scoped ids so an id cannot leak across tasks
MV_EPISODE_IDS="AIRBOT_MMK2_boxs_storage:46,Airbot_MMK2_Airbot_MMK2_storage_mango_pomegranate:66"

# BRIGHT set. The two tasks above are the DARKEST in the whole humanoid multiview
# dataset (grid mean luma ~51/50 of 255, ~36% near-black pixels) because their
# wrist cameras are unlit -- measured by scripts/_rank_multiview_brightness.py.
# These three are the brightest distinct tasks (luma 138.6 / 137.6 / 128.6, i.e.
# ~2.7x brighter) and also cover BOTH grid layouts (A: front+head, B: high+third)
# and both splits, so the panel is not just brighter but more representative.
MV_BRIGHT_EPISODE_IDS="Airbot_MMK2_doodled_line:1,AIRBOT_MMK2_mobile_car:1,Airbot_MMK2_move_pan:2"
MV_BRIGHT_RUNGS=(
  "dense|-|-|-"
  "ok|0.25|0.2|2"
  "degraded|0.30|0.10|2"
  "broken|1.00|0.00|2"
)
MV_RUNGS=(
  "degraded|0.30|0.10|2"
  "broken|1.00|0.00|2"
)

if [[ "${1:-}" == "--estimate" ]]; then
  cat <<EOF
============================================================
WorldCache quality ladder -- missing rungs only
  already published : dense, ok   (both cells)
  to generate       : degraded, broken   (both cells)

SV (humanoid singleview, 3 episodes, dense 259.4s/ep)
  degraded ~2.5x -> ~105s/ep x3 = ~5 min
  broken   ~3x   -> ~ 90s/ep x3 = ~5 min
MV (humanoid multiview, 2 episodes x 2 segments, dense 118.8s/seg)
  degraded ~2.9x -> ~41s/seg -> ~3 min
  broken   ~3.5x -> ~34s/seg -> ~2 min
  + ~90s model load per run (4 runs) = ~6 min

MV BRIGHT (3 bright tasks, full ladder incl. dense; set MV_BRIGHT=0 to skip)
  dense    1.00x -> ~119s/seg x2 segs x3 eps = ~12 min
  ok       2.23x -> ~53s/seg               = ~6 min
  degraded 2.77x -> ~43s/seg               = ~5 min
  broken   6.56x -> ~18s/seg               = ~2 min
  + ~90s load x4                           = ~6 min   subtotal ~31 min

TOTAL: ~52 min on 1 GPU (~21 min if MV_BRIGHT=0)
============================================================
EOF
  exit 0
fi

# GPU must be idle: a co-tenant would distort the timings we publish.
mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" | head -1)"
if [[ "${mem}" -ge 1000 ]]; then
  echo "ERROR: GPU ${GPU} has ${mem} MiB in use; refusing to measure." >&2
  exit 1
fi

mkdir -p "${SCRATCH}"

# Which passes to run. Scratch is keyed by rung tag, so a pass whose scratch was
# lost (e.g. the old /tmp scratch being wiped) would otherwise be recomputed even
# though its videos are already staged. Skip it explicitly instead.
#   ONLY=sv|mv|mvb   run a single pass
RUN_SV=1; RUN_MV=1; RUN_MVB="${MV_BRIGHT:-1}"
case "${ONLY:-}" in
  sv)  RUN_MV=0; RUN_MVB=0 ;;
  mv)  RUN_SV=0; RUN_MVB=0 ;;
  mvb) RUN_SV=0; RUN_MV=0;  RUN_MVB=1 ;;
  "")  ;;
  *)   echo "ERROR: ONLY must be one of sv|mv|mvb" >&2; exit 2 ;;
esac

# Counting helpers, defined BEFORE any pass guard so a single-pass run
# (ONLY=mvb) still has them. `find` on a missing directory exits 1 and `pipefail`
# propagates that through `| wc -l`, which under `set -e` would kill the script
# at the assignment -- redirecting stderr hides the message but NOT the status.
# So guard on the directory existing and return 0 explicitly.
count_mp4() {
  [[ -d "$1" ]] || { echo 0; return 0; }
  find "$1" -name '*.mp4' 2>/dev/null | wc -l
}
count_metrics() {
  [[ -d "$1" ]] || { echo 0; return 0; }
  find "$1" -name metrics.json 2>/dev/null | wc -l
}

if [[ "${RUN_SV}" == "1" ]]; then
echo "############ SINGLEVIEW ############"
for r in "${SV_RUNGS[@]}"; do
  IFS='|' read -r tag th rt pb <<<"$r"
  out="${SCRATCH}/sv_${tag}"
  log="${SCRATCH}/sv_${tag}.log"
  n_done="$(count_mp4 "${out}")"
  if [[ "${n_done}" -eq 3 ]]; then echo "  [have] sv_${tag}"; continue; fi
  echo "  [run ] sv_${tag} (thresh=${th} ret=${rt} probe=${pb})"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  SPLITS="${SV_SPLIT}" NUM_CHUNKS="2" ITER_NAME="iter_000090000" \
  EPISODE_IDS="${SV_EPISODES}" OUTPUT_ROOT="${out}" \
  WORLDCACHE_REL_L1_THRESH="${th}" WORLDCACHE_RET_RATIO="${rt}" \
  WORLDCACHE_PROBE_DEPTH="${pb}" \
    bash "${SV_SCRIPT}" >"${log}" 2>&1 || {
      echo "    ERROR (see ${log})"; tail -15 "${log}"; }
  n="$(count_mp4 "${out}")"
  echo "    -> ${n}/3 episodes"
  [[ "${n}" -ne 3 ]] && echo "    WARNING: episode selector did not bind" >&2
done
fi

if [[ "${RUN_MV}" == "1" ]]; then
echo "############ MULTIVIEW ############"
for r in "${MV_RUNGS[@]}"; do
  IFS='|' read -r tag th rt pb <<<"$r"
  out="${SCRATCH}/mv_${tag}"
  log="${SCRATCH}/mv_${tag}.log"
  n_done="$(count_metrics "${out}")"
  if [[ "${n_done}" -eq 2 ]]; then echo "  [have] mv_${tag}"; continue; fi
  echo "  [run ] mv_${tag} (thresh=${th} ret=${rt} probe=${pb})"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  EPISODE_IDS="${MV_EPISODE_IDS}" SAVE_ROOT="${out}" \
  WORLDCACHE_REL_L1_THRESH="${th}" WORLDCACHE_RET_RATIO="${rt}" \
  WORLDCACHE_PROBE_DEPTH="${pb}" \
    bash "${MV_SCRIPT}" >"${log}" 2>&1 || {
      echo "    ERROR (see ${log})"; tail -15 "${log}"; }
  n="$(count_metrics "${out}")"
  echo "    -> ${n}/2 episodes"
  [[ "${n}" -ne 2 ]] && echo "    WARNING: episode selector did not bind" >&2
done
fi

# Bright MV pass: the full ladder (dense included, since no dense reference exists
# for these episodes yet) on 3 bright tasks.
if [[ "${RUN_MVB}" == "1" ]]; then
echo "############ MULTIVIEW (bright) ############"
for r in "${MV_BRIGHT_RUNGS[@]}"; do
  IFS='|' read -r tag th rt pb <<<"$r"
  out="${SCRATCH}/mvb_${tag}"
  log="${SCRATCH}/mvb_${tag}.log"
  n_done="$(count_metrics "${out}")"
  if [[ "${n_done}" -eq 3 ]]; then echo "  [have] mvb_${tag}"; continue; fi
  echo "  [run ] mvb_${tag} (thresh=${th} ret=${rt} probe=${pb})"
  if [[ "${tag}" == "dense" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" \
    EPISODE_IDS="${MV_BRIGHT_EPISODE_IDS}" SAVE_ROOT="${out}" \
      bash "${MV_DENSE_SCRIPT}" >"${log}" 2>&1 || {
        echo "    ERROR (see ${log})"; tail -15 "${log}"; }
  else
    CUDA_VISIBLE_DEVICES="${GPU}" \
    EPISODE_IDS="${MV_BRIGHT_EPISODE_IDS}" SAVE_ROOT="${out}" \
    WORLDCACHE_REL_L1_THRESH="${th}" WORLDCACHE_RET_RATIO="${rt}" \
    WORLDCACHE_PROBE_DEPTH="${pb}" \
      bash "${MV_SCRIPT}" >"${log}" 2>&1 || {
        echo "    ERROR (see ${log})"; tail -15 "${log}"; }
  fi
  n="$(count_metrics "${out}")"
  echo "    -> ${n}/3 episodes"
  [[ "${n}" -ne 3 ]] && echo "    WARNING: expected 3 episodes" >&2
done
fi

echo ""
echo "Done. Scratch: ${SCRATCH}"
echo "Next: python3 tmp/20261005_refactor/web/assets_scripts/stage_worldcache_ladder_web.py"
