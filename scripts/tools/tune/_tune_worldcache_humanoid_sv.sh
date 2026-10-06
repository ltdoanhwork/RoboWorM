#!/usr/bin/env bash
# =============================================================================
# _tune_worldcache_humanoid_sv.sh
# Tune WorldCache on DreamGen humanoid SINGLEVIEW (Cosmos-Predict2 2B DiT):
# 3 episodes x 3 configs, plus a matched dense control, all on one idle GPU.
#
# Protocol (mirrors the humanoid-MULTIVIEW WorldCache tuning pass):
#   * Every run goes through scripts/run/dreamgen/run_humanoid_singleview_dreamgen_worldcache.sh,
#     i.e. the same entrypoint (video2world_lvg), checkpoint, resolution, fps,
#     guidance, aspect ratio and conditioning frames as the shipped dense
#     sampling-dataset pass. The dense control is that SAME script with
#     WORLDCACHE_ENABLED=0, so the only difference between control and
#     candidate is the cache flags -- speedups are apples-to-apples.
#   * Timing is read from the per-episode "LVG Total inference time" log line,
#     never from wall clock (wall includes model load + encode).
#   * Quality is fidelity to the matched DENSE PREDICTION of the same episode
#     (PhyFid-DR convention), computed by scripts/metrics/_score_vs_dense.py.
#   * Output goes to a scratch root and is NEVER written into the canonical
#     sampling_dataset tree.
#
# Usage (repo root):
#   bash scripts/tools/tune/_tune_worldcache_humanoid_sv.sh [gpu]
#   DRY_RUN=1 bash scripts/tools/tune/_tune_worldcache_humanoid_sv.sh   # cost estimate only
# =============================================================================
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

GPU="${1:-${GPU:-0}}"
SCRATCH="${SCRATCH:-/tmp/wc_tune_humanoid_sv}"
LOGDIR="${SCRATCH}/logs"
REPORT="${REPORT:-${ROOT}/results/reports/worldcache_humanoid_sv_tuning.json}"
DRY_RUN="${DRY_RUN:-0}"

SPLIT="makovian"
NUM_CHUNKS="2"
ITER_NAME="${ITER_NAME:-iter_000090000}"
OUT_ITER="${ITER_NAME}_static16fps"

# --- Episode set (3 episodes) -------------------------------------------------
# All are makovian short 2-chunk rollouts with a completed dense prediction of
# identical length (185 frames), so the dense reference and every candidate are
# frame-aligned. Chosen to span distinct manipulated objects and both arms:
#   000200 left arm  / yellow starfruit -> shelf
#   000829 right hand / green bok choy
#   001055 right arm / red apple
EPISODES="${EPISODES:-episode_000200 episode_000829 episode_001055}"
N_EPISODES="$(echo ${EPISODES} | wc -w)"

# --- Candidate configs (3) ----------------------------------------------------
# The shipped default for this cell is thresh=0.15 / ret=0.3 / probe=4 (1.89x,
# documented as the quality knee). ret_ratio is the known speed lever here, so
# the sweep walks it down and tightens the threshold to trade back quality:
#   A = shipped optimum (reference point, must reproduce ~1.89x)
#   B = lower warm-up (ret 0.25) at the shipped threshold -> faster
#   C = lower warm-up (ret 0.2) with a tighter threshold (0.12) -> fastest,
#       threshold tightened to defend the hand detail that ret 0.2 lost at 0.15
CONFIGS=(
  "A_t0.15_r0.30_p4|0.15|0.30|4"
  "B_t0.15_r0.25_p4|0.15|0.25|4"
  "C_t0.12_r0.20_p4|0.12|0.20|4"
)

DENSE_REF="${ROOT}/sampling_dataset/dense/humanoid/output/singleview/dreamgen/${SPLIT}/${OUT_ITER}"

# --- Cost estimate ------------------------------------------------------------
# Dense for this cell is ~260s / 2-chunk episode (documented). Candidates run at
# ~1.5-2.3x, so ~115-175s each.
EST_DENSE=$((260 * N_EPISODES))
EST_CAND=$((170 * N_EPISODES * ${#CONFIGS[@]}))
EST_TOTAL=$((EST_DENSE + EST_CAND))
echo "============================================================"
echo "WorldCache tuning - DreamGen humanoid SINGLEVIEW (2B DiT)"
echo "  GPU:           ${GPU}"
echo "  Episodes (${N_EPISODES}):  ${EPISODES}"
echo "  Configs (${#CONFIGS[@]}):   $(printf '%s ' "${CONFIGS[@]%%|*}")"
echo "  Dense control: 1 run (WORLDCACHE_ENABLED=0, same entrypoint)"
echo "  Scratch:       ${SCRATCH}"
echo "  Report:        ${REPORT}"
echo "  ---- GPU cost estimate ----"
echo "  dense control: ~$((EST_DENSE / 60)) min   (${N_EPISODES} x ~260s)"
echo "  candidates:    ~$((EST_CAND / 60)) min   (${#CONFIGS[@]} x ${N_EPISODES} x ~170s)"
echo "  TOTAL:         ~$((EST_TOTAL / 60)) min on 1 GPU (+ ~1 min model load per run)"
echo "============================================================"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1 -> estimate only, exiting."
  exit 0
fi

# --- Guard: GPU must be idle (a co-tenant would corrupt the timings) ----------
used_mb="$(nvidia-smi --id="${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits)"
if [[ "${used_mb}" -gt 1000 ]]; then
  echo "ERROR: GPU ${GPU} already has ${used_mb} MiB in use; timings would be skewed." >&2
  exit 1
fi

# The batch-json subsetting is done by the run script's EPISODE_IDS selector.
if ! grep -q "EPISODE_IDS" scripts/run/dreamgen/run_humanoid_singleview_dreamgen_worldcache.sh; then
  echo "ERROR: run script does not honor EPISODE_IDS; refusing to run a full sweep." >&2
  exit 1
fi

EPISODE_IDS_CSV="$(echo ${EPISODES} | tr ' ' ',')"
mkdir -p "${LOGDIR}"

# run_one <tag> <thresh> <ret> <probe> <enabled>
run_one() {
  local tag="$1" thresh="$2" ret="$3" probe="$4" enabled="$5"
  local outroot="${SCRATCH}/${tag}"
  local log="${LOGDIR}/${tag}.log"
  local outdir="${outroot}/${SPLIT}/${OUT_ITER}"

  if [[ -d "${outdir}" ]] && [[ "$(ls -1 "${outdir}"/*.mp4 2>/dev/null | wc -l)" -eq "${N_EPISODES}" ]]; then
    echo "  [SKIP] ${tag}: ${N_EPISODES} episodes already present"
    return 0
  fi

  echo "  === ${tag} (thresh=${thresh} ret=${ret} probe=${probe} enabled=${enabled}) ==="
  CUDA_VISIBLE_DEVICES="${GPU}" \
  SPLITS="${SPLIT}" NUM_CHUNKS="${NUM_CHUNKS}" \
  ITER_NAME="${ITER_NAME}" \
  EPISODE_IDS="${EPISODE_IDS_CSV}" \
  OUTPUT_ROOT="${outroot}" \
  WORLDCACHE_ENABLED="${enabled}" \
  WORLDCACHE_REL_L1_THRESH="${thresh}" \
  WORLDCACHE_RET_RATIO="${ret}" \
  WORLDCACHE_PROBE_DEPTH="${probe}" \
    bash scripts/run/dreamgen/run_humanoid_singleview_dreamgen_worldcache.sh >"${log}" 2>&1 || {
      echo "  ERROR: ${tag} failed; tail of ${log}:" >&2
      tail -25 "${log}" >&2
      return 1
    }

  local n
  n="$(ls -1 "${outdir}"/*.mp4 2>/dev/null | wc -l)"
  if [[ "${n}" -ne "${N_EPISODES}" ]]; then
    echo "  ERROR: ${tag} produced ${n} episodes, expected ${N_EPISODES} (EPISODE_IDS ignored?)" >&2
    return 1
  fi
  echo "  === ${tag} done (${n} episodes) ==="
}

echo ""
echo ">>> [1/$((1 + ${#CONFIGS[@]}))] matched dense control"
run_one "dense_control" "0.15" "0.30" "4" "0"

i=1
for spec in "${CONFIGS[@]}"; do
  IFS='|' read -r tag thresh ret probe <<<"${spec}"
  i=$((i + 1))
  echo ""
  echo ">>> [${i}/$((1 + ${#CONFIGS[@]}))] ${tag}"
  run_one "${tag}" "${thresh}" "${ret}" "${probe}" "1"
done

# --- Score every candidate against the matched dense control -----------------
echo ""
echo ">>> scoring candidates against the matched dense control"
source "${ROOT}/models/dreamgen/.venv/bin/activate"
export PYTHONPATH="${ROOT}:${ROOT}/models/dreamgen:${PYTHONPATH:-}"
DENSE_CTRL_DIR="${SCRATCH}/dense_control/${SPLIT}/${OUT_ITER}"

for spec in "${CONFIGS[@]}"; do
  IFS='|' read -r tag _ _ _ <<<"${spec}"
  echo "  --- ${tag} ---"
  CUDA_VISIBLE_DEVICES="${GPU}" python scripts/metrics/_score_vs_dense.py \
    --dense "${DENSE_CTRL_DIR}" \
    --test "${SCRATCH}/${tag}/${SPLIT}/${OUT_ITER}" \
    --out "${SCRATCH}/${tag}/quality_vs_dense.json"
done

# Sanity: the dense control must reproduce the shipped dense pass. Scoring it
# against the canonical dense output quantifies run-to-run nondeterminism, which
# is the noise floor every candidate number must be read against.
if [[ -d "${DENSE_REF}" ]]; then
  echo "  --- dense_control vs shipped dense sampling_dataset (nondeterminism floor) ---"
  CUDA_VISIBLE_DEVICES="${GPU}" python scripts/metrics/_score_vs_dense.py \
    --dense "${DENSE_REF}" \
    --test "${DENSE_CTRL_DIR}" \
    --out "${SCRATCH}/dense_control/quality_vs_shipped_dense.json" || true
fi

# --- Collect timings + quality into one report -------------------------------
echo ""
echo ">>> building report"
python scripts/tools/tune/_collect_worldcache_humanoid_sv.py \
  --scratch "${SCRATCH}" \
  --split "${SPLIT}" \
  --out-iter "${OUT_ITER}" \
  --episodes "${EPISODE_IDS_CSV}" \
  --configs "$(printf '%s;' "${CONFIGS[@]}")" \
  --out "${REPORT}"

echo ""
echo "============================================================"
echo "Done. Report: ${REPORT}"
echo "Scratch (delete when finished): ${SCRATCH}"
echo "============================================================"
