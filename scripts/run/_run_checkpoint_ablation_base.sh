#!/usr/bin/env bash
# One-off driver: run the ORIGINAL / pre-finetuning checkpoint for each
# model family x embodiment cell where (a) a genuine base checkpoint exists
# distinct from the finetuned one, and (b) the canonical dense pipeline
# already scores PSNR/SSIM/LPIPS per episode (so re-running with a different
# checkpoint auto-produces a comparable metrics.json).
#
# Skipped on purpose (see paper text for why):
#   - DreamDojo humanoid singleview: only one checkpoint ever existed
#     (2B_GR1_post-train IS the only GR1 checkpoint used everywhere).
#   - Ctrl-World / DreamGen single_arm multiview: dense already runs the
#     un-finetuned base checkpoint (checkpoint-10000.pt / GR00T-Dreams-DROID)
#     -- there is no finetuned counterpart to compare against.
#   - DreamGen humanoid/single_arm SINGLEVIEW: this pipeline never writes
#     per-episode metrics.json (only .mp4+.txt), so there is no existing
#     scorer to reuse for either checkpoint -- out of scope here.
#
# Output: checkpoint_ablation/<cell>/base_ckpt/... (NOT sampling_dataset/ --
# this is a checkpoint ablation, not an acceleration method, so it does not
# use the {method}-top-level sampling-dataset-layout convention).
set -uo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "${ROOT}"
OUT="${ROOT}/checkpoint_ablation"
mkdir -p "${OUT}"
export CUDA_VISIBLE_DEVICES=0

run_step() {
  local name="$1"; shift
  echo ""
  echo "################################################################"
  echo "# [$(date +%T)] STARTING: ${name}"
  echo "################################################################"
  if "$@"; then
    echo "[$(date +%T)] OK: ${name}"
  else
    echo "[$(date +%T)] FAILED: ${name} (exit $?) -- continuing to next cell"
  fi
}

# Cells A/B use the legacy 3-view config (CONFIG=bimanual|humanoid) so the base
# rollouts pair with the kept 3-view dense rollouts in sampling_dataset/dense/.
# They only need the DROID base ckpt (the dense default); the 3-view finetuned
# ckpts were removed on 2026-10-08 (HF doanh25032004/vgp_exp_logs).
#
# --- A) Ctrl-World bimanual multiview: base=checkpoint-10000.pt (DROID base) ---
step_ctrlworld_bimanual() {
  OUTPUT_ROOT="${OUT}/ctrlworld_bimanual_mv/base_ckpt/output" \
  INPUT_ROOT="${OUT}/ctrlworld_bimanual_mv/base_ckpt/input" \
  DATASET_NAME=bimanual_multiview_5fps DOWN_SAMPLE=6 CONFIG=bimanual \
  SPLIT=val NUM_EPISODES=2 \
    bash scripts/run/ctrlworld/run_bimanual_multiview_ctrlworld.sh
}

# --- B) Ctrl-World humanoid multiview: base=checkpoint-10000.pt ---
step_ctrlworld_humanoid() {
  OUTPUT_ROOT="${OUT}/ctrlworld_humanoid_mv/base_ckpt/output" \
  INPUT_ROOT="${OUT}/ctrlworld_humanoid_mv/base_ckpt/input" \
  DATASET_NAME=humanoid_multiview DOWN_SAMPLE=6 CONFIG=humanoid \
  SPLIT=val NUM_EPISODES=2 \
    bash scripts/run/ctrlworld/run_humanoid_multiview_ctrlworld.sh
}

# --- C) DreamGen humanoid multiview: base=Cosmos-Predict2-2B-Video2World ---
step_dreamgen_humanoid_mv() {
  local scratch="${OUT}/dreamgen_humanoid_mv/base_ckpt/multiview_root"
  local task_dir="${scratch}/makovian/AIRBOT_MMK2_mobile_car"
  mkdir -p "${task_dir}"
  local src="${ROOT}/datasets/humanoid/multiview/makovian/AIRBOT_MMK2_mobile_car"
  for d in meta videos data; do ln -sfn "${src}/${d}" "${task_dir}/${d}"; done
  DIT_PATH="${ROOT}/checkpoints/base/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt" \
  ITER_TAG="base_ckpt" \
  MULTIVIEW_ROOT="${scratch}" \
  SAVE_ROOT="${OUT}/dreamgen_humanoid_mv/base_ckpt/output" \
  MAX_EPISODES=1 MAX_SEGMENTS=2 NUM_GPUS=1 \
    bash scripts/run/dreamgen/run_humanoid_multiview_dreamgen_full.sh
}

# --- D) DreamGen bimanual multiview: base=Cosmos-Predict2-2B-Video2World ---
step_dreamgen_bimanual_mv() {
  local scratch="${OUT}/dreamgen_bimanual_mv/base_ckpt/multiview_root"
  local task_dir="${scratch}/makovian/close_cardboard_box"
  mkdir -p "${task_dir}"
  local src="${ROOT}/datasets/bimanual/multiview/makovian/close_cardboard_box"
  for d in meta videos data; do ln -sfn "${src}/${d}" "${task_dir}/${d}"; done
  DIT_PATH="${ROOT}/checkpoints/base/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt" \
  ITER_TAG="base_ckpt" \
  MULTIVIEW_ROOT="${scratch}" \
  SAVE_ROOT="${OUT}/dreamgen_bimanual_mv/base_ckpt/output" \
  MAX_EPISODES=1 MAX_SEGMENTS=2 NUM_GPUS=1 \
    bash scripts/run/dreamgen/run_bimanual_multiview_dreamgen_full.sh
}

# --- E) DreamDojo bimanual singleview: base=checkpoints/base/dreamdojo/2B_pretrain ---
step_dreamdojo_bimanual_sv() {
  CHECKPOINTS_DIR="${ROOT}/checkpoints/base/dreamdojo/2B_pretrain" \
  SAVE_ROOT="${OUT}/dreamdojo_bimanual_sv/base_ckpt/output" \
  NUM_EPISODES=2 \
    bash scripts/run/dreamdojo/run_bimanual_singleview_dreamdojo_full_episode.sh
}

# --- F) DreamDojo single_arm singleview: base=checkpoints/base/dreamdojo/2B_pretrain ---
step_dreamdojo_single_arm_sv() {
  CHECKPOINTS_DIR="${ROOT}/checkpoints/base/dreamdojo/2B_pretrain" \
  SAVE_ROOT="${OUT}/dreamdojo_single_arm_sv/base_ckpt/output" \
  NUM_EPISODES=2 \
    bash scripts/run/dreamdojo/run_single_arm_singleview_dreamdojo_dense.sh
}

run_step "Ctrl-World bimanual multiview (base ckpt)" step_ctrlworld_bimanual
run_step "Ctrl-World humanoid multiview (base ckpt)" step_ctrlworld_humanoid
run_step "DreamGen humanoid multiview (base ckpt)" step_dreamgen_humanoid_mv
run_step "DreamGen bimanual multiview (base ckpt)" step_dreamgen_bimanual_mv

echo ""
echo "SKIPPING DreamDojo bimanual/single_arm singleview base-checkpoint runs:"
echo "  ActionConditionedSetupArguments(model='2B/robot/action-cond', ...) eagerly"
echo "  resolves an UNRELATED default checkpoint from the gated HF repo"
echo "  nvidia/Cosmos-Predict2.5-2B (robot/action-cond/*_ema_bf16.pt) at construction"
echo "  time, regardless of --checkpoints-dir. Our HF token has repo-metadata access"
echo "  (model_info succeeds) but not file-level access (401 GatedRepoError) to that"
echo "  specific file. Verified this ALSO breaks the existing finetuned dense"
echo "  checkpoints-dir path in this environment/session (not something introduced by"
echo "  pointing at 2B_pretrain) -- this is a pre-existing external access blocker,"
echo "  not a bug in our checkpoint-ablation setup. Needs HF access to be granted to"
echo "  this account before DreamDojo bimanual/single_arm can run fresh inference at"
echo "  all right now (dense included)."
# run_step "DreamDojo bimanual singleview (base ckpt)" step_dreamdojo_bimanual_sv
# run_step "DreamDojo single_arm singleview (base ckpt)" step_dreamdojo_single_arm_sv

echo ""
echo "################################################################"
echo "# ALL CELLS DONE at $(date +%T)"
echo "################################################################"
