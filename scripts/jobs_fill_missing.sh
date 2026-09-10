#!/usr/bin/env bash
# Fill the remaining sampling_dataset gaps against the benchmark plan
# (sheet `distribution` of "[BENCHMARK] Accelerating Inference of Video World
# Models.xlsx", mirrored in docs/gpu_agent_fill_missing.md).
#
# Only 968 episodes are left, in two cells. Everything else is COMPLETE.
#
# Default DRY_RUN=1 prints commands. Set DRY_RUN=0 to launch.
# WAVE=all|1|2
#
# Usage:
#   bash scripts/jobs_fill_missing.sh
#   WAVE=1 DRY_RUN=0 CUDA_VISIBLE_DEVICES=0 bash scripts/jobs_fill_missing.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DRY_RUN="${DRY_RUN:-1}"
WAVE="${WAVE:-all}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

run() {
  echo ""
  echo ">>> $*"
  if [[ "${DRY_RUN}" != "0" ]]; then
    echo "    (DRY_RUN=1 — not launched)"
    return 0
  fi
  eval "$@"
}

header() {
  echo ""
  echo "================================================================"
  echo " $1"
  echo "================================================================"
}

# ---------------------------------------------------------------------------
if [[ "${WAVE}" == "all" ]]; then
  cat <<'EOF'
PLAN (sheet `distribution`) — 29343 episodes total, 968 missing.

Canonical matrix: 2 models per (embodiment, view). Ctrl-World variant differs:
  bimanual  MV -> dreamgen + ctrlworld_4view_grid (4-view)
  humanoid  MV -> dreamgen + ctrlworld_4view_grid (4-view)
  single_arm MV -> dreamgen + ctrlworld (NATIVE 3-view)
  all       SV -> dreamgen + dreamdojo
Dense per-method total = 3166.

Per-method targets encode two deliberate exclusions:
  sito / itm  = 2410 = DreamGen 1605 + Ctrl-World 805  -> SKIP DreamDojo
  fastercache = 2361 = DreamGen 1605 + DreamDojo  756  -> SKIP Ctrl-World
  everything else = 3166 (full coverage)

STATUS
  dicache      3166/3166  COMPLETE
  worldcache   3166/3166  COMPLETE
  dense        3046/3166  -120
  sito         2290/2410  -120
  itm          2290/2410  -120
  pisa         3046/3166  -120
  radial       3046/3166  -120
  svg1         3046/3166  -120
  svg2         3046/3166  -120
  fastercache  2233/2361  -128

WAVE 1  840 ep   single_arm SV DreamGen non_makovian
  for dense,sito,itm,pisa,radial,svg1,svg2 (120 each).
  Needs NUM_CHUNKS="4 6 7" — the default "2 3" skips every non_makovian clip.
  dense is EMPTY here, so run dense first; nothing else is comparable until then.

WAVE 2  128 ep   fastercache bimanual SV DreamGen non_makovian (22/150 done)

NOT IN THE PLAN (do not run):
  * sito/itm on ANY DreamDojo cell (humanoid 432, bimanual 600, single_arm 480).
    Excluded by the 2410 target. The single_arm scripts also point at the wrong
    ckpt (2B_DROID iter_100000 / 320x512 / 49 frames) and bimanual has no script.
  * fastercache on ANY Ctrl-World cell (bimanual 300, humanoid 248, single_arm 253).
    Excluded by the 2361 target; also a known no-op on this CFG-off SVD UNet.
  * sparse attn (pisa/radial/svg1/svg2) on humanoid NATIVE ctrlworld — the
    canonical humanoid variant is ctrlworld_4view_grid, already 98+150 each.

COUNTS THAT LOOK SHORT BUT ARE CORRECT (the sheet agrees):
  * humanoid MV makovian 98      -> that split has 2 tasks, non_makovian has 4.
  * humanoid SV dreamdojo 118/98 -> GR1 stride 2 needs num_frames+1=50 steps
    (delta_indices up to 98), so a clip must be >= 99 frames; 12 mak + 32 non
    are shorter. Do NOT pad to 130.
  * single_arm MV 128/129        -> episodes 015475, 016705 (mak), 015486 (non)
    lack a camera in the source; dreamgen and ctrlworld skip the same three.
  * single_arm SV 120            -> datasets/single_arm/singleview/*/meta/info.json
    says 120, not 130.
EOF
fi

# ---------------------------------------------------------------------------
WAVE1() {
  header "WAVE 1 — single_arm SV DreamGen non_makovian (NUM_CHUNKS=4 6 7)"
  # Inputs live under the view subdir:
  #   .../input/singleview/dreamgen/global/non_makovian/batch_input_{4,6,7}chunks.json
  # Their output_video currently points at the itm/ tree (shared input dir); each
  # script's prepare step rewrites it. Never pass SKIP_PREPARE=1 here.
  # Pin 90k (makovian + dicache nm already use iter_000090000_static16fps).
  local common="SPLITS=non_makovian NUM_CHUNKS='4 6 7' ITER_NAME=iter_000090000 CUDA_VISIBLE_DEVICES=${GPU}"
  run "${common} NUM_GPUS=1 bash scripts/run_single_arm_singleview_dreamgen_static16fps.sh"
  run "${common} bash scripts/run_single_arm_singleview_dreamgen_sito.sh"
  run "${common} bash scripts/run_single_arm_singleview_dreamgen_itm.sh"
  run "${common} bash scripts/run_single_arm_singleview_dreamgen_pisa.sh"
  run "${common} bash scripts/run_single_arm_singleview_dreamgen_radial.sh"
  run "${common} bash scripts/run_single_arm_singleview_dreamgen_svg1.sh"
  run "${common} bash scripts/run_single_arm_singleview_dreamgen_svg2.sh"
}

WAVE2() {
  header "WAVE 2 — fastercache bimanual SV DreamGen non_makovian remainder"
  # Do not use NUM_CHUNKS=auto here. This input dir also contains stale 9/11/33
  # chunk JSONs which overlap the current 30-fps-derived 3/5/6/18 groups. Auto
  # discovers both generations. Pin current groups so every missing episode is
  # generated once with the chunk count from prepare_singleview_dreamgen_batch.py.
  run "SPLITS=non_makovian NUM_CHUNKS='3 5 6 18' CUDA_VISIBLE_DEVICES=${GPU} \
    bash scripts/run_bimanual_singleview_dreamgen_fastercache.sh"
}

case "${WAVE}" in
  all) WAVE1; WAVE2 ;;
  1) WAVE1 ;;
  2) WAVE2 ;;
  *) echo "Unknown WAVE=${WAVE} (all|1|2)" >&2; exit 1 ;;
esac

echo ""
echo "Done. DRY_RUN=${DRY_RUN} WAVE=${WAVE} GPU=${GPU}"
if [[ "${DRY_RUN}" != "0" ]]; then
  echo "Launch: WAVE=${WAVE} DRY_RUN=0 CUDA_VISIBLE_DEVICES=${GPU} bash scripts/jobs_fill_missing.sh"
fi
