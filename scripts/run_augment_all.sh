#!/bin/bash
# Launch the FULL augment generation (243 videos) sharded across GPUs.
#
# 4 shards, balanced by measured cost rather than by episode count — episode cost
# is driven by chunk/segment count, not by how many episodes a config has:
#
#   shard 0  bimanual singleview      170 chunks              ~7.6h (4 branches:
#                                                                    +task_swap, 20/category,
#                                                                    +6 object_relocation)
#   shard 1  single_arm singleview    118 chunks              ~5.2h  + ctrlworld single_arm
#   shard 2  bimanual multiview       segment rollout         ~6h  (+3 object_relocation)
#   shard 3  single_arm multiview     14B single-shot         ~4.1h  + ctrlworld bimanual+humanoid
#                                                                    (3-view AND 4-view-grid)
#                                                                    (+4 object_relocation)
#
# Every runner is resumable (skips episodes whose output already exists), so a
# shard that dies can simply be relaunched — that is why the cheap Ctrl-World
# configs are appended to the shorter shards instead of getting their own GPU.
# ctrlworld_{bimanual,humanoid}_multiview_4view_grid are a SEPARATE checkpoint
# from the 3-view ctrlworld configs (dense keeps them as separate method dirs
# too — see .cursor/rules/ctrlworld-*-4view-grid-*-insights.mdc), not a
# replacement, so they add 20+20=40 videos on top of the original 180.
#
# object_relocation (hand-curated, see gen_manifest.py::OBJECT_RELOCATION)
# applies to every DreamGen config except humanoid; default BRANCHES in each
# run_*_dreamgen_augment.sh already includes it, so a plain shard re-run
# regenerates/covers it automatically. single_arm_singleview has 0 curated
# entries in this dataset (see augment-track-layout.mdc) — that is expected,
# not a bug.
#
# Usage:
#   bash scripts/run_augment_all.sh            # launch all 4 shards in background
#   bash scripts/run_augment_all.sh 2          # launch only shard 2 (foreground)
#   GPUS="0 1 2 3" bash scripts/run_augment_all.sh
#
# Logs: exp_logs/augment_runs/<timestamp>/shard{N}.log
# Progress: bash scripts/run_augment_all.sh --status
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# ---------------------------------------------------------------- --status mode
if [ "${1:-}" = "--status" ]; then
  echo "augment videos generated (target 243):"
  python3 - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts/augment")
import augment_paths as AP

manifest = json.loads(Path("augment_manifest.json").read_text())
grand_done = grand_tot = 0
for cfg in [c for c in manifest if not c.startswith("_")]:
    done = tot = 0
    for split in ("makovian", "non_makovian"):
        sec = manifest[cfg].get(split, {})
        branches = {"": sec} if isinstance(sec, list) else sec
        for br, ents in branches.items():
            if br.startswith("_"):
                continue
            tot += len(ents)
    # count real mp4s under this config's output base
    base = AP.base_dir(cfg, "output")
    if Path(base).is_dir():
        mp4s = [p for p in Path(base).rglob("*.mp4")
                if p.name not in ("full_gt.mp4", "full_merged.mp4")
                and not p.name.startswith("seg_")
                and p.name != "pred_all_views.mp4"]
        done = len(mp4s)
    grand_done += done
    grand_tot += tot
    bar = "#" * int(20 * done / max(tot, 1))
    print(f"  {cfg:38s} {done:3d}/{tot:3d} [{bar:<20s}]")
print(f"  {'TOTAL':38s} {grand_done:3d}/{grand_tot:3d}")
PY
  exit 0
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-${ROOT}/exp_logs/augment_runs/${STAMP}}"
mkdir -p "${LOGDIR}"

# Shard definitions. Each is a list of "gpu:script[:arg]" run sequentially.
shard_0() {
  bash scripts/run_bimanual_singleview_dreamgen_augment.sh
}
shard_1() {
  bash scripts/run_single_arm_singleview_dreamgen_augment.sh
  bash scripts/run_single_arm_multiview_ctrlworld_augment.sh both
}
shard_2() {
  bash scripts/run_bimanual_multiview_dreamgen_augment.sh
}
shard_3() {
  bash scripts/run_single_arm_multiview_dreamgen_augment.sh
  bash scripts/run_bimanual_multiview_ctrlworld_augment.sh both
  bash scripts/run_humanoid_multiview_ctrlworld_augment.sh both
  bash scripts/run_bimanual_multiview_ctrlworld_4view_grid_augment.sh both
  bash scripts/run_humanoid_multiview_ctrlworld_4view_grid_augment.sh both
}

GPUS=(${GPUS:-0 1 2 3})

run_shard() {
  local n="$1" gpu="${GPUS[$1]}"
  # NUM_SHARDS/SHARD_ID leak in from the tmux global env and would silently make a
  # run skip most episodes (see augment-track-layout rule).
  unset NUM_SHARDS SHARD_ID
  export CUDA_VISIBLE_DEVICES="${gpu}"
  echo "[shard ${n}] GPU ${gpu} -> ${LOGDIR}/shard${n}.log"
  "shard_${n}"
}

# Single shard in the foreground (for debugging one config).
if [ -n "${1:-}" ]; then
  run_shard "$1" 2>&1 | tee "${LOGDIR}/shard$1.log"
  exit "${PIPESTATUS[0]}"
fi

echo "=========================================================="
echo "AUGMENT full generation — 243 videos, 4 shards"
echo "  logs: ${LOGDIR}"
echo "  GPUs: ${GPUS[*]}"
echo "=========================================================="
for n in 0 1 2 3; do
  ( run_shard "${n}" ) > "${LOGDIR}/shard${n}.log" 2>&1 &
  echo "  shard ${n} pid $!"
  sleep 2   # stagger model loading so 4 processes don't hit the FS at once
done

echo ""
echo "All 4 shards launched. Monitor with:"
echo "  bash scripts/run_augment_all.sh --status"
echo "  tail -f ${LOGDIR}/shard0.log"
wait
echo "All shards finished. Final status:"
bash "${ROOT}/scripts/run_augment_all.sh" --status
