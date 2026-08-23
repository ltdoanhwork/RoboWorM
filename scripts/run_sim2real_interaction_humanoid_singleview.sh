#!/usr/bin/env bash
# Sim2real interaction probe on DreamGen humanoid SINGLEVIEW episodes.
#
# Differs from run_sim2real_interaction.sh (ctrlworld multiview) in three ways that
# are properties of this data, not preferences:
#   * ONE view, so the 2-camera AND-gate degenerates to trusting that view. Contact
#     is therefore only as reliable as the single projection - expect more UNKNOWN.
#   * The GT-less output tree means GT must be rebuilt onto the prediction's
#     768x432@16fps grid first; prepare_dreamgen_singleview.py does that.
#   * The ego camera is head-mounted: only "locked waist" episodes are static, so
#     that is the default filter. Background subtraction is ALSO wrong here (both
#     arms are already in frame at t=0), hence --gripper_mode gdino.
set -euo pipefail

REPO=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics
PY=$REPO/models/dreamgen/.venv/bin/python
WS=$REPO/metrics/WorldScore
TP=$WS/worldscore/benchmark/metrics/third_party
CKPT=$REPO/checkpoints/worldscore

METHOD=${METHOD:-dense}
SPLIT=${SPLIT:-makovian}
ITER=${ITER:-iter_000090000_static16fps}
WAIST=${WAIST:-locked}
LIMIT=${LIMIT:-1}
EPISODES=${EPISODES:-}

PRED_DIR=${PRED_DIR:-$REPO/sampling_dataset/$METHOD/humanoid/output/singleview/dreamgen/$SPLIT/$ITER}
SPLIT_DIR=${SPLIT_DIR:-$REPO/datasets/humanoid/singleview/$SPLIT}
PAIR_DIR=${PAIR_DIR:-$REPO/metrics/sim2real_out/_pairs/humanoid_singleview/$SPLIT}
OUT_ROOT=${OUT_ROOT:-$REPO/metrics/sim2real_out/$METHOD/humanoid_singleview_dreamgen/$SPLIT}

export PYTHONPATH="$REPO:$TP:$TP/sam2:$TP/SEA-RAFT/core:$TP/SEA-RAFT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export TQDM_DISABLE=${TQDM_DISABLE:-1}
export REPO

echo "=== [1/2] rebuilding GT onto the DreamGen grid (768x432 @16fps)"
"$PY" "$REPO/metrics/sim2real/prepare_dreamgen_singleview.py" \
  --pred_dir "$PRED_DIR" --split_dir "$SPLIT_DIR" --out "$PAIR_DIR" \
  --waist "$WAIST" ${LIMIT:+--limit $LIMIT} ${EPISODES:+--episodes $EPISODES}

echo
echo "=== [2/2] probing"
cd "$WS"
"$PY" - "$PAIR_DIR/index.json" "$OUT_ROOT" "$@" <<'PYEOF'
import json, os, subprocess, sys

index_path, out_root = sys.argv[1], sys.argv[2]
extra = sys.argv[3:]
repo = os.environ["REPO"]
ckpt = f"{repo}/checkpoints/worldscore"
rows = json.load(open(index_path))
for r in rows:
    out = os.path.join(out_root, r["episode_id"])
    print(f"\n----- {r['episode_id']} -> {out}", flush=True)
    cmd = [sys.executable, f"{repo}/metrics/sim2real/interaction_probe.py",
           "--gt", r["gt"], "--pred", r["pred"], "--metadata", r["metadata"],
           "--out", out,
           "--sam2_cfg", "configs/sam2.1/sam2.1_hiera_b+.yaml",
           "--sam2_ckpt", f"{ckpt}/sam2.1_hiera_base_plus.pt",
           "--gdino_cfg", "worldscore/benchmark/metrics/third_party/groundingdino/"
                          "config/GroundingDINO_SwinT_OGC.py",
           "--gdino_ckpt", f"{ckpt}/groundingdino_swint_ogc.pth",
           "--num_views", "1", "--gripper_mode", "gdino",
           "--view_names", "ego_view", "--static_views", "0",
           # video2world_lvg sees ONE frame + a text prompt: no actions, no states.
           # It cannot know WHEN GT grasped, so only the physics group is charged.
           "--conditioning", "text",
           "--select_referent"] + extra
    subprocess.run(cmd, check=True)
PYEOF
