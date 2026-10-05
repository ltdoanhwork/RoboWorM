#!/usr/bin/env bash
# Fetch the robot models the execution-grounded tier needs.
#
# Only the MJCFs are third-party assets; everything else in metrics/embodied_exec is
# ours. Kept out of the repo tree and sparse-checked out so this costs a few MB, not
# the whole menagerie.
#
# Verify after fetching (must print FK match: OK, residual ~3e-08 m):
#   models/dreamgen/.venv/bin/python -m metrics.embodied_exec.kinematics \
#     --groups single_arm/multiview --split makovian
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MJCF_ROOT="${MJCF_ROOT:-$REPO/checkpoints/base/mjcf}"

# add more as embodiments are registered in metrics/embodied_exec/kinematics.py
MODELS="${MODELS:-franka_emika_panda aloha}"

# GR1 (humanoid/singleview) is NOT in mujoco_menagerie; robosuite ships it. Fetched into
# a sibling root so the two upstreams stay separate clones. kinematics.py finds it
# because MJCF_SEARCH includes checkpoints/base/mjcf_robosuite.
RS_ROOT="${RS_ROOT:-$REPO/checkpoints/base/mjcf_robosuite}"
RS_ASSETS="robosuite/models/assets"
RS_PATHS="${RS_PATHS:-$RS_ASSETS/robots/gr1 $RS_ASSETS/grippers}"

mkdir -p "$(dirname "$MJCF_ROOT")"

if [ -d "$MJCF_ROOT/.git" ]; then
  echo "[info] reusing existing clone at $MJCF_ROOT"
else
  echo "[info] cloning mujoco_menagerie (sparse) -> $MJCF_ROOT"
  rm -rf "$MJCF_ROOT"
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git "$MJCF_ROOT"
fi

cd "$MJCF_ROOT"
# shellcheck disable=SC2086
git sparse-checkout set $MODELS
echo "[info] checked out: $MODELS"

for m in $MODELS; do
  if [ -d "$m" ]; then
    echo "  $m: $(ls "$m"/*.xml 2>/dev/null | wc -l) xml files"
  else
    echo "  $m: MISSING (name may differ upstream)" >&2
  fi
done

# ---- robosuite: GR1 humanoid + real gripper geometry ------------------------------
if [ -d "$RS_ROOT/.git" ]; then
  echo "[info] reusing existing robosuite clone at $RS_ROOT"
else
  echo "[info] cloning robosuite (sparse) -> $RS_ROOT"
  rm -rf "$RS_ROOT"
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/ARISE-Initiative/robosuite.git "$RS_ROOT"
fi

cd "$RS_ROOT"
# shellcheck disable=SC2086
git sparse-checkout set $RS_PATHS
# Flatten so relpaths stay short: kinematics.py asks for "gr1/robot.xml", not the full
# robosuite/models/assets/... path.
for d in "$RS_ROOT/$RS_ASSETS/robots"/*; do
  [ -d "$d" ] || continue
  ln -sfn "$d" "$RS_ROOT/$(basename "$d")"
done
ln -sfn "$RS_ROOT/$RS_ASSETS/grippers" "$RS_ROOT/grippers"
echo "[info] robosuite: $(ls -d "$RS_ROOT"/*/ 2>/dev/null | wc -l) model dirs linked"

cat <<EOF

[done] MJCF_ROOT=$MJCF_ROOT
Export it (metrics/embodied_exec/kinematics.py also checks checkpoints/mjcf, a compat symlink to checkpoints/base/mjcf):
  export MJCF_ROOT=$MJCF_ROOT
Then verify FK against dataset ground truth:
  models/dreamgen/.venv/bin/python -m metrics.embodied_exec.kinematics \\
    --groups single_arm/multiview --split makovian --max-episodes 40
EOF
