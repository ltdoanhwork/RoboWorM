#!/usr/bin/env bash
# Fetch the openly-licensed 3D descriptions of the four benchmark robots.
#
# Only the needed subdirectories are cloned (sparse-checkout, blobless) -> ~120 MB
# instead of several GB. Re-running is safe; existing checkouts are reused.
#
# Models fetched:
#   menagerie/franka_emika_panda   Apache-2.0                       (Single-Arm)
#   menagerie/aloha                BSD-3-Clause, Trossen Robotics   (Bimanual)
#   menagerie/fourier_n1           Apache-2.0, Fourier Intelligence (Humanoid single-view)
#   rdesc/humanoid/Airbot/...      BSD, discoverse-dev              (Humanoid multi-view)
#
# NOTE: the humanoid stand-in is Fourier N1, NOT the GR1 used by the dataset. The
# GR1 descriptions (FFTAI/Wiki-GRx-Models) are GPL-3.0, which would make a render
# published in the paper a derivative work. N1 is the same vendor and morphology
# under Apache-2.0. Do NOT swap it back to GR1 without a license review.
#
# Then render with:
#   MUJOCO_GL=osmesa python3 scripts/tables/render_robot_cutouts.py
set -euo pipefail

ROOT="${MODEL_ROOT:-/tmp/robot_models}"
mkdir -p "$ROOT"

sparse_clone() {
  local url="$1" dir="$2"; shift 2
  local dst="$ROOT/$dir"
  if [ -d "$dst/.git" ]; then
    # Already cloned: make sure the requested subdirs are present anyway, so
    # adding a new model to the list below works without a fresh clone.
    echo "== $dir already present, syncing sparse-checkout"
    git -C "$dst" sparse-checkout set "$@"
    git -C "$dst" checkout
    return
  fi
  echo "== cloning $dir"
  git clone --filter=blob:none --no-checkout --depth 1 "$url" "$dst"
  git -C "$dst" sparse-checkout init --cone
  git -C "$dst" sparse-checkout set "$@"
  git -C "$dst" checkout
}

sparse_clone https://github.com/google-deepmind/mujoco_menagerie.git menagerie \
  franka_emika_panda aloha trossen_vx300s fourier_n1
sparse_clone https://github.com/discoverse-dev/robot_descriptions.git rdesc \
  humanoid/Airbot

echo
echo "Models in $ROOT:"
du -sh "$ROOT"/* 2>/dev/null || true
echo
echo "Next: MUJOCO_GL=osmesa python3 scripts/tables/render_robot_cutouts.py"
