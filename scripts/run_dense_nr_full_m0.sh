#!/usr/bin/env bash
# MACHINE 0 -- full dense-NR wave. Total machine count comes from MACHINES (default 5).
#
# SUPERSEDED for the 5-box grid wave by scripts/run_dense_nr_grid.sh, which also builds
# the per-cell id subset. Keep using this one only for a WHOLE-manifest wave.
# Run this on the 0-th A100 and nothing else. Do NOT run two machines with the same
# index: the split is index-based, so a duplicate index leaves ~1/3 of the manifest
# unscored and, because every write is idempotent, nothing will error to tell you.
#
#   bash scripts/run_dense_nr_full_m0.sh
#
# Optional: GPU=1 (if this box's free card is not 0), WORKERS=6 (if the card is shared).
set -euo pipefail
MACHINE=0 exec "$(dirname "$0")/_run_dense_nr_full.sh" "$@"
