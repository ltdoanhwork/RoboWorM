#!/usr/bin/env bash
# MACHINE 0 of 3 -- full dense-NR wave.
# Run this on the 0-th A100 and nothing else. Do NOT run two machines with the same
# index: the split is index-based, so a duplicate index leaves ~1/3 of the manifest
# unscored and, because every write is idempotent, nothing will error to tell you.
#
#   bash scripts/run_dense_nr_full_m0.sh
#
# Optional: GPU=1 (if this box's free card is not 0), WORKERS=6 (if the card is shared).
set -euo pipefail
MACHINE=0 exec "$(dirname "$0")/_run_dense_nr_full.sh" "$@"
