#!/usr/bin/env bash
# EXTRA machine (the 4th+ GPU) -- joins an already-running wave by walking the id-sorted
# manifest BACKWARDS instead of claiming a stride slot.
#
# Use this when machines 0/1/2 are already running and you get another GPU. You cannot use
# a 4th stride index: the split is `rows[k*W + s :: W*MACHINES]`, so with 3 machines x 8
# workers every offset 0-23 is taken. Re-sharding to --machines 4 would require restarting
# all three running boxes (and until they restarted, the partition would be wrong).
#
#   bash scripts/run_dense_nr_full_extra.sh
#
# Optional: GPU=1, WORKERS=6.
#
# Overlap: the reverse walker only repeats work once it meets the forward front. Measured
# at 20% done, everything past 30% of the manifest was ~2% touched, so it starts in
# essentially virgin territory. Any overlap is a no-op anyway -- a worker skips ids whose
# result file already exists.
#
# Only ONE box may run this (there is a single backwards direction). A second extra GPU
# would need a real re-shard: stop everything and relaunch with MACHINES=5.
set -euo pipefail
REVERSE=1 MACHINE=0 exec "$(dirname "$0")/_run_dense_nr_full.sh" "$@"
