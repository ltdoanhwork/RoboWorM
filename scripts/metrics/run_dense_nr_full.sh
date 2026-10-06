#!/usr/bin/env bash
# One machine of the FULL dense-NR wave (whole manifest), by slot index.
#
#   bash scripts/metrics/run_dense_nr_full.sh <k>       # machine k of MACHINES (default 5)
#   bash scripts/metrics/run_dense_nr_full.sh extra     # EXTRA box: walks the manifest BACKWARDS
#
# Replaces the five copied wrappers run_dense_nr_full_m{0..4}.sh and
# run_dense_nr_full_extra.sh (2026-10-05 refactor; originals in the refactor archive).
# Behaviour is identical: `<k>` sets MACHINE=k, `extra` sets REVERSE=1 MACHINE=0, and
# every other argument is passed through to _run_dense_nr_full.sh.
#
# SUPERSEDED for the paper-grid wave by scripts/metrics/run_dense_nr_grid.sh <k>, which also
# builds the per-cell id subset. Keep using this one only for a WHOLE-manifest wave.
#
# Run each index on exactly one box. Do NOT run two machines with the same index: the
# split is index-based, so a duplicate index leaves part of the manifest unscored and,
# because every write is idempotent, nothing will error to tell you.
# Do NOT add a 4th/5th index to a wave the other machines started with MACHINES=3
# (3 machines x 8 workers is stride 24, which already covers every row): restart ALL
# boxes with MACHINES=5 instead.
#
# `extra`: joins an already-running wave without claiming a stride slot (the reverse
# walker only repeats work once it meets the forward front; a repeat is a no-op because
# a worker skips ids whose result file exists). Only ONE box may run `extra`.
#
# Optional: GPU=1 (if this box's free card is not 0), WORKERS=6 (if the card is shared).
set -euo pipefail

slot="${1:?usage: $0 <machine index 0..MACHINES-1 | extra> [args...]}"
shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$slot" == "extra" ]]; then
  REVERSE=1 MACHINE=0 exec "$HERE/_run_dense_nr_full.sh" "$@"
elif [[ "$slot" =~ ^[0-9]+$ ]]; then
  MACHINE="$slot" exec "$HERE/_run_dense_nr_full.sh" "$@"
else
  echo "error: slot must be a machine index or 'extra', got '$slot'" >&2
  exit 2
fi
