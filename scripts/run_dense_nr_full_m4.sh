#!/usr/bin/env bash
# MACHINE 4 -- full dense-NR wave (whole manifest). MACHINES defaults to 5.
#
# For the paper-grid wave use scripts/run_dense_nr_grid.sh 4 instead -- it also builds
# the per-cell id subset so the table finishes in ~1.25 days instead of ~3.5.
#
# Do NOT add this box to a wave the other machines started with MACHINES=3: 3 machines x
# 8 workers is stride 24, which already covers every row, so a 4th/5th stride index would
# duplicate 100% of its work. Restart ALL boxes with MACHINES=5.
set -euo pipefail
MACHINE=4 exec "$(dirname "$0")/_run_dense_nr_full.sh" "$@"
