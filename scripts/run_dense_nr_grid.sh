#!/usr/bin/env bash
# Launch ONE machine of the dense-NR wave that covers the FULL spec distribution.
#
#   MACHINES=<N> bash scripts/run_dense_nr_grid.sh <machine 0..N-1>
#
# Run it on each of the N boxes with a DIFFERENT index, and the SAME MACHINES value. The
# split is index-based (`rows[k*W + s :: W*MACHINES]`), so two boxes on the same index
# silently do the same work and leave 1/N of the manifest unscored -- and because every
# write is idempotent, nothing errors to tell you. Check with
# `bash scripts/dense_nr_status.sh` and `scripts/nr_shard_probe.py`.
#
# WHY THIS NOW SCORES THE WHOLE MANIFEST (changed 2026-09-21)
#   The authoritative distribution is 29,343 episodes (dense 3,166 + caching 8,693 +
#   sparse attention 12,664 + token reduction 4,820). In view-clips that is 83,804 of the
#   manifest's 86,767 rows -- the other 2,963 are legacy gt-model anchors. The spec is
#   ~97% of the manifest, so there is nothing meaningful to subset away and the whole
#   manifest is a SUPERSET of the spec.
#
#   The old default (TARGET=300 via --only-ids) *sampled* 300 clips per cell. That gives a
#   tight per-cell CI but does NOT cover the distribution, and it requires every box to
#   pass a byte-identical id file -- --only-ids is applied BEFORE the positional stride
#   partition, so boxes on different lists own misaligned slices. Dropping it removes that
#   whole failure class. Verify coverage instead with:
#     python scripts/nr_distribution_audit.py --run <RUN> --with-gt
#
#   Cost of full coverage from 37% done: 52,799 clips left
#     4 boxes 100.9 h | 5 boxes 80.7 h | 6 boxes 67.3 h   (@2.18 clips/min/box)
#
# ENV
#   MACHINES  number of boxes in this wave (default 5). MUST match on every box.
#   GPU       CUDA_VISIBLE_DEVICES on this box (default 0)
#   WORKERS   workers per GPU (default 8 = measured knee on an 80GB A100)
#   RUN       shared run dir (default metrics/WorldScore/results/dense_nr_full)
#   TARGET=N  opt back IN to per-cell sampling instead of full coverage (not the spec)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MACHINES="${MACHINES:-5}"
MACHINE="${1:?usage: MACHINES=<N> bash scripts/run_dense_nr_grid.sh <machine 0..N-1>}"
RUN="${RUN:-$REPO/metrics/WorldScore/results/dense_nr_full}"

if [ "$MACHINE" -lt 0 ] || [ "$MACHINE" -ge "$MACHINES" ]; then
  echo "ERROR: machine index $MACHINE must be in [0, $MACHINES)" >&2
  exit 1
fi

# Refuse to run against a manifest that predates the newest rendered video: that is exactly
# how the 197 re-rendered single_arm/MV/dreamgen dense episodes stayed invisible for 2 days
# (manifest built 09-17, episodes rendered 09-19).
#
# This used to be `find "$REPO/sampling_dataset" -name '*.mp4' -newer ... -print -quit`.
# It is NOT cheap: -quit can only fire once a newer file is found, so when the manifest IS
# fresh the find walks all ~170k mp4s of a 169 GB tree on a shared parallel filesystem.
# Measured: >150 s, i.e. it HUNG every launch (two stuck launches were traced to exactly
# this line). Compare against a rebuilt manifest instead -- same signal, and it reports the
# real number of missing rows rather than one example path.
REBUILD="${REBUILD_DIR:-$REPO/metrics/WorldScore/results/_manifest_rebuild}/manifest.json"
if [ -f "$REBUILD" ] && [ "$REBUILD" -nt "$RUN/manifest.json" ]; then
  DRIFT=$(python3 - "$RUN/manifest.json" "$REBUILD" <<'PY'
import json, sys
live = {r['id'] for r in json.load(open(sys.argv[1]))}
new = {r['id'] for r in json.load(open(sys.argv[2]))}
print(f'{len(new - live)} {len(live - new)}')
PY
  )
  echo "WARNING: a newer rebuilt manifest exists ($REBUILD)." >&2
  echo "  it adds $(echo "$DRIFT" | cut -d' ' -f1) rows and drops $(echo "$DRIFT" | cut -d' ' -f2) vs the live manifest." >&2
  echo "  Promote it first (stop every box, then):" >&2
  echo "    models/dreamgen/.venv/bin/python scripts/nr_promote_manifest.py \\" >&2
  echo "      --from $(dirname "$REBUILD") --to $RUN" >&2
  echo "  Set SKIP_FRESHNESS_CHECK=1 to proceed on the current manifest anyway." >&2
  [ "${SKIP_FRESHNESS_CHECK:-0}" = "1" ] || exit 1
fi

IDS=""
if [ -n "${TARGET:-}" ]; then
  # Opt-in per-cell sampling. NOT the spec -- only for a quick tighter-CI pass.
  IDS="$RUN/manifest_grid_target${TARGET}.json"
  mkdir -p "$RUN/logs"
  if [ ! -f "$IDS" ]; then
    "$REPO/models/dreamgen/.venv/bin/python" "$REPO/scripts/nr_table_coverage.py" \
      --run "$RUN" --target "$TARGET" --ids-out "$IDS" \
      >"$RUN/logs/coverage_target${TARGET}.txt" 2>&1
  fi
  echo "SUBSET MODE (target $TARGET/cell) -- this does NOT cover the spec distribution."
  echo "  ids $(python3 -c "import json;print(len(json.load(open('$IDS'))))")"
  echo "  sha $(sha256sum "$IDS" | cut -c1-12)  <- must match on all $MACHINES boxes"
else
  echo "FULL COVERAGE MODE: whole manifest ($(python3 -c \
    "import json;print(len(json.load(open('$RUN/manifest.json'))))") view-clips), \
no --only-ids."
fi
echo "machine $MACHINE of $MACHINES"

MACHINE="$MACHINE" MACHINES="$MACHINES" RUN="$RUN" ONLY_IDS="$IDS" \
  exec "$REPO/scripts/_run_dense_nr_full.sh"
