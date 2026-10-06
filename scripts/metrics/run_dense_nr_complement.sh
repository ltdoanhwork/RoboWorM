#!/usr/bin/env bash
# Score the part of the spec distribution that the CURRENTLY RUNNING wave can never reach.
#
#   bash scripts/metrics/run_dense_nr_complement.sh
#
# WHY THIS EXISTS (2026-09-21)
#   Four boxes are mid-flight on `manifest_grid_target300.json` -- a 51,072-id SAMPLE
#   (300 clips per cell). Measured against the authoritative 29,343-episode spec:
#
#     subset ids that serve the spec        49,272 of 83,804
#     spec view-clips the subset OMITS      34,532
#     ids in the subset that are NOT spec     1,800   (legacy gt-model anchors)
#     re-rendered rows inside the subset          0   <- see below
#
#   So that wave finishes at ~59% of the spec and stops. Worse, the 591 view-clips of the
#   197 re-rendered single_arm/MV/dreamgen dense episodes are in NEITHER its manifest (built
#   09-17, before the 09-19 render) nor its subset, so no amount of waiting produces them.
#
# WHY IT DOES NOT DISTURB THE RUNNING BOXES
#   `--only-ids` is applied BEFORE the positional stride partition, so a box with
#   MACHINES=1 and its own id list walks exactly that list. The list here is
#   `spec - subset`, asserted disjoint from what the four boxes walk, so the two waves
#   cannot collide. Clip `id` is `sha256(method:relative:view)[:20]` -- a pure content
#   hash -- so both waves share ONE results dir (this run dir's `results/` and `errors/`
#   are symlinks into dense_nr_full) and each skips whatever the other has already written.
#
#   It runs from a SEPARATE run dir for two reasons: it needs the rebuilt manifest (the
#   live one is stale and cannot be swapped while workers are alive, since the partition is
#   positional), and its worker logs would otherwise collide with the live wave's
#   `worker_m0_*.log`.
#
# AFTER THIS (phase 2)
#   When the four boxes finish their subset (~36 h), that is the moment to promote the
#   rebuilt manifest and relaunch ALL boxes on the whole manifest with no --only-ids:
#     python scripts/metrics/nr_promote_manifest.py --from .../_manifest_rebuild --to .../dense_nr_full
#     MACHINES=5 bash scripts/metrics/run_dense_nr_grid.sh <0..4>
#   Nothing is lost -- every result already written is skipped. Total time is the same as
#   restarting everything now (work/throughput is invariant), but no running box is touched.
#   See docs/dense_nr_full_spec_coverage.md.
#
# ENV
#   GPU      CUDA_VISIBLE_DEVICES (default 0)
#   WORKERS  workers per GPU (default 8 = measured knee on an 80GB A100)
#   REVERSE=1  walk the list backwards -- use this for a SECOND box on the complement,
#              which avoids re-sharding (there is only one backwards direction, so at most
#              one box may use it).
set -euo pipefail
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
SPEC="$REPO/metrics/WorldScore/results/dense_nr_spec"
LIVE="$REPO/metrics/WorldScore/results/dense_nr_full"
IDS="$SPEC/manifest_complement.json"

for f in "$SPEC/manifest.json" "$IDS"; do
  [ -f "$f" ] || { echo "ERROR: missing $f -- see docs/dense_nr_full_spec_coverage.md" >&2; exit 1; }
done
# The whole safety argument is disjointness; re-verify it rather than trust it.
"$REPO/models/dreamgen/.venv/bin/python" - "$IDS" "$LIVE/manifest_grid_target300.json" <<'PY'
import json, sys
comp = set(json.load(open(sys.argv[1])))
try:
    sub = set(json.load(open(sys.argv[2])))
except FileNotFoundError:
    print('note: live subset file absent; nothing to collide with')
    raise SystemExit(0)
both = comp & sub
if both:
    raise SystemExit(f'ERROR: {len(both)} ids are in BOTH the complement and the live '
                     f'subset. The two waves would duplicate work; rebuild the '
                     f'complement.')
print(f'disjointness OK: complement {len(comp)} ids, live subset {len(sub)} ids, '
      f'overlap 0')
PY

MACHINE=0 MACHINES=1 RUN="$SPEC" ONLY_IDS="$IDS" \
  exec "$REPO/scripts/metrics/_run_dense_nr_full.sh"
