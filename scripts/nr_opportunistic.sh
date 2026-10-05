#!/usr/bin/env bash
# Carve the REMAINING work into disjoint slices for opportunistic (best-effort) boxes.
#
#   bash scripts/nr_opportunistic.sh plan  <name>:<gpus>:<workers> ...
#   bash scripts/nr_opportunistic.sh launch <name> <gpu> <slice-index> <workers>
#
# WHY NOT `REVERSE=1` ON EACH BORROWED BOX
#   There is exactly ONE backwards direction. `--reverse` makes a box walk the id-sorted
#   manifest from the end as its own queue; two boxes doing that walk the SAME sequence and
#   redo each other's clips. It is the right tool for adding ONE extra machine, not twelve.
#
# WHY NOT GIVE THEM STRIDE SLOTS EITHER
#   A stride slot owns a fixed 1/MACHINES of the manifest and nobody else will ever visit it.
#   Borrowed shells (other teams' projects) can be reclaimed at any moment, and a slot whose
#   box vanished leaves a permanent hole -- with no error, since every write is idempotent.
#   Adding slots also forces a new MACHINES, i.e. a full restart of the healthy boxes.
#
# THE DESIGN HERE
#   Each opportunistic GPU gets its own run dir with `results/` and `errors/` SYMLINKED to
#   the main run, plus a private `--only-ids` slice. Consequences:
#     * slices are disjoint by construction -> no duplicated work among borrowed boxes
#     * the main 6-box forward wave still owns and covers EVERY row, so a borrowed box
#       dying costs nothing: those ids just get scored later by their real owner
#     * each dir has its own wave_mode.json -> no lock conflict with the main wave, and no
#       restart of the healthy boxes
#     * work lands in the SHARED results/, so it counts immediately
#
#   Slices are taken from the END of the remaining-id list and each box walks its slice in
#   reverse. The forward wave processes a PREFIX of each shard, so the tail is the region it
#   reaches last -- that is where a borrowed box adds the most and collides the least.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MAIN="${MAIN:-$REPO/metrics/WorldScore/results/dense_nr_full}"
OPP="${OPP:-$REPO/metrics/WorldScore/results/nr_opp}"
PY="$REPO/models/dreamgen/.venv/bin/python"
CMD="${1:?usage: nr_opportunistic.sh plan|launch ...}"; shift

if [ "$CMD" = "plan" ]; then
  # plan <total slices>  -- writes OPP/slice_<i>.json, disjoint, tail-first
  N="${1:?usage: nr_opportunistic.sh plan <total slices>}"
  mkdir -p "$OPP"
  "$PY" - "$MAIN" "$OPP" "$N" "$REPO" <<'PYEOF'
import json, os, sys
main, opp, n, repo = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
sys.path.insert(0, f'{repo}/scripts')
from run_dense_nr import excluded_row
rows = json.load(open(f'{main}/manifest.json'))
scored = {p[:-5] for p in os.listdir(f'{main}/results') if p.endswith('.json')}
# Same scope the main wave uses, so a borrowed box never scores an excluded 3-view clip.
todo = [r['id'] for r in rows if not excluded_row(r) and r['id'] not in scored]
print(f'remaining in scope: {len(todo)}')
# Manifest order == id-sorted order, which is the order the forward wave walks. Take the
# TAIL: the forward wave finishes a prefix of each shard, so the tail is least contended.
tail = todo[len(todo) // 2:]          # back half
print(f'tail (back half) used for opportunistic boxes: {len(tail)}')
per = len(tail) // n
for i in range(n):
    lo = i * per
    hi = (i + 1) * per if i < n - 1 else len(tail)
    part = tail[lo:hi][::-1]          # reverse: start furthest from the forward front
    json.dump(part, open(f'{opp}/slice_{i}.json', 'w'))
    print(f'  slice_{i}: {len(part)} ids')
# Prove disjointness rather than trusting the arithmetic.
seen, dup = set(), 0
for i in range(n):
    s = set(json.load(open(f'{opp}/slice_{i}.json')))
    dup += len(seen & s); seen |= s
assert dup == 0, f'slices overlap by {dup} ids'
print(f'disjointness OK: {len(seen)} unique ids across {n} slices, 0 overlap')
PYEOF
  exit 0
fi

if [ "$CMD" = "launch" ]; then
  NAME="${1:?name}"; GPU="${2:?gpu}"; IDX="${3:?slice index}"; W="${4:-6}"
  RUN="$OPP/$NAME-gpu$GPU"
  IDS="$OPP/slice_$IDX.json"
  [ -f "$IDS" ] || { echo "ERROR: no $IDS -- run 'plan' first" >&2; exit 1; }
  mkdir -p "$RUN"
  # Share the scored set + error reports with the main run; keep logs/mode local.
  [ -e "$RUN/results" ] || ln -s "$MAIN/results" "$RUN/results"
  [ -e "$RUN/errors" ]  || ln -s "$MAIN/errors"  "$RUN/errors"
  [ -e "$RUN/manifest.json" ] || cp -l "$MAIN/manifest.json" "$RUN/manifest.json" 2>/dev/null \
    || cp "$MAIN/manifest.json" "$RUN/manifest.json"
  echo "opportunistic $NAME gpu$GPU slice=$IDX workers=$W"
  echo "  ids   : $("$PY" -c "import json;print(len(json.load(open('$IDS'))))")"
  echo "  run   : $RUN"
  MACHINE=0 MACHINES=1 WORKERS="$W" GPU="$GPU" RUN="$RUN" ONLY_IDS="$IDS" \
    exec "$REPO/scripts/_run_dense_nr_full.sh"
fi

echo "unknown command: $CMD" >&2; exit 1
