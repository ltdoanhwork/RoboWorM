#!/usr/bin/env bash
# Core launcher for the FULL dense-NR wave, sharded across machines.
#
# Do not call this directly -- use run_dense_nr_full_m{0,1,2}.sh, which only set MACHINE.
# Keeping the logic in ONE file is deliberate: three copied-and-edited scripts is exactly
# how a shard index ends up duplicated (two machines on --machine 0 silently do the same
# 27,700 clips and leave a third of the manifest unscored, and because every write is
# idempotent nothing would ever error to tell you).
#
# HOW THE SPLIT WORKS
#   Machine k runs workers s=0..W-1, each taking rows[k*W + s :: W*MACHINES] of the
#   id-sorted manifest. That is an exact partition: verified 83,095 rows / 3 machines /
#   8 workers -> 83,095 covered, 0 duplicates, 0 missing, loads 27,703 / 27,696 / 27,696.
#   No coordination, no lock, no queue server. Every machine still walks in id order, so
#   each one stays representative of all models/mounts if you stop it early.
#
#   This supersedes --reverse (the old 2-machine trick, which relied on two queues meeting
#   in the middle and redoing the overlap). run_dense_nr.py rejects combining them.
#
# SAFETY / RESUMABILITY
#   * Each result is one atomic JSON named by the row's sha256 id, and a worker skips any
#     row whose file already exists. So restarting a machine, adding a machine later, or
#     killing one mid-flight costs at most the clips in flight.
#   * The run dir is pre-seeded with the 3,699 records already scored by the v3 runs
#     (hardlinked, not copied), so those are skipped for free.
#   * dump() writes to a hostname+pid-unique temp name before rename, so 3 machines on a
#     shared filesystem cannot interleave into one partial file.
#
# ENV OVERRIDES
#   MACHINES  total machines sharing the run (default 5)
#   WORKERS   workers per GPU (default 8; measured knee -- see comment below)
#   RUN       shared run dir (MUST be on the shared filesystem all machines mount)
#   GPU       CUDA_VISIBLE_DEVICES for this machine (default 0)
#   ONLY_IDS  JSON id list to restrict the wave to (see scripts/nr_table_coverage.py).
#             EVERY machine must pass the same file, or the stride partition stops
#             covering the subset exactly.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

: "${MACHINE:?set MACHINE=0..MACHINES-1 (use the run_dense_nr_full_m<k>.sh wrappers)}"
MACHINES="${MACHINES:-5}"
ONLY_IDS="${ONLY_IDS:-}"
# REVERSE=1 turns this box into an EXTRA machine that walks the id-sorted manifest
# BACKWARDS instead of taking a stride slot. That is the only way to add a machine to an
# already-running wave without restarting the others: the stride slots are all taken
# (3 machines x 8 workers = stride 24, offsets 0-23), so a 4th machine has no free offset.
#
# Why backwards is nearly free rather than 25% redundant: stride sharding makes each
# machine process a PREFIX of its own subsequence, so the finished set is a prefix of the
# whole manifest. Measured at 20% done -- deciles 0-10% are 100% complete, 10-20% are 73%,
# and everything past 30% is only ~2% (those are the pre-seeded v3 records). A reverse
# walker therefore starts in ~98% untouched territory and only begins duplicating work
# when the two fronts meet, i.e. when the wave is nearly finished. Overlap is harmless
# anyway: a worker skips any id whose result file already exists.
REVERSE="${REVERSE:-0}"
# 8 workers/GPU is the measured knee on an 80GB A100: throughput went 2.1 -> 3.0 clips/min
# from 5 -> 10 workers (sub-linear, GPU-bound), while 10 workers OOM-killed 13% of clips
# on 4-view-grid multiview, whose SAM2 propagation peaks far above the median. 8 + the
# expandable_segments allocator setting in run_dense_nr.sh is the reliable point.
WORKERS="${WORKERS:-8}"
RUN="${RUN:-$REPO/metrics/WorldScore/results/dense_nr_full}"
GPU="${GPU:-0}"

if [ ! -f "$RUN/manifest.json" ]; then
  echo "ERROR: no manifest at $RUN/manifest.json" >&2
  echo "Build it ONCE on any machine (CPU only, ~90s), then let all machines reuse it:" >&2
  echo "  cd $REPO && CUDA_VISIBLE_DEVICES= bash scripts/run_dense_nr.sh \\" >&2
  echo "    --out \"$RUN\" --inventory-only \\" >&2
  echo "    --methods dense dicache worldcache fastercache pisa radial svg1 svg2 sito itm" >&2
  echo "NOTE: run_dense_nr.sh cd's into metrics/WorldScore, so a RELATIVE --out lands in" >&2
  echo "      metrics/WorldScore/<your path>. Always pass an ABSOLUTE --out." >&2
  exit 1
fi

if [ "$REVERSE" = "1" ]; then
  # run_dense_nr.py rejects --reverse together with --machines>1: both mechanisms
  # partition the same manifest, and combining them would make this box walk a stride it
  # does not own from the other end. The reverse walker is its own single-machine queue.
  MACHINES=1
  MACHINE=0
fi

mkdir -p "$RUN/logs"

# ---------------------------------------------------------------------------------------
# WAVE MODE LOCK
#
# Every box in a wave must partition the SAME list of rows. The partition is positional
# (`rows[MACHINE*W + shard :: W*MACHINES]`), so three things must be identical on every
# box: MACHINES, WORKERS, and the row list itself (whole manifest vs an --only-ids
# subset). Any disagreement silently stops the union from being a partition -- some rows
# owned by nobody, others twice -- and because every write is idempotent NOTHING ERRORS.
#
# This is not hypothetical. As shipped, the two wrappers disagreed:
#   run_dense_nr_grid.sh      defaults TARGET=300  -> --only-ids, ~51k of 86,767 rows
#   run_dense_nr_multigpu.sh  hardcodes ONLY_IDS="" -> all 86,767 rows
# so the documented 5-box plan (indices 0-3 via grid + 4-7 via multigpu) would have had
# two halves walking different lists. Measured on the equivalent earlier bug: 30% of the
# remaining clips were orphaned and 24% done twice.
#
# So the FIRST box to launch pins the mode; later boxes must match or they refuse.
MODE_FILE="$RUN/wave_mode.json"
MODE_LIST="full-manifest"
[ -n "$ONLY_IDS" ] && MODE_LIST="ids:$(sha256sum "$ONLY_IDS" | cut -c1-16)"
MODE_NOW="machines=$MACHINES workers=$WORKERS list=$MODE_LIST"
# A REVERSE walker is exempt from the lock, and must be: it is NOT part of the stride
# partition at all. It forces MACHINES=1 above (run_dense_nr.py rejects --reverse with
# --machines>1), so its MODE_NOW would read `machines=1 ...` and never match a wave pinned
# at `machines=6 ...` -- the guard would block the one mechanism that is actually safe to
# add mid-wave. Its safety comes from a different property: it walks the id-sorted manifest
# backwards as its own queue, and the `dest.exists()` skip plus atomic per-id writes make
# any overlap with the forward front a no-op. It can neither orphan a row (the forward
# partition still covers all of them) nor double-score one in a way that matters.
# Only ONE reverse walker may exist -- there is a single backwards direction.
if [ "$REVERSE" = "1" ]; then
  if [ -f "$MODE_FILE" ]; then
    echo "REVERSE walker: joining wave pinned as '$(cat "$MODE_FILE")' (exempt from the"
    echo "  partition lock -- it is a separate backwards queue, not a stride slot)."
  fi
elif [ -f "$MODE_FILE" ]; then
  MODE_WAS="$(cat "$MODE_FILE")"
  if [ "$MODE_WAS" != "$MODE_NOW" ] && [ "${FORCE_MODE:-0}" != "1" ]; then
    cat >&2 <<EOF
ERROR: this box would join the wave with a DIFFERENT partition than the boxes already in it.

  wave was pinned as : $MODE_WAS
  this box wants     : $MODE_NOW

The split is positional, so a mismatch in machines / workers / row-list means some rows
are scored twice and others NEVER -- with no error, because every write is idempotent.

Fix ONE of these:
  * match the pinned mode (e.g. add FULL=1, or set TARGET / MACHINES / WORKERS to match);
  * or stop EVERY box (pkill -f 'run_dense_nr.py' on each), delete $MODE_FILE,
    and relaunch them all with the new mode.
FORCE_MODE=1 overrides -- only if every other box is already down.
EOF
    exit 1
  fi
else
  # Atomic: several boxes may launch at once.
  # Reached only when NOT a reverse walker (that branch returns above), so a reverse
  # walker never pins the mode. If it did, launching it first would pin `machines=1` and
  # then every real forward box would be refused.
  printf '%s' "$MODE_NOW" > "$MODE_FILE.$$.tmp" && mv "$MODE_FILE.$$.tmp" "$MODE_FILE"
  MODE_WAS="$(cat "$MODE_FILE")"
  if [ "$MODE_WAS" != "$MODE_NOW" ]; then
    echo "ERROR: another box pinned a different mode at the same instant:" >&2
    echo "  pinned: $MODE_WAS" >&2
    echo "  ours  : $MODE_NOW" >&2
    exit 1
  fi
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$RUN/logs/$([ "$REVERSE" = "1" ] && echo reverse || echo "machine${MACHINE}")_${STAMP}.log"
TODO=$(python3 - "$RUN" "$ONLY_IDS" <<'PY'
import json, sys, os
run, only = sys.argv[1], sys.argv[2]
rows = json.load(open(f'{run}/manifest.json'))
if only:
    keep = set(json.load(open(only)))
    rows = [r for r in rows if r['id'] in keep]
have = set(os.listdir(f'{run}/results')) if os.path.isdir(f'{run}/results') else set()
print(sum(1 for r in rows if f'{r["id"]}.json' not in have))
PY
)

cat <<EOF
=== dense-NR FULL wave ===
  mode           : $([ "$REVERSE" = "1" ] && echo 'REVERSE walker (extra machine, walks the manifest backwards)' || echo "machine $MACHINE of $MACHINES (stride shard)")
  workers        : $WORKERS   (GPU $GPU)
  run dir        : $RUN
  manifest       : $(python3 -c "import json;print(len(json.load(open('$RUN/manifest.json'))))") clips
  subset         : ${ONLY_IDS:-none (whole manifest)}
  still to score : $TODO
  log            : $LOG
EOF

# setsid puts the whole worker tree in its own process group, so it survives an ssh drop
# and can be killed as a unit: kill -- -<PGID from the line printed below>.
cd "$REPO"
setsid env CUDA_VISIBLE_DEVICES="$GPU" \
  bash scripts/run_dense_nr.sh \
    --out "$RUN" \
    --reuse-manifest \
    --workers "$WORKERS" \
    --machines "$MACHINES" \
    --machine "$MACHINE" \
    ${ONLY_IDS:+--only-ids "$ONLY_IDS"} \
    $([ "$REVERSE" = "1" ] && echo --reverse) \
  >"$LOG" 2>&1 &
PGID=$!
echo "  launched PGID $PGID"
echo
TAG=$([ "$REVERSE" = "1" ] && echo rev || echo "m${MACHINE}")
echo "monitor : tail -f $LOG"
echo "          bash scripts/dense_nr_status.sh"
echo "          grep -c '^START' $RUN/worker_${TAG}_*.log | paste -sd' '"
echo "stop    : kill -- -$PGID"
