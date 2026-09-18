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
#   MACHINES  total machines sharing the run (default 3)
#   WORKERS   workers per GPU (default 8; measured knee -- see comment below)
#   RUN       shared run dir (MUST be on the shared filesystem all machines mount)
#   GPU       CUDA_VISIBLE_DEVICES for this machine (default 0)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

: "${MACHINE:?set MACHINE=0|1|2 (use the run_dense_nr_full_m<k>.sh wrappers)}"
MACHINES="${MACHINES:-3}"
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

mkdir -p "$RUN/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$RUN/logs/machine${MACHINE}_${STAMP}.log"
TODO=$(python3 - "$RUN" <<'PY'
import json, sys, os
run = sys.argv[1]
rows = json.load(open(f'{run}/manifest.json'))
have = set(os.listdir(f'{run}/results')) if os.path.isdir(f'{run}/results') else set()
print(sum(1 for r in rows if f'{r["id"]}.json' not in have))
PY
)

cat <<EOF
=== dense-NR FULL wave ===
  machine        : $MACHINE of $MACHINES
  workers        : $WORKERS   (GPU $GPU)
  run dir        : $RUN
  manifest       : $(python3 -c "import json;print(len(json.load(open('$RUN/manifest.json'))))") clips
  still to score : $TODO   (this machine owns ~1/$MACHINES of them)
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
  >"$LOG" 2>&1 &
PGID=$!
echo "  launched PGID $PGID"
echo
echo "monitor : tail -f $LOG"
echo "          ls $RUN/results | wc -l          # global progress, all machines"
echo "          grep -c '^START' $RUN/worker_m${MACHINE}_*.log | paste -sd' '"
echo "stop    : kill -- -$PGID"
