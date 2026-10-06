#!/usr/bin/env bash
# Re-score clips that failed TRANSIENTLY (disk quota) and would otherwise be lost.
#
#   bash scripts/metrics/nr_rescore_quota_fails.sh [--dry-run]
#
# WHY THIS IS NEEDED
#   A worker walks `rows[offset::stride]` and skips a row only when `results/<id>.json`
#   exists. It never consults `errors/`. So each id is visited EXACTLY ONCE per wave:
#
#     * permanent failure (corrupt mp4, missing file) -> correctly abandoned
#     * TRANSIENT failure (disk quota) -> ALSO abandoned, silently, forever
#
#   Disk quota is transient by nature: it needs some other job to release space. The worker
#   already retries 3x with a 5/10/15-min backoff, but if the quota outlasts ~30 min the
#   clip lands in `errors/` and the wave moves on. Measured 2026-09-21: 14 clips lost this
#   way while the filesystem still had 379 TB free (it was a user/group quota blip) -- and
#   all 14 source videos were still on disk, i.e. purely lost work, not bad data.
#
# WHAT THIS DOES
#   Finds `errors/*.json` whose traceback is a quota error (EDQUOT/ENOSPC) AND whose source
#   video still exists, deletes those error stubs, and scores exactly those ids via
#   --only-ids. Removing the stub is safe: `errors/` is a report, never a skip-list.
#
# WHEN TO RUN
#   After the wave finishes (or any time quota has recovered). Check first that writes work:
#     dd if=/dev/zero of=<run>/.probe bs=1M count=20 && rm <run>/.probe
#   Safe to run while the wave is live: it only touches ids that no forward worker will
#   revisit, and every write is atomic + idempotent anyway.
set -euo pipefail
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
RUN="${RUN:-$REPO/metrics/WorldScore/results/dense_nr_full}"
PY="$REPO/models/dreamgen/.venv/bin/python"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

IDS="$RUN/manifest_quota_rescore.json"
"$PY" - "$RUN" "$IDS" "$DRY" <<'PYEOF'
import json, os, sys, glob
run, out, dry = sys.argv[1], sys.argv[2], sys.argv[3] == '1'
ids, gone, already = [], 0, 0
for p in glob.glob(f'{run}/errors/*.json'):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    tb = d.get('traceback', '')
    if 'Disk quota exceeded' not in tb and 'No space left' not in tb:
        continue
    if os.path.exists(f'{run}/results/{d["id"]}.json'):
        already += 1
        continue
    if not os.path.exists(d['path']):
        gone += 1          # source really is missing -> a permanent failure after all
        continue
    ids.append((d['id'], p))
print(f'quota failures that are re-scorable : {len(ids)}')
print(f'  already scored since (nothing to do): {already}')
print(f'  source video now missing (skip)     : {gone}')
if not ids:
    print('nothing to do'); sys.exit(1)
if dry:
    for i, _ in ids[:20]:
        print(f'  would re-score {i}')
    sys.exit(1)
json.dump([i for i, _ in ids], open(out, 'w'))
for _, p in ids:
    os.unlink(p)           # errors/ is a report, not a skip-list
print(f'wrote {out} and cleared {len(ids)} error stubs')
PYEOF

echo
echo "scoring $(python3 -c "import json;print(len(json.load(open('$IDS'))))") ids on GPU ${GPU:-0}"
cd "$REPO"
# Go through run_dense_nr.sh, NOT `python scripts/metrics/run_dense_nr.py` directly: the wrapper
# exports the PYTHONPATH for the six vendored third-party trees (SEA-RAFT/core, VFIMamba,
# groundingdino, sam2, ...) and cd's into metrics/WorldScore. Calling the .py directly dies
# with `ModuleNotFoundError: No module named 'core'` the moment a worker imports RAFT.
CUDA_VISIBLE_DEVICES="${GPU:-0}" \
  bash scripts/metrics/run_dense_nr.sh --out "$RUN" --reuse-manifest \
    --workers "${WORKERS:-4}" --only-ids "$IDS"
