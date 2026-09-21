"""Infer the LIVE shard geometry per box, from each worker's CURRENT run only.

Two earlier attempts failed, and the reasons are the useful part:

  1. gcd over a whole `worker_m<k>_<s>.log`. The file is opened in APPEND mode, so it
     holds every geometry the box has ever run (this run dir has both a --machines 3 and
     a --machines 5 wave). The gcd of a mixture is 1 = meaningless.
  2. residues of recently-written result files modulo a swept stride. When every offset
     has an owner, EVERY stride looks consistent (all residues appear), so the sweep
     cannot discriminate. It spuriously reported MACHINES=16.

What works: a worker prints `ONLY-IDS ...` (or, without a subset, nothing) exactly once
at startup, so everything after the LAST such marker belongs to the current process. A
single worker's own clip indices are an arithmetic progression with step W*MACHINES, so
the gcd of ITS successive gaps is the stride directly -- no sweep, no cross-box mixing.

Usage:
  python scripts/nr_shard_probe.py --run metrics/WorldScore/results/dense_nr_full \
      --ids metrics/WorldScore/results/dense_nr_full/manifest_grid_target300.json
"""
import argparse
import collections
import json
import math
import re
import time
from pathlib import Path

START = re.compile(r'^START (\S+) (.+)$')
# printed once per worker process, immediately before the first START
BANNER = re.compile(r'ONLY-IDS \d+ of \d+ manifest rows')


def worker_indices(log, index_of, restarts_ok=True):
    """Row indices this worker has STARTed since its last (re)start."""
    starts = []
    with log.open(errors='ignore') as fh:
        for line in fh:
            # tqdm writes \r inside a line, so a START can be mid-line
            for piece in line.split('\r'):
                piece = piece.strip()
                if restarts_ok and BANNER.search(piece):
                    starts = []          # new process -> forget the old geometry
                    continue
                m = START.match(piece)
                if m:
                    starts.append((m.group(1), m.group(2)))
    return [index_of[s] for s in starts if s in index_of]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--ids', default=None,
                    help='the --only-ids file the live boxes use. MUST match, or the row '
                         'indices differ and nothing lines up.')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--alive-minutes', type=float, default=15,
                    help='a worker log untouched for longer than this counts as dead')
    args = ap.parse_args()
    run = Path(args.run)

    manifest = json.loads((run / 'manifest.json').read_text())
    if args.ids:
        keep = set(json.loads(Path(args.ids).read_text()))
        rows = [r for r in manifest if r['id'] in keep]
    else:
        rows = manifest
    rows.sort(key=lambda r: r['id'])
    index_of = {}
    for i, r in enumerate(rows):
        index_of[(r['relative'], r['view'])] = i
    print(f'row list: {len(rows)} rows'
          + (f' (subset {Path(args.ids).name})' if args.ids else ' (whole manifest)'))

    cutoff = time.time() - args.alive_minutes * 60
    print()
    print(f'{"worker":<18}{"alive":>7}{"starts":>8}{"stride":>8}{"offset":>9}')
    per_box = collections.defaultdict(dict)
    for log in sorted(run.glob('worker_m*_*.log')):
        k, s = map(int, re.match(r'worker_m(\d+)_(\d+)\.log', log.name).groups())
        alive = log.stat().st_mtime >= cutoff
        idx = sorted(set(worker_indices(log, index_of)))
        stride = 0
        for g in (b - a for a, b in zip(idx, idx[1:])):
            stride = math.gcd(stride, g)
        off = idx[0] % stride if stride else None
        print(f'{log.name:<18}{"yes" if alive else "no":>7}{len(idx):>8}'
              f'{stride or "-":>8}{off if off is not None else "-":>9}')
        if alive and stride:
            per_box[k][s] = (stride, off)

    print()
    if not per_box:
        raise SystemExit('no live worker has enough STARTs since its last restart yet; '
                         'wait a few minutes and re-run')
    strides = collections.Counter(st for w in per_box.values() for st, _ in w.values())
    print(f'strides observed on live workers: {dict(strides)}')
    stride = strides.most_common(1)[0][0]
    machines = stride // args.workers
    print(f'=> W*MACHINES = {stride}  =>  MACHINES = {machines} (at W={args.workers})')
    if len(strides) > 1:
        print('   !! more than one stride is live -- the boxes were NOT launched with '
              'the same --machines, so the "partition" overlaps in some places and '
              'leaves holes in others. Fix that before adding a box.')

    owned = collections.defaultdict(list)
    for k, w in per_box.items():
        for s, (st, off) in w.items():
            if st == stride:
                owned[off].append(f'm{k}_{s}')
    live_machines = sorted({o // args.workers for o in owned})
    free = [m for m in range(machines) if m not in live_machines]
    print(f'=> machine indices WORKING : {live_machines}')
    print(f'=> free indices in [0,{machines}) : {free if free else "none"}')
    dup = {o: v for o, v in owned.items() if len(v) > 1}
    if dup:
        print(f'   !! offsets claimed twice: '
              + ', '.join(f'{o} by {v}' for o, v in sorted(dup.items())))

    print()
    if free:
        print(f'THIS BOX: take machine index {free[0]}.')
        print(f'  MACHINES={machines} bash scripts/run_dense_nr_grid.sh {free[0]}')
    else:
        nxt = machines + 1
        print(f'THIS BOX: no free slot. Every index in [0,{machines}) is working, and '
              f'those {stride} offsets already cover every row.')
        print(f'  A higher index is rejected (--machine must be < --machines), and '
              f'launching with MACHINES={nxt} only repeats rows the others own '
              f'(idempotent, so a no-op rather than a speedup).')
        print(f'  To use this box, restart ALL of them with MACHINES={nxt}.')


if __name__ == '__main__':
    main()
