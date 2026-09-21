"""Promote a rebuilt manifest into the live run dir, safely.

WHY THIS EXISTS
The live manifest was built 2026-09-17 and had drifted from disk in BOTH directions:
  * missing 591 view-clips (197 single_arm/MV/dreamgen dense episodes re-rendered 09-19),
    which made that spec cell read 60/128 makovian and 0/129 non_makovian;
  * counting 567 view-clips whose mp4 had since been deleted (fastercache 267, svg1 260,
    pisa 34, radial 6), which inflated fastercache to 2620 against a spec of 2361.
Rebuilding fixes both and lands every method exactly on the spec (delta +0, total 29343
episodes).

WHY IT MUST NOT BE A PLAIN `cp`
Workers partition the manifest by POSITION: `rows[k*W + s :: W*MACHINES]` over the
id-sorted list. Replacing manifest.json under a running box shifts every row's index, so
the live boxes would silently stop forming a partition -- some rows scored twice, others
never. Clip `id` is `sha256(method:relative:view)[:20]`, a pure content hash, so already
scored results stay valid across the swap; it is only the in-flight POSITION that breaks.
Hence: refuse while workers are alive.

Usage:
  python scripts/nr_promote_manifest.py \
      --from metrics/WorldScore/results/_manifest_rebuild \
      --to   metrics/WorldScore/results/dense_nr_full
  # add --force only if you accept restarting every box afterwards anyway
"""
import argparse
import json
import shutil
import time
from pathlib import Path


def live_workers(run, minutes):
    cutoff = time.time() - minutes * 60
    return sorted(p.name for p in run.glob('worker_*.log')
                  if p.stat().st_mtime >= cutoff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='src', required=True)
    ap.add_argument('--to', dest='dst', required=True)
    ap.add_argument('--alive-minutes', type=float, default=10)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)

    new = json.loads((src / 'manifest.json').read_text())
    old = json.loads((dst / 'manifest.json').read_text())
    scored = {p.stem for p in (dst / 'results').glob('*.json')}
    new_ids, old_ids = {r['id'] for r in new}, {r['id'] for r in old}

    print(f'live manifest : {len(old)} view-clips  ({dst / "manifest.json"})')
    print(f'new manifest  : {len(new)} view-clips  ({src / "manifest.json"})')
    print(f'  added   {len(new_ids - old_ids)}')
    print(f'  removed {len(old_ids - new_ids)}')
    print(f'scored results: {len(scored)}')
    orphan = scored - new_ids
    print(f'  still in the new manifest : {len(scored & new_ids)}')
    print(f'  ORPHANED by the swap      : {len(orphan)}')
    if orphan:
        print('    (these are results for clips whose mp4 was deleted; they stay on disk '
              'and are simply no longer counted as progress)')

    alive = live_workers(dst, args.alive_minutes)
    if alive and not args.force:
        print()
        print(f'REFUSING: {len(alive)} worker log(s) touched in the last '
              f'{args.alive_minutes:g} min:')
        for a in alive[:8]:
            print(f'  {a}')
        print('Workers partition the manifest BY POSITION, so swapping it under them '
              'breaks the partition silently (double-scored rows and permanent holes).')
        print('Stop every box first:')
        print("  pkill -f 'run_dense_nr.py'   # on EACH box")
        print('Then re-run this with the same arguments.')
        raise SystemExit(1)
    if alive:
        print(f'\n--force given, proceeding despite {len(alive)} live worker log(s). '
              f'Every box MUST be restarted after this.')

    backup = dst / f'manifest.json.bak.{time.strftime("%Y%m%d-%H%M%S")}'
    shutil.copy2(dst / 'manifest.json', backup)
    shutil.copy2(src / 'manifest.json', dst / 'manifest.json')
    print(f'\nbacked up old manifest -> {backup.name}')
    print(f'installed new manifest -> {dst / "manifest.json"}')

    # Any --only-ids subset built from the old manifest is now stale: it names ids that
    # were dropped and cannot name the 591 new ones. Rename rather than warn -- a warning
    # in a log nobody reads is how the old subset would get passed to --only-ids again,
    # and that silently reintroduces the 34,532-clip spec hole.
    stale = sorted(dst.glob('manifest_grid_target*.json')) + \
        sorted(dst.glob('manifest_table_subset*.json'))
    for s in stale:
        if s.name.endswith('.stale'):
            continue
        s.rename(s.with_suffix('.json.stale'))
        print(f'retired stale id subset: {s.name} -> {s.name}.stale')
    if stale:
        print('Full spec coverage needs the WHOLE manifest anyway (spec = 83,804 of '
              '86,767 view-clips), so --only-ids should not be used at all.')

    todo = sum(1 for r in new if r['id'] not in scored)
    print(f'\nremaining to score: {todo} of {len(new)} '
          f'({100 * (len(new) - todo) / len(new):.1f}% done)')
    for boxes in (4, 5, 6):
        h = todo / (2.18 * 60 * boxes)
        print(f'  {boxes} boxes @2.18 clips/min/box : {h:.1f} h ({h / 24:.2f} days)')


if __name__ == '__main__':
    main()
