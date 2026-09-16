"""Build a pilot acceleration manifest: N paired clips for chosen accel methods.

`accel_table()` in table_dense_nr.py pairs an accel clip to the DENSE record at the
same method-free `pair_key`, and requires that dense record to be in the results dir
being aggregated. div30's results only cover a sample, so a pilot must either reuse a
pair_key div30 already scored (free dense side) or score the dense clip too.

This prefers the first: it selects accel clips whose dense counterpart is ALREADY
scored in the reference run, so the pilot only pays for the accel side. Anything short
of the target is topped up with pairs where both sides must be scored.

Method choice for a pilot should span the two E7 regimes:
  - a caching method (dicache/worldcache): AMC's strong correct penalty regime
  - a blur-like method (sito/itm/svg2): AMC's known blind spot, needs the sharpness
    ratio from _exp_accel_character.py alongside

Usage:
  bash scripts/run_dense_nr.sh --out /tmp/accel_pilot --methods dense dicache sito \
      --inventory-only
  mv /tmp/accel_pilot/manifest.json /tmp/accel_pilot/full_inventory.json
  python scripts/_pilot_accel.py --out /tmp/accel_pilot --methods dicache sito --target 30
  bash scripts/run_dense_nr.sh --out /tmp/accel_pilot --workers 5 --reuse-manifest
"""
import argparse
import collections
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--methods', nargs='+', required=True)
    ap.add_argument('--target', type=int, default=30,
                    help='paired clips per accel method (pooled over model/mount)')
    ap.add_argument('--reference-run', type=Path,
                    default=ROOT / 'metrics/WorldScore/results/dense_nr_v3_div30')
    ap.add_argument('--seed', type=int, default=20260916)
    args = ap.parse_args()

    full = args.out / 'full_inventory.json'
    if not full.exists():
        raise SystemExit(f'missing {full} -- run the --inventory-only step first '
                         f'(see this file\'s docstring)')
    rows = json.loads(full.read_text())
    ref_scored = {p.stem for p in (args.reference_run / 'results').glob('*.json')}

    # (pair_key, model) -> {method: row}
    idx = collections.defaultdict(dict)
    for r in rows:
        m = r.get('method')
        if m == 'gt':
            continue
        idx[(r['pair_key'], r['model'])][m] = r

    rng = random.Random(args.seed)
    picked = {}
    report = []
    for method in args.methods:
        # Free-dense candidates first: dense side already scored in the reference run.
        free, paid = [], []
        for key, per_method in idx.items():
            a, d = per_method.get(method), per_method.get('dense')
            if not a or not d:
                continue
            if a['id'] in ref_scored:
                continue           # accel clip already scored somewhere
            (free if d['id'] in ref_scored else paid).append((a, d))
        rng.shuffle(free)
        rng.shuffle(paid)
        take_free = free[:args.target]
        take_paid = paid[:max(0, args.target - len(take_free))]
        for a, d in take_free:
            picked[a['id']] = a          # dense side comes from the reference run
        for a, d in take_paid:
            picked[a['id']] = a
            picked[d['id']] = d          # must score the dense side too
        report.append((method, len(free), len(paid), len(take_free), len(take_paid)))

    print('pilot selection (target %d paired clips per method):\n' % args.target)
    print('  %-12s %10s %10s %9s %9s' % (
        'method', 'free-dense', 'needs-both', 'take free', 'take paid'))
    for method, nf, np_, tf, tp in report:
        print('  %-12s %10d %10d %9d %9d' % (method, nf, np_, tf, tp))

    (args.out / 'manifest.json').write_text(json.dumps(list(picked.values()), indent=2))
    n_accel = sum(1 for r in picked.values() if r['method'] not in ('dense', 'gt'))
    print(f'\n{len(picked)} clips to score ({n_accel} accel + {len(picked)-n_accel} dense)')
    print('\nIMPORTANT: aggregate the pilot results TOGETHER with the reference run so')
    print('accel_table() can find each dense partner, e.g. symlink both results dirs')
    print('into one directory and point --run at it.')


if __name__ == '__main__':
    main()
