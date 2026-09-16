"""E3: build a scratch run dir holding a mount-balanced manifest of clips that are
ALREADY scored in the completed v3.2 run.

Why "already scored": E3 needs three things from one re-scoring pass, and only clips
with an existing record can give all three.
  1. a back-compatibility check -- the new probe must reproduce the stored
     `actor_flicker_raw` exactly, or the candidate comparison is against a moving
     baseline;
  2. `rho(actor_motion_raw, candidate)` per mount, which needs the motion column;
  3. the paired GT-delta (criterion 2), which needs BOTH halves of a pair.

TWO MODES
---------
`--pairs-per-mount` (the SELECTION run, 10/mount): small, cheap, whole GT/pred pairs.
Used to screen candidate statistics.

`--per-mount` (the CONFIRMATION run, 60/mount): sized so criterion 1 can actually
decide. `|rho| < 0.25` is an EQUIVALENCE test, and at n=10 an unconfounded statistic
fails it 45% of the time by noise alone while a still-confounded one (rho_true=0.35)
passes 38% of the time -- so the selection run cannot conclude anything. At n=60 the
false-fail rate drops to 5%. Use `--exclude-run` to hold out the clips already used
for selection: screening 16 candidates on the same 53 clips has an 87% chance (measured
by label permutation) that at least one passes by luck, so the winner MUST be
re-tested out-of-sample on clips that took no part in choosing it.
"""
import argparse
import collections
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path,
                    default=ROOT / 'metrics/WorldScore/results/dense_nr_v3_div30')
    ap.add_argument('--out', type=Path, default=Path('/tmp/e3_flicker/run'))
    ap.add_argument('--pairs-per-mount', type=int, default=0,
                    help='selection mode: N whole GT/pred pairs per mount')
    ap.add_argument('--per-mount', type=int, default=0,
                    help='confirmation mode: N prediction clips per mount (GT partner '
                         'added when one exists, for criterion 2)')
    ap.add_argument('--exclude-run', type=Path, default=None,
                    help='hold out every clip id already scored in this run dir, so the '
                         'confirmation sample is disjoint from the selection sample')
    ap.add_argument('--seed', type=int, default=20260916)
    args = ap.parse_args()
    if not (args.pairs_per_mount or args.per_mount):
        ap.error('give --pairs-per-mount (selection) or --per-mount (confirmation)')

    manifest = json.loads((args.src / 'manifest.json').read_text())
    scored = {p.stem for p in (args.src / 'results').glob('*.json')}
    print(f'source manifest {len(manifest)} rows, {len(scored)} scored')

    excluded = set()
    if args.exclude_run:
        excluded = {r['id'] for r in
                    json.loads((args.exclude_run / 'manifest.json').read_text())}
        print(f'holding out {len(excluded)} ids already used in {args.exclude_run}')

    # Motion column must be valid, else the clip cannot serve criterion 1.
    motion_ok = set()
    for p in (args.src / 'results').glob('*.json'):
        rec = json.loads(p.read_text())
        if rec.get('actor_motion_raw') is not None:
            motion_ok.add(rec['id'])

    by_pair = collections.defaultdict(dict)
    for r in manifest:
        if r['id'] in scored:
            by_pair[r['pair_key']][r['model']] = r

    rng = random.Random(args.seed)
    chosen, per_mount = {}, collections.Counter()
    seen_task = collections.Counter()

    if args.pairs_per_mount:
        target, need_gt = args.pairs_per_mount, True
    else:
        target, need_gt = args.per_mount, False

    cands = []
    for pk, models in by_pair.items():
        for m, pred in models.items():
            if m.startswith('gt') or pred['id'] in excluded:
                continue
            if pred['id'] not in motion_ok:
                continue
            gt = models.get(f'gt_{m}')
            if need_gt and gt is None:
                continue
            cands.append((pred, gt))
    # GT-paired clips first: they serve criterion 2 as well, at no extra cost beyond
    # their partner. Random within each tier so the sample is not path-ordered.
    rng.shuffle(cands)
    cands.sort(key=lambda c: (c[1] is None, rng.random()))

    for pred, gt in cands:
        mount = pred['camera_mount']
        if per_mount[mount] >= target:
            continue
        task = pred['relative'].rsplit('/', 2)[0]
        if seen_task[(mount, task)] >= max(2, target // 4):
            continue
        chosen[pred['id']] = pred
        if gt is not None:
            chosen[gt['id']] = gt
        per_mount[mount] += 1
        seen_task[(mount, task)] += 1

    rows = list(chosen.values())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'manifest.json').write_text(json.dumps(rows, indent=1))
    # Carry the OLD records over under a separate name so the comparison is local.
    ref = args.out / 'reference_v32'
    ref.mkdir(exist_ok=True)
    for r in rows:
        src = args.src / 'results' / f'{r["id"]}.json'
        if src.exists():
            shutil.copy2(src, ref / src.name)
    n_pred = sum(1 for r in rows if not r['model'].startswith('gt'))
    print(f'E3 manifest: {len(rows)} clips ({n_pred} pred + {len(rows) - n_pred} gt)')
    print('pred clips per mount:', dict(per_mount))
    print('models:', dict(collections.Counter(r['model'] for r in rows)))
    print(f'estimated {len(rows) * 94 / 3600:.1f} GPU-h '
          f'({len(rows) * 94 / 3600 / 3:.1f} h on 3 workers)')
    print(f'wrote {args.out}/manifest.json and {len(list(ref.glob("*.json")))} '
          f'reference records')


if __name__ == '__main__':
    main()
