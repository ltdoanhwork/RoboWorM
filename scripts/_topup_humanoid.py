"""Top up the humanoid strata that cannot carry a claim (n < MIN_PAIRS_FOR_CLAIM).

The div30 run left humanoid at 34 paired AMC clips spread over 8 (model x mount)
cells, 6 of them below n=8, so every humanoid claim was withdrawn by the n>=8 floor.
Pooling over mount was rejected: mounts disagree in sign (wrist +10.6, head -0.3) and
head is motion-confounded (rho -0.54), so pooling manufactures a wrong-signed claim.

This builds a TOP-UP manifest instead: only humanoid clips, only from pair_keys where
BOTH the prediction and its co-located GT exist (an unpaired clip cannot help a paired
claim), excluding everything div30 already scored, and allocated per (model, mount)
cell up to a target. The output is appended to the existing run directory, so the
scorer resumes into the same results dir and `table_dense_nr.py` sees one population.

Usage:
  python scripts/_topup_humanoid.py --run <run_dir> --target 12 --out /tmp/topup
  bash scripts/run_dense_nr.sh --out /tmp/topup --workers 5 --reuse-manifest
  # then aggregate: point table_dense_nr.py at a dir holding both results sets
"""
import argparse
import collections
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path,
                    default=ROOT / 'metrics/WorldScore/results/dense_nr_v3_div30')
    ap.add_argument('--target', type=int, default=12,
                    help='paired clips to reach per (model, mount) cell')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--embodiment', default='humanoid')
    ap.add_argument('--seed', type=int, default=20260916)
    args = ap.parse_args()

    full_manifest = args.out / 'full_inventory.json'
    if not full_manifest.exists():
        raise SystemExit(
            f'Build the full inventory first:\n'
            f'  bash scripts/run_dense_nr.sh --out {args.out} --methods dense '
            f'--inventory-only\n'
            f'  mv {args.out}/manifest.json {full_manifest}')
    rows = json.loads(full_manifest.read_text())

    already = {r['id'] for r in json.loads((args.run / 'manifest.json').read_text())}
    scored = {p.stem for p in (args.run / 'results').glob('*.json')}

    # Group by pair_key so we only take clips whose GT partner also exists; a clip
    # without its anchor cannot contribute to a paired delta.
    groups = collections.defaultdict(dict)
    for r in rows:
        if r['embodiment'] != args.embodiment:
            continue
        groups[r['pair_key']][r['model']] = r

    # Current paired coverage per (model, mount), counted the same way the table does.
    have = collections.Counter()
    for pk, v in groups.items():
        for model, r in v.items():
            if model.startswith('gt'):
                continue
            if f'gt_{model}' not in v:
                continue
            if r['id'] in scored and v[f'gt_{model}']['id'] in scored:
                have[(model, r['camera_mount'])] += 1

    # Candidate pairs not yet scored.
    cands = collections.defaultdict(list)
    for pk, v in sorted(groups.items()):
        for model, r in v.items():
            if model.startswith('gt'):
                continue
            anchor = v.get(f'gt_{model}')
            if not anchor:
                continue
            if r['id'] in already or anchor['id'] in already:
                continue
            cands[(model, r['camera_mount'])].append((r, anchor))

    import random
    rng = random.Random(args.seed)
    picked, plan = [], []
    for cell in sorted(set(list(have) + list(cands))):
        need = max(0, args.target - have[cell])
        pool = cands.get(cell, [])
        rng.shuffle(pool)
        take = pool[:need]
        plan.append((cell, have[cell], need, len(pool), len(take)))
        for r, anchor in take:
            picked.extend([r, anchor])

    print(f'{args.embodiment} top-up plan (target n={args.target} per cell):\n')
    print('  %-24s %-8s %5s %6s %9s %6s' % ('model', 'mount', 'have', 'need', 'available', 'take'))
    for (model, mount), h, need, avail, take in plan:
        flag = '' if take >= need else '  <-- still short'
        print('  %-24s %-8s %5d %6d %9d %6d%s' % (model, mount, h, need, avail, take, flag))

    # De-duplicate: one pair_key can be reached from several cells.
    uniq = {r['id']: r for r in picked}
    (args.out / 'manifest.json').write_text(json.dumps(list(uniq.values()), indent=2))
    n_pred = sum(1 for r in uniq.values() if not r['model'].startswith('gt'))
    print(f'\n{len(uniq)} clips to score ({n_pred} pred + {len(uniq)-n_pred} GT anchors)')
    print(f'  -> {args.out}/manifest.json')
    print(f'\nRun:\n  bash scripts/run_dense_nr.sh --out {args.out} --workers 5 --reuse-manifest')


if __name__ == '__main__':
    main()
