"""E5: build the acceleration-wave manifest — 30 clips per (method, embodiment, mount),
restricted to view-clips whose DENSE counterpart is already scored.

WHY A SEPARATE BUILDER (and not `--smoke-paired-episodes`)
---------------------------------------------------------
`smoke_manifest` pairs a prediction against its own co-located GT (`gt_<model>`). The
acceleration reference is different: per the PhyFid-DR rule the reference for a method
clip is the **dense prediction of the same episode+view**, not GT. Two consequences
that this builder enforces and the smoke sampler cannot:

* a method clip is only useful if `sampling_dataset/dense` has a SCORED result at the
  same `pair_key`. Sampling method clips freely would produce rows with nothing to
  pair against -- and dense NR results exist only for the 1,573-clip div30 sample, not
  for all ~10k dense videos, so most method clips have no partner.
* GT rows must NOT be re-emitted. The accel trees ship byte-identical GT copies, and
  `inventory()` already only takes GT from the dense tree; here we emit *only* the
  method rows, because their dense partners are already scored in the reference run.

Statistical power comes from paired clips, so the cap is per clip, not per episode.
Output is a run dir whose `manifest.json` holds only the method rows; scoring it with
`--reuse-manifest` produces records that `table_dense_nr.py` can pair against the
reference run's dense records via the method-free `pair_key`.
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

METHODS = ('worldcache', 'dicache', 'fastercache', 'itm', 'sito',
           'pisa', 'radial', 'svg1', 'svg2')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dense-run', type=Path,
                    default=ROOT / 'metrics/WorldScore/results/dense_nr_v3_div30',
                    help='run dir whose SCORED dense records are the pairing reference')
    ap.add_argument('--out', type=Path, default=ROOT / 'metrics/WorldScore/results/accel_nr_v1')
    ap.add_argument('--methods', nargs='+', default=list(METHODS))
    ap.add_argument('--per-group', type=int, default=30,
                    help='clips per (method, embodiment, camera_mount)')
    ap.add_argument('--seed', type=int, default=20260916)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    import run_dense_nr as R

    dense_manifest = json.loads((args.dense_run / 'manifest.json').read_text())
    scored = {p.stem for p in (args.dense_run / 'results').glob('*.json')}
    # Pair keys with a SCORED dense prediction. GT rows are excluded: the accel
    # reference is the dense prediction, not GT.
    dense_ok = {r['pair_key'] for r in dense_manifest
                if r['method'] == 'dense' and r['id'] in scored}
    print(f'dense reference: {len(scored)} scored records, '
          f'{len(dense_ok)} pairable pair_keys')

    tmp = args.out / '_inventory'
    tmp.mkdir(parents=True, exist_ok=True)
    from phyworld_index import _instruction_from_dir, _instruction_from_txt
    rng = random.Random(args.seed)
    rows, report = [], []
    for method in args.methods:
        tree = ROOT / 'sampling_dataset' / method
        if not tree.exists():
            print(f'  SKIP {method}: no tree at {tree}')
            continue
        # Same walk as the dense inventory, so an accel clip is inventoried by exactly
        # the same rules and the pair is comparable by construction.
        cand = [r for r in R._pred_rows(tree, method, _instruction_from_dir,
                                        _instruction_from_txt)
                if r['pair_key'] in dense_ok]
        groups = collections.defaultdict(list)
        for r in cand:
            groups[(r['embodiment'], r['camera_mount'])].append(r)
        picked = []
        for key, g in sorted(groups.items()):
            take = R.spread_sample(g, args.per_group, rng,
                                   key=lambda r: (R.task_of(r), r['split']))
            picked.extend(take)
            report.append((method, key[0], key[1], len(g), len(take)))
        rows.extend(picked)
        print(f'  {method:12} pairable {len(cand):5} -> sampled {len(picked):4} '
              f'over {len(groups)} groups')

    short = [r for r in report if r[4] < args.per_group]
    print(f'\ngroups: {len(report)}, short of {args.per_group}: {len(short)}')
    for m, emb, mount, avail, took in short:
        print(f'  SHORT {m} {emb} {mount}: only {avail} pairable')
    est_h = len(rows) * 94 / 3600
    print(f'\nTOTAL {len(rows)} clips ~= {est_h:.1f} GPU-h '
          f'(~{est_h / 5:.1f} h wall on 5 workers)')
    if args.dry_run:
        return
    R.dump(args.out / 'manifest.json', rows)
    R.dump(args.out / 'accel_selection.json',
           {'dense_run': str(args.dense_run), 'per_group': args.per_group,
            'seed': args.seed, 'methods': args.methods,
            'groups': [{'method': m, 'embodiment': e, 'camera_mount': c,
                        'pairable': a, 'sampled': t} for m, e, c, a, t in report]})
    print(f'wrote {args.out}/manifest.json')
    print(f'\nScore it with:\n  bash scripts/run_dense_nr.sh --out {args.out} '
          f'--workers 5 --reuse-manifest')


if __name__ == '__main__':
    main()
