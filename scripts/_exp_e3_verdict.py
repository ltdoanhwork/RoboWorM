"""E3 verdict: score every flicker candidate against the three pre-registered criteria.

Criteria (from docs/dense_nr_experiments.md E3; ALL must pass or the column is dropped):
  1. |rho(actor_motion_raw, candidate)| < 0.25 on EVERY mount;
  2. GT still beats pred on the paired candidate in >= 65% of pairs;
  3. a visual ordering check on hand-picked shimmering vs clean clips (done separately,
     using the clips this script prints as the extremes).

Also reports, because they decide whether a passing candidate is worth anything:
  * spread ratio p90/p10 of the candidate on GT clips -- a statistic that is
    motion-independent because it is CONSTANT has no discriminative power, and would
    pass criterion 1 trivially. This is the trap that PSNR/rho alone cannot catch.
  * Spearman with `raw`, so it is visible whether a candidate is a genuine
    re-definition or a monotone relabelling of the confounded statistic.
"""
import argparse
import collections
import json
import math
import random
import statistics
from pathlib import Path

B = 10000


def spearman(xs, ys):
    if len(xs) < 8:
        return None

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def boot_ci(d, seed=7):
    if len(d) < 5:
        return None, None
    rng = random.Random(seed)
    n = len(d)
    meds = sorted(statistics.median([d[rng.randrange(n)] for _ in range(n)])
                  for _ in range(B // 10))
    return meds[int(0.025 * len(meds))], meds[int(0.975 * len(meds))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, default=Path('/tmp/e3_flicker/run'))
    args = ap.parse_args()
    rows = [json.loads(p.read_text()) for p in sorted((args.run / 'results').glob('*.json'))]
    print(f'{len(rows)} clips scored in {args.run}')
    have = [r for r in rows if r.get('actor_flicker_variants')]
    if not have:
        raise SystemExit('no flicker variants in these records')
    cands = [k for k in have[0]['actor_flicker_variants']
             if k not in ('copy', 'flow', 'grad', 'align_pred')]

    # ---- criterion 1: rho with motion, per mount
    print('\n== C1: |rho(actor_motion_raw, candidate)| < 0.25 on EVERY mount ==')
    mounts = sorted({r['camera_mount'] for r in have})
    print(f'{"candidate":<14}' + ''.join(f'{m:>14}' for m in mounts) + f'{"verdict":>14}')
    c1 = {}
    for k in cands:
        cells, ok, unknown = [], True, False
        for m in mounts:
            g = [r for r in have if r['camera_mount'] == m
                 and r.get('actor_motion_raw') is not None
                 and r['actor_flicker_variants'].get(k) is not None]
            rho = spearman([r['actor_motion_raw'] for r in g],
                           [r['actor_flicker_variants'][k] for r in g])
            cells.append(f'{rho:+.3f}(n={len(g)})' if rho is not None
                         else f'n/a(n={len(g)})')
            if rho is None:
                # Too few clips for a rank correlation. This is NOT a pass: reporting
                # it as one is how an under-powered sweep gets mistaken for a result.
                unknown = True
            elif abs(rho) >= 0.25:
                ok = False
        c1[k] = ok and not unknown
        verdict = 'FAIL' if not ok else ('INCONCLUSIVE' if unknown else 'PASS')
        print(f'{k:<14}' + ''.join(f'{c:>14}' for c in cells) + f'{verdict:>14}')

    # ---- criterion 2: paired GT-better rate
    print('\n== C2: GT better (lower flicker) in >= 65% of pairs ==')
    by = collections.defaultdict(dict)
    for r in have:
        by[r['pair_key']][r['model']] = r
    pairs = []
    for pk, models in by.items():
        for m, pred in models.items():
            if m.startswith('gt'):
                continue
            gt = models.get(f'gt_{m}')
            if gt is not None:
                pairs.append((pred, gt))
    print(f'{len(pairs)} complete pairs')
    print(f'{"candidate":<14}{"n":>5}{"GT better":>11}{"median delta":>15}'
          f'{"CI":>26}{"verdict":>14}')
    c2 = {}
    for k in cands:
        d = [(p['actor_flicker_variants'][k] - g['actor_flicker_variants'][k])
             for p, g in pairs
             if p['actor_flicker_variants'].get(k) is not None
             and g['actor_flicker_variants'].get(k) is not None]
        if not d:
            continue
        frac = sum(1 for x in d if x > 0) / len(d)      # lower is better -> pred worse
        lo, hi = boot_ci(d)
        ok = frac >= 0.65 and len(d) >= 10
        c2[k] = ok
        ci = (f'[{lo:+.5f}, {hi:+.5f}]' if lo is not None else 'n/a')
        verdict = ('PASS' if ok else
                   ('INCONCLUSIVE' if len(d) < 10 and frac >= 0.65 else 'FAIL'))
        print(f'{k:<14}{len(d):>5}{frac:>10.1%}{statistics.median(d):>+15.5f}'
              f'{ci:>26}{verdict:>14}')

    # ---- discriminative power + independence from raw
    print('\n== Power check: a candidate that is CONSTANT passes C1 for free ==')
    print(f'{"candidate":<14}{"GT p10":>10}{"GT median":>11}{"GT p90":>10}'
          f'{"p90/p10":>9}{"rho vs raw":>12}')
    gts = [r for r in have if r['model'].startswith('gt')]
    raws = [r['actor_flicker_variants']['raw'] for r in have
            if r['actor_flicker_variants'].get('raw') is not None]
    spread = {}
    for k in cands:
        v = sorted(r['actor_flicker_variants'][k] for r in gts
                   if r['actor_flicker_variants'].get(k) is not None)
        if len(v) < 5:
            continue
        p10, med, p90 = v[int(0.1 * len(v))], statistics.median(v), v[int(0.9 * len(v))]
        spread[k] = p90 / p10 if p10 else float('inf')
        allv = [r['actor_flicker_variants'][k] for r in have
                if r['actor_flicker_variants'].get(k) is not None]
        rr = spearman(raws, allv) if len(allv) == len(raws) else None
        print(f'{k:<14}{p10:>10.5f}{med:>11.5f}{p90:>10.5f}{spread[k]:>9.2f}'
              + (f'{rr:>+12.3f}' if rr is not None else f'{"n/a":>12}'))

    # A statistic must still SEPARATE clips to be worth reporting. Requiring at least
    # a 1.5x p90/p10 spread on GT clips rejects the degenerate way to pass C1 (be
    # constant), which rho cannot detect.
    MIN_SPREAD = 1.5
    winners = [k for k in cands
               if c1.get(k) and c2.get(k) and spread.get(k, 0) >= MIN_SPREAD]
    blocked = [k for k in cands if c1.get(k) and c2.get(k) and k not in winners]
    print(f'\n== VERDICT ==\npass C1+C2 and spread >= {MIN_SPREAD}x: {winners or "NONE"}')
    if blocked:
        print(f'passed C1+C2 but too FLAT to discriminate (spread < {MIN_SPREAD}x): {blocked}')
    if not winners:
        print('No candidate passes. Per the E3 protocol the ActorFlicker column is '
              'DROPPED (or stays experimental/descriptive only). Criterion 3 (visual '
              'ordering) is only worth running on a candidate that got this far.')

    # clips for the criterion-3 visual check, on the best candidate
    if winners:
        k = winners[0]
        preds = [r for r in have if not r['model'].startswith('gt')
                 and r['actor_flicker_variants'].get(k) is not None]
        preds.sort(key=lambda r: r['actor_flicker_variants'][k])
        print(f'\n== C3 visual check candidates for `{k}` ==')
        for tag, sel in (('CLEANEST (expect no shimmer)', preds[:3]),
                         ('WORST (expect visible shimmer)', preds[-3:])):
            print(f'  {tag}:')
            for r in sel:
                print(f'    {r["actor_flicker_variants"][k]:.5f}  {r["camera_mount"]:<7}'
                      f' {r["path"]}')


if __name__ == '__main__':
    main()
