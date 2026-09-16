"""E3 follow-up: search for a motion-independent flicker statistic POST-HOC, from the
per-pair diagnostics already stored by the probe. Zero GPU.

The first E3 pass re-scored 53 clips storing, per frame-pair, the raw residual plus
`copy` (|I1-I2| before warping), `flow` (local flow magnitude), `grad` (local gradient)
and `align_pred` (grad*flow). That is enough to build and score new candidates without
touching the GPU again -- the same trick that made the E4 gate a table-time decision.

Candidates evaluated here are all of the form "residual normalized by something
measured on the SAME pixels", pooled to clip level in different ways:

* ``unexp_<p>``  sum(r) / sum(max(copy, p))            (the E3 `unexplained`, p varies)
* ``ratio_med``  median over pairs of r/max(copy,floor) (median-pooled, outlier-proof)
* ``resid_ols``  r minus the within-clip OLS fit of r on copy
* ``partial``    r/copy after ALSO dividing out flow    (two-stage)

Criterion 1 (|rho| < 0.25 vs actor_motion_raw on every mount) is evaluated at CLIP
level, same as the pre-registered criterion.
"""
import json
import math
import statistics
from pathlib import Path

RUN = Path('/tmp/e3_flicker/run')


def spearman(xs, ys):
    pairs = [(a, b) for a, b in zip(xs, ys) if a is not None and b is not None]
    if len(pairs) < 8:
        return None
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]

    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            a = 0.5 * (i + j) + 1
            for k in range(i, j + 1):
                r[o[k]] = a
            i = j + 1
        return r

    rx, ry = rk(xs), rk(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    n = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    d = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return n / d if d else None


def clip_candidates(row):
    """Build every post-hoc candidate for one clip from its per-pair diagnostics."""
    v = [p['flicker_variants'] for p in (row.get('actor_per_frame') or [])
         if p.get('flicker_variants') and p['flicker_variants'].get('raw') is not None]
    if len(v) < 3:
        return {}
    r = [x['raw'] for x in v]
    c = [x['copy'] for x in v]
    f = [x['flow'] for x in v]
    out = {}
    for p in (0.005, 0.01, 0.02, 0.05):
        out[f'unexp_{p:g}'] = sum(r) / sum(max(ci, p) for ci in c)
    out['ratio_med'] = statistics.median([ri / max(ci, 0.01) for ri, ci in zip(r, c)])
    # within-clip OLS of r on copy; keep the intercept-level residual mean.
    mc, mr = statistics.fmean(c), statistics.fmean(r)
    den = sum((ci - mc) ** 2 for ci in c)
    slope = (sum((ci - mc) * (ri - mr) for ci, ri in zip(c, r)) / den) if den else 0.0
    out['resid_ols'] = statistics.fmean([ri - slope * (ci - mc) for ci, ri in zip(c, r)])
    out['partial'] = statistics.median(
        [ri / (max(ci, 0.01) * (1.0 + fi)) for ri, ci, fi in zip(r, c, f)])
    return out


def main():
    rows = [json.loads(p.read_text()) for p in sorted((RUN / 'results').glob('*.json'))]
    have = [r for r in rows if r.get('actor_flicker_variants')
            and r.get('actor_motion_raw') is not None]
    for r in have:
        r['_cand'] = clip_candidates(r)
    have = [r for r in have if r['_cand']]
    keys = sorted(have[0]['_cand'])
    mounts = ('static', 'head', 'wrist')
    print(f'{len(have)} clips with motion + per-pair diagnostics\n')
    print('== C1 at clip level: |rho(actor_motion_raw, candidate)| < 0.25 ==')
    print(f'{"candidate":<14}' + ''.join(f'{m:>14}' for m in mounts) + f'{"verdict":>10}')
    passed = []
    for k in keys:
        cells, ok = [], True
        for m in mounts:
            g = [r for r in have if r['camera_mount'] == m]
            rho = spearman([r['actor_motion_raw'] for r in g],
                           [r['_cand'][k] for r in g])
            cells.append(f'{rho:+.3f}(n={len(g)})' if rho is not None else 'n/a')
            if rho is None or abs(rho) >= 0.25:
                ok = False
        if ok:
            passed.append(k)
        print(f'{k:<14}' + ''.join(f'{c:>14}' for c in cells)
              + f'{"PASS" if ok else "FAIL":>10}')

    # C2 + spread, only for the ones that survived C1
    by = {}
    for r in have:
        by.setdefault(r['pair_key'], {})[r['model']] = r
    print('\n== C2 (GT better >= 65%) and spread, for C1 survivors ==')
    print(f'{"candidate":<14}{"n":>5}{"GT better":>11}{"GT p90/p10":>12}{"verdict":>10}')
    winners = []
    for k in passed:
        d, gtv = [], []
        for pk, models in by.items():
            for m, pred in models.items():
                if m.startswith('gt'):
                    continue
                gt = models.get(f'gt_{m}')
                if gt is not None:
                    d.append(pred['_cand'][k] - gt['_cand'][k])
                    gtv.append(gt['_cand'][k])
        if not d:
            continue
        frac = sum(1 for x in d if x > 0) / len(d)
        gtv.sort()
        spread = gtv[int(0.9 * len(gtv))] / gtv[int(0.1 * len(gtv))] if gtv[int(0.1 * len(gtv))] else float('inf')
        ok = frac >= 0.65 and len(d) >= 10 and spread >= 1.5
        if ok:
            winners.append(k)
        print(f'{k:<14}{len(d):>5}{frac:>10.1%}{spread:>12.2f}'
              f'{"PASS" if ok else "FAIL":>10}')
    print(f'\n== VERDICT ==\nfull pass (C1 + C2 + discriminative): {winners or "NONE"}')


if __name__ == '__main__':
    main()
