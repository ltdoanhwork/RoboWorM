"""Are the per-method numbers independent of each other?

Two different questions hide under "independent":

  (1) Does method A's number depend on method B? It must not.
  (2) Does method A's number depend on dense? It does, BY DESIGN -- an acceleration
      method's whole job is to reproduce its own dense run faster, so the delta vs
      dense IS the quantity of interest, not a contamination.

But (1) has a real trap. Every method is pooled over ITS OWN set of clips, and those
sets differ (comparable_pair drops a different subset per method, and coverage is
uneven mid-wave). Then

    median over S_A of (A - dense)   vs   median over S_B of (B - dense)

is not a comparison of A and B: it is also a comparison of S_A and S_B. Algebraically
the shared dense term cancels ONLY when both are measured on the same clips.

This script re-ranks every method on the COMMON clip set (clips every method scored)
and reports whether the ranking survives.

Usage:
  python scripts/_check_method_independence.py --run metrics/WorldScore/results/dense_nr_full
"""
import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'assets' / 'scripts'))
import table_dense_nr as T  # noqa: E402

COLS = ['visual_integrity', 'active_motion_continuity', 'text_condition_alignment']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    args = ap.parse_args()
    rows = T.load_records(Path(args.run))

    dense, accel = {}, collections.defaultdict(list)
    for r in rows:
        if r.get('method') == 'dense':
            dense[(r['pair_key'], r['model'])] = r
        elif r.get('method') not in (None, 'gt'):
            accel[(r['pair_key'], r['model'])].append(r)

    # per (method, col): {clip_key: delta}
    d = collections.defaultdict(dict)
    for key, variants in accel.items():
        base = dense.get(key)
        if base is None:
            continue
        for r in variants:
            if not T.comparable_pair(r, base):
                continue
            for col in COLS:
                a, b = T.gated(base, col), T.gated(r, col)
                if a is not None and b is not None:
                    d[(r['method'], col)][key] = b - a
    methods = sorted({m for m, _ in d})

    for col in COLS:
        own = {m: d[(m, col)] for m in methods if d.get((m, col))}
        if not own:
            continue
        common = set.intersection(*(set(v) for v in own.values())) if len(own) > 1 else set()
        print('=' * 96)
        print(f'{col}   -- {len(methods)} methods, common clip set = {len(common)}')
        print('=' * 96)
        print(f'{"method":<14}{"n own":>7}{"median own":>12}{"rank":>6}   '
              f'{"n common":>9}{"median common":>15}{"rank":>6}   moved')
        own_rank = sorted(own, key=lambda m: -T.median(list(own[m].values())))
        com_rank = sorted(own, key=lambda m: -T.median([own[m][k] for k in common])) \
            if common else []
        for m in own_rank:
            vo = T.median(list(own[m].values()))
            i_own = own_rank.index(m) + 1
            if common:
                vc = T.median([own[m][k] for k in common])
                i_com = com_rank.index(m) + 1
                moved = i_com - i_own
                flag = '' if moved == 0 else f'  <<< {moved:+d}'
                print(f'{m:<14}{len(own[m]):>7}{vo:>+12.2f}{i_own:>6}   '
                      f'{len(common):>9}{vc:>+15.2f}{i_com:>6}{flag}')
            else:
                print(f'{m:<14}{len(own[m]):>7}{vo:>+12.2f}{i_own:>6}')
        # how different are the clip sets?
        sizes = [len(v) for v in own.values()]
        print(f'  own-set sizes {min(sizes)}..{max(sizes)}; '
              f'common set keeps {100 * len(common) / max(sizes):.0f}% of the largest')
        print()

    # Does the shared dense baseline dominate? Spearman between two methods' deltas
    # on their shared clips: high rho means the clip (i.e. dense) drives the delta,
    # not the method.
    print('=' * 96)
    print('Cross-method correlation of deltas on shared clips (visual_integrity)')
    print('  high rho = the CLIP explains the delta more than the METHOD does')
    print('=' * 96)
    import statistics

    def spearman(xs, ys):
        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            r = [0.0] * len(v)
            for pos, i in enumerate(order):
                r[i] = pos + 1
            return r
        rx, ry = rank(xs), rank(ys)
        mx, my = statistics.fmean(rx), statistics.fmean(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
        return num / den if den else 0.0

    rhos = []
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1:]:
            s1, s2 = d.get((m1, 'visual_integrity')), d.get((m2, 'visual_integrity'))
            if not s1 or not s2:
                continue
            shared = sorted(set(s1) & set(s2))
            if len(shared) < 50:
                continue
            rhos.append((spearman([s1[k] for k in shared], [s2[k] for k in shared]),
                         m1, m2, len(shared)))
    rhos.sort(reverse=True)
    for rho, m1, m2, n in rhos[:6]:
        print(f'  {m1:<13}{m2:<13}n={n:<6} rho {rho:+.3f}')
    print('  ...')
    for rho, m1, m2, n in rhos[-3:]:
        print(f'  {m1:<13}{m2:<13}n={n:<6} rho {rho:+.3f}')
    if rhos:
        print(f'  median rho over {len(rhos)} method pairs = '
              f'{T.median([r for r, _, _, _ in rhos]):+.3f}')


if __name__ == '__main__':
    main()
