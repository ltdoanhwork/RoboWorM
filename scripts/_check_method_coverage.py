"""Which (model, embodiment, mount) cells does each method actually cover?

`_check_method_independence.py` found the 9 methods share ZERO clips, which makes the
pooled per-method ranking (T3b) a comparison of different populations. This prints the
coverage matrix so the cause is visible, and re-ranks on the largest set of cells that
several methods DO share.

Usage:
  python scripts/_check_method_coverage.py --run metrics/WorldScore/results/dense_nr_full
"""
import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'assets' / 'scripts'))
import table_dense_nr as T  # noqa: E402

COL = 'visual_integrity'


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

    # (method) -> {cell -> [deltas]} and (method) -> {clip}
    per_cell = collections.defaultdict(lambda: collections.defaultdict(list))
    clips = collections.defaultdict(set)
    for key, variants in accel.items():
        base = dense.get(key)
        if base is None:
            continue
        a = T.gated(base, COL)
        if a is None:
            continue
        for r in variants:
            if not T.comparable_pair(r, base):
                continue
            b = T.gated(r, COL)
            if b is None:
                continue
            cell = (r['model'], r['embodiment'])
            per_cell[r['method']][cell].append(b - a)
            clips[r['method']].add((key, cell))
    methods = sorted(per_cell)
    cells = sorted({c for m in methods for c in per_cell[m]})

    print('=' * 104)
    print(f'COVERAGE: n paired clips per (model, embodiment) for {COL}')
    print('=' * 104)
    print(f'{"model / embodiment":<42}' + ''.join(f'{m[:9]:>10}' for m in methods))
    for cell in cells:
        label = f'{cell[0]} / {cell[1]}'
        print(f'{label:<42}' + ''.join(
            f'{len(per_cell[m].get(cell) or []) or "-":>10}' for m in methods))
    print(f'{"TOTAL":<42}' + ''.join(
        f'{sum(len(v) for v in per_cell[m].values()):>10}' for m in methods))
    print(f'{"cells covered":<42}' + ''.join(f'{len(per_cell[m]):>10}' for m in methods))

    print('\n' + '=' * 104)
    print('PAIRWISE CELL OVERLAP (how many cells two methods share)')
    print('=' * 104)
    print(f'{"":<14}' + ''.join(f'{m[:9]:>10}' for m in methods))
    for m1 in methods:
        line = f'{m1:<14}'
        for m2 in methods:
            n = len(set(per_cell[m1]) & set(per_cell[m2]))
            line += f'{n if m1 != m2 else "-":>10}'
        print(line)

    # Re-rank on the cells shared by the most methods.
    cell_owners = {c: {m for m in methods if c in per_cell[m]} for c in cells}
    full = [c for c, o in cell_owners.items() if len(o) == len(methods)]
    print('\n' + '=' * 104)
    print(f'cells covered by ALL {len(methods)} methods: {len(full)}')
    for k in range(len(methods), 1, -1):
        shared = [c for c, o in cell_owners.items() if len(o) >= k]
        if shared:
            ms = sorted(set.intersection(*(cell_owners[c] for c in shared))) \
                if len(shared) > 1 else sorted(cell_owners[shared[0]])
            print(f'  >= {k} methods: {len(shared)} cells, intersection of owners = {ms}')
            if len(ms) >= 4:
                print(f'\n  RE-RANK restricted to those {len(shared)} cells, '
                      f'{len(ms)} comparable methods:')
                out = []
                for m in ms:
                    vals = [v for c in shared for v in (per_cell[m].get(c) or [])]
                    if vals:
                        out.append((T.median(vals), len(vals), m))
                out.sort(reverse=True)
                for i, (v, n, m) in enumerate(out, 1):
                    print(f'    {i}. {m:<13} n={n:<5} median {v:+.2f}')
                break


if __name__ == '__main__':
    main()
