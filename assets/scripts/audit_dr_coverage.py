#!/usr/bin/env python3
"""Phase 5 coverage audit: does every grid cell have the clips the distribution promises?

The grid is 12 cells (embodiment x view x model) x 2 splits x 10 rows (dense + 9 accel
methods), scored against real video (`vs_gt`). A cell that is quietly short does not look
wrong in the spreadsheet -- it just reports a median over fewer clips -- so the count is
checked against the target distribution explicitly and every shortfall gets a reason.

TARGET (dense clips per model family, from the benchmark's own distribution):

    DreamGen    1605
    DreamDojo    756
    Ctrl-World   805
    ------------------
    total       3166

Ctrl-World is mapped the way the rest of the repo maps it: bimanual and humanoid read the
4-view grid tree (`ctrlworld_4view_grid`), single_arm reads the 3-view tree (`ctrlworld`).
Both trees exist on disk for bimanual/humanoid, so counting "all ctrlworld dirs" would
double-count 548 clips -- hence the explicit per-cell model key below.

Usage:
  models/dreamgen/.venv/bin/python assets/scripts/audit_dr_coverage.py
  models/dreamgen/.venv/bin/python assets/scripts/audit_dr_coverage.py --json out.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# (embodiment, view, model_dir) -> expected dense clips.
# Values are the per-cell dense clip counts whose family sums are 1605 / 756 / 805.
TARGET: dict[tuple[str, str, str], int] = {
    ('bimanual',   'multiview',  'dreamgen'):             300,
    ('bimanual',   'singleview', 'dreamgen'):             300,
    ('humanoid',   'multiview',  'dreamgen'):             248,
    ('humanoid',   'singleview', 'dreamgen'):             260,
    ('single_arm', 'multiview',  'dreamgen'):             257,
    ('single_arm', 'singleview', 'dreamgen'):             240,
    ('bimanual',   'singleview', 'dreamdojo'):            300,
    ('humanoid',   'singleview', 'dreamdojo'):            216,
    ('single_arm', 'singleview', 'dreamdojo'):            240,
    ('bimanual',   'multiview',  'ctrlworld_4view_grid'): 300,
    ('humanoid',   'multiview',  'ctrlworld_4view_grid'): 248,
    ('single_arm', 'multiview',  'ctrlworld'):            257,
}

FAMILY = {'dreamgen': 'DreamGen', 'dreamdojo': 'DreamDojo',
          'ctrlworld': 'Ctrl-World', 'ctrlworld_4view_grid': 'Ctrl-World'}
FAMILY_TARGET = {'DreamGen': 1605, 'DreamDojo': 756, 'Ctrl-World': 805}

# Methods that legitimately never ran a given model family, so their absence is a fact
# about the benchmark rather than a hole in this wave.
#   * SiTo / ITM are token-pruning methods wired for DreamGen and Ctrl-World only; they
#     were never run on the DreamDojo action-conditioned DiT.
#   * FasterCache is a documented no-op on Ctrl-World's CFG-off SVD UNet.
EXPECTED_ABSENT = {
    ('dreamdojo', 'sito'), ('dreamdojo', 'itm'),
    ('ctrlworld', 'fastercache'), ('ctrlworld_4view_grid', 'fastercache'),
}

# Shortfalls with a known, benign cause. The plan requires that a short cell says WHY
# rather than being left silently blank.
KNOWN_SHORT = {
    ('single_arm', 'singleview', 'dreamgen'):
        'episode_000056 makovian prediction is a 0-byte file (failed generation, '
        '"moov atom not found") -- nothing to score against, not a GT gap',
}


def model_dir(relative: str) -> str | None:
    parts = relative.split('/')
    return parts[3] if len(parts) > 4 and parts[1] == 'output' else None


def view_dir(relative: str) -> str | None:
    parts = relative.split('/')
    return parts[2] if len(parts) > 3 and parts[1] == 'output' else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='metrics/WorldScore/results/dr_full')
    ap.add_argument('--json', default='results/reports/dr_coverage_audit.json')
    args = ap.parse_args()
    run = REPO / args.run

    manifest = json.loads((run / 'manifest.json').read_text())

    # scored = a result file exists and was not skipped
    have = set()
    skipped = collections.Counter()
    results = run / 'results'
    if results.is_dir():
        with os.scandir(results) as it:
            for e in it:
                if e.name.endswith('.json'):
                    have.add(e.name[:-5])

    inv = collections.Counter()       # (emb, view, model, method) -> rows in manifest
    scored = collections.Counter()    # same, but only rows with a result on disk
    for r in manifest:
        if r['ref_kind'] != 'vs_gt':
            continue
        md, vd = model_dir(r['relative']), view_dir(r['relative'])
        if md is None or vd is None:
            continue
        key = (r['embodiment'], vd, md, r['method'])
        inv[key] += 1
        if r['id'] in have:
            scored[key] += 1

    print(f'{"cell":46s} {"dense":>6s} {"target":>7s} {"delta":>6s}  methods  scored/inv')
    print('-' * 92)
    report, fam_have = [], collections.Counter()
    for cell in sorted(TARGET):
        emb, view, md = cell
        target = TARGET[cell]
        dense_inv = inv[(emb, view, md, 'dense')]
        dense_scored = scored[(emb, view, md, 'dense')]
        methods = sorted({k[3] for k in inv if k[:3] == cell})
        tot_inv = sum(v for k, v in inv.items() if k[:3] == cell)
        tot_scored = sum(v for k, v in scored.items() if k[:3] == cell)
        delta = dense_inv - target
        fam_have[FAMILY[md]] += dense_inv
        reason = KNOWN_SHORT.get(cell)
        if delta == 0:
            flag = ''
        elif reason:
            flag = f'  <-- short: {reason}'
        else:
            flag = '  <-- SHORT (unexplained)' if delta < 0 else '  <-- OVER'
        print(f'{emb+"/"+view+"/"+md:46s} {dense_inv:6d} {target:7d} {delta:+6d}'
              f'  {len(methods):7d}  {tot_scored}/{tot_inv}{flag}')
        report.append(dict(embodiment=emb, view=view, model=md, target=target,
                           dense_inventoried=dense_inv, dense_scored=dense_scored,
                           delta=delta, reason=reason, methods=methods,
                           n_methods=len(methods),
                           rows_inventoried=tot_inv, rows_scored=tot_scored))

    print('\nby model family (dense clips):')
    for fam, tgt in FAMILY_TARGET.items():
        got = fam_have[fam]
        print(f'  {fam:12s} {got:5d} / {tgt:5d}  ({got - tgt:+d})')
    print(f'  {"TOTAL":12s} {sum(fam_have.values()):5d} / '
          f'{sum(FAMILY_TARGET.values()):5d}  '
          f'({sum(fam_have.values()) - sum(FAMILY_TARGET.values()):+d})')

    # Which of the 10 rows are missing per cell, and is that expected?
    print('\nmissing method rows:')
    all_methods = sorted({k[3] for k in inv})
    for cell in sorted(TARGET):
        emb, view, md = cell
        present = {k[3] for k in inv if k[:3] == cell}
        gone = [m for m in all_methods if m not in present]
        if not gone:
            continue
        tagged = [f'{m}{" (expected)" if (md, m) in EXPECTED_ABSENT else " (HOLE)"}'
                  for m in gone]
        print(f'  {emb}/{view}/{md}: ' + ', '.join(tagged))

    out = REPO / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        target=FAMILY_TARGET,
        family_actual={k: v for k, v in fam_have.items()},
        cells=report,
        rows_inventoried=sum(inv.values()),
        rows_scored=sum(scored.values()),
        skipped=dict(skipped),
    ), indent=2))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
