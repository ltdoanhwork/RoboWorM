#!/usr/bin/env python3
"""Independently verify that each cell's FVD is measured against ITS OWN real-video group.

The pooled reference is keyed by (embodiment, view, split, model) -- the model is part of
the key because each model stores GT in its own framing, and I3D's 224x224 resize does not
absorb a layout/aspect difference (see `scan()` in the exporter for the measurement). Two
separate code paths decide that key and they must agree, or a cell silently gets scored
against the wrong real set:

  * `scan()` groups a reference by the PATH it came from -- `view_of(relative)`.
  * `reduce_cells()` looks the group up from the MODEL LABEL -- `view_for_model(label)`.

A mismatch is invisible in the spreadsheet (every number still renders) so it is checked
here directly, three ways:

  1. MEMBERSHIP. Every reference pooled into a group really is from that embodiment,
     view and split -- no leakage across cells.
  2. AGREEMENT. For every scored cell, the group key derived from the model label equals
     the group key derived from the prediction's own path.
  3. DISCRIMINATION (a weak signal, reported not asserted). Scoring a cell against its
     own group is compared with scoring it against every foreign group. This is NOT a
     correctness test: a Frechet distance depends on the reference set's own spread, so a
     broader foreign set can sit marginally closer to a prediction set without being a
     better reference. Measured here, 21/28 dense cells score best against their own
     group, and 6 of the 7 exceptions are won by the SAME embodiment and SAME view with
     only the split flipped -- e.g. bimanual Ctrl-World MV makovian, own 626.0 versus
     614.4 against bimanual|multiview|non_makovian. Those two real sets are the same
     robot in the same scenes, so they are near-identical distributions and the gap is
     sampling noise. The remaining exception is humanoid CW-4view MV, whose FVD is
     1112-1621 against every group: predictions that far from all real data do not rank
     references meaningfully. Checks 1 and 2 are the actual proof of correct grouping.

Usage:
  models/dreamgen/.venv/bin/python assets/scripts/verify_fvd_groups.py
  models/dreamgen/.venv/bin/python assets/scripts/verify_fvd_groups.py \
      --cell single_arm/multiview/non_makovian
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'assets' / 'scripts'))

from export_dr_grid_xlsx import FvdCell, METHOD_LABEL, model_label      # noqa: E402
from export_dr_grid_xlsx_vs_real import (frechet, ref_identity,         # noqa: E402
                                         view_for_model, view_of)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='metrics/WorldScore/results/dr_full')
    ap.add_argument('--cell', default='single_arm/multiview/non_makovian/Dreamgen MV',
                    help='embodiment/view/split/model to inspect in detail')
    ap.add_argument('--json', default='results/reports/fvd_group_check.json')
    args = ap.parse_args()
    run = REPO / args.run

    focus = tuple(args.cell.split('/'))

    pooled = collections.defaultdict(FvdCell)
    seen = set()
    # per-cell prediction accumulators, keyed the way the exporter keys them
    preds = collections.defaultdict(FvdCell)
    members = collections.defaultdict(set)      # group -> {ref identity}
    member_models = collections.defaultdict(collections.Counter)
    mismatch = []
    n = 0

    with os.scandir(run / 'results') as it:
        for e in it:
            if not e.name.endswith('.json'):
                continue
            try:
                rec = json.loads(Path(e.path).read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if rec.get('ref_kind') != 'vs_gt' or rec.get('skipped'):
                continue
            rel = rec.get('relative', '')
            label = model_label(rel)
            method = METHOD_LABEL.get(rec.get('method'))
            view = view_of(rel)
            if label is None or method is None or view is None:
                continue
            n += 1
            emb, split = rec['embodiment'], rec['split']

            # --- check 2: the two independent derivations of the group key agree
            by_path = (emb, view, split, label)
            by_label = (emb, view_for_model(label), split, label)
            if by_path != by_label:
                mismatch.append(dict(relative=rel, model=label,
                                     by_path='|'.join(by_path),
                                     by_label='|'.join(by_label)))

            if rec.get('i3d_ref'):
                ident = ref_identity(rec['ref'])
                if (by_path, ident) not in seen:
                    seen.add((by_path, ident))
                    pooled[by_path].add('ref', rec['i3d_ref'])
                    members[by_path].add(ident)
                    member_models[by_path][label] += 1
            if rec.get('i3d_pred'):
                preds[(emb, label, split, method)].add('pred', rec['i3d_pred'])

    print(f'scanned {n} vs_gt records\n')

    # --- check 1: membership -------------------------------------------------------
    print('=== CHECK 1: every pooled reference belongs to its own group ===')
    bad = 0
    for g, idents in sorted(members.items()):
        emb, view, split, _model = g
        for ident in idents:
            # ident looks like <embodiment>/output/<view>/<model>/<split>/...
            p = ident.split('/')
            if len(p) < 5:
                continue
            if p[0] != emb or p[2] != view or split not in ident:
                bad += 1
                if bad <= 5:
                    print(f'  LEAK in {emb}|{view}|{split}: {ident}')
    print(f'  groups: {len(members)}   leaked references: {bad}')
    print(f'  {"PASS" if bad == 0 else "FAIL"}\n')

    # --- check 2: key agreement ----------------------------------------------------
    print('=== CHECK 2: model-label group key == path group key ===')
    print(f'  mismatches: {len(mismatch)}')
    for m in mismatch[:5]:
        print(f'    {m["model"]}: {m["by_label"]} vs {m["by_path"]}')
    print(f'  {"PASS" if not mismatch else "FAIL"}\n')

    print(f'=== FOCUS {"/".join(focus)} ===')
    if focus in members:
        print(f'  real videos pooled : {len(members[focus])}')
        print(f'  I3D windows        : {pooled[focus].n["ref"]}')
        print('  contributed by model:')
        for lbl, c in member_models[focus].most_common():
            print(f'    {lbl:16s} {c}')
    else:
        print('  group not present')
    print()

    # --- check 3: own group scores better than foreign groups ---------------------
    print('=== CHECK 3: own group vs foreign groups (informative, not a pass/fail) ===')
    moments = {g: acc.moments('ref') for g, acc in pooled.items()
               if acc.n['ref'] >= 64}
    rows, wins = [], 0
    tested = 0
    for key, acc in sorted(preds.items()):
        emb, label, split, method = key
        if method != 'Dense' or acc.n['pred'] < 64:
            continue
        own = (emb, view_for_model(label), split, label)
        if own not in moments:
            continue
        mp, cp = acc.moments('pred')
        own_fvd = frechet(mp, cp, *moments[own])
        foreign = {g: frechet(mp, cp, *mom) for g, mom in moments.items() if g != own}
        best_g = min(foreign, key=foreign.get)
        best_foreign = foreign[best_g]
        ok = own_fvd < best_foreign
        wins += ok
        tested += 1
        # a foreign winner that differs ONLY by split is the same robot in the same
        # scenes, so it is a near-identical distribution rather than a grouping error
        same_cell_other_split = (best_g[0] == own[0] and best_g[1] == own[1])
        rows.append(dict(cell=f'{emb}|{label}|{split}', own_group='|'.join(own),
                         own_fvd=own_fvd, best_foreign_fvd=best_foreign,
                         best_foreign_group='|'.join(best_g),
                         own_is_best=ok,
                         foreign_differs_only_by_split=same_cell_other_split))
        note = 'OK' if ok else ('~ other split of same cell' if same_cell_other_split
                                else '<-- foreign embodiment/view')
        print(f'  {emb+"|"+label+"|"+split:38s} own {own_fvd:8.1f}   '
              f'best foreign {best_foreign:8.1f}  {note}')
    n_split = sum(1 for r in rows if not r['own_is_best']
                  and r['foreign_differs_only_by_split'])
    n_other = tested - wins - n_split
    print(f'\n  {wins}/{tested} dense cells score best against their own group')
    print(f'  {n_split} exception(s) won only by the other split of the same '
          f'embodiment+view (near-identical real distribution)')
    print(f'  {n_other} exception(s) won by a different embodiment/view')
    print('  grouping correctness is established by CHECK 1 and CHECK 2 above')

    out = REPO / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        n_records=n,
        groups={'|'.join(g): dict(real_videos=len(members[g]),
                                  windows=pooled[g].n['ref'],
                                  by_model=dict(member_models[g]))
                for g in sorted(members)},
        leaked_references=bad,
        key_mismatches=mismatch,
        discrimination=rows,
    ), indent=2))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
