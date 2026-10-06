#!/usr/bin/env python
"""Build a POWER-TARGETED stratified manifest from the full sampling_dataset inventory.

Why this exists instead of just scoring everything: the full inventory is 87,327
view-clips = ~20 GPU-days at the measured 3.0 clips/min (10 workers, A100-80GB, GPU-bound
at 100% util / 80GB -- more workers do not help). Measured value of that spend is very
low: the pooled accel claims already resolve at n=30/method (p down to 1e-06) and a
bootstrap CI only narrows as 1/sqrt(n), so full-n buys 0.06x CI width for 250x compute
and produces no new claim.

The real gap is the PER-CELL tables, and those are cheap: capping each cell at a target n
brings 153/154 accel cells to n>=8 (the MIN_PAIRS_FOR_CLAIM floor) for ~1.8k clips.

The one thing a naive "take N per cell" gets wrong is PAIRING. Every table here is
paired, so an unpaired clip is wasted GPU:

  * an accel row needs the DENSE row at the same (pair_key, model)  -> table3/table3b
  * a dense row needs its GT anchor at the same pair_key            -> table2

So selection is dependency-aware: it prefers rows whose partner is already scored (free),
and when it must take a row whose partner is missing it pulls the partner into the
manifest too (paid) and counts that cost. Anything already scored is excluded, so the
output is a pure delta against the reference run and the whole thing stays a strict
prefix of the full inventory -- scoring it now wastes nothing if the full run happens
later.
"""
import argparse
import collections
import json
from pathlib import Path


def key_of(r):
    return (r['pair_key'], r['model'])


def gt_lookup(rows):
    """pair_key -> {model_field: row} for GT rows, matching table_dense_nr's anchor rule."""
    out = collections.defaultdict(dict)
    for r in rows:
        if r.get('method') == 'gt':
            out[r['pair_key']][r['model']] = r
    return out


def anchor_for(model, pair_key, gts):
    """Same resolution order as table_dense_nr.paired_table."""
    cand = gts.get(pair_key, {})
    row = cand.get(f'gt_{model}')
    if row is None and model.startswith('ctrlworld'):
        row = cand.get('gt')
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inventory', type=Path, required=True,
                    help='manifest.json from a --inventory-only run over ALL method trees')
    ap.add_argument('--reference-run', type=Path, required=True,
                    help='run dir whose results/ already hold scored records')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--accel-target', type=int, default=12,
                    help='n per (method, model, embodiment, mount) cell')
    ap.add_argument('--dense-target', type=int, default=40,
                    help='n per (model, embodiment, mount) cell')
    args = ap.parse_args()

    inv = json.loads((args.inventory).read_text())
    scored = {json.loads(p.read_text())['id']
              for p in (args.reference_run / 'results').glob('*.json')}
    print(f'inventory {len(inv)} clips, already scored {len(scored)}')

    by_method = collections.defaultdict(list)
    for r in inv:
        by_method[r.get('method')].append(r)
    dense_by_key = {key_of(r): r for r in by_method['dense']}
    gts = gt_lookup(inv)

    picked = {}          # id -> row
    def take(r):
        if r['id'] not in scored:
            picked[r['id']] = r

    # ---- dense track: fill each (model, embodiment, mount) cell, GT anchor mandatory.
    # A dense row without an anchor can never enter table2, so it is skipped outright
    # rather than scored and discarded later.
    dense_cells = collections.defaultdict(list)
    for r in by_method['dense']:
        dense_cells[(r['model'], r['embodiment'], r['camera_mount'])].append(r)
    d_new = d_anchor = 0
    for cell, rows in sorted(dense_cells.items()):
        have = sum(1 for r in rows if r['id'] in scored)
        # Prefer rows whose anchor is ALREADY scored: those cost 1 clip, not 2.
        rows = sorted(rows, key=lambda r: (
            anchor_for(r['model'], r['pair_key'], gts) is None,
            (anchor_for(r['model'], r['pair_key'], gts) or {}).get('id') not in scored,
            r['id']))
        for r in rows:
            if have >= args.dense_target:
                break
            if r['id'] in scored:
                continue
            a = anchor_for(r['model'], r['pair_key'], gts)
            if a is None:
                continue
            take(r); d_new += 1
            if a['id'] not in scored:
                take(a); d_anchor += 1
            have += 1
    print(f'dense track: +{d_new} dense clips, +{d_anchor} GT anchors '
          f'({len(dense_cells)} cells -> target {args.dense_target})')

    # ---- accel track: fill each (method, model, embodiment, mount) cell. A dense partner
    # is mandatory for the same reason.
    acc_cells = collections.defaultdict(list)
    for m, rows in by_method.items():
        if m in ('dense', 'gt', None):
            continue
        for r in rows:
            acc_cells[(m, r['model'], r['embodiment'], r['camera_mount'])].append(r)
    a_new = a_partner = 0
    short = []
    for cell, rows in sorted(acc_cells.items()):
        have = sum(1 for r in rows if r['id'] in scored)
        rows = sorted(rows, key=lambda r: (
            key_of(r) not in dense_by_key,
            dense_by_key.get(key_of(r), {}).get('id') not in scored
            and dense_by_key.get(key_of(r), {}).get('id') not in picked,
            r['id']))
        for r in rows:
            if have >= args.accel_target:
                break
            if r['id'] in scored:
                continue
            base = dense_by_key.get(key_of(r))
            if base is None:
                continue
            take(r); a_new += 1
            if base['id'] not in scored and base['id'] not in picked:
                take(base); a_partner += 1
            have += 1
        if have < 8:
            short.append((cell, have))
    print(f'accel track: +{a_new} accel clips, +{a_partner} extra dense partners '
          f'({len(acc_cells)} cells -> target {args.accel_target})')
    if short:
        print(f'  {len(short)} cells cannot reach n=8 even using the whole inventory:')
        for cell, n in short[:10]:
            print(f'    {cell} -> {n}')

    rows = sorted(picked.values(), key=lambda r: r['id'])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'manifest.json').write_text(json.dumps(rows))
    per = collections.Counter(r.get('method') for r in rows)
    print(f'\nTOTAL {len(rows)} clips to score '
          f'(~{len(rows) / 3.0 / 60:.1f} h at 3.0 clips/min)')
    for k, v in per.most_common():
        print(f'  {k:12s} {v:5d}')
    print(f'\nwrote {args.out / "manifest.json"}')


if __name__ == '__main__':
    main()
