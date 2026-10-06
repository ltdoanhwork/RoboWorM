#!/usr/bin/env python3
"""Export the PSNR and FVD grid measured against REAL VIDEO (vs_gt), as .xlsx.

Two sheets, `PSNR-main` and `FVD-main`, in the layout of `assets/BENCHMARK_NR_main.xlsx`:
per embodiment a Markovian block in columns A:C and a Non-Markovian block in E:G, with a
literal-value NORM mirror at J:L / N:P.

HOW THIS DIFFERS FROM export_dr_grid_xlsx.py
--------------------------------------------
That script reports FVD per (cell x method), i.e. each method's predictions against the
GT of the same clips. Here the FVD reference is instead the POOLED set of real videos for
the whole (embodiment x view x split x model) group, shared by every method in it.

The pooled reference is what makes FVD read as "how far is this method's output from real
robot video", comparable down a column, rather than a per-method quantity whose reference
silently changes with coverage. It also fixes a sample-size artefact: a method covering 98
clips and one covering 150 were previously scored against differently-sized references,
and the Frechet estimator is biased by reference size.

WHY MODEL IS IN THE POOLING KEY. Pooling across models in a group was tried first and
measured to be wrong: each model stores GT in its own framing, so for multiview the real
sets of two models differ by layout (DreamGen 768x432 2x2 grid versus Ctrl-World 1280x192
strip), and FVD between those two REAL sets reaches 1726-2895 -- larger than any model's
own FVD. See `scan()` for the full numbers.

Three details that decide whether the number means anything:

  DEDUPE THE REFERENCE BY GT PATH. One GT video appears in up to ten pairs (one per
  method). Pooling naively multiplies its windows by ten: the mean and covariance barely
  move, but `n_windows` inflates tenfold and the >=64-window guard -- the thing standing
  between a real FVD and a rank-deficient 400x400 covariance -- stops protecting anything.

  KEEP FRAMING CONSTANT WITHIN A REFERENCE. I3D's 224x224 resize normalises resolution
  but not aspect ratio or view layout, so a reference may only pool clips that share a
  framing. This is what the model term in the key enforces.

  A REAL VIDEO FLOOR ROW. FVD between two disjoint halves of the pooled real set is the
  sampling noise of the estimator: the lowest value any method could achieve. Without it
  an FVD of 300 is uninterpretable. PSNR gets `---` there instead, since GT against
  itself is infinite.

Usage:
  models/dreamgen/.venv/bin/python assets/scripts/export_dr_grid_xlsx_vs_real.py \
      --run metrics/WorldScore/results/dr_full \
      --out assets/BENCHMARK_DR_vs_real.xlsx
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

# Cap BLAS threads BEFORE numpy loads -- each cell's FVD is one 400x400 sqrtm, and
# letting openblas spread it over 64 cores made sync cost dominate (measured 0.37 s/cell
# at 4 threads versus an unbounded run that burned 7 h of CPU without finishing).
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_var, '4')

import numpy as np                                                    # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'assets' / 'scripts'))

from export_dr_grid_xlsx import (BLOCK_GAP, CENTER, EMB_LABEL,        # noqa: E402
                                 HDR, HDR_FILL, LEFT, METHOD_LABEL, METHOD_ORDER,
                                 MISSING, MODEL_ORDER, NOTE, REF_FILL, SPLITS,
                                 TXT, VAL, BOX, ABS_COLS, ABS_COLS_NM,
                                 NORM_LABEL_COL, NORM_COLS, NORM_COLS_NM,
                                 FvdCell, model_label)

FVD_MIN_WINDOWS = 64
REAL_ROW = 'Real Video'


def view_of(relative: str) -> str | None:
    parts = relative.split('/')
    return parts[2] if len(parts) > 3 and parts[1] == 'output' else None


def ref_identity(ref: str) -> str:
    """A GT file's identity, independent of which method's tree it is reached through.

    The same reference is hard-linked into every method directory, so
    `sampling_dataset/dense/.../gt_all_views.mp4` and `sampling_dataset/pisa/.../gt_all_views.mp4`
    are the same bytes under different paths. Deduplicating on the raw path therefore
    does nothing -- measured: 16,783 pairs collapsed to 16,727 "distinct" references, and
    bimanual multiview makovian reported 56,668 real windows from ~150 real videos, a
    ~10x inflation that silently defeats the >=64-window guard and biases the pooled
    covariance toward whichever clips the most methods happened to cover.

    Stripping the `sampling_dataset/<method>/` prefix gives one key per real video.
    """
    parts = ref.split('/sampling_dataset/')
    if len(parts) != 2:
        return ref
    tail = parts[1].split('/', 1)
    return tail[1] if len(tail) == 2 else parts[1]


def scan(run: Path):
    """One pass over the per-clip results, building both the per-cell and pooled views.

    Reading these files is the slow part (11 GB of I3D features), so the per-method
    accumulators and the pooled real-video reference are filled together.
    """
    cells = collections.defaultdict(lambda: {'psnr': [], 'fvd': None})
    # Pooled real reference, keyed by (embodiment, view, split, MODEL); deduped by GT path.
    #
    # WHY MODEL IS PART OF THE KEY. Pooling only by (embodiment, view, split) was measured
    # to be wrong for multiview. Each model writes its GT in its own framing -- DreamGen a
    # 768x432 2x2 grid (aspect 1.78), Ctrl-World a 1280x192 or 960x192 view strip (aspect
    # 6.67 / 5.00) -- and I3D's 224x224 resize does not absorb that. Measured FVD between
    # the REAL sets of two models in the same group, i.e. the same episodes differing only
    # in layout: bimanual MV 1726, humanoid MV 2895, single_arm MV 1626, versus 7.6 for
    # bimanual singleview where both models render the same camera. Those real-vs-real
    # distances exceed the models' own FVD (440-1621), so a pooled-across-models reference
    # measures layout mismatch rather than generation quality.
    #
    # Keying on the model makes each configuration's FVD a distance to real video in ITS
    # OWN framing, and every method inside a cell still shares one reference, which is what
    # the accel-method comparison needs. Smallest resulting group is 182 windows, still
    # above FVD_MIN_WINDOWS.
    pooled = collections.defaultdict(FvdCell)
    seen_ref: set[tuple] = set()
    # a second, independent accumulator over a disjoint half of the real videos, used for
    # the Real Video floor row
    pooled_a = collections.defaultdict(FvdCell)
    pooled_b = collections.defaultdict(FvdCell)
    audit = collections.Counter()

    n = 0
    with os.scandir(run / 'results') as it:
        for entry in it:
            if not entry.name.endswith('.json'):
                continue
            n += 1
            if n % 5000 == 0:
                print(f'  scanned {n} ...', flush=True)
            try:
                rec = json.loads(Path(entry.path).read_text())
            except (json.JSONDecodeError, OSError):
                audit['unreadable'] += 1
                continue
            if rec.get('ref_kind') != 'vs_gt':
                continue
            if rec.get('skipped'):
                audit[f'skipped:{rec["skipped"]}'] += 1
                continue
            label = model_label(rec.get('relative', ''))
            method = METHOD_LABEL.get(rec.get('method'))
            view = view_of(rec.get('relative', ''))
            if label is None or method is None or view is None:
                audit['unmapped'] += 1
                continue

            emb, split = rec['embodiment'], rec['split']
            key = (emb, label, split, method)
            cell = cells[key]

            v = rec.get('psnr')
            if v is not None and math.isfinite(v):
                cell['psnr'].append(float(v))
            elif v is not None:
                # a bit-identical pair gives PSNR=inf and would poison the median
                audit['nonfinite_psnr'] += 1

            if rec.get('i3d_pred'):
                if cell['fvd'] is None:
                    cell['fvd'] = FvdCell()
                cell['fvd'].add('pred', rec['i3d_pred'])

            if rec.get('i3d_ref'):
                gkey = (emb, view, split, label)
                ident = ref_identity(rec['ref'])
                rkey = (gkey, ident)
                if rkey not in seen_ref:       # DEDUPE: one GT serves up to 10 methods
                    seen_ref.add(rkey)
                    pooled[gkey].add('ref', rec['i3d_ref'])
                    # deterministic split-half for the noise floor (md5, not hash():
                    # PYTHONHASHSEED randomises str hashing per process, so the floor
                    # would move between runs)
                    bit = hashlib.md5(ident.encode()).digest()[0] & 1
                    (pooled_a if bit == 0 else pooled_b)[gkey].add('ref', rec['i3d_ref'])
                    audit['real_videos'] += 1
                else:
                    audit['dedup_ref_skipped'] += 1
            audit['used'] += 1
    print(f'  scanned {n} files')
    return cells, pooled, pooled_a, pooled_b, audit


def frechet(mp, cp, mr, cr) -> float:
    from scipy import linalg
    # `disp=` is deprecated in SciPy >= 1.14 and removed in 1.18; sqrtm returns the
    # matrix alone there, so accept either shape rather than pinning a SciPy version.
    covmean = linalg.sqrtm(cp.dot(cr))
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mp - mr) ** 2).sum() + np.trace(cp) + np.trace(cr)
                 - 2 * np.trace(covmean))


def view_for_model(label: str) -> str:
    return 'multiview' if label.endswith('MV') else 'singleview'


def reduce_cells(cells, pooled, pooled_a, pooled_b):
    """Grid values: PSNR median per cell, FVD against the pooled real set of its group."""
    ref_moments, ref_n = {}, {}
    for gkey, acc in pooled.items():
        if acc.n['ref'] >= FVD_MIN_WINDOWS:
            ref_moments[gkey] = acc.moments('ref')
        ref_n[gkey] = acc.n['ref']

    table, counts = {}, {}
    total = len(cells)
    for i, key in enumerate(sorted(cells), 1):
        emb, label, split, method = key
        cell = cells[key]
        vals = {'psnr': statistics.median(cell['psnr']) if cell['psnr'] else None}
        gkey = (emb, view_for_model(label), split, label)
        n_win = 0
        if cell['fvd'] is None:
            vals['fvd'], vals['fvd_reason'] = None, 'no i3d features'
        elif gkey not in ref_moments:
            vals['fvd'] = None
            vals['fvd_reason'] = f'real set <{FVD_MIN_WINDOWS} win'
        elif cell['fvd'].n['pred'] < FVD_MIN_WINDOWS:
            n_win = cell['fvd'].n['pred']
            vals['fvd'] = None
            vals['fvd_reason'] = f'<{FVD_MIN_WINDOWS} windows'
        else:
            n_win = cell['fvd'].n['pred']
            mp, cp = cell['fvd'].moments('pred')
            mr, cr = ref_moments[gkey]
            vals['fvd'], vals['fvd_reason'] = frechet(mp, cp, mr, cr), None
        counts[key] = {'n_clips': len(cell['psnr']), 'n_windows': n_win,
                       'n_real_windows': ref_n.get(gkey, 0)}
        table[key] = vals
        cell['fvd'] = None              # release ~2.5 MB of accumulators per cell
        if i % 25 == 0 or i == total:
            print(f'  reduced {i}/{total} cells', flush=True)

    # Real Video floor: two disjoint halves of the same real set against each other.
    floor = {}
    for gkey in pooled:
        a, b = pooled_a.get(gkey), pooled_b.get(gkey)
        if a is None or b is None or min(a.n['ref'], b.n['ref']) < FVD_MIN_WINDOWS:
            floor[gkey] = None
            continue
        floor[gkey] = frechet(*a.moments('ref'), *b.moments('ref'))
    return table, counts, floor, ref_n


def fmt(v, digits: int) -> str:
    return '---' if v is None else f'{v:.{digits}f}'


def models_present(table, emb) -> list[str]:
    have = {k[1] for k in table if k[0] == emb}
    return [m for m in MODEL_ORDER if m in have]


def write_block(ws, top, emb_key, emb_label, table, floor, mkey, title, digits,
                models, higher):
    """One embodiment block: absolute values plus a NORM mirror anchored on Real Video.

    Layout follows `assets/BENCHMARK_NR_main (3).xlsx`: a header row, a
    Markovian / Non-Markovian label row, then one 10-method group per model, with the
    NORM mirror carrying literal values (no formulas, so the file needs no recalc).

    The Real Video row sits inside each model's group because the real-video reference is
    pooled per (embodiment, view, split, model) -- see `scan()` for the measurement that
    forced the model into that key -- so each model block has its own floor.

    NORM divides by the Real Video floor for FVD (so 1.0 = indistinguishable from real)
    and by the Dense row for PSNR (where a real-video anchor does not exist, GT against
    itself being infinite).
    """
    groups = [(ABS_COLS, 'makovian', False), (ABS_COLS_NM, 'non_makovian', False),
              (NORM_COLS, 'makovian', True), (NORM_COLS_NM, 'non_makovian', True)]
    for (ca, cb, cc), split, norm in groups:
        ws.cell(top, ca, emb_label).font = HDR
        ws.cell(top, cb, 'Method').font = HDR
        ws.cell(top, cc, f'NORM {title}' if norm else title).font = HDR
        for c in (ca, cb, cc):
            ws.cell(top, c).alignment = CENTER
            ws.cell(top, c).fill = HDR_FILL
            ws.cell(top, c).border = BOX

        # split label row, as in the reference workbook
        lab = ws.cell(top + 1, ca,
                      'Markovian' if split == 'makovian' else 'Non \u2013 Markovian')
        lab.font = HDR
        lab.alignment = LEFT
        for c in (ca, cb, cc):
            ws.cell(top + 1, c).border = BOX

        row = top + 2
        for model in models:
            view = view_for_model(model)
            gkey = (emb_key, view, split, model)
            base = floor.get(gkey) if mkey == 'fvd' else \
                table.get((emb_key, model, split, 'Dense'), {}).get(mkey)

            # Real Video row: the estimator's own noise floor for FVD, n/a for PSNR.
            ws.cell(row, ca, model).font = TXT
            ws.cell(row, ca).alignment = LEFT
            ws.cell(row, cb, REAL_ROW).font = TXT
            ws.cell(row, cb).fill = REF_FILL
            rv = floor.get(gkey) if mkey == 'fvd' else None
            if rv is None:
                shown_rv = '---'
            elif norm:
                shown_rv = fmt(rv / base, 3) if base else '---'
            else:
                shown_rv = fmt(rv, digits)
            cell_rv = ws.cell(row, cc, shown_rv)
            cell_rv.font = VAL if rv is not None else MISSING
            cell_rv.alignment = CENTER
            for c in (ca, cb, cc):
                ws.cell(row, c).border = BOX
            row += 1

            for method in METHOD_ORDER:
                vals = table.get((emb_key, model, split, method))
                v = vals.get(mkey) if vals else None
                ws.cell(row, cb, method).font = TXT
                if norm:
                    shown = fmt(v / base, 3) if (v is not None and base) else '---'
                else:
                    shown = fmt(v, digits)
                c = ws.cell(row, cc, shown)
                c.font = VAL if v is not None else MISSING
                c.alignment = CENTER
                for cc2 in (ca, cb, cc):
                    ws.cell(row, cc2).border = BOX
                row += 1
        if norm:
            ws.cell(top, NORM_LABEL_COL, 'NORM -->').font = HDR
    return top + 2 + len(models) * (len(METHOD_ORDER) + 1)


def write_sheet(wb, table, floor, ref_n, mkey, sheet, title, digits, higher, audit):
    ws = wb.create_sheet(sheet)
    for col, width in ((1, 18), (2, 18), (3, 14), (4, 3), (5, 18), (6, 18), (7, 14),
                       (8, 3), (9, 8), (10, 18), (11, 18), (12, 14), (13, 3),
                       (14, 18), (15, 18), (16, 14)):
        ws.column_dimensions[chr(64 + col) if col <= 26 else 'A'].width = width

    top = 1
    for emb_key, emb_label in EMB_LABEL:
        models = models_present(table, emb_key)
        if not models:
            continue
        top = write_block(ws, top, emb_key, emb_label, table, floor, mkey, title,
                          digits, models, higher) + BLOCK_GAP

    note = ws.cell(top + 1, 1,
                   f'{title}: reference = REAL VIDEO (vs_gt). '
                   f'{"higher" if higher else "lower"} is better. '
                   'PSNR = median over clips. '
                   'FVD = pooled real-video set per '
                   '(embodiment x view x split x model), '
                   'GT features deduplicated by path; '
                   f'cells with <{FVD_MIN_WINDOWS} I3D windows are "---". '
                   'Real Video row = FVD between two disjoint halves of the real set '
                   '(the sampling-noise floor); not defined for PSNR.')
    note.font = NOTE
    return ws


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='metrics/WorldScore/results/dr_full')
    ap.add_argument('--out', default='assets/BENCHMARK_DR_vs_real.xlsx')
    ap.add_argument('--audit', default='results/reports/dr_vs_real_audit.json')
    args = ap.parse_args()

    run = REPO / args.run
    print(f'scanning {run} ...', flush=True)
    cells, pooled, pa, pb, audit = scan(run)
    print(f'  {len(cells)} (embodiment, model, split, method) cells', flush=True)
    print(f'  {len(pooled)} pooled real-video groups', flush=True)
    table, counts, floor, ref_n = reduce_cells(cells, pooled, pa, pb)

    from openpyxl import Workbook
    wb = Workbook()
    wb.remove(wb.active)
    write_sheet(wb, table, floor, ref_n, 'psnr', 'PSNR-main', 'PSNR (dB)', 2, True, audit)
    write_sheet(wb, table, floor, ref_n, 'fvd', 'FVD-main', 'FVD', 2, False, audit)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f'wrote {out}')

    audit_path = REPO / args.audit
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({
        'audit': dict(audit),
        'real_windows_per_group': {'|'.join(k): v for k, v in sorted(ref_n.items())},
        'floor_fvd': {'|'.join(k): v for k, v in sorted(floor.items())},
        'cells': [{'embodiment': k[0], 'model': k[1], 'split': k[2], 'method': k[3],
                   **table[k], **counts[k]} for k in sorted(table)],
    }, indent=2, default=str))
    print(f'wrote {audit_path}')
    print('\naudit:', dict(audit))


if __name__ == '__main__':
    main()
