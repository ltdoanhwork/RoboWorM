#!/usr/bin/env python3
"""Export the dense-baseline table: PSNR (per-episode) + FVD (shared real pool).

PSNR is reused unchanged from the validated DR run -- it is a per-episode distance to the
real video of the SAME episode and was never affected by the FVD reference problem.

FVD comes from `pool_perview_fvd.py`: per-camera features at a common 5 fps, scored
against ONE pooled real reference per (embodiment, camera, split) that every world model
and acceleration method shares, then averaged over the cameras common to all
configurations in the cell. That is what makes the FVD column comparable across rows.

Only cameras that passed `verify_shared_pool_fairness.py` are included: a model's own real
videos must sit at the pool's split-half noise floor, otherwise its framing penalty would
be charged to its predictions. All 38 (group, model) pairs in the final selection pass.

Emits .xlsx (PSNR / FVD sheets) and the LaTeX body for the paper.

Usage:
  models/dreamgen/.venv/bin/python assets/scripts/export_dense_table_shared_pool.py
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import statistics
import sys
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'assets' / 'scripts'))

from export_dr_grid_xlsx import (BOX, CENTER, EMB_LABEL, HDR, HDR_FILL,  # noqa: E402
                                 LEFT, METHOD_LABEL, MODEL_ORDER, TXT, VAL,
                                 model_label)

EMB_ORDER = ['single_arm', 'bimanual', 'humanoid']
EMB_NAME = dict(EMB_LABEL)
SPLIT_LABEL = {'makovian': 'Markovian', 'non_makovian': 'Non-Markovian'}


def load_psnr(run: Path, cache: Path):
    """Median PSNR per (embodiment, model, split) over dense clips.

    Cached: the scan walks ~69k result files (11 GB of I3D features inline) and takes
    ~5 min, while the values never change -- PSNR was computed per episode against that
    episode's own real video and is untouched by the FVD reference work.
    """
    if cache.exists():
        d = json.loads(cache.read_text())
        print(f'[psnr] reusing {cache}')
        return ({tuple(k.split('|')): v for k, v in d['median'].items()},
                {tuple(k.split('|')): v for k, v in d['n'].items()})
    acc = collections.defaultdict(list)
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
            if METHOD_LABEL.get(rec.get('method')) != 'Dense':
                continue
            lab = model_label(rec.get('relative', ''))
            if lab is None:
                continue
            v = rec.get('psnr')
            if v is None or not math.isfinite(v):
                continue
            acc[(rec['embodiment'], lab, rec['split'])].append(float(v))
            n += 1
    print(f'[psnr] {n} dense clips over {len(acc)} cells')
    med = {k: statistics.median(v) for k, v in acc.items()}
    cnt = {k: len(v) for k, v in acc.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(dict(
        median={'|'.join(k): v for k, v in med.items()},
        n={'|'.join(k): v for k, v in cnt.items()}), indent=2))
    return med, cnt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='metrics/WorldScore/results/dr_full')
    ap.add_argument('--pooled', default='metrics/reports/fvd_perview_pooled.json')
    ap.add_argument('--out', default='assets/BENCHMARK_dense_shared_pool.xlsx')
    ap.add_argument('--tex', default='metrics/reports/dense_table_shared_pool.tex')
    ap.add_argument('--psnr-cache', default='metrics/reports/dense_psnr_cache.json')
    ap.add_argument('--singleview', default='metrics/reports/fvd_singleview_pooled.json')
    args = ap.parse_args()

    import openpyxl

    psnr, n_clips = load_psnr(REPO / args.run, REPO / args.psnr_cache)
    pooled = json.loads((REPO / args.pooled).read_text())
    cams = pooled['common_cameras']
    fvd, fvd_n, fvd_cams = {}, {}, {}
    for k, v in pooled['cells'].items():
        emb, label, split, method = k.split('|')
        if method != 'Dense':
            continue
        fvd[(emb, label, split)] = v['fvd']
        fvd_n[(emb, label, split)] = v['n_windows']
        fvd_cams[(emb, label, split)] = v['n_cameras']

    # Single-view cells keep the FVD from the earlier pooled run: measured real-vs-real
    # between the two single-view models is 7.6-8.3, i.e. their GT is ALREADY one
    # distribution (both render the same single camera at the same resolution), so no
    # per-camera split was needed. Only the model term is dropped from the key so the
    # reference is shared, matching the multiview protocol.
    sv = REPO / args.singleview
    n_sv = 0
    if sv.exists():
        d = json.loads(sv.read_text())
        for k, v in d.get('cells', {}).items():
            emb, label, split, method = k.split('|')
            if method != 'Dense' or label.endswith('MV'):
                continue
            fvd[(emb, label, split)] = v['fvd']
            fvd_n[(emb, label, split)] = v['n_windows']
            fvd_cams[(emb, label, split)] = 1
            n_sv += 1
        print(f'[singleview] merged {n_sv} dense cells from {sv.name}')
    else:
        print(f'[singleview] {sv} missing -- single-view FVD will show ---')

    keys = sorted(set(psnr) | set(fvd))
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'dense-shared-pool'
    hdr = ['Embodiment', 'Configuration', 'Regime', 'PSNR (dB)', 'n clips',
           'FVD (shared pool)', 'n windows', 'n cameras', 'reference key']
    for c, h in enumerate(hdr, 1):
        cell = ws.cell(1, c, h)
        cell.font, cell.fill, cell.border, cell.alignment = HDR, HDR_FILL, BOX, CENTER
    r = 2
    tex_rows = []
    for emb in EMB_ORDER:
        for label in MODEL_ORDER:
            for split in ('makovian', 'non_makovian'):
                k = (emb, label, split)
                if k not in psnr and k not in fvd:
                    continue
                view = 'multiview' if label.endswith('MV') else 'singleview'
                cl = cams.get(f'{emb}|{split}') if view == 'multiview' else None
                refkey = (f'({emb}, camera, {split}) over {cl}' if cl
                          else f'({emb}, {view}, {split})')
                vals = [EMB_NAME.get(emb, emb), label, SPLIT_LABEL[split],
                        None if k not in psnr else round(psnr[k], 2), n_clips.get(k, 0),
                        None if k not in fvd else round(fvd[k], 1),
                        fvd_n.get(k, 0), fvd_cams.get(k, 0), refkey]
                for c, v in enumerate(vals, 1):
                    cell = ws.cell(r, c, v)
                    cell.border = BOX
                    cell.font = TXT if isinstance(v, str) or v is None else VAL
                    cell.alignment = LEFT if c <= 3 else CENTER
                r += 1
                tex_rows.append((emb, label, split, psnr.get(k), fvd.get(k),
                                 n_clips.get(k, 0), fvd_n.get(k, 0)))
    for col, w in zip('ABCDEFGHI', (13, 16, 15, 11, 9, 18, 11, 11, 46)):
        ws.column_dimensions[col].width = w
    out = REPO / args.out
    wb.save(out)
    print(f'wrote {out}')

    # ---- LaTeX body ----------------------------------------------------------
    L = [r'\begin{tabular}{llcccc}', r'\toprule',
         r'Embodiment & Configuration & Regime & PSNR $\uparrow$ & '
         r'FVD $\downarrow$ & $n$ \\', r'\midrule']
    last = None
    for emb, label, split, p, f, nc, nw in tex_rows:
        if last is not None and emb != last:
            L.append(r'\midrule')
        last = emb
        L.append(f'{EMB_NAME.get(emb, emb)} & {label} & {SPLIT_LABEL[split]} & '
                 f'{"---" if p is None else f"{p:.2f}"} & '
                 f'{"---" if f is None else f"{f:.1f}"} & {nw} \\\\')
    L += [r'\bottomrule', r'\end{tabular}']
    cap = (
        r'% Caption:' '\n'
        r'% PSNR is per-episode against the real video of the same episode. FVD is '
        r'computed against a fixed pooled set of real videos for each matching '
        r'embodiment $\times$ camera $\times$ regime setting, shared across all world '
        r'models and acceleration methods evaluated in that setting. Multi-view '
        r'predictions are split into their constituent camera views; FVD is computed '
        r'per camera and averaged over the cameras common to all configurations. All '
        r'clips are decimated to a common 5 fps before I3D feature extraction, and $n$ '
        r'reports the number of 16-frame windows per cell.')
    tex = REPO / args.tex
    tex.parent.mkdir(parents=True, exist_ok=True)
    tex.write_text(cap + '\n' + '\n'.join(L) + '\n')
    print(f'wrote {tex}\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
