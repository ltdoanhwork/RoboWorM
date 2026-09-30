"""Export the DR (reference-based) metrics to an .xlsx laid out like BENCHMARK_NR_main.

Four sheets, one per metric, same geometry as the NR workbook's `VisualIntegrity-main`
tab: per embodiment block, Markovian at A:C and Non-Markovian at E:G, with a `NORM -->`
mirror at J:L / N:P holding literal (not formula) ratios.

    PSNR-main    dB,  higher better
    SSIM-main    0-1, higher better
    LPIPS-main   0-1, LOWER better
    FVD-main     distribution distance, LOWER better

Four ways this differs from the NR workbook, all forced by what a reference-based
metric is:

1. NO "Real Video" row. An NR metric scores a video on its own, so GT has a score and
   the block can be normalised by it. A DR metric scores a PAIR, and GT-vs-itself is
   trivially perfect (PSNR=inf, SSIM=1, LPIPS=0, FVD=0). The normalisation anchor here
   is the DENSE row instead, and the NORM block divides by dense, not by GT.

2. Reference kind is a sheet-level choice, passed with --ref.
     vs_dense  (default, = PhyFid-DR) every accel method against the DENSE prediction
               of the same clip. This is the benchmark's question: what does the
               speedup cost relative to the model's own full-compute output. The Dense
               row is then empty BY CONSTRUCTION -- dense is the reference.
     vs_gt     every prediction against ground truth, so Dense does get a number and
               the column doubles as an absolute-quality reading. Missing wherever GT
               was never rendered -- notably ALL of Dreamgen SV (0 of 7200 pairs).

3. FVD is NOT per clip. It is a Frechet distance between two DISTRIBUTIONS of I3D
   features, so a single clip has no FVD. Each grid cell's FVD is computed over every
   I3D window of every clip in that cell, which is also how metrics/WorldScore/results/
   dr_full/fvd.json does it. A cell with fewer than --fvd-min-windows windows is left
   blank rather than shown: a 400-d covariance from a handful of windows is
   rank-deficient and its FVD measures that, not the videos.

4. PSNR is aggregated as a MEDIAN over clips, like the NR grid, but note PSNR can be
   +inf on a bit-identical pair (it happens: a cache config that skips nothing
   reproduces dense exactly). Infinities are dropped before the median and counted in
   the JSON audit, because one inf would otherwise poison a mean and silently make a
   method look perfect.

Reads the finished per-clip results in metrics/WorldScore/results/dr_full/results/
(~44.5k json, 11 GB). The I3D features are the bulk of that, so the scan streams the
covariance accumulators instead of holding features in memory.

Usage:
  models/dreamgen/.venv/bin/python assets/scripts/export_dr_grid_xlsx.py \
      --run metrics/WorldScore/results/dr_full \
      --out "sampling_dataset/data_for_web/BENCHMARK_DR_main.xlsx"
"""
import argparse
import collections
import datetime
import json
import math
import os
import statistics
from pathlib import Path

# Cap BLAS threads BEFORE numpy loads. Each cell's FVD is one 400x400 sqrtm, which
# openblas happily spreads over all 64 cores; at that size the sync cost dominates and
# the run went from minutes to hours (measured: 0.37 s/cell at 4 threads vs an unbounded
# run that burned 7 h of CPU without finishing). Must precede `import numpy`.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_var, '4')

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

REPO = Path(__file__).resolve().parents[2]

# (result key, sheet name, sheet title, decimals, higher_is_better)
METRICS = [
    ('psnr', 'PSNR-main', 'PSNR (dB)', 2, True),
    ('ssim', 'SSIM-main', 'SSIM', 4, True),
    ('lpips', 'LPIPS-main', 'LPIPS', 4, False),
    ('fvd', 'FVD-main', 'FVD', 2, False),
]

EMB_LABEL = [('bimanual', 'BIMANUAL'), ('humanoid', 'HUMANOID'),
             ('single_arm', 'SINGLE-ARM')]
SPLITS = [('makovian', 'Markovian'), ('non_makovian', 'Non \u2013 Markovian')]
METHOD_ORDER = ['Dense', 'itm', 'sito', 'World-cache', 'Faster-cache', 'Di-cache',
                'SVG2', 'SVG1', 'Radial Attention', 'PISA']
METHOD_LABEL = {'dense': 'Dense', 'itm': 'itm', 'sito': 'sito',
                'worldcache': 'World-cache', 'fastercache': 'Faster-cache',
                'dicache': 'Di-cache', 'svg2': 'SVG2', 'svg1': 'SVG1',
                'radial': 'Radial Attention', 'pisa': 'PISA'}
MODEL_ORDER = ['Dreamgen MV', 'Dreamgen SV', 'DreamDojo SV', 'Ctrl-World MV',
               'CW-4view MV']

BLOCK_GAP = 2
ROWS_PER_MODEL = len(METHOD_ORDER)
ABS_COLS = (1, 2, 3)
ABS_COLS_NM = (5, 6, 7)
NORM_LABEL_COL = 9
NORM_COLS = (10, 11, 12)
NORM_COLS_NM = (14, 15, 16)

ARIAL = 'Arial'
HDR = Font(name=ARIAL, size=12, bold=True)
TXT = Font(name=ARIAL, size=12)
VAL = Font(name=ARIAL, size=11)
MISSING = Font(name=ARIAL, size=11, color='999999')
NOTE = Font(name=ARIAL, size=9, italic=True, color='666666')
CENTER = Alignment(horizontal='center', vertical='center')
LEFT = Alignment(horizontal='left', vertical='center')
RIGHT = Alignment(horizontal='right', vertical='center')
HDR_FILL = PatternFill('solid', fgColor='D9D9D9')
REF_FILL = PatternFill('solid', fgColor='FFF2CC')
THIN = Side(style='thin', color='BFBFBF')
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #

def model_label(relative: str) -> str | None:
    """'Dreamgen MV' etc. from `<emb>/output/<view>/<model>/...`.

    The camera track is not in the model name, so it comes from the path -- DreamGen is
    rendered on both tracks and pooling them would mix a 4-view grid with one ego view.
    """
    parts = relative.split('/')
    if len(parts) < 4 or parts[1] != 'output':
        return None
    view, model = parts[2], parts[3]
    track = 'MV' if view == 'multiview' else 'SV'
    label = {'dreamgen': 'Dreamgen', 'dreamdojo': 'DreamDojo',
             'ctrlworld': 'Ctrl-World', 'ctrlworld_4view_grid': 'CW-4view'}.get(model)
    return f'{label} {track}' if label else None


class FvdCell:
    """Streaming mean/covariance for one grid cell, so 11 GB of features never
    all sit in RAM. Frechet needs only the first two moments of each set."""

    DIM = 400

    def __init__(self):
        self.n = {'pred': 0, 'ref': 0}
        self.sum = {'pred': np.zeros(self.DIM, np.float64),
                    'ref': np.zeros(self.DIM, np.float64)}
        self.outer = {'pred': np.zeros((self.DIM, self.DIM), np.float64),
                      'ref': np.zeros((self.DIM, self.DIM), np.float64)}

    def add(self, tag: str, windows) -> None:
        arr = np.asarray(windows, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.DIM:
            return
        self.n[tag] += arr.shape[0]
        self.sum[tag] += arr.sum(0)
        self.outer[tag] += arr.T @ arr

    def moments(self, tag: str):
        n = self.n[tag]
        mean = self.sum[tag] / n
        # unbiased covariance, matching np.cov(rowvar=False) used by the DR run
        cov = (self.outer[tag] - n * np.outer(mean, mean)) / (n - 1)
        return mean, cov

    def fvd(self, min_windows: int):
        if self.n['pred'] < min_windows or self.n['ref'] < min_windows:
            return None, f'<{min_windows} windows'
        from scipy import linalg
        mp, cp = self.moments('pred')
        mr, cr = self.moments('ref')
        covmean, _ = linalg.sqrtm(cp.dot(cr), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        val = (float(((mp - mr) ** 2).sum() + np.trace(cp) + np.trace(cr)
                     - 2 * np.trace(covmean)))
        return val, None


def scan(run: Path, ref_kinds: tuple[str, ...], want_fvd: bool):
    """(ref_kind, embodiment, model, split, method) -> per-metric accumulators.

    Both reference kinds are collected in ONE pass. The per-clip json carries the I3D
    features and totals 11 GB, so reading it twice to make two sheets would double the
    slowest part of the job for nothing.
    """
    cells = collections.defaultdict(
        lambda: {'psnr': [], 'ssim': [], 'lpips': [], 'fvd': None})
    audit = collections.Counter()
    results = run / 'results'
    n = 0
    with os.scandir(results) as it:
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
            ref_kind = rec.get('ref_kind')
            if ref_kind not in ref_kinds:
                continue
            if rec.get('skipped'):
                audit[f'skipped:{rec["skipped"]}'] += 1
                continue
            label = model_label(rec.get('relative', ''))
            if label is None:
                audit['unmapped_model'] += 1
                continue
            method = METHOD_LABEL.get(rec.get('method'))
            if method is None:
                audit['unmapped_method'] += 1
                continue
            key = (ref_kind, rec['embodiment'], label, rec['split'], method)
            cell = cells[key]
            for mkey in ('psnr', 'ssim', 'lpips'):
                v = rec.get(mkey)
                if v is None:
                    continue
                if not math.isfinite(v):
                    # a bit-identical pair gives PSNR=inf; one would poison an average
                    audit[f'nonfinite_{mkey}'] += 1
                    continue
                cell[mkey].append(float(v))
            if want_fvd and rec.get('i3d_pred') and rec.get('i3d_ref'):
                if cell['fvd'] is None:
                    cell['fvd'] = FvdCell()
                cell['fvd'].add('pred', rec['i3d_pred'])
                cell['fvd'].add('ref', rec['i3d_ref'])
            audit[f'used:{ref_kind}'] += 1
    print(f'  scanned {n} files')
    return cells, audit


def reduce_cells(cells, fvd_min_windows: int):
    """Grid value per (cell, metric): median over clips, FVD over pooled windows.

    Each cell's accumulator is dropped as soon as its FVD is computed. Holding all ~420
    of them costs 420 x 2 x (400x400 float64) ~= 1 GB of outer-product matrices on top of
    the feature lists, which is what killed an earlier run.
    """
    table, counts = {}, {}
    total = len(cells)
    for i, key in enumerate(sorted(cells), 1):
        cell = cells[key]
        vals = {}
        for mkey in ('psnr', 'ssim', 'lpips'):
            series = cell[mkey]
            vals[mkey] = statistics.median(series) if series else None
        if cell['fvd'] is not None:
            vals['fvd'], vals['fvd_reason'] = cell['fvd'].fvd(fvd_min_windows)
            counts[key] = {'n_clips': max(len(cell['lpips']), len(cell['ssim']),
                                          len(cell['psnr'])),
                           'n_windows': cell['fvd'].n['pred']}
            cell['fvd'] = None          # release ~2.5 MB of accumulators per cell
        else:
            vals['fvd'], vals['fvd_reason'] = None, 'no i3d features'
            counts[key] = {'n_clips': max(len(cell['lpips']), len(cell['ssim']),
                                          len(cell['psnr'])),
                           'n_windows': 0}
        table[key] = vals
        if i % 25 == 0 or i == total:
            print(f'  reduced {i}/{total} cells', flush=True)
    return table, counts


# --------------------------------------------------------------------------- #
# sheet
# --------------------------------------------------------------------------- #

def models_present(table, ref_kind, emb) -> list[str]:
    have = {key[2] for key in table if key[0] == ref_kind and key[1] == emb}
    return [m for m in MODEL_ORDER if m in have]


def write_block(ws, top, ref_kind, emb_key, emb_label, table, mkey, title, digits,
                models, with_norm):
    """One embodiment block.

    There is no "Real Video" row: an NR metric scores a video alone so GT has a score,
    but a DR metric scores a PAIR and GT-vs-itself is trivially perfect. The NORM mirror
    therefore anchors on the DENSE row, and only exists for vs_gt -- under vs_dense the
    dense row IS the reference and has no value to divide by.
    """
    groups = [(ABS_COLS, 'makovian', False), (ABS_COLS_NM, 'non_makovian', False)]
    if with_norm:
        groups += [(NORM_COLS, 'makovian', True), (NORM_COLS_NM, 'non_makovian', True)]
    for (ca, cb, cc), split, norm in groups:
        ws.cell(top, ca, emb_label).font = HDR
        ws.cell(top, cb, 'Method').font = HDR
        ws.cell(top, cc, title).font = HDR
        for c in (ca, cb, cc):
            ws.cell(top, c).alignment = CENTER
            ws.cell(top, c).fill = HDR_FILL
            ws.cell(top, c).border = BOX
        ws.cell(top + 1, ca, dict(SPLITS)[split]).font = TXT
        ws.cell(top + 1, ca).alignment = LEFT
        ws.merge_cells(start_row=top + 1, start_column=ca,
                       end_row=top + 1, end_column=cc)

        row = top + 2
        for model in models:
            first = row
            dense = (table.get((ref_kind, emb_key, model, split, 'Dense')) or {}).get(mkey)
            for method in METHOD_ORDER:
                ws.cell(row, cb, method).font = TXT
                ws.cell(row, cb).alignment = LEFT
                ws.cell(row, cb).border = BOX
                ws.cell(row, ca).border = BOX
                cell = ws.cell(row, cc)
                v = (table.get((ref_kind, emb_key, model, split, method)) or {}).get(mkey)
                if norm:
                    # ratio to this block+split's dense. Literal, like the NR workbook's
                    # NORM block, which keeps the file free of any LibreOffice recalc.
                    cell.value = (round(v / dense, 4)
                                  if (v is not None and dense not in (None, 0)) else None)
                    cell.number_format = '0.0000'
                else:
                    cell.value = round(v, digits) if v is not None else None
                    cell.number_format = f'0.{"0" * digits}'
                if cell.value is None:
                    # '---' = never rendered / below the FVD window floor / (for
                    # vs_dense) the dense row itself, which IS the reference.
                    cell.value = '---'
                    cell.font = MISSING
                else:
                    cell.font = VAL
                cell.alignment = RIGHT
                cell.border = BOX
                if method == 'Dense':
                    for c in (ca, cb, cc):
                        ws.cell(row, c).fill = REF_FILL
                row += 1
            ws.cell(first, ca, model).font = TXT
            ws.cell(first, ca).alignment = CENTER
            ws.merge_cells(start_row=first, start_column=ca,
                           end_row=first + ROWS_PER_MODEL - 1, end_column=ca)
    if with_norm:
        ws.cell(top, NORM_LABEL_COL, 'NORM \u2192 dense').font = HDR
        ws.cell(top, NORM_LABEL_COL).alignment = CENTER
    return top + 1 + len(models) * ROWS_PER_MODEL


def write_sheet(wb, table, mkey, sheet, title, digits, higher, ref_kind,
                fvd_min_windows):
    # Under vs_dense every method is measured AGAINST dense, so there is no dense value
    # to normalise by; the NORM mirror only means something for vs_gt.
    with_norm = ref_kind == 'vs_gt'
    ws = wb.create_sheet(sheet)
    row = 1
    for emb_key, emb_label in EMB_LABEL:
        models = models_present(table, ref_kind, emb_key)
        if not models:
            continue
        row = write_block(ws, row, ref_kind, emb_key, emb_label, table, mkey, title,
                          digits, models, with_norm) + BLOCK_GAP + 1

    for col, width in ((1, 17), (2, 19), (3, 13), (4, 3), (5, 17), (6, 19), (7, 13),
                       (8, 3), (9, 15), (10, 17), (11, 19), (12, 13), (13, 3),
                       (14, 17), (15, 19), (16, 13)):
        ws.column_dimensions[get_column_letter(col)].width = width

    direction = 'higher is better' if higher else 'LOWER is better'
    notes = [
        f'{title} \u2014 {direction}. Reference: {ref_kind}.',
        ('vs_dense = each method against the DENSE prediction of the SAME clip '
         '(PhyFid-DR, the benchmark\u2019s speedup-cost question). The Dense row is '
         'blank BY CONSTRUCTION \u2014 dense IS the reference \u2014 and there is no '
         'NORM mirror for the same reason.') if ref_kind == 'vs_dense' else
        ('vs_gt = each prediction against ground truth, so Dense does get a number and '
         'the sheet doubles as an absolute-quality reading. Blank wherever GT was never '
         'rendered \u2014 notably ALL of Dreamgen SV, which has 0 vs_gt pairs.'),
    ]
    if with_norm:
        notes.append('NORM blocks = value / that block+split Dense value, stored as '
                     'literals (no formulas), matching the NR workbook.')
    notes += [
        ('FVD is a distance between DISTRIBUTIONS of I3D features, so it has no '
         f'per-clip value: each cell pools every I3D window in it, and a cell with '
         f'<{fvd_min_windows} windows is left blank (a 400-d covariance from few '
         'windows is rank-deficient and its FVD measures that, not the videos).')
        if mkey == 'fvd' else
        'PSNR/SSIM/LPIPS cells are the MEDIAN over the clips of that cell.',
        "'---' = not rendered, or below the FVD window floor, or the reference row.",
        f'Generated {datetime.datetime.now():%Y-%m-%d %H:%M} from '
        f'metrics/WorldScore/results/dr_full.',
    ]
    for note in notes:
        row += 1
        ws.cell(row, 1, note).font = NOTE
    return ws


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='metrics/WorldScore/results/dr_full')
    ap.add_argument('--out', default='sampling_dataset/data_for_web/BENCHMARK_DR_main.xlsx')
    ap.add_argument('--ref', default='both', choices=['vs_dense', 'vs_gt', 'both'],
                    help='vs_dense = PhyFid-DR; vs_gt = absolute vs ground truth; '
                         'both (default) = one sheet set per reference')
    ap.add_argument('--fvd-min-windows', type=int, default=64,
                    help='below this a cell FVD is left blank (rank-deficient covariance)')
    ap.add_argument('--no-fvd', action='store_true',
                    help='skip the I3D pass (much faster; FVD sheets omitted)')
    args = ap.parse_args()

    refs = ('vs_dense', 'vs_gt') if args.ref == 'both' else (args.ref,)
    run = (REPO / args.run) if not Path(args.run).is_absolute() else Path(args.run)
    out = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    print(f'scanning {run}/results  (ref_kind={"+".join(refs)}) ...')
    cells, audit = scan(run, refs, want_fvd=not args.no_fvd)
    if not cells:
        raise SystemExit(f'no records with ref_kind in {refs} under {run}/results')
    table, counts = reduce_cells(cells, args.fvd_min_windows)

    wb = Workbook()
    wb.remove(wb.active)
    metrics = [m for m in METRICS if not (args.no_fvd and m[0] == 'fvd')]
    for ref_kind in refs:
        # PhyFid-DR (vs_dense) is the headline, so it gets the bare sheet names and
        # lands first; vs_gt is the supplementary absolute reading.
        suffix = '' if ref_kind == 'vs_dense' else '-vsGT'
        for mkey, sheet, title, digits, higher in metrics:
            name = sheet.replace('-main', f'{suffix}-main')
            write_sheet(wb, table, mkey, name, title, digits, higher, ref_kind,
                        args.fvd_min_windows)
            filled = sum(1 for k, v in table.items()
                         if k[0] == ref_kind and v.get(mkey) is not None)
            print(f'  {name:20s} {filled} cells')
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)

    audit_path = out.with_suffix('.audit.json')
    audit_path.write_text(json.dumps({
        'run': str(run.relative_to(REPO)),
        'ref_kinds': list(refs),
        'fvd_min_windows': args.fvd_min_windows,
        'aggregation': {
            'psnr_ssim_lpips': 'median over the clips of the cell; non-finite dropped',
            'fvd': 'Frechet distance over every I3D window pooled in the cell',
        },
        'scan_audit': dict(audit.most_common()),
        'cells': [
            {'ref_kind': k[0], 'embodiment': k[1], 'model': k[2], 'split': k[3],
             'method': k[4], 'n_clips': counts[k]['n_clips'],
             'n_i3d_windows': counts[k]['n_windows'],
             **{m: v.get(m) for m in ('psnr', 'ssim', 'lpips', 'fvd')},
             'fvd_reason': v.get('fvd_reason')}
            for k, v in sorted(table.items())
        ],
    }, indent=2) + '\n')
    print(f'\nwrote {out}')
    print(f'wrote {audit_path}')


if __name__ == '__main__':
    main()
