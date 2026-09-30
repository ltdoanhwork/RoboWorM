#!/usr/bin/env python3
r"""Patch the `Speedup` sheet of the NR workbook to match the paper's speedup table.

WHY THIS IS A SCRIPT AND NOT A MANUAL EDIT
------------------------------------------
`assets/BENCHMARK_NR_main (3).xlsx` is the source of record for every speedup
number that reaches a figure (`make_nr_circular_figures.py` reads the `Speedup`
tab directly). Hand-editing it leaves no record of WHAT changed, and a mistyped
cell becomes a published number. So each edit is declared here as
`(cell, expected_old, new)` and the script REFUSES to write if the old value has
drifted.

That makes the script:
  * idempotent  -- re-running finds every cell already at `new` and writes nothing;
  * safe        -- a workbook that someone else already edited fails loudly
                   instead of being silently stomped back;
  * auditable   -- the before/after table is printed, and `--check` runs the
                   verification without touching the file.

SAFE TO RE-SAVE WITH openpyxl?  Verified yes for this workbook:
  * 0 formula cells (nothing to lose to openpyxl's formula handling);
  * the 7 `xl/drawings/*.xml` parts are empty stubs (775 bytes, `<xdr:wsDr>`
    root only, no charts/images), so dropping them loses nothing.
If either of those ever stops being true, re-check before running -- openpyxl
does not round-trip charts. `--audit-parts` re-runs that check.

WHAT IS CHANGED
---------------
3 real value changes (from the LaTeX table in the paper draft):
  K6  WorldCache  / Single-arm / Ctrl-World MV  1.31x -> 1.81x
  K7  DiCache     / Single-arm / Ctrl-World MV  1.24x -> 1.74x
  F8  FasterCache / Bimanual   / DreamGen MV    1.09x -> 1.29x

`K7` deserves a note: the LaTeX source reads `\second{.74}`, i.e. `0.74x`. That
cannot be right -- it is marked SECOND BEST in a column whose best is 1.81x, and
0.74x would be a slowdown. Confirmed with the user as a dropped leading `1`.

3 hygiene fixes (no numeric meaning changes, they stop fake parser warnings):
  L5  '1,18x'  -> '1.18x'   comma decimal separator
  G12 '1.153x' -> '1.15x'   3dp where the whole sheet (and the paper) uses 2dp
  9 cells holding the literal string 'None' -> '---'
      `parse_speedup_cell` in make_nr_circular_figures.py does `float('None')`,
      which raises, so each of those emits a bogus `unparseable: 'None'` repair
      note -- burying the notes that matter. '---' is already in that module's
      `MISSING` set and is what every other tab in this workbook uses for
      "method does not apply to this cell".

Usage:
    python assets/scripts/fix_speedup_sheet.py            # apply (default)
    python assets/scripts/fix_speedup_sheet.py --check    # verify only, no write
    python assets/scripts/fix_speedup_sheet.py --audit-parts
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
XLSX = REPO / 'assets/BENCHMARK_NR_main (3).xlsx'
SHEET = 'Speedup'

# The sheet's column layout, kept here so the edit list below can be read as
# (method, embodiment, track) instead of as bare spreadsheet coordinates.
COL_TRACK = {
    'B': ('Humanoid', 'DreamGen MV'), 'C': ('Humanoid', 'Ctrl-World MV'),
    'D': ('Humanoid', 'DreamGen SV'), 'E': ('Humanoid', 'DreamDojo SV'),
    'F': ('Bimanual', 'DreamGen MV'), 'G': ('Bimanual', 'Ctrl-World MV'),
    'H': ('Bimanual', 'DreamGen SV'), 'I': ('Bimanual', 'DreamDojo SV'),
    'J': ('Single-arm', 'DreamGen MV'), 'K': ('Single-arm', 'Ctrl-World MV'),
    'L': ('Single-arm', 'DreamGen SV'), 'M': ('Single-arm', 'DreamDojo SV'),
}
ROW_METHOD = {3: 'Dense', 4: 'SiTo', 5: 'ITM', 6: 'WorldCache', 7: 'DiCache',
              8: 'FasterCache', 9: 'PISA', 10: 'Radial', 11: 'SVG1', 12: 'SVG2'}

MISSING_TOKEN = '---'

# (cell, expected_old, new, why)
EDITS = [
    # -- 3 real value changes -------------------------------------------------
    ('K6', '1.31x', '1.81x', 'paper table: WorldCache single_arm Ctrl-World MV'),
    ('K7', '1.24x', '1.74x', 'paper table: DiCache single_arm Ctrl-World MV '
                             '(LaTeX ".74" = dropped leading 1)'),
    ('F8', '1.09x', '1.29x', 'paper table: FasterCache bimanual DreamGen MV'),
    # -- hygiene: numeric formatting -----------------------------------------
    ('L5', '1,18x', '1.18x', 'comma decimal separator -> dot'),
    ('G12', '1.153x', '1.15x', '3dp -> 2dp, matching the sheet and the paper'),
    # -- hygiene: 'None' string -> the workbook's own missing token -----------
    # SiTo / ITM are not implemented for DreamDojo; FasterCache is a known
    # no-op on the CFG-off Ctrl-World SVD UNet, so it is not run there.
    ('E4', 'None', MISSING_TOKEN, "SiTo x DreamDojo SV: not applicable"),
    ('I4', 'None', MISSING_TOKEN, "SiTo x DreamDojo SV: not applicable"),
    ('M4', 'None', MISSING_TOKEN, "SiTo x DreamDojo SV: not applicable"),
    ('E5', 'None', MISSING_TOKEN, "ITM x DreamDojo SV: not applicable"),
    ('I5', 'None', MISSING_TOKEN, "ITM x DreamDojo SV: not applicable"),
    ('M5', 'None', MISSING_TOKEN, "ITM x DreamDojo SV: not applicable"),
    ('C8', 'None', MISSING_TOKEN, "FasterCache x Ctrl-World MV: not applicable"),
    ('G8', 'None', MISSING_TOKEN, "FasterCache x Ctrl-World MV: not applicable"),
    ('K8', 'None', MISSING_TOKEN, "FasterCache x Ctrl-World MV: not applicable"),
]

# The dial `make_nr_circular_figures.py` maps speedup onto. Asserted after the
# patch so a future edit that pushes a value outside the window is caught HERE,
# where the cause is obvious, rather than showing up as a silently clamped bar.
DIAL_LO, DIAL_HI = 1.00, 2.30


def describe(cell: str) -> str:
    """'K6' -> 'WorldCache / Single-arm / Ctrl-World MV'."""
    col, row = cell[0], int(cell[1:])
    emb, track = COL_TRACK[col]
    return '%s / %s / %s' % (ROW_METHOD[row], emb, track)


def parse_speed(raw):
    """Mirror of parse_speedup_cell() in make_nr_circular_figures.py.

    Kept as a local copy ON PURPOSE: this script's job is to verify the file as
    the *consumer* will see it, so it must not drift with the plotting module's
    internals. Returns (value_or_None, note_or_None).
    """
    if raw is None:
        return None, None
    if isinstance(raw, (int, float)):
        return float(raw), None
    s = str(raw).strip()
    if not s or s in {MISSING_TOKEN, '--', ''}:
        return None, None
    body = s.rstrip('xX').strip()
    note = None
    if ',' in body:
        body, note = body.replace(',', '.'), 'comma decimal: %r' % s
    try:
        return float(body), note
    except ValueError:
        return None, 'unparseable: %r' % s


def audit_parts(path: Path) -> int:
    """Re-verify that re-saving with openpyxl cannot lose anything."""
    from openpyxl import load_workbook
    wb = load_workbook(path, data_only=False)
    formulas = [(ws.title, c.coordinate) for ws in wb.worksheets
                for row in ws.iter_rows() for c in row
                if isinstance(c.value, str) and c.value.startswith('=')]
    drawings = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if 'chart' in n or 'media' in n or 'pivot' in n:
                drawings[n] = len(z.read(n))
            elif 'drawing' in n:
                drawings[n] = len(z.read(n))
    fat = {n: sz for n, sz in drawings.items() if sz > 2000 or 'chart' in n
           or 'media' in n or 'pivot' in n}
    print('formula cells        : %d' % len(formulas))
    for t, c in formulas[:5]:
        print('    %s!%s' % (t, c))
    print('drawing/chart parts  : %d (%d non-stub)' % (len(drawings), len(fat)))
    for n, sz in sorted(fat.items()):
        print('    %-32s %d bytes' % (n, sz))
    if formulas or fat:
        print('\nREFUSING: this workbook has content openpyxl may not round-trip.')
        return 1
    print('\nok: no formulas, all drawing parts are empty stubs -> safe to re-save')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--xlsx', type=Path, default=XLSX)
    ap.add_argument('--check', action='store_true',
                    help='verify only; never write')
    ap.add_argument('--audit-parts', action='store_true',
                    help='re-check that openpyxl can safely re-save this file')
    args = ap.parse_args()

    if not args.xlsx.exists():
        print('missing %s' % args.xlsx, file=sys.stderr)
        return 1
    if args.audit_parts:
        return audit_parts(args.xlsx)

    from openpyxl import load_workbook
    wb = load_workbook(args.xlsx, data_only=False)
    if SHEET not in wb.sheetnames:
        print('missing sheet %r in %s' % (SHEET, args.xlsx), file=sys.stderr)
        return 1
    ws = wb[SHEET]

    todo, already, drift = [], [], []
    for cell, old, new, why in EDITS:
        cur = ws[cell].value
        cur_s = None if cur is None else str(cur)
        if cur_s == new:
            already.append((cell, new, why))
        elif cur_s == old:
            todo.append((cell, old, new, why))
        else:
            drift.append((cell, old, cur, new))

    print('=' * 78)
    print('%s :: %s' % (args.xlsx.name, SHEET))
    print('=' * 78)
    if drift:
        print('\nDRIFT -- refusing to write. These cells hold neither the expected')
        print('old value nor the target value, so someone changed them since this')
        print('script was written. Re-read the sheet and update EDITS.\n')
        for cell, old, cur, new in drift:
            print('  %-4s %-42s expected %r, found %r (target %r)'
                  % (cell, describe(cell), old, cur, new))
        return 1

    if already:
        print('\nalready correct (%d cell%s) -- nothing to do for these:'
              % (len(already), '' if len(already) == 1 else 's'))
        for cell, new, _why in already:
            print('  %-4s %-42s = %r' % (cell, describe(cell), new))

    if todo:
        print('\n%s (%d cell%s):'
              % ('WOULD CHANGE' if args.check else 'CHANGING',
                 len(todo), '' if len(todo) == 1 else 's'))
        print('  %-4s %-42s %-9s -> %-9s %s'
              % ('cell', 'method / embodiment / track', 'old', 'new', 'why'))
        for cell, old, new, why in todo:
            print('  %-4s %-42s %-9r -> %-9r %s'
                  % (cell, describe(cell), old, new, why))
    else:
        print('\nno changes needed -- sheet already matches the target.')

    if args.check:
        print('\n--check: no file written.')
    elif todo:
        for cell, _old, new, _why in todo:
            ws[cell] = new
        wb.save(args.xlsx)
        print('\nwrote %s' % args.xlsx)

    # ---- verify by RE-READING the file from disk (never from the in-memory wb)
    print('\n' + '-' * 78)
    print('verification (re-read from disk)')
    print('-' * 78)
    wb2 = load_workbook(args.xlsx, data_only=True)
    ws2 = wb2[SHEET]

    bad = [(c, n, ws2[c].value) for c, _o, n, _w in EDITS
           if str(ws2[c].value) != n]
    if bad and not args.check:
        for c, want, got in bad:
            print('  MISMATCH %-4s want %r got %r' % (c, want, got))
        return 1

    notes, vals, missing = [], [], 0
    for row, meth in ROW_METHOD.items():
        for col in COL_TRACK:
            v, note = parse_speed(ws2['%s%d' % (col, row)].value)
            if note:
                notes.append(('%s%d' % (col, row), note))
            if v is None:
                missing += 1
            else:
                vals.append((v, '%s%d' % (col, row)))

    out = [(v, c) for v, c in vals if not (DIAL_LO <= v <= DIAL_HI)]
    print('  parsed values        : %d' % len(vals))
    print('  missing (%s)         : %d' % (MISSING_TOKEN, missing))
    print('  repair notes         : %d %s'
          % (len(notes), '' if notes else '(clean)'))
    for c, n in notes:
        print('      %-4s %s' % (c, n))
    lo, hi = min(vals), max(vals)
    print('  range                : %.2fx (%s, %s) .. %.2fx (%s, %s)'
          % (lo[0], lo[1], describe(lo[1]), hi[0], hi[1], describe(hi[1])))
    print('  dial [%.2f, %.2f]     : %d value(s) outside'
          % (DIAL_LO, DIAL_HI, len(out)))
    for v, c in out:
        print('      %-4s %-42s %.2fx OUTSIDE' % (c, describe(c), v))

    ok = not notes and not out and not bad
    print('\n%s' % ('ok: sheet is clean, fully parseable, and inside the dial'
                    if ok else 'PROBLEMS remain (see above)'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
