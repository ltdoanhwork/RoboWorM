#!/usr/bin/env python3
"""Audit whether the FVD column can support the comparisons the paper draws from it.

FVD is a distance between two fitted Gaussians in I3D feature space. Two properties
make it dangerous in a benchmark table, and this script measures both instead of
assuming them away:

1. FINITE-SAMPLE BIAS. The estimate is strongly biased upward at small n, so two
   cells measured at different n are not on the same scale even with an identical
   reference. `results/reports/fvd_sample_size_probe.json` quantifies it directly by
   subsampling one cell: the same cell measures 25.8 at n=1259 and 220.5 at n=128.

2. RESOLUTION LIMIT. Each (embodiment, camera, regime) group has a split-half floor:
   the FVD between two halves of the REAL set. A model value below that floor is
   indistinguishable from resampling the real data, and -- the part that is easy to
   miss -- a GAP between two models smaller than the floor is not a ranking, even
   when both values sit above it.

The script reports, per comparison the table makes:
  * each model's FVD against its group's floor,
  * the |A-B| gap against the same floor  <- this is what licenses a bold,
  * whether the two cells are n-matched,
  * how many cameras the multi-view average actually covers.

Run:
    models/dreamgen/.venv/bin/python assets/scripts/fvd_resolution_audit.py

Writes results/reports/fvd_resolution_audit.json and prints the table.
"""
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MV = REPO / 'results/reports/fvd_perview_pooled.json'
SV = REPO / 'results/reports/fvd_singleview_pooled.json'
PROBE = REPO / 'results/reports/fvd_sample_size_probe.json'
OUT = REPO / 'results/reports/fvd_resolution_audit.json'

# The comparisons the dense-baseline table draws: (section, embodiment, model A, model B).
# A/B are the two rows that get compared, and the labels are the ones used as keys in
# the pooled artefacts (which is why bimanual/humanoid use the 4-view Ctrl-World grid
# and single-arm the 3-view strip -- see ctrlworld_layout_reported in
# results/reports/rq1_dense_baselines.json).
PAIRS = [
    ('a', 'bimanual', 'Dreamgen MV', 'CW-4view MV'),
    ('a', 'humanoid', 'Dreamgen MV', 'CW-4view MV'),
    ('a', 'single_arm', 'Dreamgen MV', 'Ctrl-World MV'),
    ('b', 'bimanual', 'Dreamgen SV', 'DreamDojo SV'),
    ('b', 'humanoid', 'Dreamgen SV', 'DreamDojo SV'),
    ('b', 'single_arm', 'Dreamgen SV', 'DreamDojo SV'),
]
SPLITS = ('makovian', 'non_makovian')


def flatten(doc):
    """Collect every leaf cell record keyed 'cells|emb|label|split|method'."""
    out = {}

    def walk(o, path):
        if isinstance(o, dict):
            if 'fvd' in o and not isinstance(o['fvd'], dict):
                out['|'.join(path)] = o
            else:
                for k, v in o.items():
                    walk(v, path + [str(k)])

    walk(doc, [])
    return out


def main():
    mv = json.loads(MV.read_text())
    sv = json.loads(SV.read_text())
    cells = flatten(mv)
    cells.update(flatten(sv))
    cams, mv_floor, sv_floor = mv['common_cameras'], mv['floor'], sv['floor']
    poolable = sv.get('poolable', {})

    def floor_of(section, emb, split):
        """Split-half FVD of the REAL set for this group.

        Multi-view floors are per camera, so the floor of a multi-view cell is the
        mean over exactly the cameras the reported value averages over -- using all
        cameras would compare against a reference the number never touched.
        """
        if section == 'b':
            return sv_floor.get('%s|%s' % (emb, split)), None
        cl = cams.get('%s|%s' % (emb, split), [])
        vals = [mv_floor['%s|%s|%s' % (emb, c, split)] for c in cl
                if '%s|%s|%s' % (emb, c, split) in mv_floor]
        return (st.mean(vals) if vals else None), len(vals)

    rep = {
        'generated_by': 'assets/scripts/fvd_resolution_audit.py',
        'sources': [str(p.relative_to(REPO)) for p in (MV, SV, PROBE)],
        'criterion': {
            'value_resolvable': 'cell FVD > split-half floor of its own real group',
            'gap_resolvable': 'abs(A-B) > that same floor; this is what licenses a bold',
            'n_matched': 'max(nA,nB)/min(nA,nB) < 1.15',
        },
        'comparisons': [],
    }

    if PROBE.exists():
        pr = json.loads(PROBE.read_text())
        worst = max(pr['cells'],
                    key=lambda c: (c['fvd_k128'] / c['fvd_full']) if c.get('fvd_k128')
                    and c.get('fvd_full') else 0)
        rep['finite_sample_bias'] = {
            'probe_cell': worst['cell'], 'n_full': worst['n_ref'],
            'fvd_at_full_n': round(worst['fvd_full'], 1),
            'fvd_at_n128': round(worst['fvd_k128'], 1),
            'inflation': round(worst['fvd_k128'] / worst['fvd_full'], 2),
            'note': 'same cell, same reference, only n changed',
        }
        print('finite-sample bias: %s measures %.1f at n=%d but %.1f at n=128 (%.1fx)'
              % (worst['cell'], worst['fvd_full'], worst['n_ref'],
                 worst['fvd_k128'], worst['fvd_k128'] / worst['fvd_full']))

    hdr = ('%-4s %-11s %-13s %8s %8s %8s %7s %7s %6s %-5s %s')
    print('\n' + hdr % ('sec', 'emb', 'regime', 'FVD_A', 'FVD_B', 'floor',
                        'A/fl', 'B/fl', 'gap/fl', 'n-ok', 'verdict'))
    n_bad = 0
    for section, emb, a, b in PAIRS:
        for split in SPLITS:
            ra = cells.get('cells|%s|%s|%s|Dense' % (emb, a, split))
            rb = cells.get('cells|%s|%s|%s|Dense' % (emb, b, split))
            fl, ncam = floor_of(section, emb, split)
            if not (ra and rb and fl):
                continue
            na, nb = ra['n_windows'], rb['n_windows']
            gap = abs(ra['fvd'] - rb['fvd'])
            nm = max(na, nb) / min(na, nb) < 1.15
            gap_ok = gap > fl
            if not gap_ok:
                n_bad += 1
            verdict = ('bold justified' if (gap_ok and nm) else
                       ('GAP INSIDE FLOOR' if not gap_ok else 'n MISMATCH'))
            rec = {
                'section': section, 'embodiment': emb, 'regime': split,
                'model_a': a, 'model_b': b,
                'fvd_a': round(ra['fvd'], 1), 'fvd_b': round(rb['fvd'], 1),
                'n_a': na, 'n_b': nb, 'n_matched': nm,
                'floor': round(fl, 1), 'n_cameras_averaged': ncam,
                'a_over_floor': round(ra['fvd'] / fl, 2),
                'b_over_floor': round(rb['fvd'] / fl, 2),
                'gap': round(gap, 1), 'gap_over_floor': round(gap / fl, 2),
                'gap_resolvable': gap_ok,
                'shared_pool_valid': poolable.get('%s|%s' % (emb, split))
                if section == 'b' else None,
                'verdict': verdict,
            }
            rep['comparisons'].append(rec)
            print(hdr % (section, emb, split, '%.1f' % ra['fvd'], '%.1f' % rb['fvd'],
                         '%.1f' % fl, '%.2f' % rec['a_over_floor'],
                         '%.2f' % rec['b_over_floor'], '%.2f' % rec['gap_over_floor'],
                         'yes' if nm else 'NO', verdict))

    rep['summary'] = {
        'n_comparisons': len(rep['comparisons']),
        'n_gap_inside_floor': n_bad,
        'n_bold_justified': sum(1 for c in rep['comparisons']
                                if c['verdict'] == 'bold justified'),
    }
    print('\n%d of %d comparisons have a gap INSIDE the noise floor; %d support a bold'
          % (n_bad, len(rep['comparisons']), rep['summary']['n_bold_justified']))
    OUT.write_text(json.dumps(rep, indent=1) + '\n')
    print('wrote %s' % OUT.relative_to(REPO))


if __name__ == '__main__':
    main()
