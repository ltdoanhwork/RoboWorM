"""Build ONE self-contained HTML dashboard for the six NR metrics.

Self-contained means exactly that: no CDN, no fetch, no build step. The cache is
embedded as JSON and every chart is SVG drawn by ~700 lines of vanilla JS, so the file
works from a `file://` URL on a laptop with no network -- which is how these results
actually get looked at.

The dashboard is NOT a prettier version of the figures. The figures answer fixed
questions; this answers the ones a reader invents: "and on wrist only?", "what about
just DreamGen?", "which clip is driving that?". Filtering re-medians client side over
the per-clip values, so every number stays a median of the SAME gated observations the
paper tables use -- never an average of medians.

What it refuses to do is the point:
  * a metric's status badge is attached to every number, everywhere;
  * DROPPED / DIAGNOSTIC columns cannot be selected as the ranking key;
  * a CEILING_FAILED column will not render a model ranking, only its accel deltas;
  * a PAIRED_ONLY column hides its absolute medians;
  * any cell below MIN_PAIRS_FOR_CLAIM renders as "no claim", not as a number.

Usage:
  python assets/scripts/make_nr_dashboard.py --run metrics/WorldScore/results/dense_nr_full
  -> <run>/figures/nr_dashboard.html
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nr_viz_common as C  # noqa: E402

# Per-clip fields the client needs. `source_fps` / `effective_fps` / `n_sampled` are not
# display fields -- they are what `table_dense_nr.comparable_pair` gates on, and without
# them the dashboard would pair a 277-frame 16fps dense clip against a 93-frame 10fps
# accel clip and invent the same fake +10 AMC "improvement" that gate was written to kill.
CLIP_FIELDS = ['id', 'model', 'embodiment', 'split', 'camera_mount', 'method', 'view',
               'pair_key', 'instruction', 'source_fps', 'effective_fps', 'n_sampled']


def build_payload(cache_dir, run):
    cache = C.Cache(cache_dir)
    # Three interning passes, because a naive dump is ~24 MB of repeated strings:
    #   * strata values ('dreamgen', 'makovian', ...) -> index into a table
    #   * gate reasons -> index into a table (the same ~10 sentences, 40k times)
    #   * instructions -> index (one caption per episode, repeated per view+method)
    # The client reverses this once at load. Cuts the file to roughly a third.
    # Numeric fields are interned too: `source_fps` takes ~4 distinct values across 39k
    # clips, so a table lookup is smaller than the repeated float.
    tables = {f: [] for f in CLIP_FIELDS if f != 'id'}
    index = {f: {} for f in tables}

    def intern(field, value):
        if value is None:
            return -1
        t, ix = tables[field], index[field]
        if value not in ix:
            ix[value] = len(t)
            t.append(value)
        return ix[value]

    reasons, reason_ix = [], {}

    def intern_reason(value):
        if value is None:
            return -1
        if value not in reason_ix:
            reason_ix[value] = len(reasons)
            reasons.append(value)
        return reason_ix[value]

    # Round to a hair beyond the reported precision: full float64 repr is ~40% of the
    # payload and no panel reads past 4 decimals.
    round_to = {c['key']: min(c['digits'] + 3, 6) for c in cache.columns}
    clips = []
    for c in cache.clips:
        row = [c['id']] + [intern(f, c.get(f)) for f in CLIP_FIELDS if f != 'id']
        vals, grs = [], []
        any_gr = False
        for k in cache.keys:
            v = c['gated'].get(k)
            vals.append(None if v is None else round(v, round_to[k]))
            gr = intern_reason((c.get('gate_reason') or {}).get(k))
            grs.append(gr)
            any_gr = any_gr or gr >= 0
        row.append(vals)
        row.append(grs if any_gr else 0)     # 0 = "nothing gated out for this clip"
        clips.append(row)
    # Per-frame traces for the clip explorer: only the exemplars have them.
    perframe = {}
    r3 = lambda xs: None if xs is None else [None if v is None else round(v, 3)
                                             for v in xs]
    for group in cache.exemplar_groups():
        for cid in [group['dense_id']] + [a['id'] for a in group['accel']]:
            pf = cache.perframe(cid)
            if not pf:
                continue
            perframe[cid] = {
                'method': pf.get('method'), 'model': pf.get('model'),
                'embodiment': pf.get('embodiment'),
                'camera_mount': pf.get('camera_mount'),
                'fg_flow_p95': r3([p.get('fg_flow_p95') for p in pf['per_frame']]),
                'fg_flow_ego_p95': r3([p.get('fg_flow_ego_p95') for p in pf['per_frame']]),
                'arm_flow_med': r3([p.get('arm_flow_med') for p in pf['per_frame']]),
                'obj_flow_med': r3([p.get('obj_flow_med') for p in pf['per_frame']]),
                'bg_flow_med': r3([p.get('bg_flow_med') for p in pf['per_frame']]),
                'fg_cov_pct': r3([p.get('fg_cov_pct') for p in pf['per_frame']]),
                'gain': r3((pf.get('continuity') or {}).get('gain')),
                'copy_mse': r3((pf.get('continuity') or {}).get('copy_mse')),
                'vfi_mse': r3((pf.get('continuity') or {}).get('vfi_mse')),
                'gated': pf.get('gated'),
            }
    return cache, {
        'meta': cache.meta,
        'stamp': cache.stamp(),
        'columns': [{**col, 'status': cache.status(col['key']),
                     'rule': C.STATUS_RULE[cache.status(col['key'])],
                     'why': cache.why(col['key']),
                     'polarity_text': cache.polarity_arrow(col['key'])}
                    for col in cache.columns],
        'policy': cache.policy,
        'status_style': {k: {'short': v[0], 'caption': v[1], 'color': v[2]}
                         for k, v in C.STATUS_STYLE.items()},
        # Column-order contract for the interned clip rows, so the client never has to
        # hardcode positions.
        'clip_fields': CLIP_FIELDS,
        'tables': tables,
        'reasons': reasons,
        'clips': clips,
        'pooled_accel': cache.agg['t3b_accel_pooled'],
        'cell_accel': cache.agg['t3_accel_cell'],
        'paired_gt': cache.agg['t2_paired_gt'],
        'coverage': cache.agg['coverage'],
        'redundancy': cache.agg['redundancy'],
        'common_support': cache.agg.get('common_support') or {},
        'exemplars': cache.agg.get('exemplars') or [],
        'perframe': perframe,
        'labels': {
            'method': C.METHOD_LABEL, 'model': C.MODEL_LABEL,
            'embodiment': C.EMB_LABEL, 'split': C.SPLIT_LABEL,
            'method_order': C.METHOD_ORDER, 'model_order': C.MODEL_ORDER,
            'emb_order': C.EMB_ORDER, 'mount_order': C.MOUNT_ORDER,
            'family': C.FAMILY, 'family_color': C.FAMILY_COLOR,
        },
        'run': str(run),
    }


ASSETS = Path(__file__).resolve().parent / 'nr_dashboard'


def render_html(cache, payload):
    """Inline every asset into one file. `</script>` inside the JSON would close the
    embedding tag early, so it is escaped -- the classic self-contained-HTML footgun."""
    tpl = (ASSETS / 'template.html').read_text()
    data = json.dumps(payload, separators=(',', ':'), allow_nan=False)
    data = data.replace('</', '<\\/')
    subs = {
        '__TITLE__': f'NR metrics - {Path(payload["run"]).name}',
        '__CSS__': (ASSETS / 'style.css').read_text(),
        '__CORE__': (ASSETS / 'core.js').read_text(),
        '__CHARTS__': (ASSETS / 'charts.js').read_text(),
        '__VIEWS1__': (ASSETS / 'views1.js').read_text(),
        '__VIEWS2__': (ASSETS / 'views2.js').read_text(),
        '__DATA__': data,
        '__STAMP__': cache.stamp(),
        '__PARTIAL_CLASS__': 'partial' if cache.partial else '',
        '__RUN__': payload['run'],
        '__PROTOCOL__': str(cache.meta.get('protocol')),
        '__GENERATED__': str(cache.meta.get('generated_at')),
    }
    for k, v in subs.items():
        tpl = tpl.replace(k, v)
    return tpl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--cache', type=Path, default=None,
                    help='cache dir (default <run>/figures/cache)')
    ap.add_argument('--out', type=Path, default=None,
                    help='output html (default <run>/figures/nr_dashboard.html)')
    args = ap.parse_args()
    cache_dir = args.cache or args.run / 'figures' / 'cache'
    if not (cache_dir / 'agg.json').exists():
        raise SystemExit(f'no cache at {cache_dir}; run assets/scripts/nr_viz_cache.py first')
    out = args.out or args.run / 'figures' / 'nr_dashboard.html'
    out.parent.mkdir(parents=True, exist_ok=True)

    cache, payload = build_payload(cache_dir, args.run)
    print(f'cache: {cache.stamp()}')
    html = render_html(cache, payload)
    out.write_text(html)
    mb = len(html.encode()) / 1e6
    print(f'wrote {out}  ({mb:.1f} MB, {len(payload["clips"]):,} clips, '
          f'{len(payload["perframe"])} per-frame traces)')
    if mb > 80:
        print('  NOTE: large for a single file; consider --cache from a limited run '
              'for sharing')


if __name__ == '__main__':
    main()
