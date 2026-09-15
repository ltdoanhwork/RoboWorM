"""Paper figures from real NR results: distributions, motion/continuity, coverage.

Exports vector PDF/SVG, 300-DPI PNG, and the exact CSV used for plotting.
Never imputes missing metrics or merges incompatible model/view strata.
"""
import argparse
import csv
import json
from pathlib import Path

V1 = (['scene_integrity', 'entity_presence', 'task_lock', 'actor_motion_raw',
       'motion_continuity', 'actor_flicker_raw'],
      ['Visual integrity', 'Entity presence', 'Task alignment',
       'Actor motion (px/source frame)', 'Motion continuity', 'Actor flicker (0–1)'])
V2 = (['visual_integrity', 'text_condition_alignment', 'actor_motion_raw',
       'actor_motion_active_rate', 'active_motion_continuity',
       'actor_flicker_raw', 'camera_drift_raw'],
      ['Visual integrity (IQA p10 ×100)', 'Text alignment (CLIPScore)',
       'Actor activity (px/source frame)', 'Active rate',
       'Active-motion continuity (SSIM ×100)', 'Actor flicker (RGB residual)',
       'Static-camera drift (px/source frame)'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    rows = [json.loads(p.read_text()) for p in sorted((args.run/'results').glob('*.json'))]
    KEYS, LABELS = V2 if any(r.get('protocol_version') == 'dense-nr-v2-closed-world' for r in rows) else V1
    out = args.run/'figures'; out.mkdir(exist_ok=True)
    fields = ['id','model','embodiment','split','view','actor_validity','source_fps','effective_fps', *KEYS]
    with (out/'metrics.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    manifest = json.loads((args.run/'manifest.json').read_text())
    summary = dict(expected=len(manifest), completed=len(rows),
                   errors=len(list((args.run/'errors').glob('*.json'))),
                   metric_valid_counts={k:sum(r.get(k) is not None for r in rows) for k in KEYS})
    (out/'coverage.json').write_text(json.dumps(summary, indent=2))
    if not rows:
        print('No completed results; wrote coverage only.'); return
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8,
                         'pdf.fonttype':42, 'ps.fonttype':42, 'axes.spines.top':False,
                         'axes.spines.right':False, 'savefig.bbox':'tight'})
    def save(fig, stem):
        for ext in ('pdf','svg','png'): fig.savefig(out/f'{stem}.{ext}', dpi=300)
        plt.close(fig)
    def group(r): return '/'.join(r[k] for k in ('model','embodiment','split','view'))
    groups = sorted(set(map(group, rows)))
    # Facet by model/embodiment/split, retaining separate per-view distributions.
    for cell in sorted(set('/'.join(r[k] for k in ('model','embodiment','split')) for r in rows)):
        selected = [r for r in rows if group(r).startswith(cell+'/')]
        views = sorted(set(r['view'] for r in selected))
        cols = 3; nrows = (len(KEYS) + cols - 1) // cols
        fig, axs = plt.subplots(nrows, cols, figsize=(10, 2.7*nrows), constrained_layout=True)
        axs = list(np.atleast_1d(axs).flat)
        for ax, key, label in zip(axs, KEYS, LABELS):
            data, names = [], []
            for view in views:
                rr = [r for r in selected if r['view']==view and r.get(key) is not None]
                if key in ('actor_motion_raw','actor_motion_active_rate',
                           'active_motion_continuity','actor_flicker_raw'):
                    rr = [r for r in rr if r.get('actor_validity')=='robot_detected']
                vals = [r[key] for r in rr]
                if vals: data.append(vals); names.append(f'{view}\nn={len(vals)}')
            if data:
                ax.boxplot(data, tick_labels=names, showfliers=False)
                ax.tick_params(axis='x',labelrotation=35)
            else: ax.text(.5,.5,'No valid observations',ha='center',transform=ax.transAxes)
            ax.set_title(label); ax.grid(axis='y',alpha=.2)
        for ax in axs[len(KEYS):]: ax.axis('off')
        fig.suptitle('Dense: '+cell+' (partial)' if len(rows)<len(manifest) else 'Dense: '+cell)
        save(fig, 'distribution_'+cell.replace('/','_'))
    fig, ax = plt.subplots(figsize=(5.5,3.5))
    cont_key = 'active_motion_continuity' if 'active_motion_continuity' in KEYS else 'motion_continuity'
    for model in sorted(set(r['model'] for r in rows)):
        rr = [r for r in rows if r['model']==model and r.get('actor_validity')=='robot_detected'
              and r.get('actor_motion_px_per_second') is not None and r.get(cont_key) is not None]
        if rr: ax.scatter([r['actor_motion_px_per_second'] for r in rr],
                          [r[cont_key] for r in rr],s=9,alpha=.35,label=model)
    ax.set_xlabel('Actor motion (analysis px/s)'); ax.set_ylabel('Active-motion continuity' if cont_key != 'motion_continuity' else 'Motion continuity')
    ax.set_title('Dense diagnostics: pooled views; not a model ranking')
    handles, _ = ax.get_legend_handles_labels()
    if handles: ax.legend(frameon=False)
    ax.grid(alpha=.2); save(fig,'motion_continuity')
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(np.arange(len(KEYS)),[summary['metric_valid_counts'][k] for k in KEYS],color='#3977a8')
    ax.axhline(len(manifest),color='black',ls='--',label='Expected view-clips')
    ax.set_xticks(np.arange(len(KEYS)), [x.split(' (')[0] for x in LABELS], rotation=25, ha='right')
    ax.set_ylabel('Available results'); ax.legend(frameon=False)
    save(fig,'coverage')
    print(json.dumps(summary))


if __name__=='__main__': main()
