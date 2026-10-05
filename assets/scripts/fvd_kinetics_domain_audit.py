#!/usr/bin/env python3
"""How far outside Kinetics-400 does this benchmark's video sit, and what does that do to FVD?

Motivation (reviewer question, 2026-09-26): "Kinetics is used as the reference for FVD --
does that make sense here, or is it a different distribution?"

The premise needs one correction before the question can be answered: Kinetics is NOT the
reference SET in this benchmark. The reference is a pooled set of REAL robot videos from
the same embodiment x camera x regime (see .cursor/skills/fvd-shared-real-pool/SKILL.md).
Kinetics-400 enters only through the feature extractor -- and it enters much more deeply
than "pretraining data", which is what this script measures:

    checkpoints/metrics/fvd/i3d_torchscript.pt, forward(return_features=True) returns `out20`,
    the output of conv3d_0c_1x1 time-averaged -- i.e. the PRE-SOFTMAX KINETICS-400 CLASS
    LOGITS. (return_features=False would just add the softmax.)

So FVD here fits two Gaussians in a 400-dimensional space whose axes ARE 400 human-action
categories scraped from YouTube. The question is therefore not "is the reference wrong"
but "is this coordinate system able to resolve robot manipulation video at all".

What is measured, from the cached per-camera features (no GPU, no re-decoding):

1. OOD-ness of the real robot clips w.r.t. the Kinetics head itself: softmax confidence,
   how many of the 400 classes are ever the top-1, and the mass on the single most
   frequent class. An in-distribution corpus spreads over many classes with confident
   predictions; a corpus the head has no vocabulary for collapses onto a few classes.
2. Effective dimensionality of the real feature cloud (participation ratio and the number
   of PCA directions holding 95%/99% of the variance) against the 400 dimensions whose
   covariance FVD must estimate. This is the mechanism behind the sample-size bias already
   documented in fvd_sample_size_probe.json: if the cloud is d_eff-dimensional, the
   400x400 covariance is rank-deficient by construction.
3. Where the Frechet distance actually comes from: the mean term vs the covariance term,
   and how much of the mean term lives in the top few PCA directions of the real pool.
   A distance carried by a handful of directions is not "distributional realism" in 400
   semantic dimensions.
4. A scale ruler for interpretation: real-vs-real FVD across EMBODIMENTS (different scene,
   still real robot video) against dense-pred-vs-real within an embodiment. If swapping
   the scene costs far more than replacing real video with generated video, the column is
   ordering framing/scene, not physical plausibility.

Run:
    models/dreamgen/.venv/bin/python assets/scripts/fvd_kinetics_domain_audit.py

Writes metrics/reports/fvd_kinetics_domain_audit.json and prints the tables.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FEATS = REPO / 'metrics/WorldScore/results/fvd_perview/feats'
OUT = REPO / 'metrics/reports/fvd_kinetics_domain_audit.json'


def load(limit_per_group: int | None = None):
    """Return real[(emb,label)] and dense-pred[(emb,label)] window stacks.

    Real windows are deduplicated by (ref_identity, camera index): one GT video is
    hard-linked into every method tree, and counting it once per method would inflate n
    and shift every statistic (protocol rule 6).
    """
    real: dict[tuple, list] = defaultdict(list)
    pred: dict[tuple, list] = defaultdict(list)
    seen_real: set = set()
    n_files = 0
    for f in sorted(FEATS.glob('*.npz')):
        try:
            d = np.load(f, allow_pickle=True)
            meta = json.loads(str(d['meta']))
        except Exception:
            continue
        n_files += 1
        key = (meta['embodiment'], meta.get('label', '?'))
        for name in d.files:
            if name == 'meta':
                continue
            kind, _, cam = name.partition('_')
            arr = np.asarray(d[name], dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 400 or not len(arr):
                continue
            if kind == 'ref':
                ident = (meta.get('ref_identity', ''), cam)
                if ident in seen_real:
                    continue
                seen_real.add(ident)
                bucket = real
            elif kind == 'pred' and meta.get('method') == 'dense':
                bucket = pred
            else:
                continue
            if limit_per_group is not None and len(bucket[key]) >= limit_per_group:
                continue
            bucket[key].append(arr)
    stack = lambda b: {k: np.concatenate(v) for k, v in b.items() if v}
    return stack(real), stack(pred), n_files


def class_hist(x: np.ndarray) -> np.ndarray:
    """Normalised histogram over which of the 400 Kinetics classes is top-1."""
    h = np.bincount(x.argmax(1), minlength=400).astype(float)
    return h / max(h.sum(), 1)


def kinetics_occupancy(x: np.ndarray) -> dict:
    """How the Kinetics-400 head itself responds to these clips."""
    z = x - x.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    hist = Counter(x.argmax(1).tolist())
    counts = np.array(sorted(hist.values(), reverse=True), dtype=float)
    frac = counts / counts.sum()
    ent = float(-(frac * np.log(frac)).sum() / np.log(400))
    return dict(
        n_windows=int(len(x)),
        max_softmax_mean=float(p.max(1).mean()),
        max_softmax_median=float(np.median(p.max(1))),
        distinct_top1_classes=int(len(hist)),
        top1_class_mass=float(frac[0]),
        top5_class_mass=float(frac[:5].sum()),
        top10_class_mass=float(frac[:10].sum()),
        class_entropy_normalised=ent,
        # Indices only: the Kinetics-400 label map is not shipped with the torchscript
        # detector, and the argument does not depend on the names.
        top5_class_indices=[int(i) for i, _ in hist.most_common(5)],
    )


def eff_dim(x: np.ndarray) -> dict:
    """Effective dimensionality of the cloud whose 400x400 covariance FVD estimates."""
    xc = x - x.mean(0)
    lam = np.linalg.svd(xc, compute_uv=False) ** 2 / max(len(x) - 1, 1)
    lam = np.clip(lam, 0, None)
    tot = lam.sum()
    cum = np.cumsum(lam) / tot
    return dict(
        participation_ratio=float(tot ** 2 / (lam ** 2).sum()),
        dims_95pct_var=int(np.searchsorted(cum, 0.95) + 1),
        dims_99pct_var=int(np.searchsorted(cum, 0.99) + 1),
        ambient_dims=400,
    )


def sqrtm_psd(a: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh((a + a.T) / 2)
    return (v * np.sqrt(np.clip(w, 0, None))) @ v.T


def frechet(x: np.ndarray, y: np.ndarray) -> dict:
    """FVD plus the decomposition into mean and covariance terms."""
    mx, my = x.mean(0), y.mean(0)
    cx = np.cov(x, rowvar=False)
    cy = np.cov(y, rowvar=False)
    dmu = mx - my
    mean_term = float(dmu @ dmu)
    cov_term = float(np.trace(cx) + np.trace(cy) - 2 * np.trace(sqrtm_psd(cx @ cy)))
    # Share of the mean term lying in the leading PCA directions of the real cloud.
    yc = y - my
    _, _, vt = np.linalg.svd(yc, full_matrices=False)
    proj = vt @ dmu
    share = np.cumsum(proj ** 2) / max(mean_term, 1e-12)
    return dict(
        fvd=mean_term + cov_term,
        mean_term=mean_term,
        cov_term=cov_term,
        mean_term_share=mean_term / max(mean_term + cov_term, 1e-12),
        mean_term_in_top1_pc=float(share[0]),
        mean_term_in_top10_pc=float(share[min(9, len(share) - 1)]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit-per-group', type=int, default=None,
                    help='cap clips per group (debug; default = all cached)')
    args = ap.parse_args()

    real, pred, n_files = load(args.limit_per_group)
    if not real:
        raise SystemExit(f'no cached features under {FEATS}')

    rep: dict = dict(
        feature_space=dict(
            checkpoint='checkpoints/metrics/fvd/i3d_torchscript.pt',
            returns='out20 = conv3d_0c_1x1 time-averaged = pre-softmax Kinetics-400 '
                    'class logits (softmax is applied only when return_features=False)',
            dims=400,
            interpretation='each axis is one Kinetics-400 human-action category',
            reference_set='pooled REAL robot video per (embodiment, camera, regime) -- '
                          'NOT Kinetics; Kinetics enters only as the feature basis',
        ),
        n_feature_files=n_files,
        groups={}, cross_embodiment={}, pooled_real={},
    )

    all_real = np.concatenate(list(real.values()))
    rep['pooled_real'] = dict(kinetics=kinetics_occupancy(all_real),
                              effective_dim=eff_dim(all_real))

    print(f'\ncached feature files: {n_files}   real windows: {len(all_real)}')
    print('\n=== Kinetics-400 head response to REAL robot video ===')
    hdr = f"{'group':28s} {'n':>7s} {'maxP':>6s} {'#cls':>5s} {'top1%':>6s} {'H/HK':>5s} {'d95':>4s} {'PR':>6s}"
    print(hdr)
    for k in sorted(real):
        ko = kinetics_occupancy(real[k])
        ed = eff_dim(real[k])
        rep['groups']['|'.join(k)] = dict(kinetics=ko, effective_dim=ed)
        print(f"{'/'.join(k):28s} {ko['n_windows']:7d} {ko['max_softmax_mean']:6.3f} "
              f"{ko['distinct_top1_classes']:5d} {100 * ko['top1_class_mass']:6.1f} "
              f"{ko['class_entropy_normalised']:5.2f} {ed['dims_95pct_var']:4d} "
              f"{ed['participation_ratio']:6.1f}")
    p = rep['pooled_real']
    print(f"{'ALL REAL (pooled)':28s} {p['kinetics']['n_windows']:7d} "
          f"{p['kinetics']['max_softmax_mean']:6.3f} "
          f"{p['kinetics']['distinct_top1_classes']:5d} "
          f"{100 * p['kinetics']['top1_class_mass']:6.1f} "
          f"{p['kinetics']['class_entropy_normalised']:5.2f} "
          f"{p['effective_dim']['dims_95pct_var']:4d} "
          f"{p['effective_dim']['participation_ratio']:6.1f}")

    print('\n=== Where the Frechet distance comes from (dense pred vs its own real pool) ===')
    print(f"{'group':28s} {'FVD':>9s} {'mean%':>6s} {'cov%':>6s} {'top1PC%':>8s} "
          f"{'top10PC%':>9s} {'clsTV':>6s} {'sameTop1%':>10s}")
    for k in sorted(set(real) & set(pred)):
        fr = frechet(pred[k], real[k])
        # Does the KINETICS SEMANTICS distinguish real from generated at all? Total
        # variation between the two top-1 class histograms, and the fraction of class
        # mass the generated set puts on the real set's own top-5 classes.
        hr, hp = class_hist(real[k]), class_hist(pred[k])
        top5 = np.argsort(hr)[::-1][:5]
        fr['class_hist_total_variation'] = float(0.5 * np.abs(hr - hp).sum())
        fr['pred_mass_on_real_top5_classes'] = float(hp[top5].sum())
        fr['real_mass_on_real_top5_classes'] = float(hr[top5].sum())
        rep['groups'].setdefault('|'.join(k), {})['dense_vs_real'] = fr
        print(f"{'/'.join(k):28s} {fr['fvd']:9.1f} {100 * fr['mean_term_share']:6.1f} "
              f"{100 * (1 - fr['mean_term_share']):6.1f} "
              f"{100 * fr['mean_term_in_top1_pc']:8.1f} "
              f"{100 * fr['mean_term_in_top10_pc']:9.1f} "
              f"{fr['class_hist_total_variation']:6.3f} "
              f"{100 * fr['pred_mass_on_real_top5_classes']:10.1f}")

    print('\n=== Scale ruler: REAL vs REAL across embodiments (same camera family) ===')
    keys = sorted(real)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a[0] == b[0]:
                continue
            fr = frechet(real[a], real[b])
            rep['cross_embodiment'][f"{'/'.join(a)} vs {'/'.join(b)}"] = fr
            print(f"{'/'.join(a):26s} vs {'/'.join(b):26s} {fr['fvd']:9.1f}")

    # How much of the 400-class vocabulary separates three physically distinct
    # embodiments: if they share their dominant classes, the vocabulary is not
    # resolving this domain.
    by_emb = defaultdict(list)
    for (emb, _lab), v in real.items():
        by_emb[emb].append(v)
    emb_hist = {e: class_hist(np.concatenate(v)) for e, v in by_emb.items()}
    rep['embodiment_class_overlap'] = {}
    print('\n=== Kinetics vocabulary shared between DIFFERENT embodiments ===')
    embs = sorted(emb_hist)
    for i, a in enumerate(embs):
        for b in embs[i + 1:]:
            ta = set(np.argsort(emb_hist[a])[::-1][:5].tolist())
            tb = set(np.argsort(emb_hist[b])[::-1][:5].tolist())
            tv = float(0.5 * np.abs(emb_hist[a] - emb_hist[b]).sum())
            rep['embodiment_class_overlap'][f'{a} vs {b}'] = dict(
                top5_shared=len(ta & tb), shared_indices=sorted(ta & tb),
                class_hist_total_variation=tv)
            print(f'  {a:11s} vs {b:11s} top5 shared {len(ta & tb)}/5   TV {tv:.3f}')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2))
    print(f'\nwrote {OUT.relative_to(REPO)}')


if __name__ == '__main__':
    main()
