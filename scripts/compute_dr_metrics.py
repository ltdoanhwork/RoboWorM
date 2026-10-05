#!/usr/bin/env python3
"""Distortion-reference (DR) metrics over sampling_dataset: PSNR / SSIM / LPIPS / FVD.

TWO REFERENCES, BOTH STORED
---------------------------
Every accelerated clip is scored twice:

  vs_dense  the SAME episode's dense prediction.  This is the PhyFid-DR reference named in
            the phyworld-benchmark rule ("reference = dense pred of the same episode ...
            do not use GT video as the main speedup reference"), and it is the number an
            acceleration claim rests on: the generative model's own error against reality
            is present in BOTH clips and cancels, so what is left is caused by the
            acceleration method.

  vs_gt     the ground-truth video.  Already produced at inference time into each episode's
            metrics.json, but re-computed here so that (a) both references come from one
            code path with one frame-matching rule, and (b) FVD exists for it too.
            Interpretation differs: vs_gt mixes "the model cannot predict the future" with
            "the acceleration degraded it", which is why it cannot rank accelerations.

`dense` itself is scored vs_gt only -- its vs_dense would be self-comparison (PSNR = inf).

WHY THE FRAME MATCHING MATTERS
------------------------------
Pred and reference routinely differ in length (measured: num_frames_pred=165 vs
num_frames_gt=167 on ctrlworld bimanual) because the rollout stops on a chunk boundary.
Comparing frame i to frame i after a length mismatch silently drifts the pairing, so both
clips are truncated to min(len) -- never resampled, which would interpolate motion that is
precisely what the dynamics metrics measure.

Clips whose spatial size differs are SKIPPED, not resized: a re-render at another
resolution is a different measurement (the same reason table_dense_nr.comparable_pair
exists), and bilinear-resizing one side would fabricate a sharpness difference.

FVD IS A SET STATISTIC, NOT A PER-CLIP ONE
------------------------------------------
FVD compares two *distributions* of I3D features, so it is undefined for a single clip.
This script therefore emits per-clip PSNR/SSIM/LPIPS plus the I3D feature vector of every
clip; `--fvd-only` then aggregates features into one FVD per
(embodiment, view, model, split, method, reference) cell. A cell with fewer than
FVD_MIN_CLIPS clips is reported as null rather than as a number -- with ~50 clips the
covariance of a 400-d feature is severely rank-deficient and the resulting FVD is dominated
by estimator bias, not by video quality.

I3D weights: checkpoints/fvd/i3d_torchscript.pt (the StyleGAN-V torchscript export, the
de-facto standard for FVD; 400-d pre-logit features).

Usage:
  # per-clip pass (GPU, shardable exactly like run_dense_nr.py)
  python scripts/compute_dr_metrics.py --out <run> --inventory-only
  python scripts/compute_dr_metrics.py --out <run> --workers 8 --machines 14 --machine 0
  # aggregate FVD once every shard has finished
  python scripts/compute_dr_metrics.py --out <run> --fvd-only
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PROTOCOL = 'dr-v1'
I3D_CKPT = REPO / 'checkpoints/fvd/i3d_torchscript.pt'
# 16 frames is the I3D clip length the FVD reference implementations use; a video is cut
# into consecutive 16-frame windows and every window contributes one feature vector.
I3D_WINDOW = 16
I3D_SIZE = 224
FVD_MIN_CLIPS = 64          # below this the 400x400 covariance is too ill-conditioned


class CorruptVideo(Exception):
    """Source mp4 that cannot be decoded -- a data defect, not a transient fault."""


# --------------------------------------------------------------------------- #
# inventory
# --------------------------------------------------------------------------- #
def dense_twin(path: Path) -> Path | None:
    """The dense counterpart of an accelerated clip: same path, `dense` tree.

    sampling_dataset/<method>/<rest> -> sampling_dataset/dense/<rest>. Relies on every
    method tree mirroring dense's sub-path, which is the sampling-dataset-layout rule the
    run scripts already enforce.
    """
    parts = list(path.parts)
    try:
        i = parts.index('sampling_dataset')
    except ValueError:
        return None
    parts[i + 1] = 'dense'
    twin = Path(*parts)
    return twin if twin.exists() else None


def gt_for(pred: Path) -> Path | None:
    """The ground-truth video matching a prediction's framing, or None if never rendered.

    Order matters -- the reference must have the SAME layout as the prediction:

      pred_all_views.mp4  ->  gt_all_views.mp4 beside it. Verified bit-identical to
                              input/full_gt.mp4 (PSNR 138 dB), and unlike full_gt it is
                              guaranteed to share the grid framing.
      full_pred.mp4       ->  full_gt.mp4 beside it, else the mirrored input/ tree. Only
                              788 of 2,657 dense predictions have the sibling, which is why
                              the mirrored path is needed: Ctrl-World keeps GT under input/.
      <episode>.mp4       ->  <episode>_gt.mp4, else the same name under input/.

    A DreamDojo full_pred.mp4 frequently has no GT anywhere -- a rendering gap, so the clip
    gets no vs_gt row (it still gets vs_dense).
    """
    if pred.name == 'pred_all_views.mp4':
        sib = pred.with_name('gt_all_views.mp4')
        return sib if sib.exists() else None
    if pred.name == 'full_pred.mp4':
        sib = pred.with_name('full_gt.mp4')
        if sib.exists():
            return sib
        if '/output/' in str(pred):
            mirrored = Path(str(pred).replace('/output/', '/input/')
                            .replace('full_pred.mp4', 'full_gt.mp4'))
            if mirrored.exists():
                return mirrored
        return None
    for cand in (pred.parent / f'{pred.stem}_gt.mp4',
                 Path(str(pred).replace('/output/', '/input/'))):
        if cand.exists() and cand.resolve() != pred.resolve():
            return cand
    return None


def inventory(methods, embodiments=None):
    """One row per (clip, reference), using the PURE prediction and PURE reference files.

    Several mp4s sit in one episode dir and picking the wrong one silently corrupts the
    metric. Measured on this dataset:

      Ctrl-World (both multi-view layouts)
        full_pred.mp4       384x960  -- GT stacked ON TOP of the prediction. Verified:
                                        PSNR(top half, input/full_gt.mp4) = 38.4 dB while
                                        PSNR(bottom half, GT) = 23.7 dB. Using this file
                                        makes HALF the pixels identical between any two
                                        methods by construction, which inflates every
                                        vs_dense number. NEVER score it.
        pred_all_views.mp4  192x960  -- the pure prediction grid          <- score this
        gt_all_views.mp4    192x960  -- pure GT, bit-identical to
                                        input/full_gt.mp4 (PSNR 138 dB)   <- reference
      DreamGen multi-view
        full_pred.mp4       432x768  -- already pure (PSNR vs its full_gt = 20.7 dB, and
                                        it is the same height as the GT, not double)
        full_merged.mp4     432x1536 -- the side-by-side comparison, never a subject
      DreamDojo single-view
        full_pred.mp4       480x640  -- pure; GT often absent entirely

    So the rule is: `pred_all_views.mp4` WINS when present (and then full_pred.mp4 is the
    stacked merge); otherwise full_pred.mp4 / <episode>.mp4 is the prediction.

    Also excluded: seg_*.mp4 (the autoregressive segments that full_pred is the
    concatenation of -- counting them scores the same frames again and their FVD windows
    would swamp the cell) and gt_/merged files. Without these filters the first inventory
    came out 2.5x too large (25,656 full_pred vs 8,032 pred_all_views vs 20,898 seg_*).
    """
    DERIVED = ('full_gt.mp4', 'full_merged.mp4', 'gt_all_views.mp4')
    rows = []
    root = REPO / 'sampling_dataset'
    for method in methods:
        base = root / method
        if not base.is_dir():
            continue
        for pred in base.rglob('*.mp4'):
            name = pred.name
            if name in DERIVED or name.startswith('seg_'):
                continue
            if '/input/' in str(pred) or '/_fps_compare/' in str(pred):
                continue
            stacked_twin = pred.parent / 'pred_all_views.mp4'
            if name == 'full_pred.mp4' and stacked_twin.exists():
                continue                      # full_pred is the GT-stacked merge here
            if name not in ('full_pred.mp4', 'pred_all_views.mp4') and \
                    (pred.parent / 'full_pred.mp4').exists():
                continue                      # a flat-named extra beside a canonical one
            rel = pred.relative_to(base)
            emb = rel.parts[0] if rel.parts else '?'
            if embodiments and emb not in embodiments:
                continue
            refs = {}
            if method != 'dense':
                t = dense_twin(pred)
                if t:
                    refs['vs_dense'] = str(t)
            g = gt_for(pred)
            if g:
                refs['vs_gt'] = str(g)
            for ref_kind, ref_path in refs.items():
                ident = f'{PROTOCOL}|{ref_kind}|{pred}'
                rows.append(dict(
                    id=hashlib.sha256(ident.encode()).hexdigest()[:20],
                    pred=str(pred), ref=ref_path, ref_kind=ref_kind,
                    method=method, embodiment=emb,
                    relative=str(rel),
                    view_type=('multiview' if '/multiview/' in str(rel)
                               else 'singleview' if '/singleview/' in str(rel) else '?'),
                    split=('makovian' if '/makovian/' in str(rel)
                           else 'non_makovian' if '/non_makovian/' in str(rel) else '?'),
                ))
    rows.sort(key=lambda r: r['id'])
    return rows


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
_MODELS = {}


def get_models(device):
    if _MODELS:
        return _MODELS
    import torch
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    _MODELS['lpips'] = LearnedPerceptualImagePatchSimilarity(
        net_type='vgg', normalize=False).to(device).eval()
    _MODELS['i3d'] = torch.jit.load(str(I3D_CKPT)).to(device).eval()
    _MODELS['device'] = device
    return _MODELS


def read_video(path: str) -> np.ndarray:
    """(T,H,W,3) uint8. cv2 rather than imageio: it is already a dependency of the NR
    pipeline and does not need a plugin per container.

    Raises CorruptVideo for a file cv2 cannot decode -- 13 mp4s in this dataset are
    truncated (`moov atom not found`, 0-48 bytes or an exact 2 MB buffer multiple). That is
    a source-data defect, so the caller records it as a skip rather than burning four retry
    attempts on something that cannot succeed.
    """
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise CorruptVideo(f'cannot open {path}')
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not out:
        raise CorruptVideo(f'no frames in {path}')
    return np.stack(out)


def score(row, device):
    import torch
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim_f
    m = get_models(device)
    a = read_video(row['pred'])
    b = read_video(row['ref'])
    if a.shape[1:] != b.shape[1:]:
        return dict(skipped='shape_mismatch',
                    pred_shape=list(a.shape), ref_shape=list(b.shape))
    n_pred, n_ref = len(a), len(b)
    # Truncate, never resample: interpolating frames would invent motion. Keep the
    # original lengths so a reader can see how far apart the two rollouts ended.
    n = min(n_pred, n_ref)
    a, b = a[:n], b[:n]

    # Everything is computed in frame CHUNKS rather than on the whole clip at once.
    # Two reasons, both measured:
    #  * a 800-frame 480x640 clip is 737M elements, and a single tensor op over it trips
    #    `RuntimeError: Expected canUse32BitIndexMath(input) && canUse32BitIndexMath(
    #    output) to be true` -- torch indexes with int32 on several CUDA kernels;
    #  * it also needs ~3 GB of fp32 per clip per side, which with 8 workers per GPU would
    #    OOM on its own.
    CH = 16
    mses, ssim_acc, lp_acc = [], 0.0, 0.0
    feats = {'pred': [], 'ref': []}
    win_pred, win_ref = [], []
    with torch.no_grad():
        for s in range(0, n, CH):
            ca = torch.from_numpy(a[s:s + CH]).to(device).permute(0, 3, 1, 2).float() / 255
            cb = torch.from_numpy(b[s:s + CH]).to(device).permute(0, 3, 1, 2).float() / 255
            k = len(ca)
            # per-frame MSE -> per-frame PSNR -> mean, matching
            # metrics/video_metrics.calculate_psnr. A single global MSE would let one
            # blown-out frame dominate the clip.
            mses.append(((ca - cb) ** 2).mean(dim=(1, 2, 3)).cpu())
            ssim_acc += float(ssim_f(ca, cb, data_range=1.0)) * k
            lp_acc += float(m['lpips'](ca * 2 - 1, cb * 2 - 1)) * k
            win_pred.append(ca)
            win_ref.append(cb)
            # I3D consumes consecutive I3D_WINDOW-frame windows; CH is a multiple of it so
            # a window never straddles a chunk boundary.
            for tag, buf in (('pred', win_pred), ('ref', win_ref)):
                vid = torch.cat(buf) if len(buf) > 1 else buf[0]
                while len(vid) >= I3D_WINDOW:
                    w = vid[:I3D_WINDOW]
                    vid = vid[I3D_WINDOW:]
                    w = torch.nn.functional.interpolate(
                        w, size=(I3D_SIZE, I3D_SIZE), mode='bilinear',
                        align_corners=False)
                    w = (w * 2 - 1).permute(1, 0, 2, 3).unsqueeze(0)   # 1,3,T,H,W
                    feats[tag].append(m['i3d'](w, rescale=False, resize=False,
                                               return_features=True)
                                      .squeeze(0).cpu().numpy().tolist())
                buf.clear()
                if len(vid):
                    buf.append(vid)
            del ca, cb

    mse = torch.cat(mses)
    psnr = float((10 * torch.log10(1.0 / mse.clamp_min(1e-12))).mean())
    return dict(psnr=psnr, ssim=ssim_acc / n, lpips=lp_acc / n, n_frames=int(n),
                n_frames_pred=int(n_pred), n_frames_ref=int(n_ref),
                height=int(a.shape[1]), width=int(a.shape[2]),
                i3d_pred=feats['pred'], i3d_ref=feats['ref'])


def dump(path: Path, obj):
    """Atomic, hostname+pid-unique temp name so shards on a shared FS cannot interleave."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{socket.gethostname()}.{os.getpid()}.tmp')
    try:
        tmp.write_text(json.dumps(obj))
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)   # never leave a 0-byte stub counted as a result
        raise


# --------------------------------------------------------------------------- #
# FVD aggregation
# --------------------------------------------------------------------------- #
def frechet(x: np.ndarray, y: np.ndarray) -> float:
    from scipy import linalg
    mx, my = x.mean(0), y.mean(0)
    cx = np.cov(x, rowvar=False)
    cy = np.cov(y, rowvar=False)
    covmean, _ = linalg.sqrtm(cx.dot(cy), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mx - my) ** 2).sum() + np.trace(cx) + np.trace(cy)
                 - 2 * np.trace(covmean))


def aggregate_fvd(out: Path):
    import collections
    cells = collections.defaultdict(lambda: {'pred': [], 'ref': []})
    n_rec = 0
    with os.scandir(out / 'results') as it:
        for e in it:
            if not e.name.endswith('.json'):
                continue
            try:
                r = json.loads(Path(e.path).read_text())
            except Exception:
                continue
            if r.get('skipped') or not r.get('i3d_pred'):
                continue
            n_rec += 1
            key = (r['embodiment'], r['view_type'], r['split'],
                   r['method'], r['ref_kind'])
            cells[key]['pred'] += r['i3d_pred']
            cells[key]['ref'] += r['i3d_ref']

    rows = []
    for key, d in sorted(cells.items()):
        p, q = np.asarray(d['pred']), np.asarray(d['ref'])
        # Guard the estimator: a 400-d covariance from a handful of windows is
        # rank-deficient and its FVD reflects that, not the videos.
        ok = len(p) >= FVD_MIN_CLIPS and len(q) >= FVD_MIN_CLIPS
        rows.append(dict(embodiment=key[0], view_type=key[1], split=key[2],
                         method=key[3], ref_kind=key[4],
                         n_windows_pred=len(p), n_windows_ref=len(q),
                         fvd=(frechet(p, q) if ok else None),
                         reason=None if ok else f'<{FVD_MIN_CLIPS} windows'))
    dump(out / 'fvd.json', dict(protocol=PROTOCOL, min_clips=FVD_MIN_CLIPS,
                                records_used=n_rec, cells=rows))
    done = [r for r in rows if r['fvd'] is not None]
    print(f'[fvd] {len(rows)} cells, {len(done)} with enough windows '
          f'({n_rec} records) -> {out / "fvd.json"}')
    for r in sorted(done, key=lambda r: r['fvd'])[:12]:
        print(f'  {r["embodiment"]:11s} {r["view_type"]:10s} {r["split"]:13s} '
              f'{r["method"]:11s} {r["ref_kind"]:8s} FVD={r["fvd"]:9.2f} '
              f'(n={r["n_windows_pred"]})')


# --------------------------------------------------------------------------- #
def worker(args):
    import torch
    rows = json.loads((args.out / 'manifest.json').read_text())
    if args.only_ids:
        keep = set(json.loads(Path(args.only_ids).read_text()))
        rows = [r for r in rows if r['id'] in keep]
    stride = args.workers * args.machines
    offset = args.machine * args.workers + args.shard
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for row in rows[offset::stride]:
        dest = args.out / 'results' / f'{row["id"]}.json'
        if dest.exists():
            continue
        if not (os.path.exists(row['pred']) and os.path.exists(row['ref'])):
            print(f'SKIP-GONE {row["relative"]}', flush=True)
            continue
        t0 = time.time()
        print(f'START {row["ref_kind"]} {row["relative"]}', flush=True)
        for attempt in range(4):
            try:
                rec = dict(row)
                rec.update(score(row, device))
                rec['protocol'] = PROTOCOL
                rec['elapsed_s'] = time.time() - t0
                dump(dest, rec)
                print(f'DONE {row["id"]} psnr={rec.get("psnr")}', flush=True)
                break
            except CorruptVideo as exc:
                # Retrying cannot fix a truncated mp4. Record it AS A RESULT with a
                # `skipped` reason, not in errors/: that way the id is done, the shard
                # moves on, and the reason is visible in the aggregate instead of being
                # indistinguishable from a real failure.
                dump(dest, dict(row, protocol=PROTOCOL, skipped='corrupt_video',
                                detail=str(exc)))
                print(f'SKIP-CORRUPT {row["id"]}', flush=True)
                break
            except (torch.OutOfMemoryError, OSError) as exc:
                quota = isinstance(exc, OSError) and exc.errno in (errno.EDQUOT,
                                                                   errno.ENOSPC)
                if isinstance(exc, OSError) and not quota:
                    dump(args.out / 'errors' / f'{row["id"]}.json',
                         dict(row, traceback=traceback.format_exc()))
                    break
                torch.cuda.empty_cache()
                if attempt == 3:
                    dump(args.out / 'errors' / f'{row["id"]}.json',
                         dict(row, traceback=traceback.format_exc()))
                else:
                    wait = (300 if quota else 60) * (attempt + 1)
                    print(f'RETRY {"QUOTA" if quota else "OOM"} {attempt+1}/3 in {wait}s',
                          flush=True)
                    time.sleep(wait)
            except Exception:
                try:
                    dump(args.out / 'errors' / f'{row["id"]}.json',
                         dict(row, traceback=traceback.format_exc()))
                except OSError:
                    pass
                traceback.print_exc()
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--methods', nargs='+', default=[
        'dense', 'dicache', 'worldcache', 'fastercache',
        'pisa', 'radial', 'svg1', 'svg2', 'sito', 'itm'])
    ap.add_argument('--embodiments', nargs='*', default=None)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--shard', type=int, default=-1)
    ap.add_argument('--machines', type=int, default=1)
    ap.add_argument('--machine', type=int, default=0)
    ap.add_argument('--only-ids', default=None)
    ap.add_argument('--inventory-only', action='store_true')
    ap.add_argument('--reuse-manifest', action='store_true')
    ap.add_argument('--fvd-only', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'results').mkdir(exist_ok=True)
    (args.out / 'errors').mkdir(exist_ok=True)

    if args.fvd_only:
        aggregate_fvd(args.out)
        return

    mf = args.out / 'manifest.json'
    if args.inventory_only or not mf.exists():
        if args.reuse_manifest and not mf.exists():
            raise SystemExit(f'no manifest at {mf}')
        rows = inventory(args.methods, args.embodiments)
        dump(mf, rows)
        import collections
        c = collections.Counter((r['method'], r['ref_kind']) for r in rows)
        print(f'[inventory] {len(rows)} (clip, reference) pairs -> {mf}')
        for k, v in sorted(c.items()):
            print(f'  {k[0]:12s} {k[1]:8s} {v}')
        if args.inventory_only:
            return

    if args.shard >= 0:
        worker(args)
        return

    procs, logs = [], []
    tag = f'm{args.machine}' if args.machines > 1 else 'fwd'
    for i in range(args.workers):
        log = (args.out / f'worker_{tag}_{i}.log').open('a')
        cmd = [sys.executable, '-u', __file__, '--out', str(args.out),
               '--workers', str(args.workers), '--shard', str(i),
               '--machines', str(args.machines), '--machine', str(args.machine),
               '--reuse-manifest']
        if args.only_ids:
            cmd += ['--only-ids', args.only_ids]
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
        logs.append(log)
    codes = [p.wait() for p in procs]
    for lg in logs:
        lg.close()
    dump(args.out / f'queue_finished_{tag}.json',
         dict(worker_exit_codes=codes, time=time.time()))
    if any(codes):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
