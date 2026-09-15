"""Resumable dense NR inventory and persistent GPU workers; no dense reference.

Run through run_dense_nr.sh for vendored dependency paths.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
KEYS = ['visual_integrity', 'text_condition_alignment', 'actor_motion_raw',
        'actor_motion_active_rate', 'active_motion_continuity',
        'actor_flicker_raw', 'camera_drift_raw']
TEXT_CONDITIONED_MODELS = {'dreamgen', 'ctrlworld', 'ctrlworld_4view_grid'}


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def inventory(out):
    from phyworld_index import _instruction_from_dir, _instruction_from_txt
    rows = []
    for p in sorted((ROOT / 'sampling_dataset/dense').rglob('*.mp4')):
        rel = p.relative_to(ROOT / 'sampling_dataset/dense')
        parts = rel.parts
        if len(parts) < 6 or parts[1] != 'output' or parts[4] not in ('makovian', 'non_makovian'):
            continue
        if p.name == 'full_pred.mp4':
            if (p.parent / 'pred_all_views.mp4').exists():
                continue
        elif p.name != 'pred_all_views.mp4' and not ('episode_' in p.stem and p.name.endswith('.mp4')):
            continue
        if '_phyworld' in str(p) or 'seg_' in p.name:
            continue
        # Ctrl-World full_pred is a comparison panel. Only use its prediction strip.
        if parts[3].startswith('ctrlworld') and p.name == 'full_pred.mp4':
            continue
        meta = {}
        if (p.parent / 'metrics.json').exists():
            meta = json.loads((p.parent / 'metrics.json').read_text())
        instruction = _instruction_from_dir(p.parent) if p.name in ('full_pred.mp4', 'pred_all_views.mp4') else _instruction_from_txt(p.with_suffix('.txt'))
        views = list(meta.get('per_view', {}))
        if parts[2] == 'singleview':
            views, layout = ['singleview'], 'single'
        elif p.name == 'pred_all_views.mp4':
            if not views:
                raise ValueError(f'Missing view layout: {p}')
            layout = 'strip'
        elif parts[3] == 'dreamgen':
            if parts[0] == 'single_arm':
                views, layout = ['exterior_1_left', 'exterior_2_left', 'wrist_left'], 'grid'
            else:
                views, layout = ['grid_top_left', 'grid_top_right', 'left_wrist', 'right_wrist'], 'grid'
        else:
            raise ValueError(f'Unknown layout: {p}')
        for vi, view in enumerate(views):
            identity = str(rel) + ':' + view
            rows.append(dict(id=hashlib.sha256(identity.encode()).hexdigest()[:20],
                             path=str(p), relative=str(rel), model=parts[3],
                             embodiment=parts[0], split=parts[4], view=view,
                             layout=layout, view_index=vi, n_views=len(views),
                             instruction=instruction, method='dense'))
    dump(out / 'manifest.json', rows)
    print(f'INVENTORY {len(rows)} view-clips / {len(set(r["path"] for r in rows))} videos', flush=True)
    return rows


def prepare(row, dest):
    import cv2
    cap = cv2.VideoCapture(row['path'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError('Invalid FPS')
    interval = max(1, round(fps / 5))
    frames, ids = [], []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        # Conservatively exclude first conditioning frame for every video.
        if i > 0 and (i - 1) % interval == 0:
            h, w = fr.shape[:2]
            vi = row['view_index']
            if row['layout'] == 'strip':
                fr = fr[:, vi*w//row['n_views']:(vi+1)*w//row['n_views']]
            elif row['layout'] == 'grid':
                fr = fr[(vi//2)*h//2:(vi//2+1)*h//2, (vi%2)*w//2:(vi%2+1)*w//2]
            # Fixed analysis grid. Report aspect-ratio resampling explicitly.
            fr = cv2.resize(fr, (256, 192), interpolation=cv2.INTER_AREA)
            frames.append(fr[:, :, ::-1].copy()); ids.append(i)
        i += 1
    cap.release()
    if len(frames) < 3:
        raise ValueError('Fewer than three post-conditioning frames')
    import numpy as np
    arr = np.stack(frames)
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*'mp4v'), fps/interval, (256,192))
    if not writer.isOpened():
        raise RuntimeError('Cannot create analysis video')
    for fr in frames:
        writer.write(fr[:, :, ::-1])
    writer.release()
    return arr, dict(source_fps=fps, effective_fps=fps/interval, source_interval=interval,
                     analysis_width=256, analysis_height=192, sampled_source_frames=ids)


def worker(args):
    import numpy as np
    import torch
    import cv2
    torch.set_num_threads(2); cv2.setNumThreads(1)
    from compute_phyworld import score_qual, write_frames
    from metrics.quality.quality_control_probe import IQA, Detector
    from metrics.dynamics.motion_mag_smooth_probe import Flow, build_vfimamba, motion_smoothness
    from metrics.dynamics.motion_accuracy_gdino import Grounder, probe
    from metrics.phyworld.schema import EMBODIMENT_PHRASES
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    ck = ROOT / 'checkpoints/worldscore'
    tp = 'worldscore/benchmark/metrics/third_party'
    cfg = f'{tp}/groundingdino/config/GroundingDINO_SwinT_OGC.py'
    gd = str(ck / 'groundingdino_swint_ogc.pth')
    iqa = IQA('cuda'); det = Detector(cfg, gd, 'cuda')
    flow = Flow(f'{tp}/SEA-RAFT/config/eval/spring-M.json', str(ck/'Tartan-C-T-TSKH-spring540x960-M.pth'), 'cuda')
    flow.a.scale = 0
    vfi, padder = build_vfimamba('cuda')
    GlobalHydra.instance().clear(); hydra.initialize_config_module('sam2', version_base='1.2')
    sc = 'configs/sam2.1/sam2.1_hiera_b+.yaml'
    img = SAM2ImagePredictor(build_sam2(sc, str(ck/'sam2.1_hiera_base_plus.pt'), device='cuda', apply_postprocessing=False))
    vp = build_sam2_video_predictor(sc, str(ck/'sam2.1_hiera_base_plus.pt'), device='cuda', apply_postprocessing=False)
    ground = Grounder(cfg, gd, 'cuda')
    rows = json.loads((args.out/'manifest.json').read_text())
    for row in rows[args.shard::args.workers]:
        dest = args.out/'results'/f'{row["id"]}.json'
        if dest.exists():
            continue
        started = time.time()
        print(f'START {row["relative"]} {row["view"]}', flush=True)
        try:
            with tempfile.TemporaryDirectory(prefix='dense_nr_') as td:
                tmp = Path(td)
                arr, protocol = prepare(row, tmp/'analysis.mp4')
                result = {**row, **protocol, 'protocol_version': 'dense-nr-v2-closed-world',
                          'actor_flicker_definition': 'experimental FB-filtered photometric residual, not validated paper reproduction'}
                result.update(score_qual(arr, row['instruction'], row['embodiment'], iqa, det, 5))
                # Text alignment and text-derived entity coverage are condition
                # metrics, not universal NR metrics. Do not score image-only
                # methods against a text instruction they never received.
                result['text_condition_applicable'] = row['model'] in TEXT_CONDITIONED_MODELS
                if not result['text_condition_applicable'] or not row['instruction']:
                    result['task_lock'] = None
                    result['entity_presence'] = None
                result['text_condition_alignment'] = result['task_lock']
                # Phrase-derived entity detection remains a diagnostic proxy until
                # a task-object ontology declares visibility expectations.
                result['condition_entity_coverage_proxy'] = result['entity_presence']
                result['entity_presence'] = None
                paths = write_frames(arr, tmp/'frames')
                smooth = motion_smoothness(paths, vfi, padder, 'cuda', TTA=False, stride=1)
                result['motion_continuity'] = float(smooth['ssim']*100)
                actor = probe(str(tmp/'analysis.mp4'), 'pred', str(tmp), ground, img, vp, flow,
                              row['instruction'] or 'object .',
                              ' . '.join(EMBODIMENT_PHRASES[row['embodiment']])+' .',
                              'cuda', stride=1,
                              actor_motion_active_threshold=0.25 * protocol['source_interval'],
                              compute_flicker=True)
                raw = actor.get('mean_fg_flow_med', float('nan'))
                result['actor_motion_raw'] = float(raw/protocol['source_interval']) if np.isfinite(raw) else None
                result['actor_motion_analysis_px_per_frame'] = float(raw) if np.isfinite(raw) else None
                result['actor_motion_px_per_second'] = float(raw*protocol['effective_fps']) if np.isfinite(raw) else None
                p95 = actor.get('mean_fg_flow_p95', float('nan'))
                camera = actor.get('camera_drift_raw', float('nan'))
                result['actor_motion_peak95_raw'] = (float(p95/protocol['source_interval'])
                                                     if np.isfinite(p95) else None)
                is_external_static = 'wrist' not in row['view'].lower()
                result['camera_drift_applicable'] = is_external_static
                result['camera_drift_raw'] = (float(camera/protocol['source_interval'])
                                               if is_external_static and np.isfinite(camera) else None)
                # VFI continuity is only meaningful when something moves. Align
                # triple (0,1,2) to actor pairs (0,1), then use clipped localized
                # motion as a weight. Frozen clips therefore cannot win by default.
                pair_motion = [p.get('fg_flow_ego_med') / protocol['source_interval']
                               if p.get('fg_flow_ego_med') is not None else None
                               for p in actor.get('per_frame', [])]
                triple_ssim = smooth.get('ssim_per_triple', [])
                weights = []
                for j in range(len(triple_ssim)):
                    values = [x for x in pair_motion[2*j:2*j+2] if x is not None]
                    weights.append(min(float(np.mean(values)) / 0.25, 1.0) if values else 0.0)
                denom = float(np.sum(weights))
                result['active_motion_continuity'] = (float(np.dot(weights, triple_ssim) / denom * 100)
                                                      if actor['n_arm_dets'] and denom > 0 else None)
                result['active_motion_continuity_weight_sum'] = denom
                result['actor_flicker_raw'] = actor.get('actor_flicker_raw') if actor['n_arm_dets'] else None
                result['actor_n_arm_dets'] = actor['n_arm_dets']
                result['actor_validity'] = 'robot_detected' if actor['n_arm_dets'] else ('object_only' if actor['n_object_dets'] else 'invalid')
                result['paper_scale'] = {
                    'visual_integrity': 'CLIP-IQA+ p10 x100; higher is better',
                    'text_condition_alignment': 'CLIPScore [0,100]; higher is better; text-conditioned only',
                    'actor_motion_raw': 'ego-compensated px/source-frame; descriptive, not higher-is-better',
                    'actor_motion_active_rate': 'percent active actor pairs at >=0.25 px/source-frame',
                    'active_motion_continuity': 'activity-weighted VFI SSIM x100; higher is better',
                    'actor_flicker_raw': 'flow-aligned RGB residual [0,1]; lower is better',
                    'camera_drift_raw': 'background-flow px/source-frame; lower is better; static external only',
                }
                result['actor_per_frame'] = actor.get('per_frame', [])
                result['elapsed_seconds'] = time.time()-started
                # Strict JSON and independent atomic files avoid summary races.
                def clean(v):
                    if isinstance(v, dict): return {k:clean(x) for k,x in v.items()}
                    if isinstance(v, list): return [clean(x) for x in v]
                    if isinstance(v, float) and not np.isfinite(v): return None
                    return v
                dump(dest, clean(result))
                print('DONE '+row['id']+' '+str({k:result.get(k) for k in KEYS}), flush=True)
        except Exception:
            dump(args.out/'errors'/f'{row["id"]}.json', {**row, 'traceback':traceback.format_exc()})
            traceback.print_exc()
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT/'results/dense_nr_v1')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--shard', type=int, default=-1)
    ap.add_argument('--inventory-only', action='store_true')
    args = ap.parse_args(); args.out = args.out.resolve(); args.out.mkdir(parents=True, exist_ok=True)
    if args.shard >= 0:
        worker(args); return
    if args.inventory_only or not (args.out/'manifest.json').exists(): inventory(args.out)
    if args.inventory_only: return
    processes = []
    for i in range(args.workers):
        log = (args.out/f'worker_{i}.log').open('a')
        p = subprocess.Popen([sys.executable, '-u', __file__, '--out', str(args.out),
                              '--workers', str(args.workers), '--shard', str(i)], stdout=log, stderr=subprocess.STDOUT)
        processes.append(p); log.close()
    codes = [p.wait() for p in processes]
    subprocess.run([sys.executable, str(ROOT/'scripts/plot_dense_nr.py'), '--run', str(args.out)], check=True)
    dump(args.out/'queue_finished.json', {'worker_exit_codes':codes, 'time':time.time()})
    if any(codes): raise SystemExit(1)


if __name__ == '__main__': main()
