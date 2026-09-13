"""PhyWorld runner: PhyQual-NR, PhyDyn-NR, PhyFid-DR over sampling_dataset.

Intended env (see scripts/run_phyworld.sh):
  cwd = metrics/WorldScore
  PYTHONPATH includes WorldScore third_party + SEA-RAFT + VFIMamba
  python = models/dreamgen/.venv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from metrics.phyworld.schema import EMBODIMENT_PHRASES
from metrics.phyworld.scoring import (
    actor_dynamics_score,
    actor_motion_score,
    continuity_gap,
    entity_presence_score,
    kinetic_drift,
    look_drift,
    motion_continuity_score,
    phy_dyn,
    phy_fid,
    phy_qual,
    scene_integrity_score,
    task_lock_score,
)
sys.path.insert(0, str(_REPO / "scripts"))
from phyworld_index import walk  # noqa: E402

# --------------------------------------------------------------------------- #
# light helpers (no heavy imports)
# --------------------------------------------------------------------------- #


def load_video(path: str) -> np.ndarray:
    import imageio.v3 as iio
    return np.asarray(iio.imread(path, index=None))


def instruction_nouns(instruction: str, nlp) -> list[str]:
    if not instruction or nlp is None:
        return []
    stop = {"task", "robot", "arm", "gripper", "video", "view"}
    out = []
    for tok in nlp(instruction):
        if tok.pos_ in ("NOUN", "PROPN") and tok.text.lower() not in stop:
            w = tok.text.lower()
            if w not in out:
                out.append(w)
    return out[:4]


def merge_json(path: Path, payload: dict) -> dict:
    cur = {}
    if path.exists():
        try:
            cur = json.loads(path.read_text())
        except json.JSONDecodeError:
            cur = {}
    cur.update(payload)
    path.write_text(json.dumps(cur, indent=2))
    return cur


def write_frames(frames: np.ndarray, dest: Path, stride: int = 1) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    import cv2
    for i, fr in enumerate(frames):
        if i % stride:
            continue
        p = dest / f"{len(paths):05d}.jpg"
        cv2.imwrite(str(p), fr[:, :, ::-1])  # RGB -> BGR
        paths.append(str(p))
    return paths


# --------------------------------------------------------------------------- #
# tracks
# --------------------------------------------------------------------------- #


def score_qual(frames, instruction, embodiment, iqa, det, stride) -> dict:
    phrases = list(EMBODIMENT_PHRASES.get(embodiment, ["robot arm"]))
    phrases += instruction_nouns(instruction, getattr(det, "nlp", None))
    if not phrases:
        phrases = ["robot arm"]

    tmp = Path(tempfile.mkdtemp(prefix="phyworld_qual_"))
    try:
        paths = write_frames(frames, tmp, stride=stride)
        iqa_l, aes_l, cs_l, od_l = [], [], [], []
        for p in paths:
            a, b, c = iqa.frame_scores(p, instruction or "a robot performing a task")
            iqa_l.append(a); aes_l.append(b); cs_l.append(c)
            r, _ = det.rate(p, phrases)
            od_l.append(r)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    si = scene_integrity_score(float(np.mean(iqa_l)))
    tl = task_lock_score(float(np.mean(cs_l)))
    ep = entity_presence_score(float(np.mean(od_l)))
    return {
        "scene_integrity": si,
        "task_lock": tl,
        "entity_presence": ep,
        "look_score": float(np.mean(aes_l)),
        "phy_qual": phy_qual(si, tl, ep),
        "qual_n_frames": len(iqa_l),
        "qual_stride": stride,
        "entity_phrases": phrases,
        "clip_iqa+_raw": float(np.mean(iqa_l)),
        "clip_score_raw": float(np.mean(cs_l)),
        "detection_rate_raw": float(np.mean(od_l)),
    }


def score_dyn_kinetic_continuity(frames_dir_paths, flow, vfi, InputPadder, device,
                                 do_smooth, smooth_stride) -> dict:
    from metrics.dynamics.motion_mag_smooth_probe import motion_magnitude, motion_smoothness
    mag, _ = motion_magnitude(frames_dir_paths, flow)
    out = {"kinetic_budget": float(mag)}
    if do_smooth and vfi is not None:
        sm = motion_smoothness(
            frames_dir_paths, vfi, InputPadder, device,
            TTA=False, stride=smooth_stride,
        )
        out["motion_continuity"] = motion_continuity_score(sm["ssim"])
        out["motion_continuity_ssim"] = sm["ssim"]
        out["motion_continuity_mse"] = sm["mse"]
        out["motion_continuity_n"] = sm["n"]
    return out


def score_dyn_actor(video_path, embodiment, instruction, grounder, img_pred, vid_pred,
                    flow, out_dir, stride) -> dict:
    from metrics.dynamics.motion_accuracy_gdino import probe
    obj = " . ".join(instruction_nouns(instruction, None) or ["object"]) + " ."
    # prefer Detector-quality nouns if we have them later; keep prompt short
    if instruction:
        obj = instruction
    emb = " . ".join(EMBODIMENT_PHRASES.get(embodiment, ["robot arm"])) + " ."
    tag = "pred"
    r = probe(
        video_path, tag, str(out_dir), grounder, img_pred, vid_pred, flow,
        obj, emb, "cuda", max_frames=None, stride=stride,
    )
    raw = float(r.get("med_fg_vs_bg", float("nan")))
    am_raw = float(r.get("mean_fg_flow_med", float("nan")))
    return {
        "actor_dynamics_raw": raw,
        "actor_dynamics": actor_dynamics_score(raw) if raw == raw else float("nan"),
        # ActorMotion: ego-compensated subject-only flow — catches smooth-but-inactive
        # (Continuity saturates near 99 regardless; this doesn't). Report raw beside
        # PhyDyn, never folded into its mean.
        "actor_motion_raw": am_raw,
        "actor_motion": actor_motion_score(am_raw) if am_raw == am_raw else float("nan"),
        "actor_motion_active_rate": r.get("actor_motion_active_rate"),
        "actor_motion_active_threshold_px": r.get("actor_motion_active_threshold_px"),
        "actor_motion_uncorrected_raw": r.get("mean_fg_flow_med_uncorrected"),
        "actor_camera_ego_dx": r.get("mean_bg_ego_dx"),
        "actor_camera_ego_dy": r.get("mean_bg_ego_dy"),
        "actor_fg_cov_pct": r.get("mean_fg_cov_pct"),
        "actor_n_object_dets": r.get("n_object_dets"),
        "actor_n_arm_dets": r.get("n_arm_dets"),
    }


def score_fid_epe(pred_frames, dense_frames, flow_model, max_pairs) -> dict:
    from metrics.dynamics.physics_metrics import flow_vs_ref_metrics
    from metrics.dynamics.flow_backend import SeaRaftFlow
    # flow_model may be SeaRaftFlow (.flow) or unused — construct if needed
    if flow_model is None:
        flow_model = SeaRaftFlow(device="cuda")
    T = min(pred_frames.shape[0], dense_frames.shape[0])
    m = flow_vs_ref_metrics(
        pred_frames[:T], dense_frames[:T], flow_model,
        max_pairs=max_pairs, ref_name="dense",
    )
    return {
        "motion_fidelity": m["flow_epe"],
        "kinetic_budget_vs_dense_ratio": m["motion_mag_ratio"],
        "fid_num_pairs": m["num_pairs"],
    }


# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=str(_REPO / "sampling_dataset"))
    ap.add_argument("--track", default="all", choices=["qual", "dyn", "fid", "all"])
    ap.add_argument("--methods", nargs="+", default=["dense", "worldcache"])
    ap.add_argument("--embodiment", default="single_arm")
    ap.add_argument("--view", default="multiview")
    ap.add_argument("--model", default="ctrlworld")
    ap.add_argument("--split", default="makovian")
    ap.add_argument("--max-episodes", type=int, default=4)
    ap.add_argument("--episodes", nargs="*", default=None)
    ap.add_argument("--matched-only", action="store_true")
    ap.add_argument("--qual-stride", type=int, default=10)
    ap.add_argument("--smooth-stride", type=int, default=8)
    ap.add_argument("--actor-stride", type=int, default=2)
    ap.add_argument("--fid-max-pairs", type=int, default=None)
    ap.add_argument("--skip-actor", action="store_true")
    ap.add_argument("--skip-continuity", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--gdino-cfg", default="worldscore/benchmark/metrics/third_party/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    ap.add_argument("--gdino-ckpt", default=str(_REPO / "checkpoints/worldscore/groundingdino_swint_ogc.pth"))
    ap.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    ap.add_argument("--sam2-ckpt", default=str(_REPO / "checkpoints/worldscore/sam2.1_hiera_base_plus.pt"))
    ap.add_argument("--raft-cfg", default="worldscore/benchmark/metrics/third_party/SEA-RAFT/config/eval/spring-M.json")
    ap.add_argument("--raft-ckpt", default=str(_REPO / "checkpoints/worldscore/Tartan-C-T-TSKH-spring540x960-M.pth"))
    args = ap.parse_args()

    do_qual = args.track in ("qual", "all")
    do_dyn = args.track in ("dyn", "all")
    do_fid = args.track in ("fid", "all")

    ds = Path(args.dataset_root)
    rows = list(walk(
        ds, methods=args.methods, embodiment=args.embodiment,
        view=args.view, model=args.model, split=args.split,
    ))

    # matched episodes present for every requested method
    by_ep = defaultdict(dict)
    for r in rows:
        by_ep[r["episode"]][r["method"]] = r
    wanted = args.methods
    episodes = sorted(by_ep)
    if args.matched_only:
        episodes = [e for e in episodes if all(m in by_ep[e] for m in wanted)]
    if args.episodes:
        extra = [e if e.startswith("episode_") else f"episode_{e}" for e in args.episodes]
        rest = [e for e in episodes if e not in extra]
        episodes = [e for e in extra if e in by_ep] + rest
    episodes = episodes[: args.max_episodes]
    print(f"[phyworld] {len(episodes)} episodes × {wanted}: {episodes}", flush=True)

    device = "cuda"
    iqa = det = None
    raft_flow = vfi = InputPadder = None
    grounder = img_pred = vid_pred = None
    sea_raft = None

    if do_qual:
        from metrics.quality.quality_control_probe import IQA, Detector
        iqa = IQA(device)
        det = Detector(args.gdino_cfg, args.gdino_ckpt, device)

    if do_dyn:
        from metrics.dynamics.motion_mag_smooth_probe import Flow, build_vfimamba
        raft_flow = Flow(args.raft_cfg, args.raft_ckpt, device)
        # spring-M cfg uses scale=-1 (540x960); robot views are small and underflow
        # the 4-level corr pyramid. Full-res (scale=0) matches SeaRaftFlow.
        raft_flow.a.scale = 0
        if not args.skip_continuity:
            vfi, InputPadder = build_vfimamba(device)
        if not args.skip_actor:
            from metrics.dynamics.motion_accuracy_gdino import Grounder
            from sam2.build_sam import build_sam2, build_sam2_video_predictor
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            import hydra
            from hydra.core.global_hydra import GlobalHydra
            GlobalHydra.instance().clear()
            hydra.initialize_config_module("sam2", version_base="1.2")
            img_model = build_sam2(args.sam2_cfg, args.sam2_ckpt, device=device, apply_postprocessing=False)
            img_pred = SAM2ImagePredictor(img_model)
            vid_pred = build_sam2_video_predictor(args.sam2_cfg, args.sam2_ckpt, device=device, apply_postprocessing=False)
            grounder = Grounder(args.gdino_cfg, args.gdino_ckpt, device)

    if do_fid:
        from metrics.dynamics.flow_backend import SeaRaftFlow
        sea_raft = SeaRaftFlow(device=device)

    stored = {}  # (method, episode) -> payload

    # dense first so Fid can read it
    method_order = [m for m in wanted if m == "dense"] + [m for m in wanted if m != "dense"]

    for method in method_order:
        for ep in episodes:
            rec = by_ep[ep].get(method)
            if rec is None:
                print(f"[skip] {method} {ep} missing", flush=True)
                continue
            ep_dir = Path(rec["episode_dir"])
            out_path = ep_dir / "phyworld.json"
            if out_path.exists() and not args.overwrite:
                payload = json.loads(out_path.read_text())
                stored[(method, ep)] = payload
                print(f"[reuse] {method} {ep}", flush=True)
                continue

            print(f"=== {method} {ep} ===", flush=True)
            frames = load_video(rec["pred_path"])
            payload = {
                "schema": "phyworld-v0",
                "method": method,
                "episode": ep,
                "embodiment": args.embodiment,
                "view": args.view,
                "model": args.model,
                "split": args.split,
                "pred_path": rec["pred_path"],
                "instruction": rec.get("instruction") or "",
            }

            if do_qual:
                payload.update(score_qual(
                    frames, payload["instruction"], args.embodiment,
                    iqa, det, args.qual_stride,
                ))
                print(f"  PhyQual={payload['phy_qual']:.1f} "
                      f"SI={payload['scene_integrity']:.1f} "
                      f"TL={payload['task_lock']:.1f} "
                      f"EP={payload['entity_presence']:.1f}", flush=True)

            if do_dyn:
                tmp = Path(tempfile.mkdtemp(prefix="phyworld_dyn_"))
                try:
                    paths = write_frames(frames, tmp, stride=1)
                    payload.update(score_dyn_kinetic_continuity(
                        paths, raft_flow, vfi, InputPadder, device,
                        do_smooth=not args.skip_continuity,
                        smooth_stride=args.smooth_stride,
                    ))
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
                if not args.skip_actor and grounder is not None:
                    actor_out = ep_dir / "_phyworld_actor"
                    actor_out.mkdir(exist_ok=True)
                    payload.update(score_dyn_actor(
                        rec["pred_path"], args.embodiment, payload["instruction"],
                        grounder, img_pred, vid_pred, raft_flow, actor_out,
                        args.actor_stride,
                    ))
                if "actor_dynamics" in payload and "motion_continuity" in payload:
                    payload["phy_dyn"] = phy_dyn(
                        payload["actor_dynamics"], payload["motion_continuity"],
                    )
                print(f"  KineticBudget={payload.get('kinetic_budget', float('nan')):.3f} "
                      f"ActorMotion={payload.get('actor_motion_raw', float('nan')):.3f} "
                      f"Active={payload.get('actor_motion_active_rate', float('nan')):.2f} "
                      f"Actor={payload.get('actor_dynamics', float('nan')):.1f} "
                      f"Cont={payload.get('motion_continuity', float('nan')):.1f} "
                      f"PhyDyn={payload.get('phy_dyn', float('nan')):.1f}", flush=True)

            stored[(method, ep)] = payload
            merge_json(out_path, payload)

    # PhyFid-DR: speedup vs stored dense
    if do_fid:
        for method in method_order:
            if method == "dense":
                continue
            for ep in episodes:
                rec = by_ep[ep].get(method)
                drec = by_ep[ep].get("dense")
                if rec is None or drec is None:
                    continue
                pred = stored.get((method, ep), {})
                dens = stored.get(("dense", ep), {})
                print(f"=== fid {method} {ep} ===", flush=True)
                pred_fr = load_video(rec["pred_path"])
                den_fr = load_video(drec["pred_path"])
                fid = score_fid_epe(pred_fr, den_fr, sea_raft, args.fid_max_pairs)
                kd = kinetic_drift(
                    pred.get("kinetic_budget", float("nan")),
                    dens.get("kinetic_budget", float("nan")),
                )
                # ActorMotionDrift: same |log ratio| formula, subject-only magnitude.
                # Diagnostic beside PhyFid, not folded into its mean (see schema.py).
                amd = kinetic_drift(
                    pred.get("actor_motion_raw", float("nan")),
                    dens.get("actor_motion_raw", float("nan")),
                )
                cg = continuity_gap(
                    dens.get("motion_continuity_ssim", float("nan")),
                    pred.get("motion_continuity_ssim", float("nan")),
                )
                ld = look_drift(
                    pred.get("scene_integrity", float("nan")),
                    dens.get("scene_integrity", float("nan")),
                )
                fid.update({
                    "kinetic_drift": kd,
                    "actor_motion_drift": amd if amd == amd else None,
                    "continuity_gap": cg if cg == cg else None,
                    "look_drift": ld if ld == ld else None,
                })
                # self-check: dense vs dense EPE should be ~0 if we ever run it
                fid["phy_fid"] = phy_fid(
                    fid["motion_fidelity"],
                    kd if kd == kd else 0.0,
                    cg if cg == cg else 0.0,
                    ld if ld == ld else 0.0,
                )
                pred.update(fid)
                stored[(method, ep)] = pred
                merge_json(Path(rec["episode_dir"]) / "phyworld.json", pred)
                print(f"  MotionFid={fid['motion_fidelity']:.3f} "
                      f"KinDrift={kd:.3f} ActorMotionDrift={amd:.3f} "
                      f"PhyFid={fid['phy_fid']:.1f}", flush=True)

    # summaries
    def _mean(method, key):
        vals = [stored[(method, e)][key] for e in episodes
                if (method, e) in stored and key in stored[(method, e)]
                and stored[(method, e)][key] is not None
                and stored[(method, e)][key] == stored[(method, e)][key]]
        return float(np.mean(vals)) if vals else None

    tables = {"cell": {
        "embodiment": args.embodiment, "view": args.view,
        "model": args.model, "split": args.split,
        "episodes": episodes, "methods": method_order,
    }}
    for method in method_order:
        tables[method] = {
            k: _mean(method, k)
            for k in (
                "phy_qual", "scene_integrity", "task_lock", "entity_presence",
                "kinetic_budget", "actor_motion_raw", "actor_motion", "actor_motion_active_rate",
                "actor_motion_uncorrected_raw", "actor_camera_ego_dx", "actor_camera_ego_dy",
                "actor_dynamics", "motion_continuity", "phy_dyn",
                "motion_fidelity", "kinetic_drift", "actor_motion_drift",
                "continuity_gap", "look_drift", "phy_fid",
            )
        }
    out_sum = ds / "phyworld_tables.json"
    merge_json(out_sum, {f"{args.embodiment}/{args.view}/{args.model}/{args.split}": tables})
    cell_sum = (
        ds / method_order[0] / args.embodiment / "output" / args.view
        / args.model / args.split / "phyworld_summary.json"
    )
    # write a copy next to the first method's split as well as the combined file
    cell_sum.parent.mkdir(parents=True, exist_ok=True)
    cell_sum.write_text(json.dumps(tables, indent=2))
    print("\n===== PhyWorld summary =====")
    print(json.dumps(tables, indent=2))
    print(f"wrote {out_sum} and {cell_sum}")


if __name__ == "__main__":
    main()
