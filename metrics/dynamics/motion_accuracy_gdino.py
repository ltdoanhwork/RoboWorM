"""motion_accuracy probe v2 — embodiment-aware.

Difference from v1 (motion_accuracy_probe.py): instead of seeding SAM2 with ONE
hand-drawn box around the object, we use **GroundingDINO** to auto-detect BOTH the
manipulated object AND the robot embodiment (arms/grippers) from the prompt, seed a
SAM2 track for each, and take the **foreground = union(object, arms)**.

Rationale: WorldScore's motion_accuracy = max(|flow| over object) - max(|flow| over
background) assumes "background = static scene". In a fixed-camera robot clip the
robot arms live in the background and move fast, breaking that assumption. Here the
arms are moved INTO the foreground, so background is (ideally) just the static table.

We report several variants so we can see which behaves sensibly:
  - ws_fg_vs_bg      = mean_t( max(fg) - max(bg) )          # WorldScore form, fg=arm+obj
  - ws_obj_vs_bg     = mean_t( max(obj) - max(bg) )         # original object-only
  - med_fg_vs_bg     = mean_t( median(fg) - median(bg) )    # outlier-robust
  - obj/arm/bg per-frame flow (median & max) for inspection

Prompt drives grounding: object phrases come from the task text; embodiment phrases
are a fixed robot set. Everything is knowable from the sample's prompt + embodiment.
"""
import os
import sys
import json
import argparse
import shutil

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# vendored deps (added to sys.path by launcher)
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import hydra
from hydra.core.global_hydra import GlobalHydra

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from core.raft import RAFT
from core.utils.utils import load_ckpt
from core.parser import parse_args


# --------------------------------------------------------------------------- #
def decode_video_to_frames(video_path, out_dir, max_frames=None, stride=1):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            cv2.imwrite(os.path.join(out_dir, f"{saved:05d}.jpg"), frame)
            saved += 1
            if max_frames is not None and saved >= max_frames:
                break
        idx += 1
    cap.release()
    return saved


# --------------------------------------------------------------------------- #
# GroundingDINO
# --------------------------------------------------------------------------- #
class Grounder:
    def __init__(self, cfg, ckpt, device):
        a = SLConfig.fromfile(cfg)
        a.device = device
        a.bert_base_uncased_path = None
        m = build_model(a)
        sd = torch.load(ckpt, map_location="cpu")
        m.load_state_dict(clean_state_dict(sd["model"]), strict=False)
        m.eval().to(device)
        self.model = m
        self.device = device
        self.tr = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def detect(self, frame_path, caption, box_thr=0.30, txt_thr=0.25):
        pil = Image.open(frame_path).convert("RGB")
        W, H = pil.size
        t, _ = self.tr(pil, None)
        t = t.to(self.device)
        cap = caption.lower().strip()
        if not cap.endswith("."):
            cap += " ."
        out = self.model(t[None], captions=[cap])
        logits = out["pred_logits"].cpu().sigmoid()[0]
        boxes = out["pred_boxes"].cpu()[0]
        keep = logits.max(dim=1)[0] > box_thr
        logits, boxes = logits[keep], boxes[keep]
        tok = self.model.tokenizer(cap)
        dets = []
        for lg, bx in zip(logits, boxes):
            phrase = get_phrases_from_posmap(lg > txt_thr, tok, self.model.tokenizer)
            cx, cy, w, h = bx.tolist()
            xyxy = [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
            dets.append({"phrase": phrase, "conf": float(lg.max()), "xyxy": xyxy})
        return dets, (H, W)


def nms(dets, iou_thr=0.7):
    """Greedy NMS by confidence to drop duplicate boxes for the same instance."""
    def iou(a, b):
        ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
        inter = iw * ih
        ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
        return inter / ua if ua > 0 else 0.0
    dets = sorted(dets, key=lambda d: -d["conf"])
    kept = []
    for d in dets:
        if all(iou(d["xyxy"], k["xyxy"]) < iou_thr for k in kept):
            kept.append(d)
    return kept


# --------------------------------------------------------------------------- #
# SAM2
# --------------------------------------------------------------------------- #
def expand_mask(mask, pixels=10):
    k = np.ones((pixels * 2 + 1, pixels * 2 + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), k, iterations=1).astype(bool)


def seed_masks(image_predictor, frame0, boxes):
    """One SAM2 image-predictor mask per box. Returns list of bool masks + (H,W)."""
    img = cv2.cvtColor(cv2.imread(frame0), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    image_predictor.set_image(img)
    masks = []
    for bx in boxes:
        m, _, _ = image_predictor.predict(
            point_coords=None, point_labels=None,
            box=np.array(bx, dtype=np.float32)[None, :], multimask_output=False,
        )
        masks.append(m[0].astype(bool))
    return masks, (H, W)


@torch.inference_mode()
def propagate_multi(video_predictor, frames_dir, seed_masks_by_id, score_thresh=0.0, frame_idx=0):
    """Propagate several seeded object ids; return {obj_id: [mask per frame]}.

    A seed after frame 0 is propagated forward and then backward so every frame has masks.
    """
    state = video_predictor.init_state(video_path=frames_dir, async_loading_frames=False)
    for oid, m in seed_masks_by_id.items():
        video_predictor.add_new_mask(inference_state=state, frame_idx=frame_idx, obj_id=oid, mask=m)
    per_frame = {}
    for reverse in ((False, True) if frame_idx > 0 else (False,)):
        for fidx, oids, logits in video_predictor.propagate_in_video(
                state, start_frame_idx=frame_idx, reverse=reverse):
            per_frame[fidx] = {
                oid: (logits[i] > score_thresh).cpu().numpy().squeeze()
                for i, oid in enumerate(oids)
            }
    n = len(per_frame)
    out = {oid: [] for oid in seed_masks_by_id}
    for fidx in sorted(per_frame):
        for oid in seed_masks_by_id:
            out[oid].append(expand_mask(per_frame[fidx][oid]))
    return out, n


# --------------------------------------------------------------------------- #
# SEA-RAFT
# --------------------------------------------------------------------------- #
class Flow:
    def __init__(self, cfg, ckpt, device):
        args = parse_args(argparse.Namespace(cfg=cfg, path=ckpt))
        model = RAFT(args)
        load_ckpt(model, ckpt)
        model.to(device).eval()
        self.model, self.args, self.device = model, args, device

    def _img(self, f):
        im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        return torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)[None].to(self.device)

    def compute(self, f1, f2):
        with torch.no_grad():
            i1, i2 = self._img(f1), self._img(f2)
            s = self.args.scale
            i1 = F.interpolate(i1, scale_factor=2 ** s, mode="bilinear", align_corners=False)
            i2 = F.interpolate(i2, scale_factor=2 ** s, mode="bilinear", align_corners=False)
            with torch.amp.autocast(device_type="cuda"):
                out = self.model(i1, i2, iters=self.args.iters, test_mode=True)
            fl = out["flow"][-1]
            fl = F.interpolate(fl, scale_factor=0.5 ** s, mode="bilinear", align_corners=False) * (0.5 ** s)
            return fl.detach().cpu().numpy().squeeze().transpose(1, 2, 0)


def mask_resize(m, hw):
    r = cv2.resize(m.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return r.astype(bool)


def ego_compensated_magnitude(flow, foreground):
    """Subject-flow magnitudes after robust global camera-motion removal.

    Background median flow estimates a global camera translation. It is near zero
    for fixed cameras and prevents a pan from being counted as robot motion. This
    intentionally does not model parallax; report wrist views separately.
    """
    background = ~foreground
    if not foreground.any() or not background.any():
        return None, None
    ego = np.median(flow[background], axis=0)
    residual = flow[foreground] - ego
    return np.sqrt((residual ** 2).sum(axis=-1)), ego


def overlay(frame_path, masks_by_color, out_path):
    base = cv2.imread(frame_path)
    for m, color in masks_by_color:
        if m.shape[:2] != base.shape[:2]:
            m = mask_resize(m, base.shape[:2])
        tint = np.zeros_like(base); tint[:] = color
        base = np.where(m[..., None], (0.5 * base + 0.5 * tint).astype(np.uint8), base)
    cv2.imwrite(out_path, base)


# --------------------------------------------------------------------------- #
def probe(video, tag, out_dir, grounder, img_pred, vid_pred, flow,
          object_prompt, embodiment_prompt, device, max_frames=None, stride=1,
          actor_motion_active_threshold=0.25, compute_flicker=False):
    work = os.path.join(out_dir, f"{tag}_frames")
    n = decode_video_to_frames(video, work, max_frames=max_frames, stride=stride)
    print(f"[{tag}] decoded {n} frames")

    # Arm confidence sits near the 0.30 box threshold, so a frame-0-only seed flips on
    # codec noise. Seed from the frame (of four) with the most confident arm detection.
    seed_idx, arm_dets, best = 0, [], -1.0
    for fi in sorted({0, n // 4, n // 2, 3 * n // 4}):
        dets, _ = grounder.detect(os.path.join(work, f"{fi:05d}.jpg"), embodiment_prompt)
        conf = max((d["conf"] for d in dets), default=-1.0)
        if conf > best:
            seed_idx, arm_dets, best = fi, dets, conf
    frame0 = os.path.join(work, f"{seed_idx:05d}.jpg")
    obj_dets, (H, W) = grounder.detect(frame0, object_prompt)
    obj_dets = nms(obj_dets)
    arm_dets = nms(arm_dets)
    print(f"[{tag}] seed frame {seed_idx}")
    print(f"[{tag}] object dets: " + ", ".join(f"{d['phrase']}({d['conf']:.2f})" for d in obj_dets))
    print(f"[{tag}] arm dets:    " + ", ".join(f"{d['phrase']}({d['conf']:.2f})" for d in arm_dets))

    # assign SAM2 object ids: 1.. = objects, 100.. = arms
    seeds = {}
    id_role = {}
    boxes_for_seed = []
    oid = 1
    for d in obj_dets:
        boxes_for_seed.append(d["xyxy"]); id_role[oid] = "object"; oid += 1
    arm_start = 100
    for d in arm_dets:
        boxes_for_seed.append(d["xyxy"]); id_role[arm_start] = "arm"; arm_start += 1

    seed_list, (H, W) = seed_masks(img_pred, frame0, boxes_for_seed)
    ids = list(id_role.keys())
    seeds = {i: m for i, m in zip(ids, seed_list)}

    if not seeds:
        shutil.rmtree(work, ignore_errors=True)
        return {"n_object_dets": 0, "n_arm_dets": 0, "num_pairs": 0, "seed_frame": seed_idx,
                "actor_flicker_raw": None, "per_frame": []}
    masks_by_id, nprop = propagate_multi(vid_pred, work, seeds, frame_idx=seed_idx)

    obj_ids = [i for i in ids if id_role[i] == "object"]
    arm_ids = [i for i in ids if id_role[i] == "arm"]

    def union(masks_list):
        u = np.zeros((H, W), bool)
        for m in masks_list:
            u |= (m if m.shape == (H, W) else mask_resize(m, (H, W)))
        return u

    # seed overlay (object=green, arm=blue)
    ov = [(seeds[i], (0, 200, 0)) for i in obj_ids] + [(seeds[i], (255, 0, 0)) for i in arm_ids]
    overlay(frame0, ov, os.path.join(out_dir, f"{tag}_seed_overlay.png"))

    frames = sorted(os.path.join(work, f) for f in os.listdir(work) if f.endswith(".jpg"))
    per_frame = []
    for i, (f1, f2) in enumerate(zip(frames[:-1], frames[1:])):
        # `stride` decodes every Nth source frame. Convert displacement over that
        # interval back to px/source-frame so ActorMotion stays comparable to the
        # stride-1 KineticBudget and its documented units.
        fl = flow.compute(f1, f2) / float(max(1, stride))
        mag = np.sqrt(fl[..., 0] ** 2 + fl[..., 1] ** 2)
        mh = mag.shape
        obj_m = union([masks_by_id[j][i] for j in obj_ids]) if obj_ids else np.zeros((H, W), bool)
        arm_m = union([masks_by_id[j][i] for j in arm_ids]) if arm_ids else np.zeros((H, W), bool)
        if (H, W) != mh:
            obj_m = mask_resize(obj_m, mh); arm_m = mask_resize(arm_m, mh)
        fg = obj_m | arm_m
        bg = ~fg
        if fg.sum() == 0 or bg.sum() == 0:
            continue
        fg_ego_mag, ego = ego_compensated_magnitude(fl, fg)
        if fg_ego_mag is None:
            continue
        e = {
            "pair": i,
            "fg_cov_pct": float(fg.mean() * 100),
            "obj_cov_pct": float(obj_m.mean() * 100),
            "arm_cov_pct": float(arm_m.mean() * 100),
            "obj_flow_max": float(mag[obj_m].max()) if obj_m.any() else 0.0,
            "obj_flow_med": float(np.median(mag[obj_m])) if obj_m.any() else 0.0,
            "arm_flow_max": float(mag[arm_m].max()) if arm_m.any() else 0.0,
            "arm_flow_med": float(np.median(mag[arm_m])) if arm_m.any() else 0.0,
            "fg_flow_max": float(mag[fg].max()),
            "fg_flow_med": float(np.median(mag[fg])),
            "fg_flow_p95": float(np.percentile(mag[fg], 95)),
            # Wrist cameras ride the end effector: whole-frame flow is arm motion.
            "frame_flow_med": float(np.median(mag)),
            # Ego estimate is background median; unreliable when masks cover the frame.
            "bg_cov_pct": float(bg.mean() * 100),
            "fg_flow_ego_med": float(np.median(fg_ego_mag)),
            # High-percentile localized motion complements the median: small gripper
            # motions should not disappear merely because the arm mask is large.
            "fg_flow_ego_p95": float(np.percentile(fg_ego_mag, 95)),
            "bg_flow_max": float(mag[bg].max()),
            "bg_flow_med": float(np.median(mag[bg])),
            "bg_ego_dx": float(ego[0]),
            "bg_ego_dy": float(ego[1]),
        }
        if compute_flicker:
            from metrics.dynamics.actor_flicker import residual_variants
            next_fg = union([masks_by_id[j][i + 1] for j in ids])
            if next_fg.shape != mh:
                next_fg = mask_resize(next_fg, mh)
            forward = fl * max(1, stride)
            backward = flow.compute(f2, f1)
            # All E3 candidate normalizations in one pass: the optical flow and SAM2
            # masks are what cost GPU time, the variants are numpy on arrays that
            # already exist. Storing them all makes the choice of statistic a
            # post-hoc, zero-GPU decision (same principle as the E4 gate).
            variants, count = residual_variants(cv2.imread(f1), cv2.imread(f2),
                                                forward, backward, fg, next_fg)
            e["actor_flicker_raw"] = variants["raw"]
            e["flicker_valid_pixels"] = count
            e["flicker_variants"] = variants
        per_frame.append(e)
        if i % max(1, len(frames) // 6) == 0:
            overlay(f1, [(obj_m, (0, 200, 0)), (arm_m, (255, 0, 0))],
                    os.path.join(out_dir, f"{tag}_fgmask_{i:04d}.png"))

    def mean(key):
        return float(np.mean([p[key] for p in per_frame])) if per_frame else float("nan")

    variant_keys = (per_frame[0].get("flicker_variants") or {}) if per_frame else {}
    flicker_variants = None
    if variant_keys:
        flicker_variants = {}
        for k in variant_keys:
            # Clip-level mean over the pairs where the candidate is defined.
            # `unexplained` is already a ratio of sums within a pair, so averaging
            # pair values here is a mean of per-pair pooled statistics, not of
            # per-pixel ratios.
            vals = [p["flicker_variants"][k] for p in per_frame
                    if (p.get("flicker_variants") or {}).get(k) is not None]
            flicker_variants[k] = float(np.mean(vals)) if vals else None

    result = {
        "tag": tag, "video": video, "num_frames": n, "num_pairs": len(per_frame),
        "object_prompt": object_prompt, "embodiment_prompt": embodiment_prompt,
        "n_object_dets": len(obj_ids), "n_arm_dets": len(arm_ids), "seed_frame": seed_idx,
        # WorldScore form with fg = arm+object
        "ws_fg_vs_bg": float(np.mean([p["fg_flow_max"] - p["bg_flow_max"] for p in per_frame])),
        # original object-only WorldScore form
        "ws_obj_vs_bg": float(np.mean([p["obj_flow_max"] - p["bg_flow_max"] for p in per_frame])),
        # outlier-robust
        "med_fg_vs_bg": float(np.mean([p["fg_flow_med"] - p["bg_flow_med"] for p in per_frame])),
        "med_obj_vs_bg": float(np.mean([p["obj_flow_med"] - p["bg_flow_med"] for p in per_frame])),
        "mean_obj_flow_med": mean("obj_flow_med"),
        "mean_arm_flow_med": mean("arm_flow_med"),
        "mean_bg_flow_med": mean("bg_flow_med"),
        # ActorMotion (PhyWorld): subject-only kinetic budget, arm∪object mask.
        # Remove robust background flow vector first; keep uncorrected value for audit.
        "mean_fg_flow_med": mean("fg_flow_ego_med"),
        "mean_fg_flow_p95": mean("fg_flow_ego_p95"),
        "mean_fg_flow_med_uncorrected": mean("fg_flow_med"),
        # Fraction of subject frame-pairs with non-trivial residual movement.
        # Paired with ActorMotion(raw): detects one-frame spikes that inflate a mean.
        "actor_motion_active_rate": float(np.mean([
            p["fg_flow_ego_med"] >= actor_motion_active_threshold for p in per_frame
        ])) if per_frame else float("nan"),
        "actor_motion_active_threshold_px": float(actor_motion_active_threshold),
        "mean_bg_ego_dx": mean("bg_ego_dx"),
        "mean_bg_ego_dy": mean("bg_ego_dy"),
        "mean_fg_cov_pct": mean("fg_cov_pct"),
        "per_frame": per_frame,
        "actor_flicker_raw": float(np.mean([p["actor_flicker_raw"] for p in per_frame
            if p.get("actor_flicker_raw") is not None])) if any(
            p.get("actor_flicker_raw") is not None for p in per_frame) else None,
        # Clip-level mean of every E3 candidate over the pairs where it is defined.
        "actor_flicker_variants": flicker_variants,
    }
    with open(os.path.join(out_dir, f"{tag}_motion_accuracy_gdino.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{tag}] ws_fg_vs_bg={result['ws_fg_vs_bg']:.3f} ws_obj_vs_bg={result['ws_obj_vs_bg']:.3f} "
          f"med_fg_vs_bg={result['med_fg_vs_bg']:.4f} | obj_med={result['mean_obj_flow_med']:.3f} "
          f"arm_med={result['mean_arm_flow_med']:.3f} bg_med={result['mean_bg_flow_med']:.3f} "
          f"actor_motion={result['mean_fg_flow_med']:.4f} "
          f"active={result['actor_motion_active_rate']:.3f} "
          f"(uncorrected={result['mean_fg_flow_med_uncorrected']:.4f})")
    shutil.rmtree(work, ignore_errors=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    ap.add_argument("--sam2_ckpt", required=True)
    ap.add_argument("--raft_cfg", required=True)
    ap.add_argument("--raft_ckpt", required=True)
    ap.add_argument("--gdino_cfg", required=True)
    ap.add_argument("--gdino_ckpt", required=True)
    ap.add_argument("--object_prompt", default="cardboard box . box .")
    ap.add_argument("--embodiment_prompt", default="robot arm . robot gripper .")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    GlobalHydra.instance().clear()
    hydra.initialize_config_module("sam2", version_base="1.2")
    img_model = build_sam2(args.sam2_cfg, args.sam2_ckpt, device=device, apply_postprocessing=False)
    img_pred = SAM2ImagePredictor(img_model)
    vid_pred = build_sam2_video_predictor(args.sam2_cfg, args.sam2_ckpt, device=device, apply_postprocessing=False)
    flow = Flow(args.raft_cfg, args.raft_ckpt, device)
    grounder = Grounder(args.gdino_cfg, args.gdino_ckpt, device)

    results = {}
    for tag, path in [("gt", args.gt), ("pred", args.pred)]:
        results[tag] = probe(path, tag, args.out, grounder, img_pred, vid_pred, flow,
                             args.object_prompt, args.embodiment_prompt, device,
                             max_frames=args.max_frames, stride=args.stride)
    with open(os.path.join(args.out, "summary_gdino.json"), "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "per_frame"}
                   for k, v in results.items()}, f, indent=2)
    print("\n=== SUMMARY (embodiment-aware) ===")
    for tag in ("gt", "pred"):
        r = results[tag]
        print(f"{tag:5s}: ws_fg_vs_bg={r['ws_fg_vs_bg']:+.3f}  ws_obj_vs_bg={r['ws_obj_vs_bg']:+.3f}  "
              f"med_fg_vs_bg={r['med_fg_vs_bg']:+.4f}  (obj_med={r['mean_obj_flow_med']:.3f} "
              f"arm_med={r['mean_arm_flow_med']:.3f} bg_med={r['mean_bg_flow_med']:.3f})")


if __name__ == "__main__":
    main()
