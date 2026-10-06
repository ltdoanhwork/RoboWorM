"""Motion Amplitude Score (MAS): subject-localised optical-flow magnitude.

Source metric
    RBench "Motion Amplitude Score" -- Y. Deng, Z. Pan, H. Zhang, X. Li, R. Hu,
    Y. Ding, Y. Zou, Y. Zeng, D. Zhou, "Rethinking Video Generation Model for the
    Embodied World", ICML 2026. Measures how much the *subject* (robot + manipulated
    object) moves, rather than the whole frame.

Use in the paper
    RQ3 baseline (Sec. 4.2): MAS is compared against KIVA and found to track only
    surface motion. It is descriptive (the co-located reference defines the correct
    amount of motion), so it carries no higher/lower-is-better claim.

Our implementation (closed-world adaptation, as stored by the dense-NR wave)
    1. GroundingDINO (Liu et al., ECCV 2024) detects the embodiment (fixed phrases per
       embodiment, :data:`EMBODIMENT_PHRASES`) on the most confident of four seed frames,
       and the task objects (task clause) on that frame.
    2. SAM 2.1 (Ravi et al., ICLR 2025) propagates one mask per detection through the
       clip; subject = union(arm, object) masks, dilated 10 px.
    3. SEA-RAFT (Wang et al., ECCV 2024) flow between consecutive analysis frames, in
       px per *source* frame.
    4. Per pair, a statistic that depends on the camera mount (:func:`camera_mount`):
       static camera -> p95 of subject flow; head camera -> p95 of subject flow after
       removing the background-median (ego) translation, only if background >= 5 % of
       the frame; wrist camera -> whole-frame median flow (the camera rides the arm).
    5. ``motion_amplitude`` = mean over valid pairs; ``motion_amplitude_active_rate`` =
       fraction of valid pairs >= 1 px / source frame. A clip is N/A when no robot is
       detected or fewer than half the pairs are valid / tracked.
    RBench tracks points with CoTracker; we use flow inside a tracked mask instead.

Backends from ``metrics/WorldScore`` third_party (see metrics/third_party.py).
Registered as ``motion_amplitude``.
"""
import os
import json
import shutil
import tempfile
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from .base import ReferenceFreeFrameMetric
from .third_party import (GDINO_CFG, GDINO_CKPT, SAM2_CFG, SAM2_CKPT, SeaRaftFlow,
                          add_paths, write_frames)

#: GroundingDINO phrases for the embodiment part of the subject mask
EMBODIMENT_PHRASES = {
    "single_arm": ["robot arm", "gripper"],
    "bimanual": ["robot arm", "gripper"],
    "humanoid": ["robot", "hand", "arm"],
}
ACTIVE_PX = 1.0           # px/source-frame: a pair at or above this is "active"
# Head-camera ego estimate needs enough background pixels (lowered 10 -> 5 after the
# E4 audit: changes the head median by 0.00% on a fixed clip set, recovers 6% of clips).
MIN_BG_COV_PCT = 5.0
MIN_VALID_PAIR_FRAC = 0.5
STATISTIC = {"static": "subject_p95_uncorrected", "head": "subject_p95_ego_translation",
             "wrist": "frame_median"}


def camera_mount(embodiment: str, view: str) -> str:
    """static cameras need no ego compensation; head/wrist cameras move."""
    v = view.lower()
    if "wrist" in v:
        return "wrist"
    if embodiment == "humanoid" and v in ("cam_high", "cam_high_rgb", "cam_head_rgb",
                                          "singleview", "grid_top_left"):
        return "head"
    return "static"


def actor_pair_motion(per_frame, mount, interval):
    """Per-pair actor motion in px/source-frame, keyed by pair index; plus median diagnostic.

    Static: p95 of uncorrected subject flow. Subject median is ~0 because static robot
    body and objects dominate the mask; ego subtraction there removes deformable-object
    motion. Head: p95 of ego-compensated subject flow where background >= MIN_BG_COV_PCT.
    Wrist: whole-frame median flow, since the camera rides the end effector and masks
    often fill the frame.
    """
    motion, median = {}, {}
    for p in per_frame:
        k = p["pair"]
        if mount == "static":
            motion[k], median[k] = p["fg_flow_p95"], p["fg_flow_med"]
        elif mount == "wrist":
            motion[k] = median[k] = p["frame_flow_med"]
        elif p["bg_cov_pct"] >= MIN_BG_COV_PCT:
            motion[k], median[k] = p["fg_flow_ego_p95"], p["fg_flow_ego_med"]
    return ({k: v / interval for k, v in motion.items()},
            {k: v / interval for k, v in median.items()})


def summarize(actor: dict, mount: str, interval: int, n_pairs_possible: int,
              source_fps: float) -> tuple[dict, dict, int, bool]:
    """Probe output -> MAS fields (the ``actor_motion_*`` record keys of the NR wave).

    Returns ``(fields, motion, n_active, motion_ok)``; ``motion`` / ``n_active`` feed
    the activity weighting of the wave's continuity column.
    """
    per_frame = actor.get("per_frame", [])
    robot = bool(actor["n_arm_dets"])
    motion, median = actor_pair_motion(per_frame, mount, interval)
    # `per_frame` is ALREADY filtered by the probe (pairs with empty fg/bg or a failed
    # ego fit are skipped), so tracking loss is measured against the pairs that exist.
    n_pairs_possible = max(int(n_pairs_possible), 1)
    tracked_frac = len(per_frame) / n_pairs_possible
    motion_ok = bool(robot and per_frame
                     and len(motion) >= MIN_VALID_PAIR_FRAC * len(per_frame)
                     and tracked_frac >= MIN_VALID_PAIR_FRAC)
    values = np.array(list(motion.values()), dtype=float)
    n_active = int((values >= ACTIVE_PX).sum()) if motion_ok else 0
    fields = {
        "actor_motion_raw": float(values.mean()) if motion_ok else None,
        "actor_motion_median_raw": float(np.mean(list(median.values()))) if motion_ok else None,
        "actor_motion_active_rate": float(np.mean(values >= ACTIVE_PX)) if motion_ok else None,
        "actor_motion_px_per_second": float(values.mean() * source_fps) if motion_ok else None,
        "actor_motion_statistic": STATISTIC[mount],
        "actor_seed_frame": actor.get("seed_frame"),
        "actor_motion_valid_pair_frac": (len(motion) / len(per_frame)) if per_frame else 0.0,
        "actor_tracked_pair_frac": tracked_frac,
        "actor_n_active_pairs": n_active if motion_ok else None,
    }
    return fields, motion, n_active, motion_ok


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
    """GroundingDINO (Swin-T OGC) open-vocabulary box detector."""

    def __init__(self, cfg=GDINO_CFG, ckpt=GDINO_CKPT, device="cuda"):
        add_paths()
        import groundingdino.datasets.transforms as T
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

        self._phrases = get_phrases_from_posmap
        a = SLConfig.fromfile(str(cfg))
        a.device = device
        a.bert_base_uncased_path = None
        m = build_model(a)
        sd = torch.load(str(ckpt), map_location="cpu")
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
            phrase = self._phrases(lg > txt_thr, tok, self.model.tokenizer)
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


# SAM2 `init_state` materializes the WHOLE clip as one
# `torch.zeros(num_frames, 3, 1024, 1024, float32)` tensor -- 12.58 MB per frame -- and
# with `offload_video_to_cpu=False` that entire stack lives on the GPU for the whole
# propagation, even though `_get_image_feature` only ever uses `images[frame_idx]`.
#
# Measured on the dense-NR full wave (8 workers / 80GB A100): this is what fills the
# card. The 98 clips that died after all 4 OOM retries split into 16 that failed on the
# stack itself (8.72 / 6.26 / 5.81 / 3.96 / 2.88 / 2.09 GiB = 694 / 510 / 473 / 322 /
# 234 / 170 frames x 12.58 MB) and 83 SHORT clips that failed on small propagation
# allocations (44 x 280 MiB) purely because a long clip in another worker had already
# taken the card. Retrying does not help because the pressure is steady-state, not
# transient: 71% of clips that OOM'd once still died after 4 attempts, having slept
# 60+120+180s each (~12% of total worker time).
#
# Offloading is numerically inert -- validated by re-scoring a 744-frame clip and
# diffing every field against its stored record: all SAM2-derived values
# (`actor_per_frame`, 25,262 numbers, plus every `actor_*` metric) came back
# bit-identical, max abs delta 0.000e+00. The only fields that moved were the
# VFIMamba-derived continuity ones, and a CONTROL re-score with offload OFF (the exact
# config that produced the baseline) moved them MORE (4.78e-03 vs 3.96e-03), proving
# that is pre-existing GPU nondeterminism. So this does NOT bump PROTOCOL.
#
# But offloading is NOT free in production, and the cost only shows up under
# concurrency: every frame access becomes a 12.58 MB H2D copy, ~1,500 of them on a
# 744-frame forward+backward propagation. Solo on the card that cost +2.2%; with 8
# workers contending for PCIe it was +26..45% sec/frame in EVERY frame band. So offload
# only the clips that actually threaten the card and leave the short ones -- which are
# the majority (66% of scored clips are < 100 frames = < 1.3 GiB) -- on the GPU where
# they were already fine. That removes the pressure at its source, which is what also
# protects the 83 short-clip victims.
_SAM2_OFFLOAD_MIN_GIB = float(os.environ.get('NR_SAM2_OFFLOAD_MIN_GIB', '1.5'))
_SAM2_FRAME_MIB = 3 * 1024 * 1024 * 4 / 1024 / 1024  # 1024x1024x3 float32 = 12.58 MB
_SAM2_OFFLOAD_STATE = os.environ.get('NR_SAM2_OFFLOAD_STATE', '0') == '1'
_FRAME_EXT = ('.jpg', '.jpeg', '.JPG', '.JPEG', '.png')


def _should_offload_video(frames_dir):
    """Offload the frame stack to CPU only when it would be large enough to matter.

    `NR_SAM2_OFFLOAD_MIN_GIB=0` forces offload always, a huge value disables it.
    """
    if _SAM2_OFFLOAD_MIN_GIB <= 0:
        return True, 0.0
    try:
        n = sum(1 for p in os.listdir(frames_dir) if os.path.splitext(p)[-1] in _FRAME_EXT)
    except OSError:
        return False, 0.0
    gib = n * _SAM2_FRAME_MIB / 1024
    return gib >= _SAM2_OFFLOAD_MIN_GIB, gib


@torch.inference_mode()
def propagate_multi(video_predictor, frames_dir, seed_masks_by_id, score_thresh=0.0, frame_idx=0):
    """Propagate several seeded object ids; return {obj_id: [mask per frame]}.

    A seed after frame 0 is propagated forward and then backward so every frame has masks.
    """
    offload, gib = _should_offload_video(frames_dir)
    if offload:
        print(f'[sam2] frame stack {gib:.2f} GiB >= {_SAM2_OFFLOAD_MIN_GIB} GiB -> CPU', flush=True)
    state = video_predictor.init_state(video_path=frames_dir, async_loading_frames=False,
                                       offload_video_to_cpu=offload,
                                       offload_state_to_cpu=_SAM2_OFFLOAD_STATE)
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
# SEA-RAFT (shared implementation in metrics/third_party.py)
# --------------------------------------------------------------------------- #
Flow = SeaRaftFlow


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
        # interval back to px/source-frame so MAS stays comparable to the
        # stride-1 frame flow and its documented units.
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
            raise NotImplementedError(
                "actor flicker was an experimental NR column, not a paper metric; it was "
                "archived to tmp/20261005_refactor/misc/metrics/dynamics/actor_flicker.py")
        per_frame.append(e)
        if i % max(1, len(frames) // 6) == 0:
            overlay(f1, [(obj_m, (0, 200, 0)), (arm_m, (255, 0, 0))],
                    os.path.join(out_dir, f"{tag}_fgmask_{i:04d}.png"))

    def mean(key):
        return float(np.mean([p[key] for p in per_frame])) if per_frame else float("nan")

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
        # Subject-only motion (MAS-style), arm∪object mask.
        # Remove robust background flow vector first; keep uncorrected value for audit.
        "mean_fg_flow_med": mean("fg_flow_ego_med"),
        "mean_fg_flow_p95": mean("fg_flow_ego_p95"),
        "mean_fg_flow_med_uncorrected": mean("fg_flow_med"),
        # Fraction of subject frame-pairs with non-trivial residual movement.
        # Paired with the mean: detects one-frame spikes that inflate a mean.
        "actor_motion_active_rate": float(np.mean([
            p["fg_flow_ego_med"] >= actor_motion_active_threshold for p in per_frame
        ])) if per_frame else float("nan"),
        "actor_motion_active_threshold_px": float(actor_motion_active_threshold),
        "mean_bg_ego_dx": mean("bg_ego_dx"),
        "mean_bg_ego_dy": mean("bg_ego_dy"),
        "mean_fg_cov_pct": mean("fg_cov_pct"),
        "per_frame": per_frame,
    }
    with open(os.path.join(out_dir, f"{tag}_motion_amplitude.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{tag}] ws_fg_vs_bg={result['ws_fg_vs_bg']:.3f} ws_obj_vs_bg={result['ws_obj_vs_bg']:.3f} "
          f"med_fg_vs_bg={result['med_fg_vs_bg']:.4f} | obj_med={result['mean_obj_flow_med']:.3f} "
          f"arm_med={result['mean_arm_flow_med']:.3f} bg_med={result['mean_bg_flow_med']:.3f} "
          f"actor_motion={result['mean_fg_flow_med']:.4f} "
          f"active={result['actor_motion_active_rate']:.3f} "
          f"(uncorrected={result['mean_fg_flow_med_uncorrected']:.4f})")
    shutil.rmtree(work, ignore_errors=True)
    return result


def build_sam2_predictors(device="cuda", cfg=SAM2_CFG, ckpt=SAM2_CKPT):
    """SAM 2.1 image predictor (seed masks) + video predictor (propagation)."""
    add_paths()
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    GlobalHydra.instance().clear()
    hydra.initialize_config_module("sam2", version_base="1.2")
    img = SAM2ImagePredictor(build_sam2(cfg, str(ckpt), device=device, apply_postprocessing=False))
    vid = build_sam2_video_predictor(cfg, str(ckpt), device=device, apply_postprocessing=False)
    return img, vid


class MotionAmplitude(ReferenceFreeFrameMetric):
    """Registry wrapper: one clip -> MAS on the paper analysis frames.

    Needs the task text (object grounding; ``"object ."`` if empty) and, from
    ``sample.meta`` or the constructor, ``embodiment`` and ``view`` (camera mount).
    ``source_interval`` (default: from the analysis protocol) converts px per analysis
    pair to px per source frame.
    """

    name = "motion_amplitude"
    higher_is_better = {"motion_amplitude": True, "motion_amplitude_active_rate": True}

    def __init__(self, device: str = "cuda", embodiment: str = "single_arm",
                 view: str = "singleview", frame_stride: int = 1, **config: Any) -> None:
        super().__init__(device=device, embodiment=embodiment, view=view,
                         frame_stride=frame_stride, **config)
        self.embodiment, self.view = embodiment, view
        self._models = None

    def setup(self) -> None:
        flow = SeaRaftFlow(device=self.device)
        flow.a.scale = 0                     # as in the dense-NR wave
        img, vid = build_sam2_predictors(self.device)
        self._models = (Grounder(device=self.device), img, vid, flow)

    def compute(self, sample) -> dict:
        from pathlib import Path

        from .video_io import nr_analysis_frames

        m = sample.meta or {}
        emb = m.get("embodiment", self.embodiment)
        mount = camera_mount(emb, m.get("view", self.view))
        if isinstance(sample.pred, np.ndarray):
            frames, interval, fps = sample.pred, int(m.get("source_interval", 1)), float(m.get("source_fps", 5.0))
        else:
            frames, info = nr_analysis_frames(sample.pred, m.get("layout", "single"),
                                              int(m.get("view_index", 0)), int(m.get("n_views", 1)))
            interval, fps = info["source_interval"], info["source_fps"]
        from .text_alignment import task_clause

        grounder, img, vid, flow = self._models
        tmp = Path(tempfile.mkdtemp(prefix="motion_amplitude_"))
        try:
            video = tmp / "analysis.mp4"
            w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps / interval,
                                (frames.shape[2], frames.shape[1]))
            for fr in frames[::self.frame_stride]:
                w.write(fr[:, :, ::-1])
            w.release()
            actor = probe(str(video), "pred", str(tmp), grounder, img, vid, flow,
                          task_clause(sample.text) or "object .",
                          " . ".join(EMBODIMENT_PHRASES[emb]) + " .", self.device, stride=1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        fields, _, _, ok = summarize(actor, mount, interval, len(frames) - 1, fps)
        nan = float("nan")
        return {"motion_amplitude": fields["actor_motion_raw"] if ok else nan,
                "motion_amplitude_active_rate": fields["actor_motion_active_rate"] if ok else nan,
                "motion_amplitude_px_per_second": fields["actor_motion_px_per_second"] if ok else nan,
                "n_arm_dets": float(actor["n_arm_dets"]), "n_frames": float(len(frames))}

    def score_frames(self, pred, ref, text):  # pragma: no cover - compute() is overridden
        raise NotImplementedError


__all__ = ["EMBODIMENT_PHRASES", "ACTIVE_PX", "MIN_BG_COV_PCT", "MIN_VALID_PAIR_FRAC",
           "camera_mount", "actor_pair_motion", "summarize", "Grounder", "Flow",
           "build_sam2_predictors", "probe", "MotionAmplitude"]
