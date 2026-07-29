"""Standalone motion_accuracy probe (WorldScore dynamics metric).

Replicates WorldScore's MotionAccuracyMetric WITHOUT importing BaseMetric /
groundingdino / spacy / pyiqa (which the vendored metric file pulls in at import
time). We only need SAM2 (vendored) + SEA-RAFT (vendored).

Pipeline per video:
  1. decode mp4 -> JPG frames
  2. SAM2 *image* predictor seeds an object mask on frame 0 from a normalized box
  3. SAM2 *video* predictor propagates that mask through all frames (dilated +10px,
     exactly like the WorldScore metric)
  4. SEA-RAFT optical flow for each consecutive frame pair
  5. score_t = max(|flow| over object) - max(|flow| over background)
     motion_accuracy = mean_t(score_t)      # <- identical to WorldScore

We additionally log per-frame object/background flow (max/median/mean) so we can
sanity-check whether the score behaves as intended.

Run with the dreamgen venv + PYTHONPATH to the vendored sam2 and SEA-RAFT core;
see run_motion_accuracy_probe.sh.
"""
import os
import sys
import json
import argparse
import tempfile
import shutil

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# --- vendored deps (added to sys.path by the launcher) ---
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import hydra
from hydra.core.global_hydra import GlobalHydra

# SEA-RAFT core
from core.raft import RAFT
from core.utils.utils import load_ckpt
from core.parser import parse_args


# --------------------------------------------------------------------------- #
# video / frame io
# --------------------------------------------------------------------------- #
def decode_video_to_frames(video_path, out_dir, max_frames=None, stride=1):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    saved = 0
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
# SAM2 seed mask on frame 0 from a normalized xyxy box
# --------------------------------------------------------------------------- #
def seed_mask_from_box(image_predictor, frame0_path, norm_box):
    img = cv2.imread(frame0_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    x0, y0, x1, y1 = norm_box
    box = np.array([x0 * W, y0 * H, x1 * W, y1 * H], dtype=np.float32)
    image_predictor.set_image(img)
    masks, scores, _ = image_predictor.predict(
        point_coords=None, point_labels=None, box=box[None, :], multimask_output=False
    )
    mask = masks[0].astype(bool)  # (H,W)
    return mask, (H, W)


# --------------------------------------------------------------------------- #
# SAM2 video propagation (mirrors WorldScore vos_inference, box-seeded)
# --------------------------------------------------------------------------- #
def expand_mask(mask, pixels=10):
    kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return dilated.astype(bool)


@torch.inference_mode()
def propagate_masks(video_predictor, frames_dir, seed_mask, score_thresh=0.0):
    inference_state = video_predictor.init_state(
        video_path=frames_dir, async_loading_frames=False
    )
    video_predictor.add_new_mask(
        inference_state=inference_state, frame_idx=0, obj_id=1, mask=seed_mask
    )
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
        inference_state
    ):
        m = (out_mask_logits[0] > score_thresh).cpu().numpy().squeeze()
        video_segments[out_frame_idx] = m
    masks = []
    for i in sorted(video_segments):
        masks.append(expand_mask(video_segments[i]))
    return masks


# --------------------------------------------------------------------------- #
# SEA-RAFT flow
# --------------------------------------------------------------------------- #
class Flow:
    def __init__(self, cfg_path, ckpt_path, device):
        args = argparse.Namespace(cfg=cfg_path, path=ckpt_path)
        args = parse_args(args)
        model = RAFT(args)
        load_ckpt(model, ckpt_path)
        model.to(device).eval()
        self.model = model
        self.args = args
        self.device = device

    def load_image(self, imfile):
        img = cv2.imread(imfile)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)[None].to(self.device)
        return t

    def compute(self, im1, im2):
        with torch.no_grad():
            i1 = self.load_image(im1)
            i2 = self.load_image(im2)
            s = self.args.scale
            i1 = F.interpolate(i1, scale_factor=2 ** s, mode="bilinear", align_corners=False)
            i2 = F.interpolate(i2, scale_factor=2 ** s, mode="bilinear", align_corners=False)
            with torch.amp.autocast(device_type="cuda"):
                out = self.model(i1, i2, iters=self.args.iters, test_mode=True)
            flow = out["flow"][-1]
            flow = F.interpolate(flow, scale_factor=0.5 ** s, mode="bilinear", align_corners=False) * (0.5 ** s)
            flow = flow.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
        return flow


def mask_resize(mask, target_hw):
    m = (mask > 0.5).astype(np.uint8)
    r = cv2.resize(m, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (r > 0.5).astype(bool)


# --------------------------------------------------------------------------- #
# main probe
# --------------------------------------------------------------------------- #
def probe(video_path, tag, out_dir, image_predictor, video_predictor,
          flow, norm_box, device, max_frames=None, stride=1):
    work = os.path.join(out_dir, f"{tag}_frames")
    n = decode_video_to_frames(video_path, work, max_frames=max_frames, stride=stride)
    print(f"[{tag}] decoded {n} frames -> {work}")

    frame0 = os.path.join(work, "00000.jpg")
    seed, (H, W) = seed_mask_from_box(image_predictor, frame0, norm_box)
    print(f"[{tag}] seed mask coverage frame0 = {seed.mean()*100:.2f}% of image")

    # save seed overlay
    ov = cv2.imread(frame0)
    red = np.zeros_like(ov); red[..., 2] = 255
    ov = np.where(seed[..., None], (0.5 * ov + 0.5 * red).astype(np.uint8), ov)
    cv2.imwrite(os.path.join(out_dir, f"{tag}_seed_overlay.png"), ov)

    masks = propagate_masks(video_predictor, work, seed)
    print(f"[{tag}] propagated masks: {len(masks)}")

    frame_files = sorted(os.path.join(work, f) for f in os.listdir(work) if f.endswith(".jpg"))

    per_frame = []
    scores = []
    for i, (f1, f2, m) in enumerate(zip(frame_files[:-1], frame_files[1:], masks[:-1])):
        fl = flow.compute(f1, f2)
        mag = np.sqrt(fl[..., 0] ** 2 + fl[..., 1] ** 2)
        if m.shape != mag.shape:
            m = mask_resize(m, mag.shape)
        obj = mag[m]
        bg = mag[~m]
        if obj.size == 0 or bg.size == 0:
            continue
        score = float(obj.max() - bg.max())  # WorldScore motion_accuracy
        scores.append(score)
        per_frame.append({
            "pair": i,
            "obj_cov_pct": float(m.mean() * 100),
            "obj_flow_max": float(obj.max()),
            "obj_flow_med": float(np.median(obj)),
            "obj_flow_mean": float(obj.mean()),
            "bg_flow_max": float(bg.max()),
            "bg_flow_med": float(np.median(bg)),
            "bg_flow_mean": float(bg.mean()),
            "score_max_minus_max": score,
        })
        # save a few flow/mask overlays
        if i % max(1, (len(frame_files) // 6)) == 0:
            base = cv2.imread(f1)
            red = np.zeros_like(base); red[..., 2] = 255
            mov = np.where(m[..., None], (0.5 * base + 0.5 * red).astype(np.uint8), base)
            cv2.imwrite(os.path.join(out_dir, f"{tag}_mask_{i:04d}.png"), mov)

    motion_accuracy = float(np.mean(scores)) if scores else float("nan")
    # extra aggregate signals for sanity checking
    obj_med = float(np.mean([p["obj_flow_med"] for p in per_frame])) if per_frame else float("nan")
    bg_med = float(np.mean([p["bg_flow_med"] for p in per_frame])) if per_frame else float("nan")
    result = {
        "tag": tag,
        "video": video_path,
        "num_frames": n,
        "num_pairs_scored": len(scores),
        "norm_box": norm_box,
        "seed_cov_pct_frame0": float(seed.mean() * 100),
        "motion_accuracy_worldscore": motion_accuracy,
        "mean_obj_flow_median": obj_med,
        "mean_bg_flow_median": bg_med,
        "obj_minus_bg_median": obj_med - bg_med,
        "per_frame": per_frame,
    }
    with open(os.path.join(out_dir, f"{tag}_motion_accuracy.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{tag}] motion_accuracy(WorldScore) = {motion_accuracy:.4f} "
          f"| obj_med={obj_med:.3f} bg_med={bg_med:.3f}")
    # cleanup frame dir (masks/overlays kept)
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
    # normalized xyxy box around the manipulated object (cardboard box, center)
    ap.add_argument("--box", default="0.23,0.08,0.78,0.68")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm_box = [float(x) for x in args.box.split(",")]

    # build SAM2 (once) + SEA-RAFT. Use SAM2's own config module as the hydra
    # search path (the vendored `sam2` package ships a hydra plugin registering
    # its `configs/` dir), so config_name is relative to that module.
    GlobalHydra.instance().clear()
    hydra.initialize_config_module("sam2", version_base="1.2")
    image_model = build_sam2(args.sam2_cfg, args.sam2_ckpt, device=device, apply_postprocessing=False)
    image_predictor = SAM2ImagePredictor(image_model)
    video_predictor = build_sam2_video_predictor(
        args.sam2_cfg, args.sam2_ckpt, device=device, apply_postprocessing=False
    )
    flow = Flow(args.raft_cfg, args.raft_ckpt, device)

    results = {}
    for tag, path in [("gt", args.gt), ("pred", args.pred)]:
        results[tag] = probe(
            path, tag, args.out, image_predictor, video_predictor,
            flow, norm_box, device,
            max_frames=args.max_frames, stride=args.stride,
        )
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "per_frame"}
                   for k, v in results.items()}, f, indent=2)
    print("\n=== SUMMARY ===")
    for tag in ("gt", "pred"):
        r = results[tag]
        print(f"{tag:5s}: motion_accuracy={r['motion_accuracy_worldscore']:.4f} "
              f"obj_med={r['mean_obj_flow_median']:.3f} bg_med={r['mean_bg_flow_median']:.3f}")


if __name__ == "__main__":
    main()
