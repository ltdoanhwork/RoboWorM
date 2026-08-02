"""Tracking backend for the sim2real interaction metric.

Turns a ctrlworld ``*_all_views.mp4`` strip into per-view, per-role boolean mask
tracks for the two entities the physics predicates need: the manipulated
**object** and the robot **gripper**.

Three things differ from ``metrics/dynamics/motion_accuracy_gdino.py`` (which this
reuses the GroundingDINO/SAM2 machinery from):

1. **Views are split first.** ctrlworld writes 3 views tiled horizontally
   (960x192 = 3 x 320); every predicate is computed per view, and the contact
   decision is AND-gated across the two *static exterior* views.
2. **Frames are upscaled before SAM2.** SAM2 returns masks at the input frame
   resolution, so feeding 320x192 gives mask boundaries too coarse for a
   gripper-object gap measurement. We upscale (default 3x, LANCZOS) purely to get
   sub-original-pixel boundary precision.
3. **Boxes are detected ONCE on the GT frame 0 and reused for the prediction.**
   Frame 0 of a prediction is the conditioning frame, i.e. pixel-identical to GT,
   so the same box necessarily refers to the same physical entity. Detecting
   separately on each video would let GroundingDINO latch onto a different object
   in the prediction and silently make the two mask tracks incomparable. The mask
   itself is still seeded on each video's own frame 0.

The heavy deps (torch, sam2, groundingdino, spacy) are imported lazily so this
module can be imported by the pure-numpy metric code without them.
"""

import os
import re
import json
import shutil

import cv2
import numpy as np

# ctrlworld view order, from scripts/infer_single_arm_multiview_ctrlworld.py:329
CTRLWORLD_VIEW_NAMES = ["exterior_1_left", "exterior_2_left", "wrist_left"]
# only these are static-camera views; the wrist camera moves with the arm, so
# "object moved but nothing touched it" is meaningless there
CTRLWORLD_STATIC_VIEWS = [0, 1]

GRIPPER_PROMPT = "robot gripper . robotic hand . gripper fingers . robot arm ."

# SAM2 track roles. "target" = the receptacle/destination named by the instruction
# (mug, plate, cabinet); it is tracked but excluded from grasp/levitation logic.
ROLES = ("object", "gripper", "target")
SAM2_ROLES = ("object", "target")  # gripper comes from background subtraction

# How the gripper mask is obtained. The right choice is a property of the CAMERA,
# not of the model, and picking wrong silently produces an empty/garbage arm mask:
#   "bgsub" - static camera whose FIRST frames are arm-free (ctrlworld exterior
#             views): median plate + differencing. Cannot be used when the arm is
#             already in frame at t=0, because the arm lands in the plate itself.
#   "gdino" - the arm IS in frame at t=0 (GR1 humanoid ego view: both arms sit in
#             the bottom corners from the very first frame), so a frame-0 detection
#             works and is then propagated by SAM2 like any other role.
GRIPPER_MODES = ("bgsub", "gdino")


# --------------------------------------------------------------------------- #
# gripper via background subtraction (NOT GroundingDINO)
# --------------------------------------------------------------------------- #
def robot_mask_bgsub(view_frames, bg_frames=8, diff_thr=25.0, min_area_frac=0.002,
                     open_px=3, close_px=7, keep_components=2):
    """Segment the robot arm in a STATIC-camera view by median-background subtraction.

    Measured on ctrlworld single_arm: GroundingDINO cannot be used for the gripper
    here, because at frame 0 the arm is **not yet in frame** (it enters around
    t~25) - so a frame-0 detection latches onto whatever reddish clutter is in the
    background instead. Since the exterior cameras are static and the first frames
    are arm-free, ``median(first bg_frames)`` is a clean plate and everything that
    later differs from it is the arm plus whatever it carries.

    Verified on episode_000255: the mask tracks the arm exactly across t=20..80 in
    both exterior views, and GT vs prediction give near-identical foreground
    fractions (8.5%/8.8% view0, 13.9%/12.5% view1), so this is comparable between
    the two videos - which is what the paired-GT design requires.

    Returns a list of ``T`` boolean masks.
    """
    V = view_frames.astype(np.float32)
    T, H, W = V.shape[:3]
    n_bg = max(1, min(bg_frames, T))
    plate = np.median(V[:n_bg], axis=0)

    ko = np.ones((open_px, open_px), np.uint8)
    kc = np.ones((close_px, close_px), np.uint8)
    min_area = max(1, int(min_area_frac * H * W))

    masks = []
    for t in range(T):
        d = np.abs(V[t] - plate).mean(-1)
        m = (d > diff_thr).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ko)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros((H, W), bool)
        if n > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            order = np.argsort(-areas)[:keep_components]
            for j in order:
                if areas[j] >= min_area:
                    keep |= (lab == (j + 1))
        masks.append(keep)
    return masks

# words that appear in DROID instructions but are never the manipulated object
_STOPWORD_NOUNS = {
    "left", "right", "top", "bottom", "front", "back", "side", "center", "middle",
    "table", "tabletop", "desk", "counter", "surface", "floor", "robot", "arm",
    "gripper", "hand", "place", "position", "direction", "way", "time", "thing",
}


# --------------------------------------------------------------------------- #
# video / view plumbing
# --------------------------------------------------------------------------- #
def split_views(strip, num_views):
    """Split a horizontally-tiled (T, H, W*num_views, 3) strip into views."""
    T, H, W, C = strip.shape
    if W % num_views != 0:
        raise ValueError(f"width {W} not divisible by num_views {num_views}")
    vw = W // num_views
    return [strip[:, :, v * vw : (v + 1) * vw, :] for v in range(num_views)]


def decode_strip_to_views(video_path, num_views, max_frames=None):
    """Decode an mp4 strip to a list of per-view (T, H, W, 3) uint8 RGB arrays."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"decoded 0 frames from {video_path}")
    return split_views(np.stack(frames), num_views)


def write_view_frames(view_frames, out_dir, upscale=3):
    """Write a (T,H,W,3) RGB view to ``out_dir`` as SAM2-style 00000.jpg files.

    Upscaling here is what buys mask-boundary precision: SAM2 emits masks at the
    resolution of these jpgs, and 320x192 is too coarse to measure a gap.
    """
    os.makedirs(out_dir, exist_ok=True)
    T, H, W = view_frames.shape[:3]
    dsize = (W * upscale, H * upscale) if upscale != 1 else (W, H)
    for t in range(T):
        img = view_frames[t]
        if upscale != 1:
            img = cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(out_dir, f"{t:05d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return T, (dsize[1], dsize[0])


# --------------------------------------------------------------------------- #
# instruction -> object phrases
# --------------------------------------------------------------------------- #
def object_phrases_from_instruction(instruction, nlp=None, max_phrases=3):
    """Extract candidate object phrases from a DROID instruction.

    "Put the orange marker inside the white mug" -> ["orange marker", "white mug"].
    Uses spaCy noun chunks when available (same dependency as
    ``metrics/quality/quality_control_probe.py``), else a regex fallback so this
    module still works without spaCy.
    """
    text = re.sub(r"\s+", " ", instruction.strip().lower())
    phrases = []
    if nlp is not None:
        for chunk in nlp(text).noun_chunks:
            words = [t.text for t in chunk if t.pos_ not in ("DET", "PRON", "ADP")]
            if not words:
                continue
            head = words[-1]
            if head in _STOPWORD_NOUNS:
                continue
            phrases.append(" ".join(words))
    if not phrases:
        # fallback: strip leading verb + articles, keep noun-ish tokens
        toks = [w for w in re.findall(r"[a-z]+", text) if w not in _STOPWORD_NOUNS]
        toks = [w for w in toks if w not in {"the", "a", "an", "to", "on", "in", "into",
                                             "inside", "onto", "of", "and", "put", "move",
                                             "pick", "place", "up", "down", "with", "from"}]
        if toks:
            phrases = [" ".join(toks[:2])]
    seen, out = set(), []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:max_phrases]


def build_object_prompt(phrases):
    """GroundingDINO caption form: 'orange marker . white mug .'"""
    return " . ".join(phrases) + " ." if phrases else "object ."


def split_object_target(phrases):
    """First noun phrase is the manipulated object, the rest are targets.

    "Put the orange marker inside the white mug" -> ("orange marker", ["white mug"]).
    This split matters: the *manipulated* object is the only one the grasp-follow
    and levitation predicates apply to. Folding the receptacle into the same mask
    would dilute the object's motion with a stationary mug and hide violations.
    """
    if not phrases:
        return None, []
    return phrases[0], list(phrases[1:])


# --------------------------------------------------------------------------- #
# GroundingDINO (same construction as motion_accuracy_gdino.py::Grounder)
# --------------------------------------------------------------------------- #
class Grounder:
    def __init__(self, cfg, ckpt, device):
        import torch
        import groundingdino.datasets.transforms as T
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

        a = SLConfig.fromfile(cfg)
        a.device = device
        a.bert_base_uncased_path = None
        m = build_model(a)
        sd = torch.load(ckpt, map_location="cpu")
        m.load_state_dict(clean_state_dict(sd["model"]), strict=False)
        m.eval().to(device)
        self.model = m
        self.device = device
        self._torch = torch
        self._get_phrases = get_phrases_from_posmap
        self.tr = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def detect(self, frame_path, caption, box_thr=0.25, txt_thr=0.20):
        from PIL import Image

        pil = Image.open(frame_path).convert("RGB")
        W, H = pil.size
        t, _ = self.tr(pil, None)
        cap = caption.lower().strip()
        if not cap.endswith("."):
            cap += " ."
        with self._torch.no_grad():
            out = self.model(t[None].to(self.device), captions=[cap])
        logits = out["pred_logits"].cpu().sigmoid()[0]
        boxes = out["pred_boxes"].cpu()[0]
        keep = logits.max(dim=1)[0] > box_thr
        logits, boxes = logits[keep], boxes[keep]
        tok = self.model.tokenizer(cap)
        dets = []
        for lg, bx in zip(logits, boxes):
            phrase = self._get_phrases(lg > txt_thr, tok, self.model.tokenizer)
            cx, cy, w, h = bx.tolist()
            dets.append({
                "phrase": phrase,
                "conf": float(lg.max()),
                "xyxy": [(cx - w / 2) * W, (cy - h / 2) * H,
                         (cx + w / 2) * W, (cy + h / 2) * H],
            })
        return dets, (H, W)


def nms(dets, iou_thr=0.7):
    """Greedy NMS by confidence to drop duplicate boxes for the same instance."""
    def iou(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
        return inter / ua if ua > 0 else 0.0

    dets = sorted(dets, key=lambda d: -d["conf"])
    kept = []
    for d in dets:
        if all(iou(d["xyxy"], k["xyxy"]) < iou_thr for k in kept):
            kept.append(d)
    return kept


def _drop_huge(dets, hw, max_area_frac=0.5):
    """Reject boxes covering most of the frame (GDINO's 'whole scene' failure)."""
    H, W = hw
    out = []
    for d in dets:
        x0, y0, x1, y1 = d["xyxy"]
        if (max(0.0, x1 - x0) * max(0.0, y1 - y0)) / float(H * W) <= max_area_frac:
            out.append(d)
    return out


# --------------------------------------------------------------------------- #
# tracker
# --------------------------------------------------------------------------- #
class Sim2RealTracker:
    """Holds GroundingDINO + SAM2 and produces per-view {role: [mask, ...]}."""

    def __init__(self, sam2_cfg, sam2_ckpt, gdino_cfg, gdino_ckpt, device="cuda",
                 use_spacy=True):
        import torch
        import hydra
        from hydra.core.global_hydra import GlobalHydra
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self._torch = torch
        GlobalHydra.instance().clear()
        hydra.initialize_config_module("sam2", version_base="1.2")
        img_model = build_sam2(sam2_cfg, sam2_ckpt, device=device, apply_postprocessing=False)
        self.img_pred = SAM2ImagePredictor(img_model)
        self.vid_pred = build_sam2_video_predictor(
            sam2_cfg, sam2_ckpt, device=device, apply_postprocessing=False)
        self.grounder = Grounder(gdino_cfg, gdino_ckpt, device)
        self.device = device
        self.nlp = None
        self._appearance = None
        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:  # noqa: BLE001 - spaCy is optional
                print(f"[track] spaCy unavailable ({e!r}); using regex noun fallback")

    def appearance(self):
        """Lazily built CLIP scorer, shared across views/episodes (optional)."""
        if self._appearance is None:
            try:
                self._appearance = AppearanceScorer(device=self.device)
            except Exception as e:  # noqa: BLE001 - CLIP is optional
                print(f"[track] CLIP unavailable ({e!r}); referent falls back to "
                      f"geometry only")
                self._appearance = False
        return self._appearance or None

    # -- detection ---------------------------------------------------------- #
    def detect_roles(self, frame0_path, object_prompt, target_prompt=None,
                     box_thr=0.25, max_object_boxes=1, max_target_boxes=1,
                     gripper_prompt=None, max_gripper_boxes=2):
        """Detect object / target boxes on one frame.

        The gripper is only detected when ``gripper_prompt`` is given, i.e. in
        ``gripper_mode="gdino"`` (arm already in frame at t=0 - GR1 ego view). For
        the ctrlworld exterior views it stays ``None`` and the arm comes from
        :func:`robot_mask_bgsub` instead - see its docstring for why a frame-0
        detection cannot work there.

        Returns ``({role: [xyxy, ...]}, {role: [det, ...]}, (H, W))``.
        """
        obj_dets, hw = self.grounder.detect(frame0_path, object_prompt, box_thr=box_thr)
        obj_dets = _drop_huge(nms(obj_dets), hw)[:max_object_boxes]
        tgt_dets = []
        if target_prompt:
            raw, _ = self.grounder.detect(frame0_path, target_prompt, box_thr=box_thr)
            tgt_dets = _drop_huge(nms(raw), hw)[:max_target_boxes]
        dets = {"object": obj_dets, "target": tgt_dets}
        if gripper_prompt:
            raw, _ = self.grounder.detect(frame0_path, gripper_prompt, box_thr=box_thr)
            dets["gripper"] = _drop_huge(nms(raw), hw)[:max_gripper_boxes]
        boxes = {k: [d["xyxy"] for d in v] for k, v in dets.items()}
        return boxes, dets, hw

    # -- tracking ----------------------------------------------------------- #
    def track(self, frames_dir, boxes_by_role, roles=SAM2_ROLES):
        """Seed SAM2 from ``boxes_by_role`` on frame 0 of ``frames_dir``, propagate.

        Returns ``({role: [bool mask per frame]}, (H, W))`` for ``roles``. A role
        with no boxes yields all-False masks so downstream code never has to
        special-case it. ``roles`` is a parameter (not always ``SAM2_ROLES``)
        because in ``gripper_mode="gdino"`` the arm is tracked here too.
        """
        frame0 = os.path.join(frames_dir, "00000.jpg")
        n_frames = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])

        flat_boxes, id_role = [], {}
        oid = 1
        for role in roles:
            for bx in boxes_by_role.get(role, []):
                flat_boxes.append(bx)
                id_role[oid] = role
                oid += 1

        img = cv2.cvtColor(cv2.imread(frame0), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        if not flat_boxes:
            return ({r: [np.zeros((H, W), bool) for _ in range(n_frames)]
                     for r in roles}, (H, W))

        self.img_pred.set_image(img)
        seeds = {}
        for i, bx in enumerate(flat_boxes, start=1):
            m, _, _ = self.img_pred.predict(
                point_coords=None, point_labels=None,
                box=np.asarray(bx, dtype=np.float32)[None, :], multimask_output=False)
            seeds[i] = m[0].astype(bool)

        with self._torch.inference_mode():
            state = self.vid_pred.init_state(
                video_path=frames_dir, offload_video_to_cpu=True,
                offload_state_to_cpu=True, async_loading_frames=False)
            for i, m in seeds.items():
                self.vid_pred.add_new_mask(inference_state=state, frame_idx=0, obj_id=i, mask=m)
            per_frame = {}
            for fidx, oids, logits in self.vid_pred.propagate_in_video(state):
                per_frame[fidx] = {
                    o: (logits[k] > 0.0).cpu().numpy().squeeze() for k, o in enumerate(oids)
                }
            self.vid_pred.reset_state(state)

        out = {r: [] for r in roles}
        for fidx in sorted(per_frame):
            for role in roles:
                ids = [i for i, r in id_role.items() if r == role]
                u = np.zeros((H, W), bool)
                for i in ids:
                    m = per_frame[fidx].get(i)
                    if m is not None:
                        u |= np.asarray(m, dtype=bool)
                out[role].append(u)
        for role in roles:
            if not out[role]:
                out[role] = [np.zeros((H, W), bool) for _ in range(n_frames)]
        return out, (H, W)


# --------------------------------------------------------------------------- #
def _upscale_masks(masks, upscale):
    if upscale == 1:
        return masks
    out = []
    for m in masks:
        h, w = m.shape
        out.append(cv2.resize(m.astype(np.uint8), (w * upscale, h * upscale),
                              interpolation=cv2.INTER_NEAREST).astype(bool))
    return out


def motion_energy(view_frames):
    """Per-pixel accumulated inter-frame |difference|, normalized to [0, 1].

    Used to disambiguate same-category candidates: a cluttered GR1 table holds
    several green vegetables, and GroundingDINO ranks them by text confidence,
    which has no idea which one the instruction is about.
    """
    V = np.asarray(view_frames, dtype=np.float32)
    if len(V) < 2:
        return np.zeros(V.shape[1:3], np.float32)
    e = np.abs(np.diff(V, axis=0)).mean(-1).sum(0)
    m = float(e.max())
    return (e / m) if m > 0 else e


def _mask_travel(masks):
    """Net centroid displacement of a tracked mask, in units of its own diameter.

    NET start-to-end displacement, not accumulated path length: path length sums
    per-frame jitter, so a noisy mask on a static distractor can out-score the real
    referent. Scale-free (diameter-normalised) so candidates of different sizes are
    comparable, and frames where the mask drops out are skipped rather than counted
    as a jump.
    """
    cys, cxs, areas = [], [], []
    for m in masks:
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        cys.append(ys.mean())
        cxs.append(xs.mean())
        areas.append(m.sum())
    if len(cys) < 2:
        return 0.0
    diam = max(1.0, float(np.sqrt(np.median(areas))))
    # median of the first/last few centroids: robust to a bad endpoint frame
    k = max(1, min(3, len(cys) // 4))
    y0, x0 = np.median(cys[:k]), np.median(cxs[:k])
    y1, x1 = np.median(cys[-k:]), np.median(cxs[-k:])
    return float(np.hypot(y1 - y0, x1 - x0) / diam)


def _mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def _min_gap_series(obj_masks, arm_masks):
    """Per-frame boundary gap object->arm, in object-diameter units.

    Same quantity the metric later thresholds for contact, computed here only to
    decide WHICH candidate the arm interacts with.
    """
    gaps = []
    n = min(len(obj_masks), len(arm_masks))
    for i in range(n):
        o, a = obj_masks[i], arm_masks[i]
        if not o.any() or not a.any():
            gaps.append(np.nan)
            continue
        dt = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 3)
        diam = max(1.0, float(np.sqrt(o.sum())))
        gaps.append(float(dt[o].min()) / diam)
    return np.asarray(gaps, dtype=np.float64)


class AppearanceScorer:
    """CLIP check: does this tracked region actually LOOK like the named object?

    Geometry alone cannot resolve the referent on the GR1 ego view. Measured on two
    episodes with *mirror-image* geometry - each has one candidate continuously
    touching the arm and one near-but-not-touching it, and the correct answer is the
    touching one in episode_000829 but the non-touching one in episode_001065 - so
    any contact/travel threshold that fixes one breaks the other. Appearance is the
    only independent signal available.
    """

    def __init__(self, name="ViT-B/32", device="cuda"):
        import clip
        self._torch = __import__("torch")
        self.model, self.preprocess = clip.load(name, device=device)
        self.tokenize = clip.tokenize
        self.device = device

    def score(self, frames, masks, positive, negatives, n_samples=4, pad=0.35):
        """Mean P(positive) over ``n_samples`` frames, softmax'd against negatives.

        ``masks`` may be at a different (upscaled) resolution than ``frames``; the
        box is rescaled rather than assumed to match. Getting this wrong silently
        crops outside the image and returns NaN for every candidate.
        """
        from PIL import Image
        texts = [positive] + list(negatives)
        n = min(len(frames), len(masks))
        if n == 0:
            return float("nan")
        idx = sorted({int(round(i * (n - 1) / max(1, n_samples - 1)))
                      for i in range(n_samples)})
        fh, fw = np.asarray(frames[0]).shape[:2]
        mh, mw = masks[0].shape[:2]
        sy, sx = fh / float(mh), fw / float(mw)
        crops = []
        for t in idx:
            m = masks[t]
            if not m.any():
                continue
            ys, xs = np.nonzero(m)
            y0, y1 = ys.min() * sy, ys.max() * sy
            x0, x1 = xs.min() * sx, xs.max() * sx
            py = (y1 - y0 + 1) * pad + 4
            px = (x1 - x0 + 1) * pad + 4
            sub = np.asarray(frames[t])[
                max(0, int(y0 - py)):min(fh, int(y1 + py) + 1),
                max(0, int(x0 - px)):min(fw, int(x1 + px) + 1)]
            if sub.size:
                crops.append(self.preprocess(Image.fromarray(sub)))
        if not crops:
            return float("nan")
        with self._torch.no_grad():
            f = self.model.encode_image(self._torch.stack(crops).to(self.device))
            t = self.model.encode_text(self.tokenize(texts).to(self.device))
            f = f / f.norm(dim=-1, keepdim=True)
            t = t / t.norm(dim=-1, keepdim=True)
            p = (100.0 * f @ t.T).softmax(-1)[:, 0]
        return float(p.mean().cpu())


APPEARANCE_NEGATIVES = ["a robotic gripper hand", "a wooden table",
                        "a kitchen appliance", "a blurry background"]


def pick_referent(tracker, frames_dir, dets, motion, upscale, prefilter=4,
                  role="object", arm_masks=None, arm_iou_veto=0.5,
                  max_area_frac=0.04, tau_gap=0.35, appearance=None,
                  object_text=None, frames=None, appearance_band=0.5):
    """Pick which same-category candidate the instruction actually refers to.

    GroundingDINO returns every candidate and ranks them by text confidence, which
    has no idea which one the instruction is about. On the cluttered GR1 table that
    put the *static* pepper first (conf 0.43) on episode_000829 while the bok choy
    actually being manipulated scored 0.17; tracking the distractor left GT with zero
    contact frames, i.e. a pick-up task scored as "no interaction".

    Two hard filters run first, each earned from a real false result:

    * ``max_area_frac`` - a manipulable object is small. Without it a box covering
      the table edge / chair (13% of frame) won episode_001065 and, being static and
      adjacent to the arm, satisfied every term for a false-perfect 1.000.
    * ``arm_iou_veto`` - a candidate that IS the arm can never be the manipulated
      object. On episode_001065 the "slice" box covered the hand (IoU 0.88, area 5.1%
      vs the arm's 5.4%), so the gap was measured to itself, always 0, and the
      episode scored a false-clean 0.975. A *grasped* object stays a small island
      inside the arm mask and keeps a low IoU (0.13 on episode_000829), so this does
      not reject the legitimate case.

    Ranking is then **interaction-first**: contact fraction, then travel, then text
    confidence. CLIP appearance is still computed and logged as ``look`` for
    diagnostics, but it is deliberately NOT used to rank, because on this data it is
    wrong in a *systematic* direction. The manipulated object is usually already held
    and partly occluded by the hand, so CLIP consistently prefers a clean, static
    twin sitting on the table: ranking by appearance across 6 locked-waist GR1
    episodes left 5 of 6 with zero GT contact (rejected, unmeasurable), versus 5 of 6
    measurable when ranking by interaction. An unmeasurable metric is not a safe
    metric, so contact leads.

    The one case interaction gets wrong is episode_001065, where a dragged crumb
    touches the arm more than the target toast does; that episode scores a
    false-clean 0.975 and is the known residual failure. It is contained by the two
    hard filters above plus the ``object_is_arm`` validity check, and ``look`` is
    reported alongside so this class of pick stays auditable.

    This reads GT only, and the winning box is reused to seed the prediction exactly
    as before, so it resolves identity without leaking how good the prediction is.
    """
    if not dets:
        return dets, []
    pool = sorted(dets, key=lambda d: -d["conf"])[:max(1, prefilter)]

    ranked = []
    for d in pool:
        masks, _ = tracker.track(frames_dir, {role: [d["xyxy"]]}, roles=(role,))
        mk = masks[role]
        area = float(np.mean([m.mean() for m in mk]))
        iou, contact_frac = 0.0, float("nan")
        if arm_masks is not None:
            n = min(len(mk), len(arm_masks))
            if n:
                iou = float(np.median([_mask_iou(mk[i], arm_masks[i])
                                       for i in range(n)]))
                gap = _min_gap_series(mk, arm_masks)
                if np.isfinite(gap).any():
                    contact_frac = float(np.nanmean(gap < tau_gap))
        look = float("nan")
        if appearance is not None and frames is not None and object_text:
            look = appearance.score(frames, mk, object_text, APPEARANCE_NEGATIVES)
        ranked.append({
            "det": d, "conf": d["conf"], "phrase": d["phrase"],
            "area_frac": area, "arm_iou": iou, "look": look,
            "travel": _mask_travel(mk), "contact_frac": contact_frac,
            "too_big": area > max_area_frac, "is_arm": iou > arm_iou_veto,
        })

    ok = [r for r in ranked if not r["too_big"] and not r["is_arm"]] or ranked

    def key(r):
        cf = 0.0 if np.isnan(r["contact_frac"]) else r["contact_frac"]
        return (-cf, -r["travel"], -r["conf"])

    ok.sort(key=key)
    ranked.sort(key=lambda r: (r["too_big"] or r["is_arm"], key(r)))
    return [ok[0]["det"]], ranked


def pick_acting_arms(dets, motion, upscale, rel_thr=0.4):
    """Keep only the arm(s) that actually move; drop the idle one.

    In the GR1 ego view both arms are in frame from t=0, so detecting "robot arm"
    returns both and their union becomes one blob spanning the table. A bloated arm
    mask makes the gripper-object gap collapse to 0 everywhere, which silently
    fakes contact for any object it happens to sweep past.

    Motion energy separates them cleanly (measured on episode_000829: acting right
    arm 0.212 vs idle left arm 0.101). Arms within ``rel_thr`` of the best are kept
    too, so bimanual episodes do not lose the second hand.
    """
    if len(dets) <= 1:
        return dets, []
    scored = []
    for d in dets:
        x0, y0, x1, y1 = [int(v / upscale) for v in d["xyxy"]]
        sub = motion[max(0, y0):y1 + 1, max(0, x0):x1 + 1]
        scored.append((float(sub.mean()) if sub.size else 0.0, d))
    scored.sort(key=lambda t: -t[0])
    best = scored[0][0]
    keep = [d for s, d in scored if best <= 0 or s >= rel_thr * best]
    return keep, [{"phrase": d["phrase"], "conf": d["conf"], "motion": s,
                   "kept": (best <= 0 or s >= rel_thr * best)} for s, d in scored]


def track_episode(tracker, gt_video, pred_video, work_dir, instruction,
                  num_views=3, upscale=3, max_frames=None, box_thr=0.25,
                  bg_frames=8, diff_thr=25.0, gripper_mode="bgsub",
                  view_names=None, static_views=None, select_referent=False,
                  referent_box_thr=0.15, referent_prefilter=4,
                  referent_tau_gap=0.35, use_appearance=True):
    """Track object / target / gripper in GT and pred.

    Object/target detection happens ONCE per view on the GT frame 0; the boxes are
    reused to seed the prediction, since frame 0 is the shared conditioning frame.

    ``gripper_mode`` selects how the arm is segmented, which depends on the camera
    (see :data:`GRIPPER_MODES`): ``"bgsub"`` for a static camera whose first frames
    are arm-free (ctrlworld exterior), ``"gdino"`` when the arm is already in frame
    at t=0 (GR1 humanoid ego view) - there the arm is detected on frame 0 and
    propagated by SAM2 like the other roles.

    ``view_names`` / ``static_views`` default to the ctrlworld layout; pass them
    explicitly for other embodiments (e.g. a single ego view).

    ``select_referent`` resolves which same-category candidate the instruction means
    by tracked participation instead of text confidence (see :func:`pick_referent`);
    required on cluttered scenes like the GR1 table.

    Returns:
        {"views": [{"gt": {role: [mask]}, "pred": {role: [mask]}, "hw": (H, W),
                    "dets": {...}, "frames_dir": {...}}, ...],
         "num_frames": T, ...}
    """
    if gripper_mode not in GRIPPER_MODES:
        raise ValueError(f"gripper_mode must be one of {GRIPPER_MODES}")
    if view_names is None:
        view_names = CTRLWORLD_VIEW_NAMES
    if static_views is None:
        static_views = CTRLWORLD_STATIC_VIEWS

    phrases = object_phrases_from_instruction(instruction, nlp=tracker.nlp)
    obj_phrase, target_phrases = split_object_target(phrases)
    object_prompt = build_object_prompt([obj_phrase] if obj_phrase else [])
    target_prompt = build_object_prompt(target_phrases) if target_phrases else None
    print(f"[track] instruction={instruction!r}")
    print(f"[track] object prompt -> {object_prompt!r} | target prompt -> {target_prompt!r}")
    print(f"[track] gripper mode -> {gripper_mode}")

    gt_views = decode_strip_to_views(gt_video, num_views, max_frames=max_frames)
    pred_views = decode_strip_to_views(pred_video, num_views, max_frames=max_frames)
    T = min(gt_views[0].shape[0], pred_views[0].shape[0])

    out_views = []
    for v in range(num_views):
        gt_dir = os.path.join(work_dir, f"view{v}_gt")
        pred_dir = os.path.join(work_dir, f"view{v}_pred")
        write_view_frames(gt_views[v][:T], gt_dir, upscale=upscale)
        write_view_frames(pred_views[v][:T], pred_dir, upscale=upscale)
        sam2_roles = SAM2_ROLES + (("gripper",) if gripper_mode == "gdino" else ())
        boxes, dets, hw = tracker.detect_roles(
            os.path.join(gt_dir, "00000.jpg"), object_prompt,
            target_prompt=target_prompt,
            box_thr=referent_box_thr if select_referent else box_thr,
            max_object_boxes=8 if select_referent else 1,
            gripper_prompt=GRIPPER_PROMPT if gripper_mode == "gdino" else None)

        referent = []
        referent_pick = None
        if select_referent:
            motion = motion_energy(gt_views[v][:T])
            # Candidate selection runs at NATIVE resolution on its own frame dir.
            # Upscaling exists only to sharpen the final gap measurement; probing
            # ~5 candidates at 3x cost ~7 SAM2 propagations over 2304x1296 and made
            # one episode take minutes. Native res is ~9x cheaper per propagation
            # and is plenty to decide *which* object is the referent.
            sel_dir = os.path.join(work_dir, f"view{v}_sel")
            write_view_frames(gt_views[v][:T], sel_dir, upscale=1)
            sel_scale = 1.0 / upscale

            def _down(box):
                return [float(c) * sel_scale for c in box]

            arm_gt = None
            if gripper_mode == "gdino" and dets.get("gripper"):
                if len(dets["gripper"]) > 1:
                    dets["gripper"], arm_rank = pick_acting_arms(
                        dets["gripper"], motion, upscale)
                    for r in arm_rank:
                        print(f"[track]   arm cand {r['phrase']!r} conf={r['conf']:.2f} "
                              f"motion={r['motion']:.3f} "
                              f"{'kept' if r['kept'] else 'dropped (idle)'}")
                am, _ = tracker.track(
                    sel_dir, {"gripper": [_down(d["xyxy"]) for d in dets["gripper"]]},
                    roles=("gripper",))
                arm_gt = am["gripper"]

            sel_dets = [dict(d, xyxy=_down(d["xyxy"])) for d in dets["object"]]
            picked, referent = pick_referent(
                tracker, sel_dir, sel_dets, motion, 1,
                prefilter=referent_prefilter, arm_masks=arm_gt,
                tau_gap=referent_tau_gap,
                appearance=tracker.appearance() if use_appearance else None,
                object_text=(f"a photo of {obj_phrase}" if obj_phrase else None),
                frames=gt_views[v][:T])
            # map back: keep the ORIGINAL (upscaled) box of the winner
            keep_i = sel_dets.index(picked[0])
            dets["object"] = [dets["object"][keep_i]]
            for r in referent:
                mark = "<-- picked" if r["det"] is picked[0] else ""
                flag = (" VETO(is arm)" if r["is_arm"]
                        else " VETO(too big)" if r["too_big"] else "")
                cf, lk = r["contact_frac"], r["look"]
                print(f"[track]   obj cand {r['phrase']!r} conf={r['conf']:.2f} "
                      f"area={r['area_frac']*100:.2f}% arm_iou={r['arm_iou']:.2f} "
                      f"look={'nan' if np.isnan(lk) else format(lk, '.2f')} "
                      f"contact={'nan' if np.isnan(cf) else format(cf, '.2f')} "
                      f"travel={r['travel']:.2f}{flag} {mark}")
            boxes = {k: [d["xyxy"] for d in v_] for k, v_ in dets.items()}
            won = referent[0] if referent and referent[0]["det"] is picked[0] else \
                next((r for r in referent if r["det"] is picked[0]), None)
            if won is not None:
                def _f(x):
                    return None if x is None or np.isnan(x) else float(x)
                referent_pick = {
                    "phrase": won["phrase"], "conf": _f(won["conf"]),
                    "look": _f(won["look"]), "contact_frac": _f(won["contact_frac"]),
                    "travel": _f(won["travel"]), "n_candidates": len(referent),
                }
            shutil.rmtree(sel_dir, ignore_errors=True)

        name = view_names[v] if v < len(view_names) else f"view_{v}"
        desc = " ".join(
            f"{r}=[" + (", ".join(f"{d['phrase']}({d['conf']:.2f})" for d in dets[r]) or "-") + "]"
            for r in sam2_roles)
        print(f"[track] view {v} ({name}) {desc}")

        gt_masks, _ = tracker.track(gt_dir, boxes, roles=sam2_roles)
        pred_masks, _ = tracker.track(pred_dir, boxes, roles=sam2_roles)

        if gripper_mode == "bgsub":
            # gripper: background subtraction at native res, then upscale to match
            gt_masks["gripper"] = _upscale_masks(
                robot_mask_bgsub(gt_views[v][:T], bg_frames=bg_frames, diff_thr=diff_thr),
                upscale)
            pred_masks["gripper"] = _upscale_masks(
                robot_mask_bgsub(pred_views[v][:T], bg_frames=bg_frames, diff_thr=diff_thr),
                upscale)

        cov = {r: float(np.mean([m.mean() for m in gt_masks[r]]) * 100) for r in ROLES}
        print(f"[track]   GT coverage%: " + " ".join(f"{r}={cov[r]:.2f}" for r in ROLES))

        out_views.append({
            "view": v,
            "view_name": name,
            "is_static": v in static_views,
            "hw": hw,
            "gt": gt_masks,
            "pred": pred_masks,
            "dets": {k: [{"phrase": d["phrase"], "conf": d["conf"], "xyxy": d["xyxy"]}
                         for d in dets[k]] for k in dets},
            "frames_dir": {"gt": gt_dir, "pred": pred_dir},
            "referent_pick": referent_pick,
        })

    return {
        "views": out_views,
        "num_frames": int(T),
        "num_views": num_views,
        "upscale": upscale,
        "object_prompt": object_prompt,
        "target_prompt": target_prompt,
        "object_phrases": phrases,
        "gripper_mode": gripper_mode,
        "select_referent": bool(select_referent),
        "view_names": [w["view_name"] for w in out_views],
        # kept so a cached tracks.npz can regenerate its frames for visualisation
        "gt_video": os.path.abspath(gt_video),
        "pred_video": os.path.abspath(pred_video),
    }


def save_tracks(tracks, path):
    """Persist masks as bit-packed npz so calibration never re-runs the tracker.

    Tracking one episode costs ~100 s of GPU; threshold sweeps and violation mining
    need to re-read the same masks dozens of times, so the masks - not the scores -
    are the artifact worth caching. ``packbits`` keeps a 130-frame 3-view episode
    at ~1 MB instead of ~60 MB.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrays, meta_views = {}, []
    for vw in tracks["views"]:
        v = vw["view"]
        for tag in ("gt", "pred"):
            for role in ROLES:
                masks = vw[tag][role]
                arrays[f"v{v}_{tag}_{role}"] = np.packbits(
                    np.stack(masks).astype(bool), axis=None)
        meta_views.append({k: vw[k] for k in
                           ("view", "view_name", "is_static", "hw", "dets",
                            "frames_dir")
                           if k in vw})
        meta_views[-1]["referent_pick"] = vw.get("referent_pick")
    mh, mw = tracks["views"][0]["gt"][ROLES[0]][0].shape[:2]
    meta = {k: tracks[k] for k in tracks if k != "views"}
    meta["views"] = meta_views
    meta["mask_hw"] = [int(mh), int(mw)]
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)
    return path


def load_tracks(path):
    """Inverse of :func:`save_tracks`."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    H, W = meta["mask_hw"]
    T = meta["num_frames"]
    n = T * H * W
    views = []
    for mv in meta["views"]:
        v = mv["view"]
        vw = dict(mv)
        for tag in ("gt", "pred"):
            vw[tag] = {}
            for role in ROLES:
                flat = np.unpackbits(z[f"v{v}_{tag}_{role}"], count=n).astype(bool)
                vw[tag][role] = list(flat.reshape(T, H, W))
        views.append(vw)
    out = {k: meta[k] for k in meta if k not in ("views", "mask_hw")}
    out["views"] = views
    return out


def rebuild_frames(tracks, video_gt, video_pred, work_dir):
    """Re-extract the upscaled frames a cached ``tracks.npz`` refers to.

    Frames are derivable from the mp4s while masks are not, so batch runs keep only
    ``tracks.npz`` (~1 MB/episode instead of ~35 MB) and call this when a panel or
    clip actually needs pixels. Mutates and returns ``tracks`` with the new dirs.
    """
    nv = tracks["num_views"]
    up = tracks["upscale"]
    T = tracks["num_frames"]
    gt_views = decode_strip_to_views(video_gt, nv, max_frames=T)
    pred_views = decode_strip_to_views(video_pred, nv, max_frames=T)
    for vw in tracks["views"]:
        v = vw["view"]
        gt_dir = os.path.join(work_dir, f"view{v}_gt")
        pred_dir = os.path.join(work_dir, f"view{v}_pred")
        write_view_frames(gt_views[v][:T], gt_dir, upscale=up)
        write_view_frames(pred_views[v][:T], pred_dir, upscale=up)
        vw["frames_dir"] = {"gt": gt_dir, "pred": pred_dir}
    return tracks


def cleanup_work(work_dir):
    shutil.rmtree(work_dir, ignore_errors=True)
