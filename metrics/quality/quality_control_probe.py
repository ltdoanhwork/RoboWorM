"""WorldScore Quality + Controllability probe: subjective_quality, content_alignment,
object_control — on our paired robot sample.

Metrics (same models/definitions as WorldScore's aspect_info):

  subjective_quality  = mean over frames of  clipiqa+   (CLIP-IQA+, [0,1] higher better)
                      + mean over frames of  laion_aes  (LAION aesthetic, [0,10] higher better)
                        -> WorldScore averages the two normalized halves.
  content_alignment   = mean over frames of  clipscore(frame, task prompt)  ([0,2.5] higher better)
  object_control      = mean over frames of  detection-rate: fraction of the expected
                        object phrases that GroundingDINO finds in the frame ([0,1]).

Each raw score is ALSO normalized to [0,1] with WorldScore's own constants
(z-score->range for clipiqa+/clip_aesthetic/clip_score; min-max for object_detection)
so the numbers are directly comparable to published WorldScore values.

Frames are processed one at a time (WorldScore stacks all frames at 512x512, which
OOMs on our long clips). Use --stride to subsample.
"""
import os
import sys
import json
import argparse
import re

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ---- WorldScore normalization constants (verbatim from benchmark/utils/utils.py) ----
NORM = {
    "clip_iqa+":       {"avg": 0.5842, "std": 0.0441, "z_max": 1.8703, "z_min": -1.8342,
                        "range": [0.25, 0.75], "higher_is_better": True},
    "clip_aesthetic":  {"avg": 5.5952, "std": 0.2561, "z_max": 1.9741, "z_min": -2.9342,
                        "range": [0.25, 0.75], "higher_is_better": True},
    "clip_score":      {"avg": 26.67, "std": 0.8875, "z_max": 1.2741, "z_min": -1.5950,
                        "range": [0.25, 0.75], "higher_is_better": True},
    "object_detection": {"empirical_max": 1, "empirical_min": 0, "higher_is_better": True},
}


def norm_z(score, k):
    c = NORM[k]
    x = ((score - c["avg"]) / c["std"] - c["z_min"]) / (c["z_max"] - c["z_min"])
    if not c["higher_is_better"]:
        x = 1 - x
    v = c["range"][0] + (c["range"][1] - c["range"][0]) * x
    return float(max(0.0, min(1.0, v)))


def norm_minmax(score, k):
    c = NORM[k]
    s = max(c["empirical_min"], min(c["empirical_max"], score))
    v = (s - c["empirical_min"]) / (c["empirical_max"] - c["empirical_min"])
    if not c["higher_is_better"]:
        v = 1 - v
    return float(v)


def decode(video, out_dir, stride=1):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    i = saved = 0
    paths = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            p = os.path.join(out_dir, f"{saved:05d}.jpg")
            cv2.imwrite(p, fr)
            paths.append(p)
            saved += 1
        i += 1
    cap.release()
    return paths


# --------------------------------------------------------------------------- #
# subjective_quality + content_alignment  (pyiqa: clipiqa+, laion_aes, clipscore)
# --------------------------------------------------------------------------- #
class IQA:
    """Mirrors WorldScore's metric classes exactly:
      clip_iqa+      -> pyiqa 'clipiqa+'   (IQAPytorchMetric, Resize512+ToTensor)
      clip_aesthetic -> pyiqa 'laion_aes'  (IQAPytorchMetric, Resize512+ToTensor)
      clip_score     -> torchmetrics CLIPScore(clip-vit-base-patch16), range [0,100],
                        fed uint8 (img*255) like WorldScore's torchmetrics wrapper.
    """

    def __init__(self, device):
        import pyiqa
        from torchmetrics.multimodal.clip_score import CLIPScore
        self.device = device
        self.pre = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
        self.clipiqa = pyiqa.create_metric("clipiqa+").to(device)
        self.aes = pyiqa.create_metric("laion_aes").to(device)
        self.clipscore = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)

    def _img(self, path):
        return self.pre(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def frame_scores(self, path, caption):
        t = self._img(path)
        iqa = float(self.clipiqa(t).item())
        aes = float(self.aes(t).item())
        u8 = (t[0] * 255).to(torch.uint8)
        cs = float(self.clipscore(u8, caption).detach().item())
        return iqa, aes, cs


# --------------------------------------------------------------------------- #
# object_control (GroundingDINO detection rate)
# --------------------------------------------------------------------------- #
class Detector:
    """Mirrors WorldScore's ObjectDetectionMetric (box/text threshold 0.4, phrase matching)."""

    def __init__(self, cfg, ckpt, device, box_thr=0.4, txt_thr=0.4):
        import groundingdino.datasets.transforms as T
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict
        import spacy
        a = SLConfig.fromfile(cfg)
        a.device = device
        a.bert_base_uncased_path = None
        m = build_model(a)
        sd = torch.load(ckpt, map_location="cpu")
        m.load_state_dict(clean_state_dict(sd["model"]), strict=False)
        m.eval().to(device)
        self.model, self.device = m, device
        self.box_thr, self.txt_thr = box_thr, txt_thr
        self.tr = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.nlp = spacy.load("en_core_web_sm")
        from groundingdino.util.utils import get_phrases_from_posmap
        self._gp = get_phrases_from_posmap

    @torch.no_grad()
    def phrases(self, path, caption):
        pil = Image.open(path).convert("RGB")
        t, _ = self.tr(pil, None)
        cap = caption.lower().strip()
        if not cap.endswith("."):
            cap += "."
        out = self.model(t[None].to(self.device), captions=[cap])
        logits = out["pred_logits"].cpu().sigmoid()[0]
        keep = logits.max(dim=1)[0] > self.box_thr
        logits = logits[keep]
        tok = self.model.tokenizer(cap)
        return [self._gp(lg > self.txt_thr, tok, self.model.tokenizer) for lg in logits]

    def _nouns(self, phrase):
        doc = self.nlp(phrase)
        nouns = [t.text.lower() for t in doc
                 if t.pos_ in ("NOUN", "PROPN") or t.tag_ in ("NN", "NNS", "NNP", "NNPS")]
        return nouns

    def rate(self, path, prompt_list):
        """Fraction of expected phrases detected (WorldScore two-pass matching)."""
        caption = ", ".join(prompt_list)
        preds = [re.sub(r"\s*([_\-*/+])\s*", r"\1", re.sub(r"\(.*?\)", "", p).strip().lower())
                 for p in self.phrases(path, caption)]
        remaining = set(prompt_list)
        seen = set()
        count = 0
        # pass 1: exact
        for r in preds:
            if r in seen or r not in remaining:
                continue
            seen.add(r)
            count += 1
            remaining.discard(r)
        # pass 2: partial (noun overlap / subword)
        for r in preds:
            if r in seen or not remaining:
                continue
            if "##" in r:
                rr = r.replace("##", "")
                hit = next((p for p in remaining if rr and rr in p), None)
            else:
                rn = self._nouns(r)
                hit = next((p for p in remaining
                            if any(n in self._nouns(p) for n in rn)), None)
            if hit:
                count += 1
                remaining.discard(hit)
                seen.add(r)
        return min(count / max(1, len(prompt_list)), 1.0), preds


# --------------------------------------------------------------------------- #
def probe(video, tag, out_dir, iqa, det, caption, objects, stride):
    work = os.path.join(out_dir, f"_{tag}_frames")
    frames = decode(video, work, stride=stride)
    iqa_l, aes_l, cs_l, od_l = [], [], [], []
    sample_preds = None
    for j, f in enumerate(frames):
        a, b, c = iqa.frame_scores(f, caption)
        iqa_l.append(a); aes_l.append(b); cs_l.append(c)
        r, preds = det.rate(f, objects)
        od_l.append(r)
        if j == 0:
            sample_preds = preds
    import shutil
    shutil.rmtree(work, ignore_errors=True)

    m_iqa, m_aes, m_cs, m_od = (float(np.mean(x)) for x in (iqa_l, aes_l, cs_l, od_l))
    n_iqa, n_aes = norm_z(m_iqa, "clip_iqa+"), norm_z(m_aes, "clip_aesthetic")
    return {
        "tag": tag, "video": video, "n_frames_eval": len(frames), "stride": stride,
        "caption": caption, "objects": objects,
        "frame0_detected_phrases": sample_preds,
        "subjective_quality": {
            "clip_iqa+": m_iqa, "clip_aesthetic": m_aes,
            "clip_iqa+_norm": n_iqa, "clip_aesthetic_norm": n_aes,
            "score_normalized": (n_iqa + n_aes) / 2,  # WorldScore averages the two
        },
        "content_alignment": {
            "clip_score": m_cs, "score_normalized": norm_z(m_cs, "clip_score"),
        },
        "object_control": {
            "object_detection": m_od, "score_normalized": norm_minmax(m_od, "object_detection"),
        },
        "per_frame": {"clip_iqa+": iqa_l, "clip_aesthetic": aes_l,
                      "clip_score": cs_l, "object_detection": od_l},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True, help="tag=path ...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--caption", required=True, help="task prompt for clip_score")
    ap.add_argument("--objects", required=True,
                    help="comma-separated expected objects, e.g. 'robot arm, cardboard box'")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--gdino_cfg", required=True)
    ap.add_argument("--gdino_ckpt", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    objects = [o.strip().lower() for o in args.objects.split(",") if o.strip()]

    iqa = IQA(device)
    det = Detector(args.gdino_cfg, args.gdino_ckpt, device)

    res = {}
    for spec in args.clips:
        tag, path = spec.split("=", 1)
        print(f"=== {tag}", flush=True)
        res[tag] = probe(path, tag, args.out, iqa, det, args.caption, objects, args.stride)
        r = res[tag]
        print(f"  subjective_quality: clipiqa+={r['subjective_quality']['clip_iqa+']:.4f} "
              f"aes={r['subjective_quality']['clip_aesthetic']:.3f} "
              f"-> norm={r['subjective_quality']['score_normalized']:.4f}")
        print(f"  content_alignment : clip_score={r['content_alignment']['clip_score']:.3f} "
              f"-> norm={r['content_alignment']['score_normalized']:.4f}")
        print(f"  object_control    : det_rate={r['object_control']['object_detection']:.4f} "
              f"-> norm={r['object_control']['score_normalized']:.4f}")
        print(f"  frame0 phrases: {r['frame0_detected_phrases']}", flush=True)

    with open(os.path.join(args.out, "quality_control_summary.json"), "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "per_frame"}
                   for k, v in res.items()}, f, indent=2)
    with open(os.path.join(args.out, "quality_control_perframe.json"), "w") as f:
        json.dump({k: v["per_frame"] for k, v in res.items()}, f)
    print("\nWrote", os.path.join(args.out, "quality_control_summary.json"))


if __name__ == "__main__":
    main()
