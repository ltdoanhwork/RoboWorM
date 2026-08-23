"""Reference-free metrics + summary table for the AUGMENT benchmark set.

The augmented inputs (edited object / free prompt) have NO ground-truth video to
compare against, so we use reference-free signals that still separate a good
world model from a bad one:

  - motion_magnitude   : mean |frame_{t+1} - frame_t| over the clip (0-255 scale).
                         Too low => frozen / under-moving; reasonable => the model
                         actually animates the scene.
  - temporal_consistency: mean SSIM between consecutive frames. High (~1) = smooth;
                         a melting / flickering generation drops sharply.
  - clip_text_score    : mean CLIP cosine(image, prompt) over frames (only if a
                         CLIP model is available and a prompt is provided). Higher
                         = the video matches the (possibly edited) instruction.

Works on any predicted mp4. Discovers episodes from an output tree per model:
  - dreamgen : sampling_dataset/augment/single_arm/output/singleview/dreamgen/{split}/<iter>_static16fps/global/episode_*.mp4 (+ .txt prompt sidecar)
  - dreamdojo: sampling_dataset/augment/single_arm/output/singleview/dreamdojo/{split}/<iter>/episode_*/full_pred.mp4
  - ctrlworld: sampling_dataset/augment/single_arm/output/multiview/ctrlworld/{split}/episode_*/pred_all_views.mp4

Writes per-episode augment_metrics.json next to each video and a combined
augment_metrics_summary.json + a printed markdown table.

Runs in models/dreamgen/.venv (torch + torchvision SSIM + optional CLIP).

Usage:
  python scripts/augment/compute_augment_metrics.py --root sampling_dataset/augment/single_arm
  python scripts/augment/compute_augment_metrics.py --root ... --no-clip
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def read_video(path: Path) -> np.ndarray:
    from decord import VideoReader, cpu
    vr = VideoReader(str(path), ctx=cpu(0), num_threads=2)
    try:
        return vr.get_batch(range(len(vr))).asnumpy()
    except Exception:
        return vr.get_batch(range(len(vr))).numpy()


def motion_magnitude(frames: np.ndarray) -> float:
    if frames.shape[0] < 2:
        return 0.0
    d = np.abs(frames[1:].astype(np.float32) - frames[:-1].astype(np.float32))
    return float(d.mean())


def temporal_consistency(frames: np.ndarray) -> float:
    """Mean SSIM between consecutive frames (grayscale, windowed)."""
    if frames.shape[0] < 2:
        return 1.0
    g = frames.astype(np.float32).mean(axis=-1) / 255.0  # (T,H,W)
    # lightweight global SSIM approximation per consecutive pair
    scores = []
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    for a, b in zip(g[:-1], g[1:]):
        mu_a, mu_b = a.mean(), b.mean()
        va, vb = a.var(), b.var()
        cov = ((a - mu_a) * (b - mu_b)).mean()
        ssim = ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / (
            (mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
        scores.append(float(ssim))
    return float(np.mean(scores))


class ClipScorer:
    def __init__(self, model_name="openai/clip-vit-base-patch16"):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.dev).eval()
        self.proc = CLIPProcessor.from_pretrained(model_name)

    def score(self, frames: np.ndarray, prompt: str, max_frames: int = 16) -> float:
        from PIL import Image
        torch = self.torch
        idx = np.linspace(0, frames.shape[0] - 1, min(max_frames, frames.shape[0])).astype(int)
        imgs = [Image.fromarray(frames[i]) for i in idx]
        with torch.no_grad():
            inp = self.proc(text=[prompt], images=imgs, return_tensors="pt",
                            padding=True, truncation=True).to(self.dev)
            out = self.model(**inp)
            img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            txt_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            sims = (img_emb @ txt_emb.T).squeeze(-1)
        return float(sims.mean().item())


def read_prompt_sidecar(mp4: Path) -> str | None:
    txt = mp4.with_suffix(".txt")
    if txt.exists():
        content = txt.read_text()
        m = re.search(r"\[Prompt\]\s*(.+?)\s*(\[Negative Prompt\]|$)", content, re.S)
        return (m.group(1).strip() if m else content.strip())
    return None


def discover(root: Path):
    """Yield (model, split, episode_id, mp4_path, prompt_or_none)."""
    # dreamgen singleview
    dg = root / "output" / "singleview" / "dreamgen"
    if dg.exists():
        for mp4 in sorted(dg.glob("*/*/global/episode_*.mp4")):
            split = mp4.parents[2].name
            ep = mp4.stem.replace("episode_", "")
            yield "dreamgen", split, ep, mp4, read_prompt_sidecar(mp4)
    # dreamdojo singleview
    dd = root / "output" / "singleview" / "dreamdojo"
    if dd.exists():
        for mp4 in sorted(dd.glob("*/*/episode_*/full_pred.mp4")):
            split = mp4.parents[3].name
            ep = mp4.parent.name.replace("episode_", "")
            yield "dreamdojo", split, ep, mp4, None
    # ctrlworld multiview
    cw = root / "output" / "multiview" / "ctrlworld"
    if cw.exists():
        for mp4 in sorted(cw.glob("*/episode_*/pred_all_views.mp4")):
            split = mp4.parents[1].name
            ep = mp4.parent.name.replace("episode_", "")
            yield "ctrlworld", split, ep, mp4, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="sampling_dataset/augment/single_arm")
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--out", default=None, help="Summary json path (default under root).")
    args = ap.parse_args()

    root = Path(args.root)
    clip = None
    if not args.no_clip:
        try:
            clip = ClipScorer()
        except Exception as e:
            print(f"[augment_metrics] CLIP unavailable ({e}); skipping clip_text_score.")

    rows = []
    for model, split, ep, mp4, prompt in discover(root):
        try:
            frames = read_video(mp4)
        except Exception as e:
            print(f"  [WARN] cannot read {mp4}: {e}")
            continue
        m = {
            "model": model, "split": split, "episode": ep,
            "num_frames": int(frames.shape[0]),
            "motion_magnitude": round(motion_magnitude(frames), 4),
            "temporal_consistency": round(temporal_consistency(frames), 4),
        }
        if clip is not None and prompt:
            try:
                m["clip_text_score"] = round(clip.score(frames, prompt), 4)
            except Exception:
                pass
        (mp4.parent / "augment_metrics.json").write_text(json.dumps(m, indent=2))
        rows.append(m)
        print(f"  {model}/{split}/{ep}: motion={m['motion_magnitude']:.2f} "
              f"tc={m['temporal_consistency']:.3f}"
              + (f" clip={m.get('clip_text_score'):.3f}" if 'clip_text_score' in m else ""))

    out_path = Path(args.out) if args.out else root / "augment_metrics_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate per model
    summary = {}
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        agg = {"num_episodes": len(sub)}
        for k in ["motion_magnitude", "temporal_consistency", "clip_text_score"]:
            vals = [r[k] for r in sub if k in r]
            if vals:
                agg[k] = round(float(np.mean(vals)), 4)
        summary[model] = agg
    out_path.write_text(json.dumps({"per_model": summary, "episodes": rows}, indent=2))

    # Markdown table
    print("\n| model | eps | motion_mag | temporal_consist | clip_text |")
    print("|---|---|---|---|---|")
    for model, agg in summary.items():
        print(f"| {model} | {agg['num_episodes']} | {agg.get('motion_magnitude','-')} "
              f"| {agg.get('temporal_consistency','-')} | {agg.get('clip_text_score','-')} |")
    print(f"\n[augment_metrics] summary -> {out_path}")


if __name__ == "__main__":
    main()
