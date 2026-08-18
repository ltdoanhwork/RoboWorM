"""Object-editing helper for augmenting benchmark inputs.

Given an input image (or a batch of video frames) and a natural-language edit
instruction like "replace the red cup with a blue bowl", produce an edited image
where only the object is swapped. Used to build "harder" augmented inputs for the
video-world benchmark (Ctrl-World / DreamDojo need the object swapped in EVERY
frame; DreamGen only needs the first frame).

Backends (env EDIT_MODEL, default 'ip2p'):
  - ip2p        : timbrooks/instruct-pix2pix (InstructPix2Pix, works on diffusers 0.33.1)
  - sdxl_ip2p   : diffusers/sdxl-instructpix2pix-768 (higher-res InstructPix2Pix)
  - qwen        : Qwen/Qwen-Image-Edit   (needs diffusers >= 0.35, QwenImageEditPipeline)
  - flux_kontext: black-forest-labs/FLUX.1-Kontext-dev (needs diffusers >= 0.34)

Frame-consistency helpers (important because we edit frames independently):
  - Fixed seed across all frames of a clip (--seed).
  - Optional SAM-based masking (--use-mask): segment the object region from the
    FIRST frame's edit-diff, then only composite the edited object back over the
    original frame so the background/robot stays pixel-identical (kills flicker).
    Falls back to full-frame edit if SAM is unavailable.

Runs in models/dreamgen/.venv (has diffusers + torch + transformers).

Usage (single image):
  python scripts/augment/edit_object.py \
    --input first.png --output edited.png \
    --instruction "replace the red cup with a blue bowl" --seed 0

Usage (batch of frames in a dir -> dir):
  python scripts/augment/edit_object.py \
    --input frames_dir/ --output edited_dir/ \
    --instruction "replace the marker with a pencil" --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype() -> torch.dtype:
    return torch.float16 if torch.cuda.is_available() else torch.float32


class ObjectEditor:
    """Wraps a diffusers image-edit pipeline behind a uniform .edit(img, instr)."""

    def __init__(self, model_key: str, steps: int, guidance: float, image_guidance: float):
        self.model_key = model_key
        self.steps = steps
        self.guidance = guidance
        self.image_guidance = image_guidance
        self.pipe = self._load()

    def _load(self):
        import diffusers

        dev, dt = _device(), _dtype()
        if self.model_key == "ip2p":
            from diffusers import StableDiffusionInstructPix2PixPipeline
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                "timbrooks/instruct-pix2pix", torch_dtype=dt, safety_checker=None,
            )
        elif self.model_key == "sdxl_ip2p":
            from diffusers import StableDiffusionXLInstructPix2PixPipeline
            pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
                "diffusers/sdxl-instructpix2pix-768", torch_dtype=dt,
            )
        elif self.model_key == "qwen":
            if not hasattr(diffusers, "QwenImageEditPipeline"):
                raise RuntimeError(
                    "QwenImageEditPipeline not in diffusers "
                    f"{diffusers.__version__}; upgrade to >=0.35 in a dedicated venv."
                )
            from diffusers import QwenImageEditPipeline
            pipe = QwenImageEditPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit", torch_dtype=torch.bfloat16,
            )
        elif self.model_key == "flux_kontext":
            if not hasattr(diffusers, "FluxKontextPipeline"):
                raise RuntimeError(
                    "FluxKontextPipeline not in diffusers "
                    f"{diffusers.__version__}; upgrade to >=0.34 in a dedicated venv."
                )
            from diffusers import FluxKontextPipeline
            pipe = FluxKontextPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16,
            )
        else:
            raise ValueError(f"Unknown EDIT_MODEL '{self.model_key}'")

        pipe = pipe.to(dev)
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def edit(self, image: Image.Image, instruction: str, seed: int) -> Image.Image:
        gen = torch.Generator(device=_device()).manual_seed(seed)
        w, h = image.size
        kwargs = dict(
            prompt=instruction,
            image=image,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            generator=gen,
        )
        # InstructPix2Pix pipelines expose image_guidance_scale (fidelity to input).
        if self.model_key in ("ip2p", "sdxl_ip2p"):
            kwargs["image_guidance_scale"] = self.image_guidance
        out = self.pipe(**kwargs).images[0]
        if out.size != (w, h):
            out = out.resize((w, h), Image.LANCZOS)
        return out


def _try_load_sam():
    """Load a SAM-family predictor for optional object masking. Returns callable
    (image_rgb_np, box_or_point) -> bool mask, or None if unavailable."""
    try:
        from transformers import SamModel, SamProcessor
    except Exception:
        return None
    try:
        dev = _device()
        model = SamModel.from_pretrained("facebook/sam-vit-base").to(dev).eval()
        proc = SamProcessor.from_pretrained("facebook/sam-vit-base")
    except Exception:
        return None

    def _mask_from_box(img_np: np.ndarray, box):
        with torch.no_grad():
            inputs = proc(Image.fromarray(img_np), input_boxes=[[box]], return_tensors="pt").to(dev)
            out = model(**inputs)
            masks = proc.image_processor.post_process_masks(
                out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )[0][0]
            m = masks[0].numpy().astype(bool)
        return m

    return _mask_from_box


def _diff_box(orig: Image.Image, edited: Image.Image, pad: float = 0.12):
    """Bounding box of the region that changed most between orig and edited."""
    a = np.asarray(orig.convert("RGB"), dtype=np.int16)
    b = np.asarray(edited.convert("RGB"), dtype=np.int16)
    d = np.abs(a - b).sum(axis=2)
    thr = max(20, np.percentile(d, 97))
    ys, xs = np.where(d >= thr)
    if len(xs) < 10:
        return None
    h, w = d.shape
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
    return [max(0, x0 - px), max(0, y0 - py), min(w - 1, x1 + px), min(h - 1, y1 + py)]


def _feather(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """Soft-edge a boolean mask into a float alpha in [0,1]."""
    try:
        from scipy.ndimage import gaussian_filter, binary_dilation
        m = binary_dilation(mask, iterations=2).astype(np.float32)
        return np.clip(gaussian_filter(m, sigma=radius), 0.0, 1.0)
    except Exception:
        return mask.astype(np.float32)


def _composite(orig: Image.Image, edited: Image.Image, alpha: np.ndarray) -> Image.Image:
    a = np.asarray(orig.convert("RGB"), dtype=np.float32)
    b = np.asarray(edited.convert("RGB"), dtype=np.float32)
    al = alpha[..., None]
    out = a * (1 - al) + b * al
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def _collect_inputs(inp: Path):
    if inp.is_dir():
        files = sorted(p for p in inp.iterdir() if p.suffix.lower() in IMG_EXTS)
        return files
    return [inp]


def main():
    ap = argparse.ArgumentParser(description="Object-editing helper (diffusers).")
    ap.add_argument("--input", required=True, help="Input image or dir of frames.")
    ap.add_argument("--output", required=True, help="Output image or dir.")
    ap.add_argument("--instruction", required=True, help="Edit instruction.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=os.environ.get("EDIT_MODEL", "qwen"),
                    choices=["ip2p", "sdxl_ip2p", "qwen", "flux_kontext"])
    ap.add_argument("--steps", type=int, default=int(os.environ.get("EDIT_STEPS", "30")))
    ap.add_argument("--guidance", type=float, default=float(os.environ.get("EDIT_GUIDANCE", "7.5")))
    ap.add_argument("--image-guidance", type=float,
                    default=float(os.environ.get("EDIT_IMAGE_GUIDANCE", "1.5")),
                    help="InstructPix2Pix image_guidance_scale (higher = closer to input).")
    ap.add_argument("--use-mask", action="store_true",
                    help="Composite edited object over original via SAM/diff mask (reduces flicker).")
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    files = _collect_inputs(inp)
    if not files:
        print(f"[edit_object] no input images at {inp}", file=sys.stderr)
        sys.exit(1)

    is_batch = inp.is_dir()
    if is_batch:
        outp.mkdir(parents=True, exist_ok=True)
    else:
        outp.parent.mkdir(parents=True, exist_ok=True)

    editor = ObjectEditor(args.model, args.steps, args.guidance, args.image_guidance)
    sam = _try_load_sam() if args.use_mask else None
    if args.use_mask and sam is None:
        print("[edit_object] SAM unavailable; falling back to diff-box mask.", file=sys.stderr)

    shared_alpha = None  # reuse first-frame mask across the clip for temporal stability
    for i, f in enumerate(files):
        orig = Image.open(f).convert("RGB")
        edited = editor.edit(orig, args.instruction, args.seed)

        if args.use_mask:
            if shared_alpha is None:
                box = _diff_box(orig, edited)
                mask = None
                if box is not None:
                    if sam is not None:
                        try:
                            mask = sam(np.asarray(orig), box)
                        except Exception:
                            mask = None
                    if mask is None:
                        mask = np.zeros(np.asarray(orig).shape[:2], dtype=bool)
                        mask[box[1]:box[3], box[0]:box[2]] = True
                if mask is not None:
                    shared_alpha = _feather(mask)
            if shared_alpha is not None:
                edited = _composite(orig, edited, shared_alpha)

        out_path = (outp / f.name) if is_batch else outp
        edited.save(out_path)
        print(f"  [{i+1}/{len(files)}] {f.name} -> {out_path}")

    print(f"[edit_object] done: {len(files)} image(s), model={args.model}, seed={args.seed}")


if __name__ == "__main__":
    main()
