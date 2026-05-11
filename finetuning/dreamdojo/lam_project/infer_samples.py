import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from tqdm import trange

from lam.dataset import OriginalVideoDataset
from lam.model import LAM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LAM inference on multiple samples.")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml.")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Checkpoint path override.")
    parser.add_argument("--num_samples", type=int, default=64, help="Number of samples.")
    parser.add_argument("--seed", type=int, default=32, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    parser.add_argument("--output_dir", type=str, default="lam_infer_outputs", help="Output directory.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(config: Dict[str, Any], ckpt_path: str, device: torch.device) -> LAM:
    model_cfg = dict(config["model"])
    model_cfg.pop("optimizer", None)
    model = LAM(**model_cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def build_dataset(config: Dict[str, Any]) -> OriginalVideoDataset:
    data_cfg = config["data"]
    return OriginalVideoDataset(
        dataset_paths=data_cfg["dataset_paths"],
        split="test",
        padding=data_cfg.get("padding", "repeat"),
        randomize=True,
        num_frames=data_cfg.get("num_frames", 2),
        output_format=data_cfg.get("output_format", "t h w c"),
        color_aug=False,
    )


def draw_label(img: np.ndarray, text: str) -> np.ndarray:
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle((0, 0, 150, 24), fill=(0, 0, 0))
    draw.text((6, 6), text, fill=(255, 255, 255))
    return np.array(pil_img)


def save_compare_image(video: torch.Tensor, recon: torch.Tensor, out_path: Path) -> None:
    # video: [T,H,W,C], recon: [T-1,H,W,C], expected T=2 for current setup
    gt_seq = video.clamp(0, 1).cpu()
    recon_seq = recon.clamp(0, 1).cpu()

    t0 = (gt_seq[0] * 255.0).numpy().astype(np.uint8)
    gt_t1 = (gt_seq[1] * 255.0).numpy().astype(np.uint8)
    pred_t1 = (recon_seq[0] * 255.0).numpy().astype(np.uint8)

    t0 = draw_label(t0, "t0 (input)")
    pred_t1 = draw_label(pred_t1, "pred_t1")
    gt_t1 = draw_label(gt_t1, "gt_t1")

    canvas = np.concatenate([t0, pred_t1, gt_t1], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ckpt_path = args.ckpt_path or config.get("ckpt_path")
    if not ckpt_path:
        raise ValueError("Missing checkpoint path. Use --ckpt_path or set ckpt_path in config.")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = build_model(config, ckpt_path, device)
    dataset = build_dataset(config)

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    metrics: List[Dict[str, Any]] = []
    sample_count = min(args.num_samples, len(dataset))

    for i in trange(sample_count, desc="Running inference"):
        sample_idx = random.randint(0, len(dataset) - 1)
        batch = dataset[sample_idx]
        video = batch["videos"].unsqueeze(0).to(device)

        outputs = model.lam({"videos": video})
        recon = outputs["recon"]
        gt_future = video[:, 1:]
        mse = ((gt_future - recon) ** 2).mean().item()

        gt_img = gt_future.clamp(0, 1).reshape(-1, *gt_future.shape[2:]).permute(0, 3, 1, 2)
        recon_img = recon.clamp(0, 1).reshape(-1, *recon.shape[2:]).permute(0, 3, 1, 2)
        psnr = (-10.0 * torch.log10(torch.clamp((gt_img - recon_img).pow(2).mean(), min=1e-10))).item()

        save_compare_image(
            video[0].detach().cpu(),
            recon[0].detach().cpu(),
            images_dir / f"sample_{i:04d}_datasetidx_{sample_idx}.png",
        )

        metrics.append(
            {"sample_id": i, "dataset_index": sample_idx, "mse": mse, "psnr": psnr}
        )

    summary = {
        "num_samples": sample_count,
        "checkpoint": str(ckpt_path),
        "mean_mse": float(np.mean([m["mse"] for m in metrics])) if metrics else None,
        "mean_psnr": float(np.mean([m["psnr"] for m in metrics])) if metrics else None,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "samples": metrics}, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved images to: {images_dir}")
    print(f"Saved metrics to: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
