from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from finetuning.config import add_config_arguments, config_from_args
from finetuning.data.dataset_robo_ctrl_world import RoboCtrlWorldDataset
from finetuning.models.ctrl_world_robo import CtrlWorldRobo, load_checkpoint_flexible
from finetuning.train_ctrl_world import decode_stacked_latents


@torch.no_grad()
def run_one(model: CtrlWorldRobo, batch: dict, output_path: Path) -> None:
    cfg = model.cfg
    device = next(model.parameters()).device
    latents = batch["latent"].to(device)
    actions = batch["action"].to(device)
    texts = list(batch["text"]) if cfg.text_cond else None
    embodiment_id = batch["embodiment_id"].to(device) if "embodiment_id" in batch else None
    history = latents[:, : cfg.num_history]
    current = latents[:, cfg.num_history]
    _, pred_future = model.generate_latents(
        current,
        history,
        actions,
        texts,
        embodiment_id=embodiment_id,
        output_type="latent",
    )
    pred_full = torch.cat([history, pred_future], dim=1)[0].float()
    gt_full = latents[0].float()
    pred_video = decode_stacked_latents(model.vae, pred_full, cfg)
    gt_video = decode_stacked_latents(model.vae, gt_full, cfg)
    comparison = np.concatenate([gt_video, pred_video], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, comparison, fps=cfg.fps, macro_block_size=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay-evaluate a trained Robo Ctrl-World checkpoint.")
    add_config_arguments(parser)
    parser.add_argument("--ckpt", default=None, help="Alias for --ckpt-path.")
    parser.add_argument("--mode", choices=["train", "val"], default="val")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()
    cfg = config_from_args(args)
    ckpt = args.ckpt or cfg.ckpt_path or str(Path(cfg.output_dir) / "latest.pt")
    save_dir = Path(args.save_dir or Path(cfg.output_dir) / "replay_eval")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RoboCtrlWorldDataset(cfg, mode=args.mode)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = CtrlWorldRobo(cfg)
    missing, unexpected = load_checkpoint_flexible(model, ckpt)
    print(f"[eval] loaded {ckpt}; missing/skipped={len(missing)} unexpected={len(unexpected)}")
    model.to(device=device, dtype=cfg.torch_dtype)
    model.eval()

    for idx, batch in enumerate(tqdm(loader, total=min(args.num_samples, len(dataset)), desc="eval replay")):
        if idx >= args.num_samples:
            break
        name = f"{idx:04d}_{batch['dataset_name'][0]}_{batch['episode_id'][0]}_{int(batch['start_idx'][0])}.mp4"
        run_one(model, batch, save_dir / name)
    print(f"[eval] wrote videos to {save_dir}")


if __name__ == "__main__":
    main()

