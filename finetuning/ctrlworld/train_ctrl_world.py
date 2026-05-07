from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import einops
import imageio.v2 as imageio
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from finetuning.config import FinetuneConfig, add_config_arguments, config_from_args
from finetuning.data.dataset_robo_ctrl_world import RoboCtrlWorldDataset
from finetuning.models.ctrl_world_robo import CtrlWorldRobo, load_checkpoint_flexible


def save_checkpoint(accelerator: Accelerator, model: CtrlWorldRobo, output_dir: Path, step: int) -> None:
    if not accelerator.is_main_process:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    ckpt = unwrapped.state_dict()
    step_path = output_dir / f"checkpoint-{step}.pt"
    latest_path = output_dir / "latest.pt"
    torch.save(ckpt, step_path)
    latest_path.unlink(missing_ok=True)
    try:
        os.symlink(step_path.name, latest_path)
    except OSError:
        torch.save(ckpt, latest_path)
    (output_dir / "latest_checkpoint.txt").write_text(str(step_path.name), encoding="utf-8")


@torch.no_grad()
def decode_stacked_latents(vae, latents: torch.Tensor, cfg: FinetuneConfig, max_frames: int | None = None) -> np.ndarray:
    # latents: [T, C, V*H, W]
    if max_frames is not None:
        latents = latents[:max_frames]
    slot_h = cfg.latent_slot_height
    views = einops.rearrange(latents, "t c (v h) w -> v t c h w", v=cfg.n_views, h=slot_h)
    flat = einops.rearrange(views, "v t c h w -> (v t) c h w").to(vae.device)
    decoded = []
    for i in range(0, flat.shape[0], cfg.decode_chunk_size):
        chunk = flat[i : i + cfg.decode_chunk_size] / vae.config.scaling_factor
        decoded.append(vae.decode(chunk, num_frames=chunk.shape[0]).sample.detach().cpu())
    video = torch.cat(decoded, dim=0)
    video = einops.rearrange(video, "(v t) c h w -> v t h w c", v=cfg.n_views)
    video = ((video / 2.0 + 0.5).clamp(0, 1) * 255).to(torch.uint8).numpy()

    frames = []
    for t in range(video.shape[1]):
        view_imgs = [video[v, t] for v in range(cfg.n_views)]
        if cfg.n_views == 4:
            top = np.concatenate(view_imgs[:2], axis=1)
            bottom = np.concatenate(view_imgs[2:4], axis=1)
            frames.append(np.concatenate([top, bottom], axis=0))
        else:
            frames.append(np.concatenate(view_imgs, axis=1))
    return np.stack(frames)


@torch.no_grad()
def save_validation_sample(
    accelerator: Accelerator,
    model: CtrlWorldRobo,
    batch: dict,
    cfg: FinetuneConfig,
    output_dir: Path,
    step: int,
) -> None:
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.eval()
    device = accelerator.device
    latents = batch["latent"][:1].to(device)
    actions = batch["action"][:1].to(device)
    texts = [batch["text"][0]] if cfg.text_cond else None
    embodiment_id = batch["embodiment_id"][:1].to(device) if "embodiment_id" in batch else None
    history = latents[:, : cfg.num_history]
    current = latents[:, cfg.num_history]

    try:
        _, pred_future = unwrapped.generate_latents(
            current,
            history,
            actions,
            texts,
            embodiment_id=embodiment_id,
            output_type="latent",
        )
        pred_full = torch.cat([history, pred_future], dim=1)[0].float()
        gt_full = latents[0].float()
        pred_video = decode_stacked_latents(unwrapped.vae, pred_full, cfg)
        gt_video = decode_stacked_latents(unwrapped.vae, gt_full, cfg)
        comparison = np.concatenate([gt_video, pred_video], axis=1)
        sample_dir = output_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(sample_dir / f"step_{step:08d}.mp4", comparison, fps=cfg.fps, macro_block_size=1)
    except Exception as exc:
        sample_dir = output_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"error": str(exc), "batch": {k: str(v) for k, v in batch.items() if k not in {"latent", "action"}}}, sample_dir / f"step_{step:08d}_failed.pt")
    finally:
        unwrapped.train()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a RoboCOIN multiview Ctrl-World style world model.")
    add_config_arguments(parser)
    parser.add_argument("--resume", default=None, help="Checkpoint to resume/load. Defaults to cfg.ckpt_path.")
    parser.add_argument("--no-sample-video", action="store_true")
    args = parser.parse_args()
    cfg = config_from_args(args)
    os.environ.setdefault("WANDB_MODE", "offline")

    logger = get_logger(__name__, log_level="INFO")
    set_seed(cfg.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with="wandb",
        project_dir=cfg.output_dir,
    )

    train_dataset = RoboCtrlWorldDataset(cfg, mode="train")
    val_dataset = RoboCtrlWorldDataset(cfg, mode="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = CtrlWorldRobo(cfg)
    resume_path = args.resume or cfg.ckpt_path or cfg.init_ckpt_path
    if resume_path:
        missing, unexpected = load_checkpoint_flexible(model, resume_path)
        logger.info(f"Loaded compatible weights from {resume_path}; missing/skipped={len(missing)} unexpected={len(unexpected)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    output_dir = Path(cfg.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
        try:
            accelerator.init_trackers("ctrl_world_robo", config=cfg.to_dict())
        except Exception:
            pass

    total_batch_size = cfg.train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Train windows = {len(train_dataset)}")
    logger.info(f"  Val windows = {len(val_dataset)}")
    logger.info(f"  Total batch size = {total_batch_size}")
    logger.info(f"  Max steps = {cfg.max_train_steps}")

    global_step = 0
    train_loss = 0.0
    val_iter = iter(val_loader)
    progress = tqdm(range(cfg.max_train_steps), disable=not accelerator.is_local_main_process)

    while global_step < cfg.max_train_steps:
        for batch in train_loader:
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss, _ = model(batch)
                avg_loss = accelerator.gather(loss.detach().repeat(cfg.train_batch_size)).mean()
                train_loss += avg_loss.item() / cfg.gradient_accumulation_steps
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                if global_step % cfg.log_steps == 0:
                    value = train_loss / cfg.log_steps
                    progress.set_postfix({"loss": value})
                    accelerator.log({"train_loss": value}, step=global_step)
                    train_loss = 0.0
                if global_step % cfg.checkpointing_steps == 0:
                    save_checkpoint(accelerator, model, output_dir, global_step)
                if global_step % cfg.validation_steps == 0:
                    try:
                        val_batch = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        val_batch = next(val_iter)
                    with torch.no_grad(), accelerator.autocast():
                        val_loss, _ = model(val_batch)
                    val_loss_mean = accelerator.gather(val_loss.detach().repeat(1)).mean().item()
                    accelerator.log({"val_loss": val_loss_mean}, step=global_step)
                    if not args.no_sample_video:
                        save_validation_sample(accelerator, model, val_batch, cfg, output_dir, global_step)

                if global_step >= cfg.max_train_steps:
                    break

    save_checkpoint(accelerator, model, output_dir, global_step)
    accelerator.end_training()


if __name__ == "__main__":
    main()
