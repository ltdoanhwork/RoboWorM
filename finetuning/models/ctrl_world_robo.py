from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import einops
import torch
import torch.nn as nn

from finetuning.config import FinetuneConfig, REPO_ROOT


CTRL_WORLD_DIR = REPO_ROOT / "models" / "Ctrl-World"
if str(CTRL_WORLD_DIR) not in sys.path:
    sys.path.insert(0, str(CTRL_WORLD_DIR))

from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline  # noqa: E402
from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline  # noqa: E402
from models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel  # noqa: E402


class ActionEncoderRobo(nn.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_size: int = 1024,
        text_cond: bool = True,
        use_embodiment_embedding: bool = True,
        num_embodiments: int = 16,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.text_cond = text_cond
        self.use_embodiment_embedding = use_embodiment_embedding
        self.action_encode = nn.Sequential(
            nn.Linear(action_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.kaiming_normal_(self.action_encode[0].weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.action_encode[2].weight, mode="fan_in", nonlinearity="relu")
        self.embodiment_embed = nn.Embedding(num_embodiments, hidden_size) if use_embodiment_embedding else None

    def forward(
        self,
        action: torch.Tensor,
        texts: list[str] | None = None,
        text_tokenizer: Any | None = None,
        text_encoder: nn.Module | None = None,
        frame_level_cond: bool = True,
        embodiment_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not frame_level_cond:
            action = einops.rearrange(action, "b t d -> b 1 (t d)")
        hidden = self.action_encode(action)

        if texts is not None and self.text_cond:
            if text_tokenizer is None or text_encoder is None:
                raise ValueError("text_tokenizer and text_encoder are required when text conditioning is enabled")
            with torch.no_grad():
                tokenized = text_tokenizer(texts, padding="max_length", return_tensors="pt", truncation=True)
                tokenized = tokenized.to(text_encoder.device)
                text_outputs = text_encoder(**tokenized)
                text_hidden = text_outputs.text_embeds
                text_hidden = einops.repeat(text_hidden, "b c -> b 1 (n c)", n=2)
            hidden = hidden + text_hidden.to(hidden.dtype)

        if self.embodiment_embed is not None and embodiment_id is not None:
            emb = self.embodiment_embed(embodiment_id.to(hidden.device))
            hidden = hidden + emb[:, None, :].to(hidden.dtype)
        return hidden


class CtrlWorldRobo(nn.Module):
    def __init__(self, cfg: FinetuneConfig):
        super().__init__()
        self.cfg = cfg
        self.pipeline = StableVideoDiffusionPipeline.from_pretrained(cfg.svd_model_path)
        unet = UNetSpatioTemporalConditionModel()
        unet.load_state_dict(self.pipeline.unet.state_dict(), strict=False)
        self.pipeline.unet = unet

        self.unet = self.pipeline.unet
        self.vae = self.pipeline.vae
        self.image_encoder = self.pipeline.image_encoder
        self.scheduler = self.pipeline.scheduler

        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.unet.requires_grad_(True)
        self.unet.enable_gradient_checkpointing()

        from transformers import AutoTokenizer, CLIPTextModelWithProjection

        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(cfg.clip_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.clip_model_path, use_fast=False)
        self.text_encoder.requires_grad_(False)

        self.action_encoder = ActionEncoderRobo(
            action_dim=cfg.action_dim,
            hidden_size=1024,
            text_cond=cfg.text_cond,
            use_embodiment_embedding=cfg.use_embodiment_embedding,
            num_embodiments=cfg.num_embodiments,
        )

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        latents = batch["latent"].to(self.unet.device)  # [B, T, C, V*H, W]
        action = batch["action"].to(self.unet.device)
        texts = list(batch["text"]) if cfg.text_cond else None
        embodiment_id = batch.get("embodiment_id")
        if embodiment_id is not None:
            embodiment_id = embodiment_id.to(self.unet.device)

        bsz = latents.shape[0]
        if latents.shape[1:] != (cfg.sequence_length, 4, cfg.latent_height, cfg.latent_slot_width):
            raise ValueError(f"Unexpected latent shape: {latents.shape}")
        if action.shape[1:] != (cfg.sequence_length, cfg.action_dim):
            raise ValueError(f"Unexpected action shape: {action.shape}")

        noise_aug_strength = 0.0
        p_mean = 0.7
        p_std = 1.6
        num_history = cfg.num_history

        current_img = latents[:, num_history]
        sigma_img = torch.rand([bsz, 1, 1, 1], device=latents.device) * 0.2
        current_img = 1 / (sigma_img**2 + 1) ** 0.5 * (current_img + torch.randn_like(current_img) * sigma_img)
        condition_latent = einops.repeat(current_img, "b c h w -> b t c h w", t=cfg.sequence_length)
        if cfg.his_cond_zero:
            condition_latent[:, :num_history] = 0.0

        action_hidden = self.action_encoder(
            action,
            texts,
            self.tokenizer,
            self.text_encoder,
            frame_level_cond=cfg.frame_level_cond,
            embodiment_id=embodiment_id,
        )
        uncond_hidden = torch.zeros_like(action_hidden)
        cond_mask = (torch.rand(action_hidden.shape[0], device=latents.device) > 0.05).unsqueeze(1).unsqueeze(2)
        action_hidden = action_hidden * cond_mask + uncond_hidden * (~cond_mask)

        rnd_normal = torch.randn([bsz, 1, 1, 1, 1], device=latents.device)
        sigma = (rnd_normal * p_std + p_mean).exp()
        c_skip = 1 / (sigma**2 + 1)
        c_out = -sigma / (sigma**2 + 1) ** 0.5
        c_in = 1 / (sigma**2 + 1) ** 0.5
        c_noise = (sigma.log() / 4).reshape([bsz])
        loss_weight = (sigma**2 + 1) / sigma**2
        noisy_latents = latents + torch.randn_like(latents) * sigma

        sigma_h = torch.randn([bsz, num_history, 1, 1, 1], device=latents.device) * 0.3
        history = latents[:, :num_history]
        noisy_history = 1 / (sigma_h**2 + 1) ** 0.5 * (history + sigma_h * torch.randn_like(history))
        input_latents = torch.cat([noisy_history, c_in * noisy_latents[:, num_history:]], dim=1)
        input_latents = torch.cat([input_latents, condition_latent / self.vae.config.scaling_factor], dim=2)

        added_time_ids = self.pipeline._get_add_time_ids(
            cfg.fps,
            cfg.motion_bucket_id,
            noise_aug_strength,
            action_hidden.dtype,
            bsz,
            1,
            False,
        ).to(latents.device)

        model_pred = self.unet(
            input_latents,
            c_noise,
            encoder_hidden_states=action_hidden,
            added_time_ids=added_time_ids,
            frame_level_cond=cfg.frame_level_cond,
        ).sample
        predict_x0 = c_out * model_pred + c_skip * noisy_latents
        loss = ((predict_x0[:, num_history:] - latents[:, num_history:]) ** 2 * loss_weight).mean()
        return loss, torch.tensor(0.0, device=latents.device, dtype=loss.dtype)

    @torch.no_grad()
    def generate_latents(
        self,
        current_latent: torch.Tensor,
        history: torch.Tensor,
        action: torch.Tensor,
        texts: list[str] | None,
        embodiment_id: torch.Tensor | None = None,
        output_type: str = "latent",
    ):
        cfg = self.cfg
        action_hidden = self.action_encoder(
            action,
            texts if cfg.text_cond else None,
            self.tokenizer,
            self.text_encoder,
            frame_level_cond=cfg.frame_level_cond,
            embodiment_id=embodiment_id,
        )
        return CtrlWorldDiffusionPipeline.__call__(
            self.pipeline,
            image=current_latent,
            text=action_hidden,
            width=cfg.image_width,
            height=cfg.image_height * cfg.n_views,
            num_frames=cfg.num_frames,
            history=history,
            num_inference_steps=cfg.num_inference_steps,
            decode_chunk_size=cfg.decode_chunk_size,
            max_guidance_scale=cfg.guidance_scale,
            fps=cfg.fps,
            motion_bucket_id=cfg.motion_bucket_id,
            mask=None,
            output_type=output_type,
            return_dict=False,
            frame_level_cond=cfg.frame_level_cond,
            his_cond_zero=cfg.his_cond_zero,
        )


def load_checkpoint_flexible(model: nn.Module, ckpt_path: str | os.PathLike[str]) -> tuple[list[str], list[str]]:
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        target_key = key.removeprefix("module.")
        if target_key in current and tuple(current[target_key].shape) == tuple(value.shape):
            compatible[target_key] = value
        else:
            skipped.append(target_key)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return list(missing) + skipped, list(unexpected)

