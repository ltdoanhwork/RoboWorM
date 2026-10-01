from __future__ import annotations

import types
from typing import List, Optional, Tuple, Union

import torch
import torch.amp as amp

from methods.cache_strategy import stats as cache_stats
from methods.cache_strategy.common import (
    DiCacheConfig,
    _maybe_concat_condition_mask,
    _maybe_embed_action,
    initialize_dicache_state,
    prepare_cache_runtime_model,
)

try:
    try:
        from cosmos_predict2.conditioner import DataType
    except ImportError:
        from cosmos_predict2._src.predict2.conditioner import DataType
except Exception:  # pragma: no cover - test fallback for minimal environments
    from enum import Enum

    class DataType(Enum):
        VIDEO = "video"
        IMAGE = "image"

try:
    try:
        from imaginaire.utils import log
    except ImportError:
        from cosmos_predict2._src.imaginaire.utils import log
except Exception:  # pragma: no cover - test fallback for minimal environments
    class _FallbackLog:
        @staticmethod
        def info(*args, **kwargs):
            pass

    log = _FallbackLog()


def dicache_mini_train_dit_forward(
    self,
    x_B_C_T_H_W: torch.Tensor,
    timesteps_B_T: torch.Tensor,
    crossattn_emb: torch.Tensor,
    fps: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    data_type: Optional[DataType] = DataType.VIDEO,
    intermediate_feature_ids: Optional[List[int]] = None,
    img_context_emb: Optional[torch.Tensor] = None,
    condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    del intermediate_feature_ids
    assert isinstance(data_type, DataType), f"Expected DataType, got {type(data_type)}."

    if kwargs.get("timestep_scale") is None and hasattr(self, "timestep_scale"):
        timesteps_B_T = timesteps_B_T * self.timestep_scale
    x_B_C_T_H_W = _maybe_concat_condition_mask(
        self,
        x_B_C_T_H_W,
        is_video=data_type == DataType.VIDEO,
        condition_video_input_mask_B_C_T_H_W=condition_video_input_mask_B_C_T_H_W,
    )

    x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = self.prepare_embedded_sequence(
        x_B_C_T_H_W,
        fps=fps,
        padding_mask=padding_mask,
    )

    if self.crossattn_proj is not None:
        crossattn_emb = self.crossattn_proj(crossattn_emb)

    if img_context_emb is not None:
        assert self.extra_image_context_dim is not None
        img_context_emb = self.img_context_proj(img_context_emb)
        context_input = (crossattn_emb, img_context_emb)
    else:
        context_input = crossattn_emb

    with amp.autocast("cuda", enabled=getattr(self, "use_wan_fp32_strategy", False), dtype=torch.float32):
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D, adaln_lora_B_T_3D = _maybe_embed_action(
            self,
            t_embedding_B_T_D,
            adaln_lora_B_T_3D,
            kwargs,
        )
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

    self.affline_scale_log_info = {"t_embedding_B_T_D": t_embedding_B_T_D.detach()}
    self.affline_emb = t_embedding_B_T_D
    self.crossattn_emb = crossattn_emb

    if extra_pos_emb is not None:
        assert x_B_T_H_W_D.shape == extra_pos_emb.shape

    block_kwargs = {
        "emb_B_T_D": t_embedding_B_T_D,
        "crossattn_emb": context_input,
        "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
        "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
        "extra_per_block_pos_emb": extra_pos_emb,
    }

    skip_forward = False
    ori_x = x_B_T_H_W_D
    residual_x = None
    current_idx = self.cnt % 2
    test_x = x_B_T_H_W_D.clone()

    # Block-work accounting; see methods/cache_strategy/stats.py. The probe's
    # `probe_depth` blocks are paid before the gate decides, so they count as executed
    # work even on a step that is then skipped.
    _dc_gated = self.cnt >= int(self.dicache_num_steps * self.dicache_ret_ratio)
    _dc_exec = int(self.dicache_probe_depth) if _dc_gated else 0

    if self.cnt >= int(self.dicache_num_steps * self.dicache_ret_ratio):
        for blk in self.blocks[: self.dicache_probe_depth]:
            test_x = blk(test_x, **block_kwargs)

        if self.previous_input[current_idx] is not None and self.previous_internal_states[current_idx] is not None:
            delta_y = (test_x - self.previous_internal_states[current_idx]).abs().mean() / (
                self.previous_internal_states[current_idx].abs().mean() + 1e-8
            )
            self.accumulated_rel_l1_distance[current_idx] += delta_y

            if self.accumulated_rel_l1_distance[current_idx] < self.dicache_rel_l1_thresh:
                skip_forward = True
                self.resume_flag[current_idx] = False
                residual_x = self.residual_cache[current_idx]
            else:
                self.resume_flag[current_idx] = True
                self.accumulated_rel_l1_distance[current_idx] = 0
                self.previous_internal_states[current_idx] = test_x.clone()

    if skip_forward:
        if len(self.residual_window[current_idx]) >= 2:
            current_residual = test_x - x_B_T_H_W_D
            numer = (current_residual - self.probe_residual_window[current_idx][-2]).abs().mean()
            denom = (
                self.probe_residual_window[current_idx][-1]
                - self.probe_residual_window[current_idx][-2]
            ).abs().mean()
            gamma = (numer / denom).clip(1, 2) if denom > 1e-6 else 1.0
            x_B_T_H_W_D = x_B_T_H_W_D + self.residual_window[current_idx][-2] + gamma * (
                self.residual_window[current_idx][-1] - self.residual_window[current_idx][-2]
            )
        else:
            x_B_T_H_W_D = x_B_T_H_W_D + residual_x

        self.previous_internal_states[current_idx] = test_x
        self.previous_input[current_idx] = ori_x
    else:
        if self.resume_flag[current_idx]:
            x_B_T_H_W_D = test_x
            remaining = self.blocks[self.dicache_probe_depth :]
        else:
            remaining = self.blocks

        for i, blk in enumerate(remaining):
            x_B_T_H_W_D = blk(x_B_T_H_W_D, **block_kwargs)
            _dc_exec += 1
            real_idx = i if not self.resume_flag[current_idx] else i + self.dicache_probe_depth
            if real_idx == self.dicache_probe_depth - 1:
                self.previous_internal_states[current_idx] = x_B_T_H_W_D.clone()

        residual_x = x_B_T_H_W_D - ori_x
        self.residual_cache[current_idx] = residual_x
        probe_residual = (
            self.previous_internal_states[current_idx] - ori_x
            if self.previous_internal_states[current_idx] is not None
            else residual_x
        )
        self.previous_input[current_idx] = ori_x

        if len(self.residual_window[current_idx]) <= 2:
            self.residual_window[current_idx].append(residual_x)
            self.probe_residual_window[current_idx].append(probe_residual)
        else:
            self.residual_window[current_idx][-2] = self.residual_window[current_idx][-1]
            self.residual_window[current_idx][-1] = residual_x
            self.probe_residual_window[current_idx][-2] = self.probe_residual_window[current_idx][-1]
            self.probe_residual_window[current_idx][-1] = probe_residual

    x_out = self.final_layer(
        x_B_T_H_W_D,
        t_embedding_B_T_D,
        adaln_lora_B_T_3D=adaln_lora_B_T_3D,
    )
    x_B_C_Tt_Hp_Wp = self.unpatchify(x_out)

    self.cnt += 1
    cache_stats.record_forward(
        blocks_executed=_dc_exec, skipped=skip_forward, gated=_dc_gated
    )
    if self.cnt >= self.dicache_num_steps:
        initialize_dicache_state(self, num_steps=self.dicache_num_steps)

    return x_B_C_Tt_Hp_Wp


def apply_dicache(model, config: DiCacheConfig):
    model.dicache_enabled = True
    model.dicache_num_steps = config.num_steps
    model.dicache_rel_l1_thresh = config.rel_l1_thresh
    model.dicache_ret_ratio = config.ret_ratio
    model.dicache_probe_depth = config.probe_depth

    initialize_dicache_state(model, num_steps=config.num_steps)
    prepare_cache_runtime_model(model)
    model.forward = types.MethodType(dicache_mini_train_dit_forward, model)
    cache_stats.configure(
        "dicache",
        num_blocks=len(model.blocks),
        config={
            "num_steps": config.num_steps,
            "rel_l1_thresh": config.rel_l1_thresh,
            "ret_ratio": config.ret_ratio,
            "probe_depth": config.probe_depth,
        },
    )

    log.info(
        f"[DiCache] Applied: steps={config.num_steps} thresh={config.rel_l1_thresh} "
        f"ret_ratio={config.ret_ratio} probe_depth={config.probe_depth}"
    )
    return model
