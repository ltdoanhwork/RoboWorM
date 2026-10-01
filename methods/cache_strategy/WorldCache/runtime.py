from __future__ import annotations

import types
from typing import List, Optional, Tuple, Union

import torch
import torch.amp as amp
import torch.nn.functional as F
from einops import rearrange

from methods.cache_strategy import stats as cache_stats
from methods.cache_strategy.common import (
    WorldCacheConfig,
    _maybe_concat_condition_mask,
    _maybe_embed_action,
    initialize_worldcache_state,
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


def estimate_optical_flow(prev_img_tensor, curr_img_tensor, scale_factor=0.5):
    """GPU-native Lucas-Kanade optical flow. (B,C,H,W) -> (B,H,W,2)."""
    original_h, original_w = prev_img_tensor.shape[2], prev_img_tensor.shape[3]
    if scale_factor != 1.0:
        prev_img_tensor = F.interpolate(
            prev_img_tensor,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
        )
        curr_img_tensor = F.interpolate(
            curr_img_tensor,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
        )
    i1 = prev_img_tensor.mean(dim=1, keepdim=True)
    i2 = curr_img_tensor.mean(dim=1, keepdim=True)
    k_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=i1.dtype,
        device=i1.device,
    ).view(1, 1, 3, 3)
    k_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=i1.dtype,
        device=i1.device,
    ).view(1, 1, 3, 3)
    i_x = F.conv2d(i1, k_x, padding=1)
    i_y = F.conv2d(i1, k_y, padding=1)
    i_t = i2 - i1
    win = 21
    avg = torch.nn.AvgPool2d(kernel_size=win, stride=1, padding=win // 2)
    s_ix2 = avg(i_x * i_x)
    s_iy2 = avg(i_y * i_y)
    s_ixiy = avg(i_x * i_y)
    s_ixit = avg(i_x * i_t)
    s_iyit = avg(i_y * i_t)
    det = s_ix2 * s_iy2 - s_ixiy * s_ixiy + 1e-6
    u = -(s_iy2 * s_ixit - s_ixiy * s_iyit) / det
    v = -(s_ix2 * s_iyit - s_ixiy * s_ixit) / det
    flow = torch.cat((u, v), dim=1)
    if scale_factor != 1.0:
        flow = F.interpolate(
            flow,
            size=(original_h, original_w),
            mode="bilinear",
            align_corners=False,
        ) * (1.0 / scale_factor)
    return flow.permute(0, 2, 3, 1)


def warp_feature(feature_tensor, flow_tensor):
    if flow_tensor is None:
        return feature_tensor
    bsz, _channels, height, width = feature_tensor.shape
    if flow_tensor.ndim == 3:
        flow_tensor = flow_tensor.unsqueeze(0).expand(bsz, -1, -1, -1)
    elif flow_tensor.ndim == 4 and flow_tensor.shape[0] == 1 and bsz > 1:
        flow_tensor = flow_tensor.expand(bsz, -1, -1, -1)
    flow_tensor = flow_tensor.to(device=feature_tensor.device, dtype=feature_tensor.dtype)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=feature_tensor.device, dtype=feature_tensor.dtype),
        torch.arange(width, device=feature_tensor.device, dtype=feature_tensor.dtype),
        indexing="ij",
    )
    xx = xx.unsqueeze(0).expand(bsz, -1, -1)
    yy = yy.unsqueeze(0).expand(bsz, -1, -1)
    gx = 2.0 * (xx + flow_tensor[..., 0]) / max(width - 1, 1) - 1.0
    gy = 2.0 * (yy + flow_tensor[..., 1]) / max(height - 1, 1) - 1.0
    grid = torch.stack((gx, gy), dim=3)
    return F.grid_sample(
        feature_tensor,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=True,
    )


def compute_hf_drift(prev_img, curr_img):
    channels = prev_img.shape[1]
    kernel = torch.tensor(
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
        dtype=prev_img.dtype,
        device=prev_img.device,
    ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    prev_hf = F.conv2d(prev_img, kernel, padding=1, groups=channels)
    curr_hf = F.conv2d(curr_img, kernel, padding=1, groups=channels)
    return (prev_hf - curr_hf).abs().mean()


def compute_saliency_map(features):
    if features.ndim == 5:
        bsz, time, height, width, channels = features.shape
        features = rearrange(features, "b t h w d -> b (t d) h w")
    saliency = torch.std(features, dim=1, keepdim=True)
    bsz = saliency.shape[0]
    saliency_flat = saliency.view(bsz, -1)
    mn = saliency_flat.min(dim=1, keepdim=True)[0].view(bsz, 1, 1, 1)
    mx = saliency_flat.max(dim=1, keepdim=True)[0].view(bsz, 1, 1, 1)
    return (saliency - mn) / (mx - mn + 1e-6)


def compute_optimal_gamma(delta_curr, delta_prev):
    bsz = delta_curr.shape[0]
    delta_curr_flat = delta_curr.view(bsz, -1)
    delta_prev_flat = delta_prev.view(bsz, -1)
    numer = (delta_curr_flat * delta_prev_flat).sum(dim=1, keepdim=True)
    denom = (delta_prev_flat * delta_prev_flat).sum(dim=1, keepdim=True)
    gamma = torch.clamp(numer / (denom + 1e-6), 0.2, 1.8)
    return gamma.view(bsz, 1, 1, 1, 1)


def worldcache_mini_train_dit_forward(
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

    batch, time, _height, _width, _dim = x_B_T_H_W_D.shape
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

    is_parallel_cfg = getattr(self, "worldcache_parallel_cfg", False)
    current_idx = 0 if is_parallel_cfg else self.cnt % 2
    test_x = x_B_T_H_W_D.clone()

    # Block-work accounting (methods/cache_strategy/stats.py). `_wc_gated` records
    # whether the probe actually ran this forward, because `probe_depth` block
    # invocations are paid BEFORE the gate decides -- so a step the gate declines to
    # skip costs num_blocks + probe_depth, more block work than dense. Measuring that
    # honestly is the whole point of a baseline tune.
    _wc_gated = self.cnt >= int(self.worldcache_num_steps * self.worldcache_ret_ratio)
    _wc_num_blocks = len(self.blocks)
    _wc_probe = int(self.worldcache_probe_depth)
    _wc_exec = _wc_probe if _wc_gated else 0   # probe blocks, paid before the gate decides

    if self.cnt >= int(self.worldcache_num_steps * self.worldcache_ret_ratio):
        probe_depth = self.worldcache_probe_depth
        for blk in self.blocks[:probe_depth]:
            test_x = blk(test_x, **block_kwargs)

        if self.previous_input[current_idx] is not None and self.previous_internal_states[current_idx] is not None:
            delta_x = (x_B_T_H_W_D - self.previous_input[current_idx]).abs().mean() / (
                self.previous_input[current_idx].abs().mean() + 1e-8
            )
            delta_y = (test_x - self.previous_internal_states[current_idx]).abs().mean() / (
                self.previous_internal_states[current_idx].abs().mean() + 1e-8
            )
            self.accumulated_rel_l1_distance[current_idx] += delta_y

            if getattr(self, "worldcache_saliency_enabled", False):
                diff_map = rearrange(
                    (test_x - self.previous_internal_states[current_idx]).abs(),
                    "b t h w d -> b (t d) h w",
                ).mean(dim=1, keepdim=True)
                saliency_map = compute_saliency_map(self.previous_internal_states[current_idx])
                beta = getattr(self, "worldcache_saliency_weight", 5.0)
                weighted_drift = (diff_map * (1.0 + beta * saliency_map)).mean()
                denom = self.previous_internal_states[current_idx].abs().mean() + 1e-6
                weighted_rel = weighted_drift / denom
                self.accumulated_rel_l1_distance[current_idx] += weighted_rel - delta_y
                delta_y = weighted_rel

            if getattr(self, "worldcache_aduc_enabled", False) and not is_parallel_cfg:
                actual_step = self.cnt // 2
                step_ratio = actual_step / max(getattr(self, "worldcache_num_steps", 35), 1)
                if current_idx == 1 and step_ratio > getattr(self, "worldcache_aduc_start", 0.5):
                    if len(self.previous_output) > 1 and self.previous_output[1] is not None:
                        self.cnt += 1
                        cache_stats.record_forward(
                            blocks_executed=_wc_exec, skipped=True, gated=_wc_gated
                        )
                        return self.previous_output[1]

            input_velocity = delta_x
            alpha = getattr(self, "worldcache_motion_sensitivity", 5.0)
            dynamic_thresh = self.worldcache_rel_l1_thresh / (1.0 + alpha * input_velocity)

            if getattr(self, "worldcache_dynamic_decay", False):
                num_steps = max(getattr(self, "worldcache_num_steps", 35), 1)
                u_ratio = num_steps / 35.0
                base_mult = (u_ratio**2) / 6.0 + u_ratio / 2.0 + 10.0 / 3.0
                dynamic_thresh *= 1.0 + base_mult * (self.cnt / num_steps)

            hf_ok = True
            if getattr(self, "worldcache_hf_enabled", False):
                current_input = rearrange(x_B_T_H_W_D, "b t h w d -> b (t d) h w")
                previous_input = rearrange(self.previous_input[current_idx], "b t h w d -> b (t d) h w")
                hf_drift = compute_hf_drift(previous_input, current_input)
                if hf_drift > getattr(self, "worldcache_hf_thresh", 0.01):
                    hf_ok = False

            if self.accumulated_rel_l1_distance[current_idx] < dynamic_thresh and hf_ok:
                if is_parallel_cfg and batch > 1:
                    diff_full = (test_x - self.previous_internal_states[current_idx]).abs()
                    half_batch = batch // 2
                    drift_cond = diff_full[:half_batch].mean() / (
                        self.previous_internal_states[current_idx][:half_batch].abs().mean() + 1e-6
                    )
                    drift_uncond = diff_full[half_batch:].mean() / (
                        self.previous_internal_states[current_idx][half_batch:].abs().mean() + 1e-6
                    )
                    if drift_cond < dynamic_thresh and drift_uncond < dynamic_thresh:
                        skip_forward = True
                        self.worldcache_step_skipped_count += 1
                        self.resume_flag[current_idx] = False
                        residual_x = self.residual_cache[current_idx]
                    else:
                        self.resume_flag[current_idx] = True
                        self.accumulated_rel_l1_distance[current_idx] = 0
                        self.previous_internal_states[current_idx] = test_x.clone()
                else:
                    skip_forward = True
                    self.worldcache_step_skipped_count += 1
                    self.resume_flag[current_idx] = False
                    residual_x = self.residual_cache[current_idx]
            else:
                self.resume_flag[current_idx] = True
                self.accumulated_rel_l1_distance[current_idx] = 0
                self.previous_internal_states[current_idx] = test_x.clone()

    if skip_forward:
        if len(self.residual_window[current_idx]) >= 2:
            current_residual = test_x - x_B_T_H_W_D
            if getattr(self, "worldcache_osi_enabled", False):
                dt = current_residual - self.probe_residual_window[current_idx][-2]
                ds = (
                    self.probe_residual_window[current_idx][-1]
                    - self.probe_residual_window[current_idx][-2]
                )
                gamma = compute_optimal_gamma(dt, ds)
            else:
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

        if (
            getattr(self, "worldcache_flow_enabled", False)
            and self.previous_input[current_idx] is not None
            and residual_x is not None
        ):
            current_input = rearrange(x_B_T_H_W_D, "b t h w d -> b (t d) h w")
            previous_input = rearrange(self.previous_input[current_idx], "b t h w d -> b (t d) h w")
            flow = estimate_optical_flow(
                previous_input,
                current_input,
                scale_factor=getattr(self, "worldcache_flow_scale", 0.5),
            )
            if flow is not None:
                residual_input = rearrange(residual_x, "b t h w d -> b (t d) h w")
                warped_residual = warp_feature(residual_input, flow)
                warped = rearrange(warped_residual, "b (t d) h w -> b t h w d", t=time)
                x_B_T_H_W_D = x_B_T_H_W_D - residual_x + warped
    else:
        if self.resume_flag[current_idx]:
            x_B_T_H_W_D = test_x
            remaining = self.blocks[self.worldcache_probe_depth :]
        else:
            remaining = self.blocks

        for i, blk in enumerate(remaining):
            x_B_T_H_W_D = blk(x_B_T_H_W_D, **block_kwargs)
            _wc_exec += 1
            real_idx = i if not self.resume_flag[current_idx] else i + self.worldcache_probe_depth
            if real_idx == self.worldcache_probe_depth - 1:
                self.previous_internal_states[current_idx] = x_B_T_H_W_D.clone()

        residual_x = x_B_T_H_W_D - ori_x
        self.residual_cache[current_idx] = residual_x
        probe_residual = (
            self.previous_internal_states[current_idx] - ori_x
            if self.previous_internal_states[current_idx] is not None
            else residual_x
        )
        self.probe_residual_cache[current_idx] = probe_residual
        self.previous_input[current_idx] = ori_x
        self.previous_output[current_idx] = x_B_T_H_W_D

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
        blocks_executed=_wc_exec, skipped=skip_forward, gated=_wc_gated
    )
    if self.cnt >= self.worldcache_num_steps:
        rate = self.worldcache_step_skipped_count / max(self.worldcache_num_steps, 1)
        log.info(
            f"[WorldCache] Skipped {self.worldcache_step_skipped_count}/{self.worldcache_num_steps} "
            f"({rate:.1%}) | alpha={getattr(self, 'worldcache_motion_sensitivity', 'N/A')}"
        )
        initialize_worldcache_state(self, num_steps=self.worldcache_num_steps)

    if current_idx < len(self.previous_output):
        self.previous_output[current_idx] = x_B_C_Tt_Hp_Wp.clone()

    return x_B_C_Tt_Hp_Wp


def apply_worldcache(model, config: WorldCacheConfig):
    model.worldcache_enabled = True
    model.worldcache_num_steps = config.num_steps
    model.worldcache_rel_l1_thresh = config.rel_l1_thresh
    model.worldcache_ret_ratio = config.ret_ratio
    model.worldcache_probe_depth = config.probe_depth
    model.worldcache_motion_sensitivity = config.motion_sensitivity
    model.worldcache_flow_enabled = config.flow_enabled
    model.worldcache_flow_scale = config.flow_scale
    model.worldcache_hf_enabled = config.hf_enabled
    model.worldcache_hf_thresh = config.hf_thresh
    model.worldcache_saliency_enabled = config.saliency_enabled
    model.worldcache_saliency_weight = config.saliency_weight
    model.worldcache_osi_enabled = config.osi_enabled
    model.worldcache_dynamic_decay = config.dynamic_decay
    model.worldcache_aduc_enabled = config.aduc_enabled
    model.worldcache_aduc_start = config.aduc_start
    model.worldcache_parallel_cfg = config.parallel_cfg

    initialize_worldcache_state(model, num_steps=config.num_steps)
    prepare_cache_runtime_model(model)
    model.forward = types.MethodType(worldcache_mini_train_dit_forward, model)
    cache_stats.configure(
        "worldcache",
        num_blocks=len(model.blocks),
        config={
            "num_steps": config.num_steps,
            "rel_l1_thresh": config.rel_l1_thresh,
            "ret_ratio": config.ret_ratio,
            "probe_depth": config.probe_depth,
            "motion_sensitivity": config.motion_sensitivity,
            "dynamic_decay": config.dynamic_decay,
        },
    )

    log.info(
        f"[WorldCache] Applied: steps={config.num_steps} thresh={config.rel_l1_thresh} "
        f"ret_ratio={config.ret_ratio} probe_depth={config.probe_depth} "
        f"alpha={config.motion_sensitivity} flow={config.flow_enabled}({config.flow_scale}) "
        f"hf={config.hf_enabled} saliency={config.saliency_enabled} osi={config.osi_enabled} "
        f"decay={config.dynamic_decay} aduc={config.aduc_enabled}({config.aduc_start}) "
        f"parallel_cfg={config.parallel_cfg}"
    )
    return model
