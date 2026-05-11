from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .runtime import ITMTokenMerger, infer_ctrl_world_heat_map_shape


def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch_size, seq_len, channels = x.shape
    return x.reshape(batch_size, seq_len, num_heads, channels // num_heads).permute(0, 2, 1, 3).contiguous()


class SVDITMAttnProcessor:
    MIN_TOKENS = 128

    def __init__(
        self,
        *,
        itm_state: dict,
        layer_idx: int,
        start_layer_idx: int = 0,
        compress_ratio: float | None = None,
        prune_from_step: int = 1,
        merge_from_step: int = 2,
        max_downsample_ratio: int = 4,
    ) -> None:
        self.itm_state = itm_state
        self.layer_idx = layer_idx
        self.start_layer_idx = start_layer_idx
        self.max_downsample_ratio = max_downsample_ratio
        self.runtime = ITMTokenMerger(
            group_mode="full_2d_framewise",
            compress_ratio=compress_ratio,
            prune_from_step=prune_from_step,
            merge_from_step=merge_from_step,
            merge_attn=True,
            merge_crossattn=False,
            merge_mlp=False,
        )

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, query_len, _ = hidden_states.shape
        num_heads = attn.heads
        kv_source = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key_len = kv_source.shape[1]

        phase = self.itm_state.get("phase", "disabled")
        heat_map = self.itm_state.get("heat_map")
        use_itm = (
            phase != "disabled"
            and heat_map is not None
            and self.layer_idx >= self.start_layer_idx
            and query_len == key_len
            and query_len >= self.MIN_TOKENS
        )

        plan = None
        merge_mode = self.itm_state.get("merge_mode", "mean")
        if use_itm:
            geometry = infer_ctrl_world_heat_map_shape(query_len)
            if geometry is not None:
                downsample_ratio = 72 // int(geometry.H)
                if downsample_ratio <= self.max_downsample_ratio:
                    heat = self._expand_heat_map(heat_map, batch_size=batch_size, height=geometry.H, width=geometry.W)
                    plan = self.runtime.prepare(hidden_states, heat_map=heat)

        if plan is not None:
            hidden_states = self.runtime.merge(hidden_states, plan, mode=merge_mode)
            kv_source = hidden_states

        q = _split_heads(attn.to_q(hidden_states), num_heads)
        k = _split_heads(attn.to_k(kv_source), num_heads)
        v = _split_heads(attn.to_v(kv_source), num_heads)
        hidden_states = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0)
        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous().reshape(hidden_states.shape[0], -1, q.shape[1] * q.shape[-1])

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if plan is not None:
            hidden_states = self.runtime.unmerge(hidden_states, plan, mode=merge_mode)
        return hidden_states

    def _expand_heat_map(self, heat_map: torch.Tensor, *, batch_size: int, height: int, width: int) -> torch.Tensor:
        if heat_map.ndim != 4:
            raise ValueError(f"Ctrl-World ITM heat map must have shape (B, F, H, W), got {tuple(heat_map.shape)}.")
        expanded = heat_map.reshape(-1, heat_map.shape[-2], heat_map.shape[-1])
        resized = F.interpolate(expanded.unsqueeze(1), size=(height, width), mode="area").squeeze(1)
        if resized.shape[0] == batch_size:
            return resized
        if resized.shape[0] * 2 == batch_size:
            return torch.cat((resized, resized), dim=0)
        raise ValueError(
            f"Unable to align Ctrl-World ITM heat map batch {resized.shape[0]} with attention batch {batch_size}."
        )
