from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .runtime import SiToTokenPruner


def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch_size, seq_len, channels = x.shape
    return x.reshape(batch_size, seq_len, num_heads, channels // num_heads).permute(0, 2, 1, 3).contiguous()


class SVDSiToAttnProcessor:
    MIN_TOKENS: int = 128
    _GEOMETRY_BY_SEQ_LEN = {
        # 3-view vertical strip (72x40)
        2880: (72, 40, 1),
        720: (36, 20, 2),
        180: (18, 10, 4),
        # 4-view 2x2 grid (48x80)
        3840: (48, 80, 1),
        960: (24, 40, 2),
        240: (12, 20, 4),
    }

    def __init__(
        self,
        *,
        layer_idx: int,
        start_layer_idx: int = 0,
        prune_ratio: float | None = None,
        patch_h: int = 2,
        patch_w: int = 2,
        noise_alpha: float = 0.1,
        sim_beta: float = 1.0,
        max_downsample_ratio: int = 4,
        plan_recompute_every: int = 0,
    ) -> None:
        self.layer_idx = layer_idx
        self.start_layer_idx = start_layer_idx
        self.max_downsample_ratio = max_downsample_ratio
        # plan_recompute_every > 0 caches the prune plan and only re-derives it
        # every N forward calls of this layer. The plan (which tokens to keep /
        # how to recover) is dominated by spatial structure that barely changes
        # across denoise steps, so reusing it removes the expensive `prepare()`
        # (score/argmax/patch/similarity) from ~all but 1 of every N calls.
        self.plan_recompute_every = int(plan_recompute_every)
        self._plan_cache: dict[int, object] = {}
        self._call_count: dict[int, int] = {}
        self.pruner = SiToTokenPruner(
            group_mode="full_2d",
            prune_ratio=prune_ratio,
            patch_h=patch_h,
            patch_w=patch_w,
            noise_alpha=noise_alpha,
            sim_beta=sim_beta,
            layer_idx=layer_idx,
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

        use_sito, geometry = self._should_use_sito(query_len=query_len, key_len=key_len)
        if use_sito:
            hidden_states, plan = self._run_sito_attention(attn, hidden_states, geometry=geometry)
        else:
            q = _split_heads(attn.to_q(hidden_states), num_heads)
            k = _split_heads(attn.to_k(kv_source), num_heads)
            v = _split_heads(attn.to_v(kv_source), num_heads)
            hidden_states = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0)
            hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous().reshape(batch_size, query_len, -1)
            plan = None

        hidden_states = hidden_states.to(dtype=hidden_states.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if plan is not None:
            hidden_states = self.pruner.recover(hidden_states, plan)
        return hidden_states

    def _run_sito_attention(self, attn, hidden_states: torch.Tensor, *, geometry: tuple[int, int, int]) -> tuple[torch.Tensor, object]:
        group_h, group_w, _ = geometry
        seq_len = hidden_states.shape[1]
        plan = self._get_plan(hidden_states, group_h=group_h, group_w=group_w, seq_len=seq_len)
        if plan is None:
            q = _split_heads(attn.to_q(hidden_states), attn.heads)
            k = _split_heads(attn.to_k(hidden_states), attn.heads)
            v = _split_heads(attn.to_v(hidden_states), attn.heads)
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            out = out.permute(0, 2, 1, 3).contiguous().reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
            return out, None

        pruned_states = self.pruner.prune(hidden_states, plan)
        q = _split_heads(attn.to_q(pruned_states), attn.heads)
        k = _split_heads(attn.to_k(pruned_states), attn.heads)
        v = _split_heads(attn.to_v(pruned_states), attn.heads)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(pruned_states.shape[0], pruned_states.shape[1], -1)
        return out, plan

    def _get_plan(self, hidden_states: torch.Tensor, *, group_h: int, group_w: int, seq_len: int):
        video_size = type("SpatialSize", (), {"T": 1, "H": group_h, "W": group_w})
        if self.plan_recompute_every <= 0:
            return self.pruner.prepare(hidden_states, video_size=video_size)

        count = self._call_count.get(seq_len, 0)
        cached = self._plan_cache.get(seq_len)
        # Recompute on the first call and every N-th call; otherwise reuse the
        # cached plan to skip the expensive prepare() step.
        if cached is None or (count % self.plan_recompute_every) == 0:
            cached = self.pruner.prepare(hidden_states, video_size=video_size)
            self._plan_cache[seq_len] = cached
        self._call_count[seq_len] = count + 1
        return cached

    def _should_use_sito(self, *, query_len: int, key_len: int) -> tuple[bool, tuple[int, int, int] | None]:
        if query_len != key_len:
            return False, None
        if self.layer_idx < self.start_layer_idx:
            return False, None
        if query_len < self.MIN_TOKENS:
            return False, None
        geometry = self._GEOMETRY_BY_SEQ_LEN.get(query_len)
        if geometry is None:
            return False, None
        if geometry[2] > self.max_downsample_ratio:
            return False, None
        return True, geometry
