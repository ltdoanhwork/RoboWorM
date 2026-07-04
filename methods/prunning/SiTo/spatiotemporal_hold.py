from __future__ import annotations

"""Spatio-temporal prune-once-hold acceleration for Ctrl-World.

Ctrl-World's spatial `BasicTransformerBlock` alone is only ~16% of UNet time, so
pruning tokens around it cannot make the model much faster. The
`TransformerSpatioTemporalModel`, however, runs BOTH a spatial block and a
temporal block per layer, and the temporal block's effective batch dimension is
``H*W`` (one time-sequence per spatial token). So pruning spatial tokens ONCE at
the module input and holding them compressed through the spatial AND temporal
blocks shrinks both — targeting ~38% of UNet time instead of ~16%.

We override `TransformerSpatioTemporalModel.forward` (monkeypatch per instance)
so tokens are pruned right after `proj_in`, kept compressed through every
spatial/temporal block, and un-merged just before `proj_out`. The recover is a
single `index_select` (cheap), and the prune plan is computed once per episode
(cached), so the selection overhead is amortized to near zero.
"""

import types
from typing import Optional

import torch

try:
    from diffusers.models.transformers.transformer_temporal import (
        TransformerSpatioTemporalModel,
        TransformerTemporalModelOutput,
    )
except Exception:  # pragma: no cover - import path fallback
    from diffusers.models.transformer_temporal import (  # type: ignore
        TransformerSpatioTemporalModel,
        TransformerTemporalModelOutput,
    )


_GEOMETRY_BY_TOKENS = {2880: (72, 40), 720: (36, 20), 180: (18, 10)}


class _SpatioTemporalPruner:
    """Builds and caches a token-keep plan for one TransformerSpatioTemporalModel."""

    def __init__(self, *, keep_ratio: float, min_tokens: int, max_downsample_ratio: int, recompute_every: int, similarity_recover: bool = True):
        self.keep_ratio = keep_ratio
        self.min_tokens = min_tokens
        self.max_downsample_ratio = max_downsample_ratio
        self.recompute_every = int(recompute_every)
        self.similarity_recover = similarity_recover
        self._keep_idx = None
        self._gather = None
        self._n = None
        self._calls = 0

    def eligible(self, num_tokens: int) -> bool:
        if num_tokens < self.min_tokens:
            return False
        geom = _GEOMETRY_BY_TOKENS.get(num_tokens)
        if geom is None:
            return False
        downsample = 72 // geom[0]
        return downsample <= self.max_downsample_ratio

    def plan(self, hidden_states: torch.Tensor):
        """hidden_states: (batch_frames, N, C). Returns (keep_idx, gather_idx)."""
        n = hidden_states.shape[1]
        need = self._keep_idx is None or self._n != n
        if self.recompute_every > 0 and not need:
            if (self._calls % self.recompute_every) != 0:
                self._calls += 1
                return self._keep_idx, self._gather
        with torch.no_grad():
            feat = hidden_states.float().mean(dim=0)  # (N, C) batch-averaged token features
            score = feat.norm(dim=-1)  # importance = magnitude
            keep = max(int(round(n * self.keep_ratio)), 1)
            keep_idx = torch.topk(score, keep, sorted=False).indices.sort().values
            gather = torch.empty(n, dtype=torch.long, device=hidden_states.device)
            gather[keep_idx] = torch.arange(keep, device=hidden_states.device)
            if keep < n:
                mask = torch.ones(n, dtype=torch.bool, device=hidden_states.device)
                mask[keep_idx] = False
                pruned = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                if self.similarity_recover:
                    # ITM-style: each pruned token recovers from the most SIMILAR
                    # kept token (cosine), not the spatially nearest index. This
                    # preserves appearance far better at the same keep ratio.
                    kf = torch.nn.functional.normalize(feat.index_select(0, keep_idx), dim=-1)
                    pf = torch.nn.functional.normalize(feat.index_select(0, pruned), dim=-1)
                    sim = torch.mm(pf, kf.t())  # (pruned, keep)
                    gather[pruned] = sim.argmax(dim=1)
                else:
                    pos = torch.searchsorted(keep_idx, pruned, right=True) - 1
                    gather[pruned] = pos.clamp(min=0)
        self._keep_idx, self._gather, self._n = keep_idx, gather, n
        self._calls += 1
        return keep_idx, gather


def _make_forward(pruner: _SpatioTemporalPruner):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        batch_frames, _, height, width = hidden_states.shape
        num_frames = image_only_indicator.shape[-1]
        batch_size = batch_frames // num_frames

        time_context = encoder_hidden_states
        time_context_first_timestep = time_context[None, :].reshape(
            batch_size, num_frames, -1, time_context.shape[-1]
        )[:, 0]
        time_context = time_context_first_timestep[:, None].broadcast_to(
            batch_size, height * width, time_context.shape[-2], time_context.shape[-1]
        )
        # kept below (after we know the pruned token count)

        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        num_tokens = height * width
        use_prune = pruner.eligible(num_tokens)
        keep_idx = gather = None
        if use_prune:
            keep_idx, gather = pruner.plan(hidden_states)
            hidden_states = hidden_states.index_select(1, keep_idx)
            tokens_now = keep_idx.numel()
        else:
            tokens_now = num_tokens

        # time_context batch dim must match (batch_size * tokens_now)
        if use_prune:
            tc = time_context.reshape(batch_size, height * width, time_context.shape[-2], time_context.shape[-1])
            tc = tc.index_select(1, keep_idx)
            time_context = tc.reshape(batch_size * tokens_now, -1, time_context.shape[-1])
        else:
            time_context = time_context.reshape(batch_size * height * width, -1, time_context.shape[-1])

        num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
        num_frames_emb = num_frames_emb.repeat(batch_size, 1).reshape(-1)
        t_emb = self.time_proj(num_frames_emb).to(dtype=hidden_states.dtype)
        emb = self.time_pos_embed(t_emb)[:, None, :]

        for block, temporal_block in zip(self.transformer_blocks, self.temporal_transformer_blocks):
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states_mix = hidden_states + emb
            hidden_states_mix = temporal_block(
                hidden_states_mix,
                num_frames=num_frames,
                encoder_hidden_states=time_context,
            )
            hidden_states = self.time_mixer(
                x_spatial=hidden_states,
                x_temporal=hidden_states_mix,
                image_only_indicator=image_only_indicator,
            )

        if use_prune:
            hidden_states = hidden_states.index_select(1, gather)

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        output = hidden_states + residual

        if not return_dict:
            return (output,)
        return TransformerTemporalModelOutput(sample=output)

    return forward


def enable_spatiotemporal_hold(
    unet,
    *,
    keep_ratio: float = 0.5,
    min_tokens: int = 128,
    max_downsample_ratio: int = 1,
    recompute_every: int = 999999,
    similarity_recover: bool = True,
) -> int:
    """Monkeypatch every TransformerSpatioTemporalModel to prune-once-hold.

    Returns the number of patched modules.
    """
    count = 0
    for module in unet.modules():
        if isinstance(module, TransformerSpatioTemporalModel):
            pruner = _SpatioTemporalPruner(
                keep_ratio=keep_ratio,
                min_tokens=min_tokens,
                max_downsample_ratio=max_downsample_ratio,
                recompute_every=recompute_every,
                similarity_recover=similarity_recover,
            )
            module._orig_forward = module.forward
            module.forward = types.MethodType(_make_forward(pruner), module)
            count += 1
    print(
        f"[ST-hold] Enabled spatio-temporal prune-once-hold on {count} modules "
        f"(keep_ratio={keep_ratio}, max_downsample_ratio={max_downsample_ratio}, "
        f"recompute_every={recompute_every}, similarity_recover={similarity_recover})"
    )
    return count


def disable_spatiotemporal_hold(unet) -> None:
    for module in unet.modules():
        if hasattr(module, "_orig_forward"):
            module.forward = module._orig_forward
            delattr(module, "_orig_forward")
    print("[ST-hold] Disabled.")
