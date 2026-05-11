"""
SVG1 Stripe Sparse Attention Processor for Ctrl-World SVD UNet.

Ported from WanAttn_SVGAttn_Processor2_0 (Sparse-VideoGen/svg/models/wan/attention.py)
with these adaptations for SVD UNet:

1. No joint text tokens — SVD keeps spatial and cross-attention separate.
   We skip cross-attention (encoder_hidden_states is not None) → fallback SDPA.

2. No joint timestep argument — SVD processor API does not receive the
   denoising timestep. We replace timestep-based warmup with layer-index
   based warmup (first_layers_fp).

3. FlexAttention → masked SDPA — torch 2.3 in this env does not have
   torch.nn.attention.flex_attention. We use F.scaled_dot_product_attention
   with a precomputed boolean mask (same math, slightly higher memory).

4. Online profiling — retained from SVG1: at each call we sample
   num_sampled_rows query rows, compute MSE vs full attention for each
   candidate mask, and pick the better pattern. MSE sampling itself uses
   full matmul on the sampled slice, so it is always correct.

5. Dynamic mask per T — masks are built lazily and cached per sequence length.
   The precomputed class-level masks were for T=2880 only; now each layer
   (L0=2880, L1=720, L2=180) gets its own correctly-sized [T,T] mask.

Ctrl-World SVD UNet spatial token counts (72x40 latent, 3-view stacked):
  - Temporal self-attn: T=11 frames      → SDPA fallback (T < MIN_TOKENS)
  - Mid-block (9x5):   T=45              → SDPA fallback (T < MIN_TOKENS)
  - L2 spatial (18x10): T=180            → SVG1 sparse
  - L1 spatial (36x20): T=720            → SVG1 sparse
  - L0 spatial (72x40): T=2880 (BOTTLENECK) → SVG1 sparse
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import torch
import torch.nn.functional as F


class SVDSVGAttnProcessor:
    """
    Drop-in replacement for diffusers AttnProcessor2_0, powered by SVG1.

    Class-level attributes (shared across all instances, set by adapter.py):
        sparsity:        float  — fraction of tokens attended per row.
        num_sampled_rows: int  — rows sampled for MSE profiling.
        first_layers_fp: int   — layers below this index use full SDPA.
        num_views, H_per_view, W: geometry for view-stripe mask at T=T_full.
    """

    # ── shared class-level state (set by adapter.enable_svg) ──────────────
    sparsity: float = 0.25
    num_sampled_rows: int = 64
    first_layers_fp: int = 2
    MIN_TOKENS: int = 128   # skip temporal (T=11) and mid (T=45)
    # Geometry — set by adapter so masks can be rebuilt for any T
    num_views: int = 3
    H_per_view: int = 24
    W: int = 40

    def __init__(self, layer_idx: int, num_layers: int):
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        # Per-instance mask cache keyed by T → List[Tensor([T,T] bool)]
        self._mask_cache: dict = {}

    # ── diffusers callback ─────────────────────────────────────────────────
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,                  # (B, T, D)
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        B, T_q, D = hidden_states.shape
        H = attn.heads

        # ── project Q, K, V ────────────────────────────────────────────────
        q = attn.to_q(hidden_states)
        kv_src = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        k = attn.to_k(kv_src)
        v = attn.to_v(kv_src)
        T_kv = kv_src.shape[1]

        q = _split_heads(q, H)   # (B, H, T_q, D_head)
        k = _split_heads(k, H)
        v = _split_heads(v, H)

        # ── decide: full vs sparse ──────────────────────────────────────────
        is_self_attn = (T_q == T_kv)
        long_enough  = (T_q >= self.MIN_TOKENS)
        early_layer  = (self.layer_idx < self.first_layers_fp)

        use_sparse = is_self_attn and long_enough and (not early_layer)

        if use_sparse:
            hidden_states = self._svg1_attention(q, k, v, T_q)
        else:
            hidden_states = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=0.0
            )

        # ── merge heads → output proj ───────────────────────────────────────
        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous().reshape(B, T_q, -1)
        hidden_states = hidden_states.to(q.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    # ── SVG1 core logic ────────────────────────────────────────────────────
    def _svg1_attention(
        self,
        q: torch.Tensor,   # (B, H, T, D_h)
        k: torch.Tensor,
        v: torch.Tensor,
        T: int,
    ) -> torch.Tensor:
        """
        Online profiling: sample MSE between candidate masks, pick the best,
        then apply masked SDPA with the winning mask.
        """
        masks = self._get_masks_for_T(T, device=q.device)
        best_mask_idx = self._select_best_mask(q, k, v, T, masks)
        return self._masked_sparse_attention(q, k, v, best_mask_idx, T, masks)

    # ── mask cache: lazily build [T,T] bool masks per T ───────────────────
    def _get_masks_for_T(self, T: int, device: torch.device) -> List[torch.Tensor]:
        """
        Return two [T, T] boolean attention masks for the given sequence length.
        Cached per T so the build cost is paid only once per layer.
        """
        if T in self._mask_cache:
            return [m.to(device) for m in self._mask_cache[T]]

        from .mask_utils import _build_diagonal_mask, _build_view_stripe_mask, sparsity_to_bandwidth

        bandwidth = sparsity_to_bandwidth(T, self.sparsity)
        mask0 = _build_diagonal_mask(T, bandwidth)            # [T, T] bool

        # View-stripe mask only makes sense when T equals the full L0 resolution.
        # For smaller layers (L1=720, L2=180), use a wider diagonal as candidate 1.
        T_full = self.num_views * self.H_per_view * self.W   # e.g. 3*24*40 = 2880
        if T == T_full:
            mask1 = _build_view_stripe_mask(self.num_views, self.H_per_view, self.W)
        else:
            wider_bw = min(T, sparsity_to_bandwidth(T, min(self.sparsity * 2, 0.6)))
            mask1 = _build_diagonal_mask(T, wider_bw)

        self._mask_cache[T] = [mask0, mask1]
        return [mask0.to(device), mask1.to(device)]

    def _select_best_mask(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        T: int,
        masks: List[torch.Tensor],   # each [T, T] bool, on correct device
    ) -> int:
        """
        Sample num_sampled_rows rows from q, compare per-mask MSE vs full attn.
        Returns the index of the winning mask (scalar int).
        """
        B, H, _, D_h = q.shape
        n_rows = min(self.num_sampled_rows, T)
        sampled_rows = torch.randint(0, T, (n_rows,), device=q.device)

        q_sampled = q[:, :, sampled_rows, :]                  # (B, H, n_rows, D_h)
        scale = D_h ** -0.5
        scores_full = torch.matmul(q_sampled, k.transpose(-2, -1)) * scale  # (B, H, n_rows, T)
        attn_full   = F.softmax(scores_full, dim=-1)
        out_full    = torch.matmul(attn_full, v)               # (B, H, n_rows, D_h)

        num_masks = len(masks)
        mses = torch.zeros(num_masks, B, H, device=q.device, dtype=q.dtype)

        for mask_idx, mask in enumerate(masks):
            # mask: [T, T] bool — index sampled rows → (n_rows, T) ← same T as scores
            mask_rows = mask[sampled_rows, :]                  # (n_rows, T)

            scores_m = scores_full.masked_fill(
                ~mask_rows.unsqueeze(0).unsqueeze(0),          # (1, 1, n_rows, T) ✓
                float("-inf"),
            )
            attn_m = F.softmax(scores_m, dim=-1)
            attn_m = torch.nan_to_num(attn_m, nan=0.0)
            out_m  = torch.matmul(attn_m, v)                   # (B, H, n_rows, D_h)

            mse = ((out_m - out_full) ** 2).mean(dim=(-2, -1))  # (B, H)
            mses[mask_idx] = mse

        # Global majority vote across all B × H
        best_per_bh = mses.argmin(dim=0)          # (B, H)
        flat = best_per_bh.flatten()
        counts = torch.bincount(flat, minlength=num_masks)
        return int(counts.argmax().item())         # scalar

    def _masked_sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        best_mask_idx: int,
        T: int,
        masks: List[torch.Tensor],   # each [T, T] bool, on correct device
    ) -> torch.Tensor:
        """Apply the winning [T,T] bool mask as an additive float mask in SDPA."""
        full_mask = masks[best_mask_idx]   # [T, T] bool

        # True → attend (0.0), False → mask out (-inf)
        additive = torch.zeros(T, T, device=q.device, dtype=q.dtype)
        additive = additive.masked_fill(~full_mask, float("-inf"))

        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=additive,
            dropout_p=0.0,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """(B, T, D) → (B, H, T, D//H)"""
    B, T, D = x.shape
    return x.reshape(B, T, num_heads, D // num_heads).permute(0, 2, 1, 3).contiguous()
