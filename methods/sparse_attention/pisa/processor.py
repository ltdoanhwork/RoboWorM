"""
SVD-compatible AttnProcessor wrapping the PISA kernel.

Ctrl-World's SVD UNet uses standard diffusers Attention modules:
- Spatial cross-attention: hidden_states (B*F, T_hw, D) + encoder_hidden_states (B*F, T_ctx, D)
- Temporal self-attention: hidden_states (B, F, D) — T=25 frames, very small

Design decisions
----------------
- For cross-attention (T_q != T_kv): always fall back to SDPA (PISA is for
  self-attention; approximate cross-attn would lose action conditioning fidelity).
- For self-attention with T < MIN_TOKENS_FOR_PISA: fall back to SDPA.
  With SVD default resolution (320x192), spatial T≈60 which is borderline.
- block_size is auto-clamped to T to avoid kernel errors.
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn.functional as F


class SVDPISAAttnProcessor:
    """
    Drop-in replacement for diffusers AttnProcessor2_0, powered by PISA.

    Falls back transparently to torch SDPA when:
      - Cross-attention (T_q != T_kv): PISA is designed for self-attention
      - T_q < MIN_TOKENS_FOR_PISA: overhead not worth it
      - layer_idx < start_layer_idx: user-specified skip
    """

    # Below this token count, PISA overhead > gain; use SDPA instead.
    MIN_TOKENS_FOR_PISA: int = 48

    def __init__(
        self,
        attn_fn,
        density: float = 0.5,
        block_size: int = 16,
        layer_idx: int = 0,
        start_layer_idx: int = 0,
    ):
        self.attn_fn = attn_fn
        self.density = density
        self.block_size = block_size
        self.layer_idx = layer_idx
        self.start_layer_idx = start_layer_idx

    # ------------------------------------------------------------------
    # diffusers calls processor(attn_module, hidden_states, ...)
    # ------------------------------------------------------------------
    def __call__(
        self,
        attn,                                         # Attention module
        hidden_states: torch.Tensor,                  # (B, T_q, D)
        encoder_hidden_states: Optional[torch.Tensor] = None,  # (B, T_kv, D) or None
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        B, T_q, _ = hidden_states.shape
        H = attn.heads

        # --- Project Q, K, V ------------------------------------------
        q = attn.to_q(hidden_states)
        kv_src = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        k = attn.to_k(kv_src)
        v = attn.to_v(kv_src)

        T_kv = kv_src.shape[1]

        # --- Reshape to (B, H, T, D_head) ------------------------------
        q = self._split_heads(q, H)   # (B, H, T_q, D_head)
        k = self._split_heads(k, H)   # (B, H, T_kv, D_head)
        v = self._split_heads(v, H)   # (B, H, T_kv, D_head)

        # --- Decide whether to use PISA --------------------------------
        is_self_attn = (T_q == T_kv)
        layer_eligible = (self.layer_idx >= self.start_layer_idx)
        long_enough = (T_q >= self.MIN_TOKENS_FOR_PISA)

        use_pisa = is_self_attn and layer_eligible and long_enough

        if use_pisa:
            # block_size must be ≤ T — clamp silently
            bs = min(self.block_size, T_q)
            hidden_states = self.attn_fn(
                q, k, v,
                density=self.density,
                block_size=bs,
            )
        else:
            # Standard SDPA (fused, no overhead)
            hidden_states = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=0.0,
            )

        # --- Merge heads: (B, H, T_q, D_head) → (B, T_q, D) ----------
        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        hidden_states = hidden_states.reshape(B, T_q, -1)
        hidden_states = hidden_states.to(q.dtype)

        # Output projection (linear + dropout)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    # ------------------------------------------------------------------
    @staticmethod
    def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """(B, T, D) → (B, H, T, D//H)"""
        B, T, D = x.shape
        return x.reshape(B, T, num_heads, D // num_heads).permute(0, 2, 1, 3).contiguous()
