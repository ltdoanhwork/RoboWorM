from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch_size, seq_len, channels = x.shape
    return x.reshape(batch_size, seq_len, num_heads, channels // num_heads).permute(0, 2, 1, 3).contiguous()


class SVDFasterCacheAttnProcessor:
    ELIGIBLE_SEQ_LENS = {180, 720, 2880}

    def __init__(
        self,
        *,
        runtime_state: dict,
        layer_idx: int,
        first_layers_fp: int = 2,
    ) -> None:
        self.runtime_state = runtime_state
        self.layer_idx = layer_idx
        self.first_layers_fp = first_layers_fp

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        kv_source = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        batch_size, query_len, _ = hidden_states.shape
        key_len = kv_source.shape[1]
        if not self._is_eligible(query_len=query_len, key_len=key_len):
            return self._dense_forward(
                attn,
                hidden_states,
                kv_source=kv_source,
                attention_mask=attention_mask,
            )

        history = self.runtime_state.setdefault("block_histories", {}).setdefault(self.layer_idx, [])
        step_idx = int(self.runtime_state.get("current_step_idx", -1))
        start_step = int(self.runtime_state.get("resolved_start_step", 0))
        block_interval = max(int(self.runtime_state.get("block_interval", 3)), 1)
        is_anchor_step = step_idx < start_step or ((step_idx - start_step) % block_interval == 0)
        if (
            not is_anchor_step
            and len(history) >= 2
            and history[-1].shape == hidden_states.shape
            and history[-2].shape == hidden_states.shape
        ):
            latest = history[-1]
            previous = history[-2]
            return latest + (latest - previous) * float(self.runtime_state.get("block_alpha", 0.3))

        out = self._dense_forward(
            attn,
            hidden_states,
            kv_source=kv_source,
            attention_mask=attention_mask,
        )
        history.append(out.detach().clone())
        if len(history) > 2:
            del history[:-2]
        return out

    def _is_eligible(self, *, query_len: int, key_len: int) -> bool:
        if self.layer_idx < self.first_layers_fp:
            return False
        if query_len != key_len:
            return False
        if query_len not in self.ELIGIBLE_SEQ_LENS:
            return False
        return True

    @staticmethod
    def _dense_forward(
        attn,
        hidden_states: torch.Tensor,
        *,
        kv_source: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        num_heads = attn.heads
        q = _split_heads(attn.to_q(hidden_states), num_heads)
        k = _split_heads(attn.to_k(kv_source), num_heads)
        v = _split_heads(attn.to_v(kv_source), num_heads)
        hidden_states = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0)
        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous().reshape(hidden_states.shape[0], -1, attn.heads * q.shape[-1])
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
