from __future__ import annotations

import math
import types
from dataclasses import dataclass, field
from typing import Any

import torch

try:
    from cosmos_predict2._src.imaginaire.utils import log
except Exception:  # pragma: no cover - test fallback for minimal environments
    class _FallbackLog:
        @staticmethod
        def info(*args, **kwargs):
            pass

    log = _FallbackLog()

from methods.cache_strategy.common import FasterCacheConfig


def resolve_fastercache_start_step(config: FasterCacheConfig, num_steps: int) -> int:
    if config.start_step > 0:
        return config.start_step
    return int(math.ceil(max(num_steps, 1) * 0.3))


def _history_for_shape(history: list[torch.Tensor], shape: torch.Size) -> bool:
    return len(history) >= 2 and history[-1].shape == shape and history[-2].shape == shape


def _split_fft_bands(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = x.reshape(x.shape[0], -1)
    freq = torch.fft.rfft(flat, dim=-1)
    split_idx = max(freq.shape[-1] // 4, 1)
    lf = freq.clone()
    hf = freq.clone()
    lf[..., split_idx:] = 0
    hf[..., :split_idx] = 0
    return lf, hf


def _compose_from_fft_bands(
    reference: torch.Tensor,
    delta_lf: torch.Tensor,
    delta_hf: torch.Tensor,
) -> torch.Tensor:
    ref_flat = reference.reshape(reference.shape[0], -1)
    ref_freq = torch.fft.rfft(ref_flat, dim=-1)
    recon = torch.fft.irfft(ref_freq + delta_lf + delta_hf, n=ref_flat.shape[-1], dim=-1)
    return recon.reshape_as(reference).type_as(reference)


def _clone_detached(x: torch.Tensor) -> torch.Tensor:
    return x.detach().clone()


def compute_fastercache_deltas(
    conditional: torch.Tensor,
    unconditional: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    lf_cond, hf_cond = _split_fft_bands(conditional)
    lf_uncond, hf_uncond = _split_fft_bands(unconditional)
    return _clone_detached(lf_uncond - lf_cond), _clone_detached(hf_uncond - hf_cond)


def reconstruct_fastercache_output(
    conditional: torch.Tensor,
    delta_lf: torch.Tensor,
    delta_hf: torch.Tensor,
) -> torch.Tensor:
    return _compose_from_fft_bands(conditional, delta_lf, delta_hf)


@dataclass
class FasterCacheRuntimeState:
    config: FasterCacheConfig
    cfg_mode: str
    total_steps: int = 0
    resolved_start_step: int = 0
    call_idx: int = 0
    current_step_idx: int = 0
    current_branch: str = "cond"
    pending_cond_output: torch.Tensor | None = None
    delta_lf: torch.Tensor | None = None
    delta_hf: torch.Tensor | None = None
    block_histories: dict[tuple[str, int], list[torch.Tensor]] = field(default_factory=dict)
    block_seq_len_min: int = 2048
    block_alpha: float = 0.3

    def reset(self, num_steps: int) -> None:
        self.total_steps = int(num_steps)
        self.resolved_start_step = resolve_fastercache_start_step(self.config, num_steps)
        self.call_idx = 0
        self.current_step_idx = 0
        self.current_branch = "cond"
        self.pending_cond_output = None
        self.delta_lf = None
        self.delta_hf = None
        self.block_histories.clear()

    def is_model_anchor_step(self, step_idx: int) -> bool:
        if step_idx < self.resolved_start_step:
            return True
        interval = max(int(self.config.model_interval), 1)
        return (step_idx - self.resolved_start_step) % interval == 0

    def is_block_anchor_step(self, step_idx: int) -> bool:
        if step_idx < self.resolved_start_step:
            return True
        interval = max(int(self.config.block_interval), 1)
        return (step_idx - self.resolved_start_step) % interval == 0


def initialize_fastercache_state(model, config: FasterCacheConfig, *, cfg_mode: str) -> FasterCacheRuntimeState:
    state = FasterCacheRuntimeState(config=config, cfg_mode=cfg_mode)
    state.reset(num_steps=1)
    model.fastercache_state = state
    model.fastercache_enabled = True
    model.fastercache_config = config
    model.has_fastercache_backend = True
    return state


def reset_fastercache_state(model, config: FasterCacheConfig, num_steps: int) -> int:
    state = getattr(model, "fastercache_state", None)
    if state is None:
        state = initialize_fastercache_state(model, config, cfg_mode="sequential")
    state.reset(num_steps=num_steps)
    return num_steps


def _get_dit_block_start_idx(num_blocks: int) -> int:
    if num_blocks >= 36:
        return 6
    if num_blocks >= 28:
        return 4
    if num_blocks >= 20:
        return 3
    return max(2, num_blocks // 6)


def _dit_layer_is_eligible(model, layer_idx: int) -> bool:
    num_blocks = int(getattr(model, "num_blocks", len(getattr(model, "blocks", []))))
    start_idx = _get_dit_block_start_idx(num_blocks)
    return start_idx <= layer_idx < max(num_blocks - 2, start_idx)


def _run_dit_model_shortcut(state: FasterCacheRuntimeState) -> bool:
    return (
        state.current_branch == "uncond"
        and not state.is_model_anchor_step(state.current_step_idx)
        and state.pending_cond_output is not None
        and state.delta_lf is not None
        and state.delta_hf is not None
    )


def _patch_dit_block_self_attention(model) -> None:
    state: FasterCacheRuntimeState = model.fastercache_state
    for layer_idx, block in enumerate(model.blocks):
        attn = getattr(block, "self_attn", None)
        if attn is None or hasattr(attn, "_fastercache_original_forward"):
            continue

        original_forward = attn.forward
        attn._fastercache_original_forward = original_forward
        attn._fastercache_layer_idx = layer_idx

        def _wrapped_forward(
            self,
            x: torch.Tensor,
            context: torch.Tensor | None = None,
            rope_emb: torch.Tensor | None = None,
            video_size=None,
            attention_runtime_context: dict | None = None,
            **kwargs,
        ):
            del attention_runtime_context
            if context is not None:
                return self._fastercache_original_forward(
                    x,
                    context=context,
                    rope_emb=rope_emb,
                    video_size=video_size,
                    **kwargs,
                )

            runtime_state: FasterCacheRuntimeState = model.fastercache_state
            seq_len = int(x.shape[1])
            layer_history = runtime_state.block_histories.setdefault(
                (runtime_state.current_branch, self._fastercache_layer_idx), []
            )

            can_reuse = (
                _dit_layer_is_eligible(model, self._fastercache_layer_idx)
                and seq_len >= runtime_state.block_seq_len_min
                and not runtime_state.is_block_anchor_step(runtime_state.current_step_idx)
                and _history_for_shape(layer_history, x.shape)
            )
            if can_reuse:
                latest = layer_history[-1]
                previous = layer_history[-2]
                return latest + (latest - previous) * runtime_state.block_alpha

            out = self._fastercache_original_forward(
                x,
                context=context,
                rope_emb=rope_emb,
                video_size=video_size,
                **kwargs,
            )
            layer_history.append(_clone_detached(out))
            if len(layer_history) > 2:
                del layer_history[:-2]
            return out

        attn.forward = types.MethodType(_wrapped_forward, attn)


def apply_fastercache(model, config: FasterCacheConfig, *, cfg_mode: str = "sequential"):
    if hasattr(model, "_fastercache_original_forward"):
        initialize_fastercache_state(model, config, cfg_mode=cfg_mode)
        return model

    state = initialize_fastercache_state(model, config, cfg_mode=cfg_mode)
    original_forward = model.forward
    model._fastercache_original_forward = original_forward
    _patch_dit_block_self_attention(model)

    def _wrapped_forward(self, *args, **kwargs):
        runtime_state: FasterCacheRuntimeState = self.fastercache_state
        if kwargs.get("use_cuda_graphs", False):
            raise ValueError("[FasterCache] FasterCache is incompatible with CUDA Graphs in v1.")

        runtime_state.current_step_idx = runtime_state.call_idx // 2
        runtime_state.current_branch = "cond" if runtime_state.call_idx % 2 == 0 else "uncond"

        if _run_dit_model_shortcut(runtime_state):
            output = _compose_from_fft_bands(
                runtime_state.pending_cond_output,
                runtime_state.delta_lf,
                runtime_state.delta_hf,
            )
        else:
            output = self._fastercache_original_forward(*args, **kwargs)

        if runtime_state.current_branch == "cond":
            runtime_state.pending_cond_output = _clone_detached(output)
        elif runtime_state.pending_cond_output is not None:
            cond_output = runtime_state.pending_cond_output
            runtime_state.delta_lf, runtime_state.delta_hf = compute_fastercache_deltas(cond_output, output)
            runtime_state.pending_cond_output = None

        runtime_state.call_idx += 1
        return output

    model.forward = types.MethodType(_wrapped_forward, model)
    log.info(
        f"[FasterCache] Applied DiT runtime: start_step={config.start_step} model_interval={config.model_interval} block_interval={config.block_interval} cfg_mode={cfg_mode}"
    )
    return model
