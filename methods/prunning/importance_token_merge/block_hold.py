from __future__ import annotations

"""Block-level "prune-once-hold" acceleration for Ctrl-World.

Motivation
----------
Profiling Ctrl-World showed that pruning tokens *around each attention* cannot
speed the model up: spatial self-attention is only ~7% of wall time, and the
merge/recover bookkeeping costs about as much as the attention FLOPs it saves.

The winning strategy (micro-benchmarked at 1.3-2.0x on the largest spatial
stage) is to prune the tokens ONCE at the input of a whole
``BasicTransformerBlock`` and keep them compressed through the ENTIRE block
(self-attn + cross-attn + feed-forward), only un-merging at the block output.
That way the feed-forward — the dominant cost of the block — also runs at the
reduced token count, and the prune/unmerge overhead is paid once per block
instead of once per attention.

This is implemented as forward hooks on ``BasicTransformerBlock`` so it works
without editing diffusers internals. Cross-attention keeps its full
``encoder_hidden_states`` (only the spatial query tokens are compressed), so
conditioning is untouched.
"""

from typing import Any

import torch

from .runtime import ITMTokenMerger, infer_ctrl_world_heat_map_shape


class _BlockHoldController:
    """Holds the merge plan for one BasicTransformerBlock across a forward pass."""

    def __init__(
        self,
        *,
        itm_state: dict,
        layer_idx: int,
        start_layer_idx: int,
        compress_ratio: float | None,
        max_downsample_ratio: int,
        self_importance: bool = True,
        plan_recompute_every: int = 0,
    ) -> None:
        self.itm_state = itm_state
        self.layer_idx = layer_idx
        self.start_layer_idx = start_layer_idx
        self.max_downsample_ratio = max_downsample_ratio
        self.self_importance = self_importance
        # The merge plan (which tokens to keep/merge) is driven by spatial
        # structure that is stable across denoise steps. ITM's prepare() is very
        # expensive (~12ms/block: randperm + argsort + similarity matmul), so
        # recomputing it every step is what makes block-hold slower than dense.
        # Caching it and only recomputing every N calls amortizes that cost.
        self.plan_recompute_every = int(plan_recompute_every)
        self._plan_cache: dict[int, Any] = {}
        self._call_count: dict[int, int] = {}
        self.merger = ITMTokenMerger(
            group_mode="full_2d_framewise",
            compress_ratio=compress_ratio,
            prune_from_step=0,
            merge_from_step=0,
            merge_attn=True,
            dst_mode="evenly_k",
            use_rand=False,
        )
        self._plan = None

    def _eligible(self, seq_len: int) -> Any | None:
        if self.layer_idx < self.start_layer_idx:
            return None
        if not self.self_importance:
            if self.itm_state.get("phase", "disabled") == "disabled":
                return None
            if self.itm_state.get("heat_map") is None:
                return None
        geometry = infer_ctrl_world_heat_map_shape(seq_len)
        if geometry is None:
            return None
        downsample_ratio = 72 // int(geometry.H)
        if downsample_ratio > self.max_downsample_ratio:
            return None
        return geometry

    def pre_hook(self, module, args, kwargs):
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if hidden_states is None or hidden_states.ndim != 3:
            self._plan = None
            return None
        batch_size, seq_len, _ = hidden_states.shape
        geometry = self._eligible(seq_len)
        if geometry is None:
            self._plan = None
            return None

        if self.self_importance:
            # Derive per-token importance directly from the hidden states (feature
            # L2 magnitude), so no classifier-free-guidance heat map is needed.
            # This lets ITM run at guidance_scale=1 (single batch) instead of
            # being forced to CFG=2, which otherwise doubles every UNet call.
            heat = hidden_states.float().norm(dim=-1).reshape(batch_size, geometry.H, geometry.W)
        else:
            heat = self._expand_heat_map(
                self.itm_state["heat_map"], batch_size=batch_size, height=geometry.H, width=geometry.W
            )
        if heat is None:
            self._plan = None
            return None

        plan = self._get_plan(hidden_states, heat=heat, seq_len=seq_len)
        if plan is None:
            self._plan = None
            return None

        self._plan = plan
        merged = self.merger.merge(hidden_states, plan, mode="mean")
        if args:
            new_args = (merged,) + tuple(args[1:])
            return new_args, kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = merged
        return args, kwargs

    def post_hook(self, module, args, kwargs, output):
        if self._plan is None:
            return output
        plan = self._plan
        self._plan = None
        if isinstance(output, tuple):
            unmerged = self.merger.unmerge(output[0], plan, mode="mean")
            return (unmerged,) + tuple(output[1:])
        return self.merger.unmerge(output, plan, mode="mean")

    def _get_plan(self, hidden_states, *, heat, seq_len):
        if self.plan_recompute_every <= 0:
            return self.merger.prepare(hidden_states, heat_map=heat)
        count = self._call_count.get(seq_len, 0)
        cached = self._plan_cache.get(seq_len)
        if cached is None or (count % self.plan_recompute_every) == 0:
            cached = self.merger.prepare(hidden_states, heat_map=heat)
            self._plan_cache[seq_len] = cached
        self._call_count[seq_len] = count + 1
        return cached

    def _expand_heat_map(self, heat_map, *, batch_size, height, width):
        import torch.nn.functional as F

        if heat_map.ndim != 4:
            return None
        expanded = heat_map.reshape(-1, heat_map.shape[-2], heat_map.shape[-1])
        resized = F.interpolate(expanded.unsqueeze(1), size=(height, width), mode="area").squeeze(1)
        if resized.shape[0] == batch_size:
            return resized
        if batch_size % resized.shape[0] == 0:
            return resized.repeat(batch_size // resized.shape[0], 1, 1)
        if resized.shape[0] % batch_size == 0:
            return resized[:: resized.shape[0] // batch_size]
        return None


def enable_itm_block_hold(
    unet,
    *,
    itm_state: dict,
    start_layer_idx: int = 0,
    compress_ratio: float | None = None,
    max_downsample_ratio: int = 4,
    self_importance: bool = True,
    plan_recompute_every: int = 0,
) -> list:
    """Attach prune-once-hold hooks to every spatial BasicTransformerBlock.

    When ``self_importance`` is True (default), per-token importance is derived
    from the hidden-state feature magnitude, so no CFG heat map is needed and the
    model can run at guidance_scale=1 (single batch). Set it False to reuse the
    classifier-free-guidance heat map produced by the pipeline.

    Returns the list of hook handles so callers can remove them later.
    """
    from diffusers.models.attention import BasicTransformerBlock

    itm_state.setdefault("phase", "disabled")
    itm_state.setdefault("heat_map", None)
    # Only expose _itm_state to the pipeline when we actually rely on its
    # CFG-driven heat map. In self-importance mode the hooks are fully
    # self-contained, and setting _itm_state would make the pipeline demand
    # classifier-free guidance (defeating the single-batch speedup).
    if not self_importance:
        unet._itm_state = itm_state

    handles = []
    block_idx = 0
    for module in unet.modules():
        if isinstance(module, BasicTransformerBlock):
            controller = _BlockHoldController(
                itm_state=itm_state,
                layer_idx=block_idx,
                start_layer_idx=start_layer_idx,
                compress_ratio=compress_ratio,
                max_downsample_ratio=max_downsample_ratio,
                self_importance=self_importance,
                plan_recompute_every=plan_recompute_every,
            )
            handles.append(module.register_forward_pre_hook(controller.pre_hook, with_kwargs=True))
            handles.append(module.register_forward_hook(controller.post_hook, with_kwargs=True))
            block_idx += 1

    unet._itm_block_hold_handles = handles
    print(
        f"[ITM-block] Enabled prune-once-hold on {block_idx} BasicTransformerBlocks "
        f"(start_layer_idx={start_layer_idx}, max_downsample_ratio={max_downsample_ratio}, "
        f"self_importance={self_importance})"
    )
    return handles


def disable_itm_block_hold(unet) -> None:
    for handle in getattr(unet, "_itm_block_hold_handles", []):
        handle.remove()
    if hasattr(unet, "_itm_block_hold_handles"):
        delattr(unet, "_itm_block_hold_handles")
    if hasattr(unet, "_itm_state"):
        delattr(unet, "_itm_state")
    print("[ITM-block] Disabled prune-once-hold hooks.")
