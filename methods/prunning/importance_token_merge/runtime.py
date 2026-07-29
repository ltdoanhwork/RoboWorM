from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

import torch
import torch.nn.functional as F


GroupMode = Literal["per_frame", "full_2d_framewise"]
MergeMode = Literal["mean", "none"]
ITMPhase = Literal["disabled", "prune", "merge"]

_AUTO_COMPRESS_SCHEDULE = (
    (2048, 0.35),
    (512, 0.25),
    (128, 0.15),
)


@dataclass(frozen=True)
class ITMLocalPlan:
    token_count: int
    a_idx: torch.Tensor
    b_idx: torch.Tensor
    unm_idx: torch.Tensor
    src_idx: torch.Tensor
    dst_assign_idx: torch.Tensor
    output_idx: torch.Tensor

    @property
    def merged_token_count(self) -> int:
        return int(self.output_idx.numel())


@dataclass(frozen=True)
class ITMRuntimePlan:
    group_mode: GroupMode
    local_plans: tuple[ITMLocalPlan, ...]
    group_count: int
    original_batch: int
    original_tokens_per_group: int
    merged_tokens_per_group: int

    @property
    def original_length(self) -> int:
        return self.group_count * self.original_tokens_per_group

    @property
    def merged_length(self) -> int:
        return self.group_count * self.merged_tokens_per_group


def resolve_itm_auto_compress_ratio(tokens_per_group: int) -> float:
    for min_tokens, ratio in _AUTO_COMPRESS_SCHEDULE:
        if tokens_per_group >= min_tokens:
            return ratio
    return 0.0


def _default_start_layer_idx(num_blocks: int) -> int:
    return 6 if num_blocks >= 36 else 4


def _validate_itm_args(
    *,
    compress_ratio: float | None,
    sx: int,
    sy: int,
    kk: float,
    pp: float,
    start_layer_idx: int,
    keep_last_n_dense: int,
    group_mode: GroupMode,
    prune_from_step: int,
    merge_from_step: int,
) -> None:
    if compress_ratio is not None and not 0.0 <= compress_ratio < 0.75:
        raise ValueError(f"`compress_ratio` must be in [0, 0.75), got {compress_ratio}.")
    if sx <= 0 or sy <= 0:
        raise ValueError(f"`sx` and `sy` must be positive, got {(sx, sy)}.")
    if not 0.0 < kk < 1.0:
        raise ValueError(f"`kk` must be in (0, 1), got {kk}.")
    if pp <= 0.0:
        raise ValueError(f"`pp` must be positive, got {pp}.")
    if start_layer_idx < 0:
        raise ValueError(f"`start_layer_idx` must be non-negative, got {start_layer_idx}.")
    if keep_last_n_dense < 0:
        raise ValueError(f"`keep_last_n_dense` must be non-negative, got {keep_last_n_dense}.")
    if prune_from_step < 0:
        raise ValueError(f"`prune_from_step` must be non-negative, got {prune_from_step}.")
    if merge_from_step < prune_from_step:
        raise ValueError(
            f"`merge_from_step` must be >= `prune_from_step`, got {merge_from_step} < {prune_from_step}."
        )
    if group_mode not in {"per_frame", "full_2d_framewise"}:
        raise ValueError(
            f"`group_mode` must be 'per_frame' or 'full_2d_framewise', got {group_mode!r}."
        )


def build_itm_parameters(
    num_blocks: int,
    *,
    start_layer_idx: int | None = None,
    keep_last_n_dense: int = 2,
    compress_ratio: float | None = None,
    prune_from_step: int = 1,
    merge_from_step: int = 2,
    merge_attn: bool = True,
    merge_crossattn: bool = False,
    merge_mlp: bool = False,
    group_mode: GroupMode = "per_frame",
    sx: int = 2,
    sy: int = 2,
    kk: float = 0.25,
    pp: float = 1.0,
    dst_mode: str = "random",
    use_rand: bool = True,
) -> list[dict[str, Any] | None]:
    if num_blocks <= 0:
        raise ValueError(f"`num_blocks` must be positive, got {num_blocks}.")
    resolved_start = _default_start_layer_idx(num_blocks) if start_layer_idx is None else start_layer_idx
    _validate_itm_args(
        compress_ratio=compress_ratio,
        sx=sx,
        sy=sy,
        kk=kk,
        pp=pp,
        start_layer_idx=resolved_start,
        keep_last_n_dense=keep_last_n_dense,
        group_mode=group_mode,
        prune_from_step=prune_from_step,
        merge_from_step=merge_from_step,
    )
    dense_tail_start = max(num_blocks - keep_last_n_dense, resolved_start)
    return [
        None
        if layer_idx < resolved_start or layer_idx >= dense_tail_start
        else {
            "layer_idx": layer_idx,
            "group_mode": group_mode,
            "compress_ratio": compress_ratio,
            "prune_from_step": prune_from_step,
            "merge_from_step": merge_from_step,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp,
            "sx": sx,
            "sy": sy,
            "kk": kk,
            "pp": pp,
            "dst_mode": dst_mode,
            "use_rand": use_rand,
        }
        for layer_idx in range(num_blocks)
    ]


def compute_itm_heat_map(
    cond_tensor: torch.Tensor,
    uncond_tensor: torch.Tensor,
    *,
    patch_spatial: int,
) -> torch.Tensor:
    if patch_spatial <= 0:
        raise ValueError(f"`patch_spatial` must be positive, got {patch_spatial}.")
    if cond_tensor.shape != uncond_tensor.shape:
        raise ValueError(
            f"`cond_tensor` and `uncond_tensor` must have the same shape, got {cond_tensor.shape} and {uncond_tensor.shape}."
        )
    if cond_tensor.ndim != 5:
        raise ValueError(
            f"`cond_tensor` and `uncond_tensor` must have shape (B, C, T, H, W), got {cond_tensor.shape}."
        )

    heat_map = torch.abs(cond_tensor - uncond_tensor).mean(dim=1)  # (B, T, H, W)
    batch_size, frames, height, width = heat_map.shape
    token_h = height // patch_spatial
    token_w = width // patch_spatial
    heat_map = F.interpolate(
        heat_map.reshape(batch_size * frames, 1, height, width),
        size=(token_h, token_w),
        mode="area",
    )
    return heat_map.reshape(batch_size, frames, token_h, token_w)


class ITMTokenMerger:
    def __init__(
        self,
        *,
        group_mode: GroupMode = "per_frame",
        compress_ratio: float | None = None,
        prune_from_step: int = 1,
        merge_from_step: int = 2,
        merge_attn: bool = True,
        merge_crossattn: bool = False,
        merge_mlp: bool = False,
        sx: int = 2,
        sy: int = 2,
        kk: float = 0.25,
        pp: float = 1.0,
        dst_mode: str = "random",
        use_rand: bool = True,
        layer_idx: int | None = None,
    ) -> None:
        _validate_itm_args(
            compress_ratio=compress_ratio,
            sx=sx,
            sy=sy,
            kk=kk,
            pp=pp,
            start_layer_idx=0,
            keep_last_n_dense=0,
            group_mode=group_mode,
            prune_from_step=prune_from_step,
            merge_from_step=merge_from_step,
        )
        self.group_mode = group_mode
        self.compress_ratio = compress_ratio
        self.prune_from_step = prune_from_step
        self.merge_from_step = merge_from_step
        self.merge_attn = merge_attn
        self.merge_crossattn = merge_crossattn
        self.merge_mlp = merge_mlp
        self.sx = sx
        self.sy = sy
        self.kk = kk
        self.pp = pp
        self.dst_mode = dst_mode
        self.use_rand = use_rand
        self.layer_idx = layer_idx

    def get_phase(self, diffusion_step_idx: int | None) -> ITMPhase:
        if diffusion_step_idx is None:
            return "disabled"
        if diffusion_step_idx < self.prune_from_step:
            return "disabled"
        if diffusion_step_idx < self.merge_from_step:
            return "prune"
        return "merge"

    def get_merge_mode(self, diffusion_step_idx: int | None) -> MergeMode:
        return "none" if self.get_phase(diffusion_step_idx) == "prune" else "mean"

    def is_active(self, diffusion_step_idx: int | None) -> bool:
        return self.get_phase(diffusion_step_idx) != "disabled"

    def prepare(
        self,
        hidden_states: torch.Tensor,
        *,
        heat_map: torch.Tensor | None,
        video_size: Any | None = None,
    ) -> ITMRuntimePlan | None:
        if heat_map is None:
            return None
        if hidden_states.ndim != 3:
            raise ValueError(f"`hidden_states` must have shape (B, N, D), got {tuple(hidden_states.shape)}.")
        if self.group_mode == "per_frame":
            if video_size is None:
                raise ValueError("`video_size` is required for per-frame ITM.")
            return self._prepare_per_frame(hidden_states, heat_map=heat_map, video_size=video_size)
        return self._prepare_full_2d_framewise(hidden_states, heat_map=heat_map)

    def merge(self, hidden_states: torch.Tensor, plan: ITMRuntimePlan | None, *, mode: MergeMode) -> torch.Tensor:
        if plan is None:
            return hidden_states
        if plan.group_mode == "per_frame":
            batch_size, _, channels = hidden_states.shape
            grouped = hidden_states.view(
                batch_size,
                plan.group_count,
                plan.original_tokens_per_group,
                channels,
            )
            merged_groups = [
                self._merge_local(grouped[:, group_idx], plan.local_plans[group_idx], mode=mode)
                for group_idx in range(plan.group_count)
            ]
            return torch.cat(merged_groups, dim=1)

        return self._merge_local(hidden_states, plan.local_plans[0], mode=mode)

    def unmerge(self, hidden_states: torch.Tensor, plan: ITMRuntimePlan | None, *, mode: MergeMode) -> torch.Tensor:
        if plan is None:
            return hidden_states
        if plan.group_mode == "per_frame":
            batch_size, _, channels = hidden_states.shape
            grouped = hidden_states.view(
                batch_size,
                plan.group_count,
                plan.merged_tokens_per_group,
                channels,
            )
            unmerged_groups = [
                self._unmerge_local(grouped[:, group_idx], plan.local_plans[group_idx], mode=mode)
                for group_idx in range(plan.group_count)
            ]
            return torch.cat(unmerged_groups, dim=1)

        return self._unmerge_local(hidden_states, plan.local_plans[0], mode=mode)

    def merge_rope(self, rope_emb: torch.Tensor | None, plan: ITMRuntimePlan | None) -> torch.Tensor | None:
        if rope_emb is None or plan is None:
            return rope_emb
        if rope_emb.shape[0] != plan.original_length:
            return rope_emb
        if plan.group_mode == "per_frame":
            offsets = []
            base = 0
            for local_plan in plan.local_plans:
                offsets.append(local_plan.output_idx + base)
                base += local_plan.token_count
            merged_idx = torch.cat(offsets, dim=0)
            return rope_emb.index_select(0, merged_idx.to(rope_emb.device))
        return rope_emb.index_select(0, plan.local_plans[0].output_idx.to(rope_emb.device))

    def _prepare_per_frame(
        self,
        hidden_states: torch.Tensor,
        *,
        heat_map: torch.Tensor,
        video_size: Any,
    ) -> ITMRuntimePlan | None:
        batch_size, seq_len, channels = hidden_states.shape
        frames = int(video_size.T)
        height = int(video_size.H)
        width = int(video_size.W)
        tokens_per_group = height * width
        if frames * tokens_per_group != seq_len:
            raise ValueError(
                f"Invalid per-frame geometry: seq_len={seq_len}, video_size={(frames, height, width)}."
            )
        if heat_map.shape != (batch_size, frames, height, width):
            raise ValueError(
                f"Expected ITM heat map shape {(batch_size, frames, height, width)}, got {tuple(heat_map.shape)}."
            )
        grouped_states = hidden_states.view(batch_size, frames, tokens_per_group, channels)
        state_summary = grouped_states.mean(dim=0)
        heat_summary = heat_map.mean(dim=0)
        local_plans = []
        for frame_idx in range(frames):
            local_plan = self._build_local_plan(
                state_summary[frame_idx],
                heat_summary[frame_idx],
            )
            if local_plan is None:
                return None
            local_plans.append(local_plan)
        merged_tokens = local_plans[0].merged_token_count
        return ITMRuntimePlan(
            group_mode="per_frame",
            local_plans=tuple(local_plans),
            group_count=frames,
            original_batch=batch_size,
            original_tokens_per_group=tokens_per_group,
            merged_tokens_per_group=merged_tokens,
        )

    def _prepare_full_2d_framewise(
        self,
        hidden_states: torch.Tensor,
        *,
        heat_map: torch.Tensor,
    ) -> ITMRuntimePlan | None:
        batch_size, seq_len, channels = hidden_states.shape
        if heat_map.ndim != 3:
            raise ValueError(
                f"`heat_map` must have shape (B, H, W) for Ctrl-World ITM, got {tuple(heat_map.shape)}."
            )
        if heat_map.shape[0] != batch_size:
            raise ValueError(
                f"Heat map batch size mismatch: expected {batch_size}, got {heat_map.shape[0]}."
            )
        height, width = int(heat_map.shape[1]), int(heat_map.shape[2])
        if height * width != seq_len:
            raise ValueError(
                f"Heat map geometry {(height, width)} does not match seq_len={seq_len}."
            )
        local_plan = self._build_local_plan(hidden_states.mean(dim=0), heat_map.mean(dim=0))
        if local_plan is None:
            return None
        return ITMRuntimePlan(
            group_mode="full_2d_framewise",
            local_plans=(local_plan,),
            group_count=1,
            original_batch=batch_size,
            original_tokens_per_group=seq_len,
            merged_tokens_per_group=local_plan.merged_token_count,
        )

    def _build_local_plan(self, metric_summary: torch.Tensor, heat_summary: torch.Tensor) -> ITMLocalPlan | None:
        token_count = int(metric_summary.shape[0])
        compress_ratio = self.compress_ratio
        if compress_ratio is None:
            compress_ratio = resolve_itm_auto_compress_ratio(token_count)
        if compress_ratio <= 0.0:
            return None
        remove_count = int(token_count * compress_ratio)
        if remove_count <= 0:
            return None

        heat_flat = heat_summary.reshape(-1).to(metric_summary.device, dtype=torch.float32)
        if heat_flat.numel() != token_count:
            raise ValueError(
                f"Heat map token count mismatch: expected {token_count}, got {heat_flat.numel()}."
            )

        num_dst = max(int(token_count * self.kk), 1)
        set0_size = min(max(int((token_count - remove_count) * self.pp), num_dst), token_count)

        heat_idx = heat_flat.argsort(descending=True)
        set0_idx = heat_idx[:set0_size]
        heat_threshold = heat_flat[set0_idx[-1]]

        if self.dst_mode == "random":
            if self.use_rand:
                perm = torch.randperm(set0_size, device=metric_summary.device)
            else:
                perm = torch.arange(set0_size, device=metric_summary.device)
            b_idx = set0_idx.index_select(0, perm[:num_dst])
        elif self.dst_mode == "top_k":
            b_idx = set0_idx[:num_dst]
        elif self.dst_mode == "bottom_k":
            b_idx = set0_idx[-num_dst:]
        elif self.dst_mode == "evenly_k":
            stride = max(set0_size // num_dst, 1)
            b_idx = set0_idx[::stride][:num_dst]
        else:
            raise ValueError(f"Unsupported `dst_mode`: {self.dst_mode!r}.")

        dst_mask = torch.zeros(token_count, dtype=torch.bool, device=metric_summary.device)
        dst_mask[b_idx] = True
        a_idx = torch.arange(token_count, device=metric_summary.device)[~dst_mask]

        src_metric = F.normalize(metric_summary.index_select(0, a_idx).float(), dim=-1)
        dst_metric = F.normalize(metric_summary.index_select(0, b_idx).float(), dim=-1)
        similarity = torch.mm(src_metric, dst_metric.t())
        node_max, node_idx = similarity.max(dim=-1)

        src_heat = heat_flat.index_select(0, a_idx)
        low_importance_mask = src_heat < heat_threshold
        node_max = node_max.clone()
        node_max[low_importance_mask] = node_max[low_importance_mask] + 100000.0

        remove_count = min(remove_count, a_idx.numel())
        edge_idx = node_max.argsort(descending=True)
        src_idx = edge_idx[:remove_count]
        unm_idx = edge_idx[remove_count:]
        dst_assign_idx = node_idx.index_select(0, src_idx)
        output_idx = torch.cat((a_idx.index_select(0, unm_idx), b_idx), dim=0)

        return ITMLocalPlan(
            token_count=token_count,
            a_idx=a_idx,
            b_idx=b_idx,
            unm_idx=unm_idx,
            src_idx=src_idx,
            dst_assign_idx=dst_assign_idx,
            output_idx=output_idx,
        )

    def _merge_local(self, hidden_states: torch.Tensor, local_plan: ITMLocalPlan, *, mode: MergeMode) -> torch.Tensor:
        batch_size, _, channels = hidden_states.shape
        src_pool = hidden_states.index_select(1, local_plan.a_idx)
        dst = hidden_states.index_select(1, local_plan.b_idx)
        unm = src_pool.index_select(1, local_plan.unm_idx)
        if mode != "none":
            src = src_pool.index_select(1, local_plan.src_idx)
            dst = dst.scatter_reduce(
                1,
                local_plan.dst_assign_idx.view(1, -1, 1).expand(batch_size, -1, channels),
                src,
                reduce=mode,
            )
        return torch.cat((unm, dst), dim=1)

    def _unmerge_local(self, hidden_states: torch.Tensor, local_plan: ITMLocalPlan, *, mode: MergeMode) -> torch.Tensor:
        batch_size, _, channels = hidden_states.shape
        unm_len = int(local_plan.unm_idx.numel())
        unm = hidden_states[:, :unm_len, :]
        dst = hidden_states[:, unm_len:, :]
        src = dst.index_select(1, local_plan.dst_assign_idx)

        out = hidden_states.new_zeros(batch_size, local_plan.token_count, channels)
        out.scatter_(
            1,
            local_plan.b_idx.view(1, -1, 1).expand(batch_size, -1, channels),
            dst,
        )
        out.scatter_(
            1,
            local_plan.a_idx.index_select(0, local_plan.unm_idx).view(1, -1, 1).expand(batch_size, -1, channels),
            unm,
        )
        out.scatter_(
            1,
            local_plan.a_idx.index_select(0, local_plan.src_idx).view(1, -1, 1).expand(batch_size, -1, channels),
            src,
        )
        return out


def build_itm_attention_context(
    *,
    runtime_context: dict[str, Any] | None,
    itm_state: dict[str, Any] | None,
    diffusion_step_idx: int,
) -> dict[str, Any]:
    context = {} if runtime_context is None else dict(runtime_context)
    if itm_state is None:
        return context
    context["itm"] = {
        "diffusion_step_idx": diffusion_step_idx,
        "phase": itm_state.get("phase", "disabled"),
        "merge_mode": itm_state.get("merge_mode", "mean"),
        "heat_map": itm_state.get("heat_map"),
    }
    return context


def infer_ctrl_world_heat_map_shape(seq_len: int) -> SimpleNamespace | None:
    geometry = {
        # 3-view vertical strip (72x40)
        2880: (72, 40),
        720: (36, 20),
        180: (18, 10),
        # 4-view 2x2 grid (48x80)
        3840: (48, 80),
        960: (24, 40),
        240: (12, 20),
    }.get(seq_len)
    if geometry is None:
        return None
    return SimpleNamespace(H=geometry[0], W=geometry[1])
