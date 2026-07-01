from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F


GroupMode = Literal["full_2d", "per_frame", "temporal"]

_DEFAULT_PATCH_H = 2
_DEFAULT_PATCH_W = 2
_DEFAULT_NOISE_ALPHA = 0.1
_DEFAULT_SIM_BETA = 1.0
_AUTO_PRUNE_SCHEDULE = (
    (2048, 0.45),
    (512, 0.35),
    (128, 0.20),
)


@dataclass(frozen=True)
class SiToRuntimePlan:
    keep_indices: torch.Tensor
    pruned_indices: torch.Tensor
    replacement_keep_positions: torch.Tensor
    original_length: int


def resolve_sito_auto_prune_ratio(tokens_per_group: int) -> float:
    for min_tokens, prune_ratio in _AUTO_PRUNE_SCHEDULE:
        if tokens_per_group >= min_tokens:
            return prune_ratio
    return 0.0


def _default_start_layer_idx(num_blocks: int) -> int:
    return 6 if num_blocks >= 36 else 4


def _validate_sito_common(
    *,
    patch_h: int,
    patch_w: int,
    prune_ratio: float | None,
    start_layer_idx: int,
    keep_last_n_dense: int,
    group_mode: GroupMode,
) -> None:
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError(f"`patch_h` and `patch_w` must be positive, got {(patch_h, patch_w)}.")
    if start_layer_idx < 0:
        raise ValueError(f"`start_layer_idx` must be non-negative, got {start_layer_idx}.")
    if keep_last_n_dense < 0:
        raise ValueError(f"`keep_last_n_dense` must be non-negative, got {keep_last_n_dense}.")
    if group_mode not in {"full_2d", "per_frame", "temporal"}:
        raise ValueError(f"`group_mode` must be 'full_2d', 'per_frame', or 'temporal', got {group_mode!r}.")
    if prune_ratio is not None:
        if group_mode == "temporal":
            if not 0.0 <= prune_ratio < 1.0:
                raise ValueError(
                    f"For `group_mode='temporal'`, `prune_ratio` must satisfy 0 <= prune_ratio < 1, got {prune_ratio}."
                )
        else:
            max_prune_ratio = 1.0 - 1.0 / float(patch_h * patch_w)
            if not 0.0 <= prune_ratio < max_prune_ratio:
                raise ValueError(
                    "`prune_ratio` must satisfy 0 <= prune_ratio < "
                    f"{max_prune_ratio:.4f} for patch size {(patch_h, patch_w)}, got {prune_ratio}."
                )


def build_sito_parameters(
    num_blocks: int,
    *,
    start_layer_idx: int | None = None,
    keep_last_n_dense: int = 2,
    prune_ratio: float | None = None,
    patch_h: int = _DEFAULT_PATCH_H,
    patch_w: int = _DEFAULT_PATCH_W,
    noise_alpha: float = _DEFAULT_NOISE_ALPHA,
    sim_beta: float = _DEFAULT_SIM_BETA,
    group_mode: GroupMode = "per_frame",
) -> list[dict[str, Any] | None]:
    if num_blocks <= 0:
        raise ValueError(f"`num_blocks` must be positive, got {num_blocks}.")
    resolved_start = _default_start_layer_idx(num_blocks) if start_layer_idx is None else start_layer_idx
    _validate_sito_common(
        patch_h=patch_h,
        patch_w=patch_w,
        prune_ratio=prune_ratio,
        start_layer_idx=resolved_start,
        keep_last_n_dense=keep_last_n_dense,
        group_mode=group_mode,
    )

    dense_tail_start = max(num_blocks - keep_last_n_dense, resolved_start)
    return [
        None
        if (layer_idx < resolved_start or layer_idx >= dense_tail_start)
        else {
            "layer_idx": layer_idx,
            "group_mode": group_mode,
            "prune_ratio": prune_ratio,
            "patch_h": patch_h,
            "patch_w": patch_w,
            "noise_alpha": noise_alpha,
            "sim_beta": sim_beta,
        }
        for layer_idx in range(num_blocks)
    ]


class SiToTokenPruner:
    def __init__(
        self,
        *,
        group_mode: GroupMode = "per_frame",
        prune_ratio: float | None = None,
        patch_h: int = _DEFAULT_PATCH_H,
        patch_w: int = _DEFAULT_PATCH_W,
        noise_alpha: float = _DEFAULT_NOISE_ALPHA,
        sim_beta: float = _DEFAULT_SIM_BETA,
        layer_idx: int | None = None,
    ) -> None:
        _validate_sito_common(
            patch_h=patch_h,
            patch_w=patch_w,
            prune_ratio=prune_ratio,
            start_layer_idx=0,
            keep_last_n_dense=0,
            group_mode=group_mode,
        )
        self.group_mode = group_mode
        self.prune_ratio = prune_ratio
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.noise_alpha = noise_alpha
        self.sim_beta = sim_beta
        self.layer_idx = layer_idx

    def _resolve_prune_ratio(self, tokens_per_group: int) -> float:
        prune_ratio = self.prune_ratio
        if prune_ratio is None:
            prune_ratio = resolve_sito_auto_prune_ratio(tokens_per_group)
        max_prune_ratio = 1.0 - 1.0 / float(self.patch_h * self.patch_w)
        return float(min(max(prune_ratio, 0.0), max(0.0, max_prune_ratio - 1e-6)))

    def prepare(
        self,
        hidden_states: torch.Tensor,
        *,
        video_size: Any | None = None,
    ) -> SiToRuntimePlan | None:
        if hidden_states.ndim != 3:
            raise ValueError(f"`hidden_states` must have shape (B, N, D), got {tuple(hidden_states.shape)}.")
        if self.group_mode == "temporal":
            if video_size is None:
                raise ValueError("`video_size` is required for `group_mode='temporal'`.")
            return self._build_temporal_plan(hidden_states, video_size=video_size)
        if self.group_mode == "per_frame":
            if video_size is None:
                raise ValueError("`video_size` is required for `group_mode='per_frame'`.")
            return self._build_per_frame_plan(hidden_states, video_size=video_size)
        return self._build_full_2d_plan(hidden_states, video_size=video_size)

    def prune(self, hidden_states: torch.Tensor, plan: SiToRuntimePlan | None) -> torch.Tensor:
        if plan is None:
            return hidden_states
        return hidden_states.index_select(dim=1, index=plan.keep_indices)

    def recover(self, hidden_states: torch.Tensor, plan: SiToRuntimePlan | None) -> torch.Tensor:
        if plan is None:
            return hidden_states

        batch_size, _, hidden_dim = hidden_states.shape
        full_states = hidden_states.new_empty(batch_size, plan.original_length, hidden_dim)
        full_states[:, plan.keep_indices, :] = hidden_states
        if plan.pruned_indices.numel() > 0:
            propagated = hidden_states.index_select(dim=1, index=plan.replacement_keep_positions)
            full_states[:, plan.pruned_indices, :] = propagated
        return full_states

    def prune_rope(self, rope_emb: torch.Tensor | None, plan: SiToRuntimePlan | None) -> torch.Tensor | None:
        if rope_emb is None or plan is None:
            return rope_emb
        if rope_emb.shape[0] != plan.original_length:
            return rope_emb
        return rope_emb.index_select(dim=0, index=plan.keep_indices)

    def _build_full_2d_plan(self, hidden_states: torch.Tensor, *, video_size: Any | None) -> SiToRuntimePlan | None:
        _, seq_len, _ = hidden_states.shape
        if video_size is not None and hasattr(video_size, "H") and hasattr(video_size, "W"):
            group_h = int(video_size.H)
            group_w = int(video_size.W)
            if group_h * group_w == seq_len:
                return self._build_group_plan(hidden_states, group_h=group_h, group_w=group_w)
        side = int(math.isqrt(seq_len))
        if side * side != seq_len:
            return None
        return self._build_group_plan(hidden_states, group_h=side, group_w=side)

    def _build_per_frame_plan(self, hidden_states: torch.Tensor, *, video_size: Any) -> SiToRuntimePlan | None:
        _, seq_len, _ = hidden_states.shape
        group_t = int(video_size.T)
        group_h = int(video_size.H)
        group_w = int(video_size.W)
        per_frame_tokens = group_h * group_w
        if group_t <= 0 or per_frame_tokens <= 0 or group_t * per_frame_tokens != seq_len:
            raise ValueError(
                f"Invalid video geometry for SiTo: got seq_len={seq_len}, "
                f"video_size={(group_t, group_h, group_w)}."
            )

        keep_indices: list[torch.Tensor] = []
        pruned_indices: list[torch.Tensor] = []
        replacement_keep_positions: list[torch.Tensor] = []
        keep_base = 0

        frame_tokens = hidden_states.view(hidden_states.shape[0], group_t, per_frame_tokens, hidden_states.shape[-1])
        for frame_idx in range(group_t):
            local_plan = self._build_group_plan(frame_tokens[:, frame_idx], group_h=group_h, group_w=group_w)
            if local_plan is None:
                return None
            frame_offset = frame_idx * per_frame_tokens
            keep_indices.append(local_plan.keep_indices + frame_offset)
            pruned_indices.append(local_plan.pruned_indices + frame_offset)
            replacement_keep_positions.append(local_plan.replacement_keep_positions + keep_base)
            keep_base += int(local_plan.keep_indices.numel())

        return SiToRuntimePlan(
            keep_indices=torch.cat(keep_indices, dim=0),
            pruned_indices=torch.cat(pruned_indices, dim=0),
            replacement_keep_positions=torch.cat(replacement_keep_positions, dim=0),
            original_length=seq_len,
        )

    def _build_temporal_plan(self, hidden_states: torch.Tensor, *, video_size: Any) -> SiToRuntimePlan | None:
        """Prune temporally-redundant tokens, recovering from the nearest kept frame.

        Tokens sharing the same spatial position ``(h, w)`` across the flattened
        time/view axis ``t`` (``t = V * T`` for multiview) form a temporal group.
        A token is a pruning candidate when it is very similar to the *previous*
        frame at the same position (low temporal change). Frame 0 of every position
        is always kept. Crucially, each pruned token is recovered from the
        **nearest preceding kept frame at the same spatial position** (not a fixed
        ``t=0`` anchor), so motion is tracked instead of being reset to the first
        frame. Selection is fully vectorized.
        """
        _, seq_len, _ = hidden_states.shape
        group_t = int(video_size.T)
        group_h = int(video_size.H)
        group_w = int(video_size.W)
        per_frame_tokens = group_h * group_w
        if group_t <= 0 or per_frame_tokens <= 0 or group_t * per_frame_tokens != seq_len:
            raise ValueError(
                f"Invalid video geometry for SiTo temporal: got seq_len={seq_len}, "
                f"video_size={(group_t, group_h, group_w)}."
            )
        # Need at least 2 frames to have temporal redundancy to exploit.
        if group_t < 2:
            return None

        # Temporal groups are small (group_t frames); the spatial auto-schedule
        # (keyed on 2048/512/128 tokens) does not apply. Fall back to a sensible
        # default fraction of the temporal axis when no explicit ratio is set.
        if self.prune_ratio is None:
            prune_ratio = 0.30
        else:
            max_prune_ratio = 1.0 - 1.0 / float(group_t)
            prune_ratio = float(min(max(self.prune_ratio, 0.0), max(0.0, max_prune_ratio - 1e-6)))
        if prune_ratio <= 0.0:
            return None

        device = hidden_states.device
        # token_summary: (N, D) averaged over batch and L2-normalized per token.
        token_summary = F.normalize(hidden_states.float(), dim=-1).mean(dim=0)
        # Reshape to (t, hw, D); flattened index = t * per_frame_tokens + hw.
        grid = token_summary.view(group_t, per_frame_tokens, token_summary.shape[-1])

        # Redundancy score = similarity of each frame (t>=1) to the PREVIOUS frame
        # at the same spatial position. High similarity => low temporal change =>
        # safe to prune (and cheap to recover from the temporal neighbor).
        prev_feat = grid[:-1]  # (t-1, hw, D): frames 0..t-2
        curr_feat = grid[1:]  # (t-1, hw, D): frames 1..t-1
        sim_to_prev = (curr_feat * prev_feat).sum(dim=-1)  # (t-1, hw)
        if self.noise_alpha > 0:
            sim_to_prev = sim_to_prev + self.noise_alpha * torch.randn_like(sim_to_prev)

        num_candidates = sim_to_prev.numel()
        target_prune = min(int(round(seq_len * prune_ratio)), num_candidates)
        if target_prune <= 0:
            return None

        # Candidate token indices (only frames 1..t-1 are prunable; frame 0 kept).
        cand_t = torch.arange(1, group_t, device=device).view(-1, 1).expand(group_t - 1, per_frame_tokens)
        cand_hw = torch.arange(per_frame_tokens, device=device).view(1, -1).expand(group_t - 1, per_frame_tokens)
        cand_token_idx = (cand_t * per_frame_tokens + cand_hw).reshape(-1)

        sim_flat = sim_to_prev.reshape(-1)
        # Most similar to previous frame == most redundant -> prune first.
        prune_order = sim_flat.argsort(descending=True)
        prune_positions = prune_order[:target_prune]
        pruned_indices = cand_token_idx.index_select(0, prune_positions)

        keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        keep_mask[pruned_indices] = False
        keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)

        # For each pruned token, find the nearest PRECEDING kept frame at the same
        # spatial position. Build a (group_t, per_frame_tokens) kept-frame table and
        # take a cumulative "last kept frame index" along time. Frame 0 is always
        # kept, so a valid predecessor always exists.
        kept_grid = keep_mask.view(group_t, per_frame_tokens)  # (t, hw) bool
        frame_ids = torch.arange(group_t, device=device).view(group_t, 1).expand(group_t, per_frame_tokens)
        # last_kept[t, hw] = max frame index <= t that is kept at position hw.
        last_kept = torch.cummax(torch.where(kept_grid, frame_ids, torch.full_like(frame_ids, -1)), dim=0).values

        pruned_t = pruned_indices // per_frame_tokens
        pruned_hw = pruned_indices % per_frame_tokens
        # Nearest preceding kept frame for each pruned token = last_kept at (t-1, hw)
        # (the predecessor row), guaranteed >= 0 because frame 0 is kept.
        src_frame = last_kept[(pruned_t - 1).clamp(min=0), pruned_hw]
        replacement_src_token = src_frame * per_frame_tokens + pruned_hw

        # Map original kept-token index -> position within the kept list.
        position_in_keep = torch.empty(seq_len, dtype=torch.long, device=device)
        position_in_keep[keep_indices] = torch.arange(keep_indices.numel(), device=device)
        replacement_keep_positions = position_in_keep.index_select(0, replacement_src_token)

        return SiToRuntimePlan(
            keep_indices=keep_indices,
            pruned_indices=pruned_indices,
            replacement_keep_positions=replacement_keep_positions,
            original_length=seq_len,
        )

    def _build_group_plan(self, hidden_states: torch.Tensor, *, group_h: int, group_w: int) -> SiToRuntimePlan | None:
        tokens_per_group = group_h * group_w
        prune_ratio = self._resolve_prune_ratio(tokens_per_group)
        if prune_ratio <= 0.0:
            return None

        device = hidden_states.device
        patch_indices, remainder_indices = self._build_patch_index_layout(group_h=group_h, group_w=group_w, device=device)
        if patch_indices.numel() == 0:
            return None

        token_summary = F.normalize(hidden_states.float(), dim=-1).mean(dim=0)
        mean_feature = token_summary.mean(dim=0, keepdim=True)
        scores = self.sim_beta * torch.matmul(token_summary, mean_feature.transpose(0, 1)).squeeze(-1)
        if self.noise_alpha > 0:
            scores = scores + self.noise_alpha * torch.randn_like(scores)

        anchors_in_patch = scores.index_select(0, patch_indices.reshape(-1)).view_as(patch_indices).argmax(dim=-1)
        anchor_indices = patch_indices.gather(dim=1, index=anchors_in_patch.unsqueeze(-1)).squeeze(-1)

        source_mask = torch.ones_like(patch_indices, dtype=torch.bool)
        source_mask.scatter_(1, anchors_in_patch.unsqueeze(-1), False)
        source_indices = patch_indices[source_mask]
        if source_indices.numel() == 0:
            return None

        anchor_features = token_summary.index_select(0, anchor_indices)
        source_features = token_summary.index_select(0, source_indices)
        source_to_anchor = torch.matmul(source_features, anchor_features.transpose(0, 1))
        best_similarity, best_anchor_idx = source_to_anchor.max(dim=1)

        max_prune = int(source_indices.numel())
        target_prune = min(int(round(tokens_per_group * prune_ratio)), max_prune)
        if target_prune <= 0:
            return None

        prune_order = best_similarity.argsort(descending=True)
        source_prune_positions = prune_order[:target_prune]
        pruned_indices = source_indices.index_select(0, source_prune_positions)

        keep_mask = torch.ones(tokens_per_group, dtype=torch.bool, device=device)
        keep_mask[pruned_indices] = False
        keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)

        if remainder_indices.numel() > 0:
            keep_indices = torch.cat((keep_indices, remainder_indices), dim=0).unique(sorted=True)

        # For each pruned token, find the most similar token in the FULL kept set
        # (anchors + unmerged sources), matching the original SiTo recovery logic.
        kept_features = token_summary.index_select(0, keep_indices)
        pruned_features = token_summary.index_select(0, pruned_indices)
        sim_to_kept = torch.matmul(pruned_features, kept_features.transpose(0, 1))
        replacement_keep_positions = sim_to_kept.argmax(dim=1)

        return SiToRuntimePlan(
            keep_indices=keep_indices,
            pruned_indices=pruned_indices,
            replacement_keep_positions=replacement_keep_positions,
            original_length=tokens_per_group,
        )

    def _build_patch_index_layout(
        self,
        *,
        group_h: int,
        group_w: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        crop_h = (group_h // self.patch_h) * self.patch_h
        crop_w = (group_w // self.patch_w) * self.patch_w
        if crop_h == 0 or crop_w == 0:
            return (
                torch.empty((0, self.patch_h * self.patch_w), dtype=torch.long, device=device),
                torch.arange(group_h * group_w, device=device, dtype=torch.long),
            )

        index_grid = torch.arange(group_h * group_w, device=device, dtype=torch.long).view(group_h, group_w)
        cropped = index_grid[:crop_h, :crop_w]
        patch_indices = (
            cropped.view(crop_h // self.patch_h, self.patch_h, crop_w // self.patch_w, self.patch_w)
            .permute(0, 2, 1, 3)
            .reshape(-1, self.patch_h * self.patch_w)
        )

        crop_mask = torch.zeros((group_h, group_w), dtype=torch.bool, device=device)
        crop_mask[:crop_h, :crop_w] = True
        remainder_indices = index_grid[~crop_mask]
        return patch_indices, remainder_indices
