from __future__ import annotations

import importlib
import importlib.util
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:
    create_block_mask = None
    flex_attention = None


def _vendor_svg_root() -> Path:
    return Path(__file__).resolve().parent.parent / "Sparse-VideoGen" / "svg"


@lru_cache(maxsize=1)
def _load_vendor_svg_package_name() -> str:
    package_dir = _vendor_svg_root()
    init_file = package_dir / "__init__.py"
    if not init_file.exists():
        raise ImportError(f"[SVG] Missing vendored Sparse-VideoGen package at {package_dir}.")

    package_name = "dreamgen_svg_vendor"
    if package_name in sys.modules:
        return package_name

    spec = importlib.util.spec_from_file_location(
        package_name,
        init_file,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"[SVG] Could not load vendored package spec from {init_file}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return package_name


def _import_vendor_svg_module(module_name: str):
    package_name = _load_vendor_svg_package_name()
    try:
        return importlib.import_module(f"{package_name}.{module_name}")
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise ImportError(
            "[SVG] Missing runtime dependency while importing vendored Sparse-VideoGen module "
            f"{module_name!r}: {missing}."
        ) from exc


@lru_cache(maxsize=1)
def _load_svg2_modules():
    flashinfer_patch = _import_vendor_svg_module("flashinfer_patch")
    kmeans_utils = _import_vendor_svg_module("kmeans_utils")
    permute_mod = _import_vendor_svg_module("kernels.triton.permute")
    flashinfer_patch.apply_flashinfer_patch()
    return {
        "kmeans_utils": kmeans_utils,
        "permute_mod": permute_mod,
    }


@lru_cache(maxsize=1)
def _load_svg1_modules():
    placement_mod = _import_vendor_svg_module("models.cosmos.placement")
    return {
        "placement_mod": placement_mod,
    }


def _dense_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)


# flex_attention must be called from within a torch.compile(backend="inductor") context
# so that is_dynamo_compiling()=True and the efficient Triton kernel is used.
# Calling it standalone falls back to sdpa_dense which materializes the full S×S matrix.
@torch.compile(backend="inductor", fullgraph=False)
def _compiled_flex_attention(q, k, v, block_mask):
    return flex_attention(q, k, v, block_mask=block_mask)


def _should_use_dense_warmup(attention_runtime_context: dict[str, Any] | None, dense_step_frac: float) -> bool:
    if dense_step_frac <= 0 or attention_runtime_context is None:
        return False
    step_idx = attention_runtime_context.get("diffusion_step_idx")
    num_sampling_steps = attention_runtime_context.get("num_sampling_steps")
    if step_idx is None or num_sampling_steps in (None, 0):
        return False
    dense_steps = int(math.floor(float(dense_step_frac) * int(num_sampling_steps)))
    return dense_steps > 0 and int(step_idx) < dense_steps


def _sample_rows(seq_len: int, max_row: int, num_rows: int, device: torch.device) -> torch.Tensor:
    upper = min(seq_len, max_row)
    if upper <= 0:
        raise RuntimeError("[SVG1] No valid rows available for online profiling.")
    return torch.randint(0, upper, (min(num_rows, upper),), device=device)


def _frame_patch_indices(row_indices: torch.Tensor, frame_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    frame_ids = torch.div(row_indices, frame_size, rounding_mode="floor")
    patch_ids = torch.remainder(row_indices, frame_size)
    return frame_ids, patch_ids


def _generate_temporal_head_mask_mod(context_length: int, num_frames: int, token_per_frame: int, mul: float):
    def round_to_multiple(idx: float) -> int:
        return math.ceil(idx / 128) * 128

    def temporal_mask_mod(b, h, q_idx, kv_idx):
        del b, h
        two_frame = round_to_multiple(mul * token_per_frame)
        temporal_head_mask = torch.abs(q_idx - kv_idx) <= two_frame
        first_frame_mask = kv_idx < token_per_frame
        return first_frame_mask | temporal_head_mask

    return temporal_mask_mod


def _sparsity_to_width(sparsity: float, context_length: int, num_frame: int, frame_size: int) -> float:
    seq_len = context_length + num_frame * frame_size
    total_elements = seq_len**2
    adjusted_sparsity = (sparsity * total_elements - 2 * seq_len * context_length) / total_elements
    width = seq_len * (1 - math.sqrt(max(0.0, 1.0 - adjusted_sparsity)))
    return width / frame_size


def _build_svg1_attention_mask(mask_name: str, *, sample_mse_max_row: int, context_length: int, num_frame: int, frame_size: int):
    attention_mask = torch.zeros(
        (context_length + num_frame * frame_size, context_length + num_frame * frame_size),
        dtype=torch.bool,
        device="cpu",
    )
    pixel_attn_mask = torch.zeros_like(attention_mask, dtype=torch.bool, device="cpu")
    pixel_attn_mask[:, :frame_size] = 1

    block_size = 128
    block_thres = frame_size * 2
    num_block = math.ceil(num_frame * frame_size / block_size)
    for i in range(num_block):
        for j in range(num_block):
            if abs(i - j) < block_thres // block_size:
                pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1

    if mask_name == "spatial":
        attention_mask = pixel_attn_mask
    elif mask_name == "temporal":
        attention_mask = (
            pixel_attn_mask.reshape(frame_size, num_frame, frame_size, num_frame)
            .permute(1, 0, 3, 2)
            .reshape(frame_size * num_frame, frame_size * num_frame)
        )
    else:
        raise ValueError(f"[SVG1] Unsupported mask name {mask_name!r}.")
    return attention_mask[:sample_mse_max_row]


class DreamGenSVG1Runtime:
    def __init__(self, *, sparsity: float, num_sampled_rows: int, sample_mse_max_row: int):
        if not 0 < sparsity <= 1:
            raise ValueError(f"[SVG1] `sparsity` must be in (0, 1], got {sparsity}.")
        if num_sampled_rows <= 0 or sample_mse_max_row <= 0:
            raise ValueError("[SVG1] `num_sampled_rows` and `sample_mse_max_row` must be positive.")
        if flex_attention is None or create_block_mask is None:
            raise RuntimeError(
                "[SVG1] This Torch build does not provide `torch.nn.attention.flex_attention`. "
                "Use a Torch build with flex attention support."
            )

        _load_svg1_modules()
        self.sparsity = float(sparsity)
        self.num_sampled_rows = int(num_sampled_rows)
        self.sample_mse_max_row = int(sample_mse_max_row)
        self._mask_metadata_cache: dict[tuple[int, int, int], dict[str, int]] = {}
        self._attention_mask_cache: dict[tuple[int, int, int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
        self._block_mask_cache: dict[tuple[torch.dtype, torch.device, int, int, int, int, int], Any] = {}

    def _validate(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, video_size) -> None:
        if not query.is_cuda:
            raise RuntimeError("[SVG1] CUDA tensors are required.")
        if video_size is None:
            raise RuntimeError("[SVG1] `video_size` is required for DreamGen self-attention.")
        if query.shape != key.shape or key.shape != value.shape:
            raise RuntimeError(
                "[SVG1] Only self-attention is supported "
                f"(query={tuple(query.shape)}, key={tuple(key.shape)}, value={tuple(value.shape)})."
            )
        expected_seq_len = video_size.T * video_size.H * video_size.W
        if query.shape[1] != expected_seq_len:
            raise RuntimeError(
                "[SVG1] Sequence length does not match video geometry "
                f"(seq_len={query.shape[1]}, expected={expected_seq_len}, video_size={video_size})."
            )

    def _mask_metadata(self, *, seq_len: int, num_frame: int, frame_size: int) -> dict[str, int]:
        key = (seq_len, num_frame, frame_size)
        if key not in self._mask_metadata_cache:
            width_frame = _sparsity_to_width(self.sparsity, 0, num_frame, frame_size)
            self._mask_metadata_cache[key] = {
                "context_length": 0,
                "prompt_length": 0,
                "diag_width": width_frame,
                "multiplier": width_frame,
            }
        return self._mask_metadata_cache[key]

    def _attention_masks(self, *, seq_len: int, num_frame: int, frame_size: int, device: torch.device):
        key = (seq_len, num_frame, frame_size, device)
        if key not in self._attention_mask_cache:
            spatial = _build_svg1_attention_mask(
                "spatial",
                sample_mse_max_row=min(self.sample_mse_max_row, seq_len),
                context_length=0,
                num_frame=num_frame,
                frame_size=frame_size,
            ).to(device=device)
            temporal = _build_svg1_attention_mask(
                "temporal",
                sample_mse_max_row=min(self.sample_mse_max_row, seq_len),
                context_length=0,
                num_frame=num_frame,
                frame_size=frame_size,
            ).to(device=device)
            self._attention_mask_cache[key] = (spatial, temporal)
        return self._attention_mask_cache[key]

    def _block_mask(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
        num_heads: int,
        head_dim: int,
        seq_len: int,
        num_frame: int,
        frame_size: int,
    ):
        metadata = self._mask_metadata(seq_len=seq_len, num_frame=num_frame, frame_size=frame_size)
        cache_key = (dtype, device, num_heads, head_dim, seq_len, num_frame, frame_size)
        if cache_key not in self._block_mask_cache:
            mask_mod = _generate_temporal_head_mask_mod(
                metadata["context_length"],
                num_frame,
                frame_size,
                metadata["multiplier"],
            )
            block_mask = create_block_mask(mask_mod, None, None, seq_len, seq_len, device=device, _compile=True)
            warm_q = torch.zeros((1, num_heads, seq_len, head_dim), dtype=dtype, device=device)
            warm_k = torch.zeros_like(warm_q)
            warm_v = torch.zeros_like(warm_q)
            _ = _compiled_flex_attention(warm_q, warm_k, warm_v, block_mask=block_mask)
            self._block_mask_cache[cache_key] = block_mask
        return self._block_mask_cache[cache_key]

    def _sample_mse_profile(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attention_masks: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, num_heads, seq_len, dim = query.shape
        num_sampled_rows = min(self.num_sampled_rows, seq_len, attention_masks[0].shape[0])
        sampled_rows = _sample_rows(seq_len, min(self.sample_mse_max_row, seq_len), num_sampled_rows, query.device)
        sampled_q = query[:, :, sampled_rows, :]
        sampled_qk_scores = torch.matmul(sampled_q, key.transpose(-2, -1)) / math.sqrt(dim)
        sampled_attn_weights = F.softmax(sampled_qk_scores.float(), dim=-1).to(query.dtype)
        sampled_golden_hidden_states = torch.matmul(sampled_attn_weights, value)

        sampled_mses = torch.zeros(len(attention_masks), bsz, num_heads, device=query.device, dtype=query.dtype)
        for mask_idx, attn_mask in enumerate(attention_masks):
            sampled_attention_mask = attn_mask[sampled_rows, :]
            sampled_attention_scores = sampled_qk_scores.masked_fill(sampled_attention_mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
            sampled_attn_weights = F.softmax(sampled_attention_scores.float(), dim=-1).to(query.dtype)
            sampled_hidden_states = torch.matmul(sampled_attn_weights, value)
            sampled_mses[mask_idx] = torch.mean((sampled_hidden_states - sampled_golden_hidden_states) ** 2, dim=(2, 3))
        return torch.argmin(sampled_mses, dim=0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        video_size,
        attention_runtime_context: dict[str, Any] | None = None,
        dense_step_frac: float = 0.0,
    ) -> torch.Tensor:
        self._validate(query, key, value, video_size=video_size)

        q = rearrange(query, "b s h d -> b h s d").contiguous()
        k = rearrange(key, "b s h d -> b h s d").contiguous()
        v = rearrange(value, "b s h d -> b h s d").contiguous()

        if _should_use_dense_warmup(attention_runtime_context, dense_step_frac):
            return rearrange(_dense_attention(q, k, v), "b h s d -> b s h d").contiguous()

        frame_size = video_size.H * video_size.W
        attention_masks = self._attention_masks(
            seq_len=q.shape[2],
            num_frame=video_size.T,
            frame_size=frame_size,
            device=q.device,
        )
        best_mask_idx = self._sample_mse_profile(q, k, v, attention_masks=attention_masks)

        modules = _load_svg1_modules()
        placement_mod = modules["placement_mod"]
        query_out = torch.empty_like(q)
        key_out = torch.empty_like(k)
        value_out = torch.empty_like(v)
        output_hidden_states = torch.empty_like(q)

        placement_mod.cosmos_sparse_head_placement(
            q,
            k,
            v,
            query_out,
            key_out,
            value_out,
            best_mask_idx,
            0,
            video_size.T,
            frame_size,
        )

        block_mask = self._block_mask(
            dtype=q.dtype,
            device=q.device,
            num_heads=q.shape[1],
            head_dim=q.shape[3],
            seq_len=q.shape[2],
            num_frame=video_size.T,
            frame_size=frame_size,
        )
        hidden_states = _compiled_flex_attention(query_out, key_out, value_out, block_mask=block_mask)
        placement_mod.cosmos_hidden_states_placement(
            hidden_states,
            output_hidden_states,
            best_mask_idx,
            0,
            video_size.T,
            frame_size,
        )
        return rearrange(output_hidden_states, "b h s d -> b s h d").contiguous()


class DreamGenSVG2Runtime:
    def __init__(
        self,
        *,
        num_q_centroids: int,
        num_k_centroids: int,
        top_p_kmeans: float,
        min_kc_ratio: float,
        kmeans_iter_init: int,
        kmeans_iter_step: int,
    ):
        if num_q_centroids <= 0 or num_k_centroids <= 0:
            raise ValueError("[SVG2] Centroid counts must be positive.")
        if not 0 < top_p_kmeans <= 1:
            raise ValueError(f"[SVG2] `top_p_kmeans` must be in (0, 1], got {top_p_kmeans}.")
        if not 0 <= min_kc_ratio <= 1:
            raise ValueError(f"[SVG2] `min_kc_ratio` must be in [0, 1], got {min_kc_ratio}.")
        if kmeans_iter_init < 0 or kmeans_iter_step < 0:
            raise ValueError("[SVG2] K-Means iteration counts must be non-negative.")
        _load_svg2_modules()

        self.num_q_centroids = num_q_centroids
        self.num_k_centroids = num_k_centroids
        self.top_p_kmeans = top_p_kmeans
        self.min_kc_ratio = min_kc_ratio
        self.kmeans_iter_init = kmeans_iter_init
        self.kmeans_iter_step = kmeans_iter_step
        self._q_centroids = None
        self._k_centroids = None
        self._centroid_shape = None

    def _validate(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, video_size) -> None:
        if not query.is_cuda:
            raise RuntimeError("[SVG2] CUDA tensors are required.")
        if video_size is None:
            raise RuntimeError("[SVG2] `video_size` is required for DreamGen self-attention.")
        if query.shape != key.shape or key.shape != value.shape:
            raise RuntimeError(
                "[SVG2] Only self-attention is supported "
                f"(query={tuple(query.shape)}, key={tuple(key.shape)}, value={tuple(value.shape)})."
            )
        expected_seq_len = video_size.T * video_size.H * video_size.W
        if query.shape[1] != expected_seq_len:
            raise RuntimeError(
                "[SVG2] Sequence length does not match video geometry "
                f"(seq_len={query.shape[1]}, expected={expected_seq_len}, video_size={video_size})."
            )
        if query.shape[0] != 1:
            raise RuntimeError(f"[SVG2] Effective batch size must be 1, got batch={query.shape[0]}.")

    def _maybe_reset_centroids(self, q_shape: tuple[int, ...]) -> None:
        if self._centroid_shape != q_shape:
            self._q_centroids = None
            self._k_centroids = None
            self._centroid_shape = q_shape

    def _cluster(self, q_flat: torch.Tensor, k_flat: torch.Tensor):
        modules = _load_svg2_modules()
        batch_kmeans = modules["kmeans_utils"].batch_kmeans_Euclid

        self._maybe_reset_centroids(tuple(q_flat.shape))
        qlabels, qcentroids, qcluster_sizes, _ = batch_kmeans(
            q_flat,
            n_clusters=self.num_q_centroids,
            max_iters=self.kmeans_iter_init if self._q_centroids is None else self.kmeans_iter_step,
            init_centroids=self._q_centroids,
        )
        klabels, kcentroids, kcluster_sizes, _ = batch_kmeans(
            k_flat,
            n_clusters=self.num_k_centroids,
            max_iters=self.kmeans_iter_init if self._k_centroids is None else self.kmeans_iter_step,
            init_centroids=self._k_centroids,
        )
        self._q_centroids = qcentroids
        self._k_centroids = kcentroids
        return qlabels, qcentroids, qcluster_sizes, klabels, kcentroids, kcluster_sizes

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        video_size,
        attention_runtime_context: dict[str, Any] | None = None,
        dense_step_frac: float = 0.0,
    ) -> torch.Tensor:
        self._validate(query, key, value, video_size=video_size)

        q = rearrange(query, "b s h d -> b h s d").contiguous()
        k = rearrange(key, "b s h d -> b h s d").contiguous()
        v = rearrange(value, "b s h d -> b h s d").contiguous()

        if _should_use_dense_warmup(attention_runtime_context, dense_step_frac):
            return rearrange(_dense_attention(q, k, v), "b h s d -> b s h d").contiguous()

        modules = _load_svg2_modules()
        kmeans_utils = modules["kmeans_utils"]
        permute_mod = modules["permute_mod"]

        bsz, num_heads, seq_len, head_dim = q.shape
        q_flat = q.reshape(bsz * num_heads, seq_len, head_dim)
        k_flat = k.reshape(bsz * num_heads, seq_len, head_dim)
        v_flat = v.reshape(bsz * num_heads, seq_len, head_dim)

        qlabels, qcentroids, qcluster_sizes, klabels, kcentroids, kcluster_sizes = self._cluster(q_flat, k_flat)

        dynamic_map = kmeans_utils.identify_dynamic_map(
            qcentroids.view(bsz, num_heads, self.num_q_centroids, head_dim),
            kcentroids.view(bsz, num_heads, self.num_k_centroids, head_dim),
            qcluster_sizes.view(bsz, num_heads, self.num_q_centroids),
            kcluster_sizes.view(bsz, num_heads, self.num_k_centroids),
            self.top_p_kmeans,
            self.min_kc_ratio,
        )

        q_permuted, q_sorted_indices = permute_mod.permute_tensor_by_labels_triton(q, qlabels.view(bsz, num_heads, seq_len), dim=2)
        k_permuted, k_sorted_indices = permute_mod.permute_tensor_by_labels_triton(k, klabels.view(bsz, num_heads, seq_len), dim=2)
        v_permuted, _ = permute_mod.permute_tensor_by_labels_triton(
            v,
            klabels.view(bsz, num_heads, seq_len),
            dim=2,
            sorted_indices=k_sorted_indices,
        )

        output_permuted = kmeans_utils.dynamic_block_sparse_fwd_flashinfer(
            q_permuted,
            k_permuted,
            v_permuted,
            dynamic_map,
            qcluster_sizes.view(bsz, num_heads, self.num_q_centroids),
            kcluster_sizes.view(bsz, num_heads, self.num_k_centroids),
            is_cpu=False,
        )
        attn_output = permute_mod.apply_inverse_permutation_triton(output_permuted, q_sorted_indices, dim=2)
        return rearrange(attn_output, "b h s d -> b s h d").contiguous()


class DreamGenSVGRuntime:
    def __init__(self, *, variant: str, **kwargs):
        if variant == "svg1":
            self.runtime = DreamGenSVG1Runtime(
                sparsity=kwargs["sparsity"],
                num_sampled_rows=kwargs["num_sampled_rows"],
                sample_mse_max_row=kwargs["sample_mse_max_row"],
            )
        elif variant == "svg2":
            self.runtime = DreamGenSVG2Runtime(
                num_q_centroids=kwargs["num_q_centroids"],
                num_k_centroids=kwargs["num_k_centroids"],
                top_p_kmeans=kwargs["top_p_kmeans"],
                min_kc_ratio=kwargs["min_kc_ratio"],
                kmeans_iter_init=kwargs["kmeans_iter_init"],
                kmeans_iter_step=kwargs["kmeans_iter_step"],
            )
        else:
            raise ValueError(f"[SVG] Unsupported variant {variant!r}.")
        self.variant = variant
        self.dense_step_frac = float(kwargs.get("dense_step_frac", 0.0))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        video_size,
        attention_runtime_context: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        # Early dense check: bypass inner runtime entirely to avoid rearrange overhead.
        # Input is (B, S, H, D); SDPA expects (B, H, S, D), but we can use
        # the transpose view without .contiguous() since flash-attn handles strides.
        if _should_use_dense_warmup(attention_runtime_context, self.dense_step_frac):
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            return out.transpose(1, 2)
        return self.runtime.forward(
            query,
            key,
            value,
            video_size=video_size,
            attention_runtime_context=attention_runtime_context,
            dense_step_frac=self.dense_step_frac,
        )
