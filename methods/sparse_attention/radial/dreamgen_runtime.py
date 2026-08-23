from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F


def _pad_bshd(tensor: torch.Tensor, pad_len: int) -> torch.Tensor:
    if pad_len <= 0:
        return tensor.contiguous()
    return F.pad(tensor, (0, 0, 0, 0, 0, pad_len)).contiguous()


@lru_cache(maxsize=1)
def _load_radial_attn_module():
    module_path = (
        Path(__file__).resolve().parent.parent
        / "radial-attention"
        / "radial_attn"
        / "attn_mask.py"
    )
    if not module_path.exists():
        raise ImportError(f"[Radial] Missing vendored module at {module_path}.")

    spec = importlib.util.spec_from_file_location("dreamgen_radial_attn_mask", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"[Radial] Could not load module spec from {module_path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise ImportError(
            "[Radial] Missing runtime dependency while loading vendored radial-attention module: "
            f"{missing}."
        ) from exc
    return module


@dataclass
class _RadialPlan:
    workspace: torch.Tensor
    wrapper: object


class DreamGenRadialRuntime:
    def __init__(self, *, block_size: int = 64, decay_factor: float = 1.0, model_type: str = "hunyuan"):
        if block_size not in (64, 128):
            raise ValueError(f"[Radial] Only block_size 64 or 128 is supported, got {block_size}.")
        if model_type not in {"hunyuan", "wan"}:
            raise ValueError(f"[Radial] Unsupported model_type {model_type!r}.")

        self.block_size = block_size
        self.decay_factor = decay_factor
        self.model_type = model_type

        self._module = _load_radial_attn_module()
        self._flashinfer = getattr(self._module, "flashinfer", None)
        if self._flashinfer is None:
            raise ImportError("[Radial] flashinfer is required but was not imported by the vendored module.")

        self._mask_cache: dict[tuple, torch.Tensor] = {}
        self._plan_cache: dict[tuple, _RadialPlan] = {}

    def _validate_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        video_size,
    ) -> None:
        if not query.is_cuda:
            raise RuntimeError("[Radial] CUDA tensors are required.")
        if video_size is None:
            raise RuntimeError("[Radial] `video_size` is required for DreamGen Radial self-attention.")
        if query.shape != key.shape or key.shape != value.shape:
            raise RuntimeError(
                "[Radial] Only self-attention is supported by the DreamGen Radial backend "
                f"(query={tuple(query.shape)}, key={tuple(key.shape)}, value={tuple(value.shape)})."
            )
        expected_seq_len = video_size.T * video_size.H * video_size.W
        if query.shape[1] != expected_seq_len:
            raise RuntimeError(
                "[Radial] Sequence length does not match video geometry "
                f"(seq_len={query.shape[1]}, expected={expected_seq_len}, video_size={video_size})."
            )

    def _get_mask(
        self,
        *,
        query: torch.Tensor,
        seq_len: int,
        padded_seq_len: int,
        video_size,
    ) -> torch.Tensor:
        key = (
            query.device.type,
            query.device.index,
            seq_len,
            padded_seq_len,
            video_size.T,
            video_size.H,
            video_size.W,
            self.block_size,
            float(self.decay_factor),
            self.model_type,
        )
        if key not in self._mask_cache:
            mask = self._module.gen_log_mask_shrinked(
                query,
                padded_seq_len,
                seq_len,
                video_size.T,
                sparse_type="radial",
                block_size=self.block_size,
                decay_factor=self.decay_factor,
                model_type=self.model_type,
            )
            self._mask_cache[key] = mask[: padded_seq_len // self.block_size, : padded_seq_len // self.block_size]
        return self._mask_cache[key]

    def _get_plan(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        seq_len: int,
        padded_seq_len: int,
        video_size,
    ):
        num_heads = query.shape[2]
        head_dim = query.shape[3]
        plan_key = (
            query.device.type,
            query.device.index,
            seq_len,
            padded_seq_len,
            video_size.T,
            video_size.H,
            video_size.W,
            self.block_size,
            float(self.decay_factor),
            self.model_type,
            num_heads,
            head_dim,
            query.dtype,
            key.dtype,
        )
        if plan_key not in self._plan_cache:
            mask = self._get_mask(query=query, seq_len=seq_len, padded_seq_len=padded_seq_len, video_size=video_size)
            workspace = torch.empty(128 * 1024 * 1024, device=query.device, dtype=torch.uint8)
            wrapper = self._flashinfer.BlockSparseAttentionWrapper(workspace, backend="fa2")
            indptr = self._module.get_indptr_from_mask(mask, query)
            indices = self._module.get_indices_from_mask(mask, query)
            wrapper.plan(
                indptr=indptr,
                indices=indices,
                M=padded_seq_len,
                N=padded_seq_len,
                R=self.block_size,
                C=self.block_size,
                num_qo_heads=num_heads,
                num_kv_heads=num_heads,
                head_dim=head_dim,
                q_data_type=query.dtype,
                kv_data_type=key.dtype,
            )
            self._plan_cache[plan_key] = _RadialPlan(workspace=workspace, wrapper=wrapper)
        return self._plan_cache[plan_key].wrapper

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, video_size) -> torch.Tensor:
        self._validate_inputs(query, key, value, video_size=video_size)

        seq_len = query.shape[1]
        pad_len = (-seq_len) % self.block_size
        padded_query = _pad_bshd(query, pad_len)
        padded_key = _pad_bshd(key, pad_len)
        padded_value = _pad_bshd(value, pad_len)
        padded_seq_len = padded_query.shape[1]

        wrapper = self._get_plan(
            query=padded_query,
            key=padded_key,
            seq_len=seq_len,
            padded_seq_len=padded_seq_len,
            video_size=video_size,
        )

        outputs = []
        for batch_idx in range(padded_query.shape[0]):
            outputs.append(wrapper.run(padded_query[batch_idx], padded_key[batch_idx], padded_value[batch_idx]))

        output = torch.stack(outputs, dim=0).contiguous()
        if pad_len > 0:
            output = output[:, :seq_len]
        # flashinfer's block-sparse wrapper can return fp16 even when planned with bf16;
        # cast back so downstream projections see the model dtype.
        if output.dtype != query.dtype:
            output = output.to(query.dtype)
        return output.contiguous()
