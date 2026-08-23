# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from neophilia; Author: Qsh (qsh.zh27@gmail.com)
# MIT License

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, List, Tuple, Union

import torch
import torch.distributed as dist
from einops import rearrange
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.nn import Module

from cosmos_predict2._src.predict2.modules.neighborhood_attn import NeighborhoodAttention
from cosmos_predict2._src.predict2.networks.attention import attention


@lru_cache(maxsize=1)
def _load_svg_runtime_class():
    runtime_path = (
        Path(__file__).resolve().parents[6]
        / "methods"
        / "sparse_attention"
        / "svg"
        / "dreamdojo_runtime.py"
    )
    if not runtime_path.exists():
        raise ImportError(f"[SVG] Missing DreamDojo runtime at {runtime_path}.")

    spec = importlib.util.spec_from_file_location("dreamdojo_svg_runtime", runtime_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"[SVG] Could not load module spec from {runtime_path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DreamDojoSVGRuntime


@lru_cache(maxsize=1)
def _load_piecewise_sparse_attention():
    kernel_path = (
        Path(__file__).resolve().parents[6]
        / "methods"
        / "sparse_attention"
        / "pisa"
        / "piecewise_sparse_attn_tma.py"
    )
    if not kernel_path.exists():
        raise ImportError(f"[PISA] Missing vendored kernel at {kernel_path}.")

    spec = importlib.util.spec_from_file_location("dreamdojo_pisa_kernel", kernel_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"[PISA] Could not load module spec from {kernel_path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.piecewise_sparse_attention


@lru_cache(maxsize=1)
def _load_radial_runtime_class():
    runtime_path = (
        Path(__file__).resolve().parents[6]
        / "methods"
        / "sparse_attention"
        / "radial"
        / "dreamgen_runtime.py"
    )
    if not runtime_path.exists():
        raise ImportError(f"[Radial] Missing runtime at {runtime_path}.")

    spec = importlib.util.spec_from_file_location("dreamdojo_radial_runtime", runtime_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"[Radial] Could not load module spec from {runtime_path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DreamGenRadialRuntime


def post_all2all(local_seq_2_local_head, seq_world_size):
    def post_func(input):
        # b, s, n, h
        if local_seq_2_local_head:
            output = rearrange(input, "w bs seq h d -> bs (w seq) h d")
        else:
            output = rearrange(input, "w bs s h d -> bs s (w h) d", w=seq_world_size)

        return output

    return post_func


def single_all_to_all(input, local_seq_2_local_head, group, async_op=False):
    seq_world_size = dist.get_world_size(group)

    # b, s, n, h
    if local_seq_2_local_head:
        bs, local_seq_len, num_total_head, head_dim = input.shape
        assert num_total_head % seq_world_size == 0, (
            f"Number of heads ({num_total_head}) must be divisible by the sequence parallel size ({seq_world_size})!"
        )
        input_t = rearrange(
            input, "bs seq_len (w h) d -> w bs seq_len h d", w=seq_world_size, h=num_total_head // seq_world_size
        ).contiguous()
        post_all2all_fun = post_all2all(local_seq_2_local_head, seq_world_size)
    else:
        bs, global_seq_len, num_local_head, head_dim = input.shape
        input_t = rearrange(
            input, "bs (w s) h d -> w bs s h d", w=seq_world_size, s=global_seq_len // seq_world_size
        ).contiguous()
        post_all2all_fun = post_all2all(local_seq_2_local_head, seq_world_size)

    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group, async_op=async_op)

    res = post_all2all_fun(output)
    return res


def async_a2a_communicate(
    a2a_inputs: Union[torch.Tensor, List[torch.Tensor]],
    cp_size: int,
    cp_group: ProcessGroup,
    cp_stream: torch.cuda.Stream,
    local_seq_2_local_head: bool,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    A2A communication for context parallelism. best used in communicate qkv
    Modified from Nvidia Transformer Engine.
    """
    a2a_inputs = [a2a_inputs] if not isinstance(a2a_inputs, list) else a2a_inputs
    a2a_outputs, a2a_reqs = [None] * len(a2a_inputs), [None] * len(a2a_inputs)
    a2a_post_fns = [None] * len(a2a_inputs)
    if local_seq_2_local_head:
        for i in range(len(a2a_inputs) + 2):
            if 0 < i < len(a2a_inputs) + 1:
                a2a_outputs[i - 1] = torch.empty_like(a2a_inputs[i - 1])
                a2a_reqs[i - 1] = torch.distributed.all_to_all_single(
                    a2a_outputs[i - 1], a2a_inputs[i - 1], group=cp_group, async_op=True
                )
                a2a_post_fns[i - 1] = post_all2all(local_seq_2_local_head, cp_size)
            if i > 1:
                with torch.cuda.stream(cp_stream):
                    a2a_reqs[i - 2].wait()
                    a2a_outputs[i - 2] = a2a_post_fns[i - 2](a2a_outputs[i - 2])
            if i < len(a2a_inputs):
                a2a_inputs[i] = rearrange(
                    a2a_inputs[i], "bs seq_len (w h) d -> w bs seq_len h d", w=cp_size
                ).contiguous()
    else:
        for i in range(len(a2a_inputs) + 2):
            if 0 < i < len(a2a_inputs) + 1:
                a2a_outputs[i - 1] = torch.empty_like(a2a_inputs[i - 1])
                a2a_reqs[i - 1] = torch.distributed.all_to_all_single(
                    a2a_outputs[i - 1], a2a_inputs[i - 1], group=cp_group, async_op=True
                )
                a2a_post_fns[i - 1] = post_all2all(local_seq_2_local_head, cp_size)
            if i < len(a2a_inputs):
                a2a_inputs[i] = rearrange(a2a_inputs[i], "bs (w s) h d -> w bs s h d", w=cp_size).contiguous()
            if i > 1:
                with torch.cuda.stream(cp_stream):
                    a2a_reqs[i - 2].wait()
                    a2a_outputs[i - 2] = a2a_post_fns[i - 2](a2a_outputs[i - 2])
    torch.cuda.current_stream().wait_stream(cp_stream)
    return a2a_outputs[0] if len(a2a_inputs) == 1 else a2a_outputs


class _SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor, local_seq_2_local_head: bool) -> Tensor:
        ctx.group = group
        res = single_all_to_all(input, local_seq_2_local_head, group, False)
        ctx.local_seq_2_local_head = local_seq_2_local_head
        return res

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None]:
        return (None, _SeqAllToAll.apply(ctx.group, *grad_output, not ctx.local_seq_2_local_head), None)


class _SeqAllToAllQKV(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cp_size: int,
        cp_stream: torch.cuda.Stream,
        local_seq_2_local_head: bool,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        ctx.group = group
        ctx.cp_size = cp_size
        ctx.cp_stream = cp_stream
        ctx.local_seq_2_local_head = local_seq_2_local_head
        q, k, v = async_a2a_communicate([q, k, v], cp_size, group, cp_stream, local_seq_2_local_head)
        return q, k, v

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, Tensor, Tensor, None, None, None]:
        q_grad, k_grad, v_grad = _SeqAllToAllQKV.apply(
            ctx.group, *grad_output, ctx.cp_size, ctx.cp_stream, not ctx.local_seq_2_local_head
        )
        return (None, q_grad, k_grad, v_grad, None, None, None)


class DistributedAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
    """

    def __init__(
        self,
        local_attention: Union[Module, Callable],
    ) -> None:
        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention
        self.pg = None
        self.stream = None

    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        if self.pg is None:
            return self.local_attn(query, key, value, *args, **kwargs)
        pg_size = dist.get_world_size(self.pg)
        if pg_size < 2:
            return self.local_attn(query, key, value, *args, **kwargs)

        query_layer, key_layer, value_layer = _SeqAllToAllQKV.apply(
            self.pg, query, key, value, pg_size, self.stream, True
        )
        context_layer = self.local_attn(query_layer, key_layer, value_layer, *args, **kwargs)

        output = _SeqAllToAll.apply(self.pg, context_layer, False)
        return output

    def set_context_parallel_group(self, group, stream):
        self.pg = group
        self.stream = stream


class MinimalA2AAttnOp(DistributedAttention):
    def __init__(self, *args, **kwargs):
        del args, kwargs
        super(MinimalA2AAttnOp, self).__init__(attention)

    def set_context_parallel_group(self, process_group, ranks, stream, cp_comm_type: str = "p2p"):
        del ranks
        super().set_context_parallel_group(process_group, stream)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs) -> Tensor:
        results = super().forward(query, key, value, *args, **kwargs)
        return rearrange(results, "b ... h l -> b ... (h l)")


class NattenA2AAttnOp(MinimalA2AAttnOp):
    def __init__(self, *args, **kwargs):
        super(NattenA2AAttnOp, self).__init__(None)
        self.natten_op = NeighborhoodAttention(*args, **kwargs, base_attn_op=attention)
        self.local_attn = self.natten_op


class PISALocalAttnOp(torch.nn.Module):
    def __init__(self, density: float = 0.5, block_size: int = 64, use_bias: bool = False):
        super().__init__()
        self.density = density
        self.block_size = block_size
        self.use_bias = use_bias

    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs) -> Tensor:
        del args, kwargs
        if query.shape[1] != key.shape[1] or key.shape[1] != value.shape[1]:
            raise RuntimeError(
                "[PISA] Cross-attention is not supported by the config-driven backend "
                f"(T_q={query.shape[1]}, T_k={key.shape[1]}, T_v={value.shape[1]})."
            )

        piecewise_sparse_attention = _load_piecewise_sparse_attention()
        q = rearrange(query, "b s h d -> b h s d").contiguous()
        k = rearrange(key, "b s h d -> b h s d").contiguous()
        v = rearrange(value, "b s h d -> b h s d").contiguous()
        out = piecewise_sparse_attention(
            q,
            k,
            v,
            density=self.density,
            block_size=min(self.block_size, query.shape[1]),
            scale=None,
            use_bias=self.use_bias,
        )
        # No .contiguous() here: PISAA2AAttnOp immediately flattens (h d), which copies anyway.
        return out.transpose(1, 2)


class PISAA2AAttnOp(DistributedAttention):
    def __init__(self, density: float = 0.5, block_size: int = 64, use_bias: bool = False):
        super(PISAA2AAttnOp, self).__init__(
            PISALocalAttnOp(density=density, block_size=block_size, use_bias=use_bias)
        )

    def set_context_parallel_group(self, process_group, ranks, stream, cp_comm_type: str = "p2p"):
        del cp_comm_type
        world_size = 1 if process_group is None else dist.get_world_size(process_group)
        rank_count = 1 if ranks is None else len(ranks)
        if world_size > 1 or rank_count > 1:
            raise RuntimeError("[PISA] Context parallel / multi-GPU is not supported in DreamDojo v1.")
        del ranks
        super().set_context_parallel_group(process_group, stream)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs) -> Tensor:
        results = super().forward(query, key, value, *args, **kwargs)
        return rearrange(results, "b ... h d -> b ... (h d)")


class RadialLocalAttnOp(torch.nn.Module):
    def __init__(self, block_size: int = 64, decay_factor: float = 1.0, model_type: str = "hunyuan"):
        super().__init__()
        radial_runtime_cls = _load_radial_runtime_class()
        self.runtime = radial_runtime_cls(block_size=block_size, decay_factor=decay_factor, model_type=model_type)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs) -> Tensor:
        del args
        video_size = kwargs.pop("video_size", None)
        kwargs.pop("attention_runtime_context", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise RuntimeError(f"[Radial] Unexpected keyword arguments: {unexpected}.")
        return self.runtime.forward(query, key, value, video_size=video_size)


class RadialA2AAttnOp(DistributedAttention):
    def __init__(self, block_size: int = 64, decay_factor: float = 1.0, model_type: str = "hunyuan"):
        super(RadialA2AAttnOp, self).__init__(
            RadialLocalAttnOp(block_size=block_size, decay_factor=decay_factor, model_type=model_type)
        )

    def set_context_parallel_group(self, process_group, ranks, stream, cp_comm_type: str = "p2p"):
        del cp_comm_type
        world_size = 1 if process_group is None else dist.get_world_size(process_group)
        rank_count = 1 if ranks is None else len(ranks)
        if world_size > 1 or rank_count > 1:
            raise RuntimeError("[Radial] Context parallel / multi-GPU is not supported in DreamDojo v1.")
        del ranks
        super().set_context_parallel_group(process_group, stream)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs) -> Tensor:
        results = super().forward(query, key, value, *args, **kwargs)
        return rearrange(results, "b ... h d -> b ... (h d)")


class SVGLocalAttnOp(torch.nn.Module):
    def __init__(self, **svg_parameters):
        super().__init__()
        svg_runtime_cls = _load_svg_runtime_class()
        self.runtime = svg_runtime_cls(**svg_parameters)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs) -> Tensor:
        del args
        video_size = kwargs.pop("video_size", None)
        attention_runtime_context = kwargs.pop("attention_runtime_context", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise RuntimeError(f"[SVG] Unexpected keyword arguments: {unexpected}.")
        return self.runtime.forward(
            query,
            key,
            value,
            video_size=video_size,
            attention_runtime_context=attention_runtime_context,
        )


class SVGA2AAttnOp(DistributedAttention):
    def __init__(self, **svg_parameters):
        super(SVGA2AAttnOp, self).__init__(SVGLocalAttnOp(**svg_parameters))

    def set_context_parallel_group(self, process_group, ranks, stream, cp_comm_type: str = "p2p"):
        del cp_comm_type
        world_size = 1 if process_group is None else dist.get_world_size(process_group)
        rank_count = 1 if ranks is None else len(ranks)
        if world_size > 1 or rank_count > 1:
            raise RuntimeError("[SVG] Context parallel / multi-GPU is not supported in DreamDojo v1.")
        del ranks
        super().set_context_parallel_group(process_group, stream)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs) -> Tensor:
        results = super().forward(query, key, value, *args, **kwargs)
        return rearrange(results, "b ... h d -> b ... (h d)")
