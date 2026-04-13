# Copyright (c) 2025-2026, Haopeng Li

import functools
import warnings

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:
    try:
        from triton.tools import tensor_descriptor

        TensorDescriptor = tensor_descriptor.TensorDescriptor
    except ImportError:
        TensorDescriptor = None


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def _safe_kernel_config():
    if supports_host_descriptor():
        return dict(num_warps=8, num_stages=4)
    return dict(num_warps=4, num_stages=2)


def contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(
            ctx,
            *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
            **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()},
        )

    return wrapper


@triton.jit
def chunk_reduce_kernel(
    q,
    k,
    v,
    qc,
    kc,
    vc,
    T, H,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    i_b = i_bh // H
    i_h = i_bh % H

    q_base = q + i_b * stride_qb + i_h * stride_qh
    k_base = k + i_b * stride_kb + i_h * stride_kh
    v_base = v + i_b * stride_vb + i_h * stride_vh

    i_k = i_kv // tl.cdiv(V, BV)
    i_v = i_kv % tl.cdiv(V, BV)

    block_size = tl.minimum(BT, T - i_t * BT)

    p_q = tl.make_block_ptr(q_base, (T, K), (stride_qt, stride_qd), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k_base, (T, K), (stride_kt, stride_kd), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v_base, (T, V), (stride_vt, stride_vd), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))

    b_qc = tl.sum(b_q, axis=0) / block_size
    b_kc = tl.sum(b_k, axis=0) / block_size
    b_vc = tl.sum(b_v, axis=0)

    p_qc = tl.make_block_ptr(qc + i_bh * N * K + i_t * K, (K,), (1,), (i_k * BK,), (BK,), (0,))
    p_kc = tl.make_block_ptr(kc + i_bh * N * K + i_t * K, (K,), (1,), (i_k * BK,), (BK,), (0,))
    p_vc = tl.make_block_ptr(vc + i_bh * N * V + i_t * V, (V,), (1,), (i_v * BV,), (BV,), (0,))

    tl.store(p_qc, b_qc.to(p_qc.dtype.element_ty), boundary_check=(0,))
    tl.store(p_kc, b_kc.to(p_kc.dtype.element_ty), boundary_check=(0,))
    tl.store(p_vc, b_vc.to(p_vc.dtype.element_ty), boundary_check=(0,))


@triton.jit
def fused_chunk_reduce_kernel(
    q,
    k,
    v,
    qc,
    kc,
    vc,
    h_out,
    T, H,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    i_b = i_bh // H
    i_h = i_bh % H

    q_base = q + i_b * stride_qb + i_h * stride_qh
    k_base = k + i_b * stride_kb + i_h * stride_kh
    v_base = v + i_b * stride_vb + i_h * stride_vh

    i_k = i_kv // tl.cdiv(V, BV)
    i_v = i_kv % tl.cdiv(V, BV)

    block_size = tl.minimum(BT, T - i_t * BT)

    p_q = tl.make_block_ptr(q_base, (T, K), (stride_qt, stride_qd), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k_base, (T, K), (stride_kt, stride_kd), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v_base, (T, V), (stride_vt, stride_vd), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))

    b_qc = tl.sum(b_q, axis=0) / block_size
    b_kc = tl.sum(b_k, axis=0) / block_size
    b_vc = tl.sum(b_v, axis=0)

    # Algebraic decomposition to avoid tl.sum→tl.dot chain that crashes
    # Triton 3.3.0 MLIR inliner. Math:
    #   (K - mean_K)^T @ V = K^T @ V - mean_K[:,None] * sum_V[None,:]
    b_kt = tl.trans(b_k).to(b_v.dtype)
    b_ktv = tl.dot(b_kt, b_v)                           # [BK,BV] f32
    b_correction = b_kc[:, None] * b_vc.to(tl.float32)[None, :]  # [BK,BV] f32
    b_hc = b_ktv - b_correction

    p_qc = tl.make_block_ptr(qc + i_bh * N * K + i_t * K, (K,), (1,), (i_k * BK,), (BK,), (0,))
    p_kc = tl.make_block_ptr(kc + i_bh * N * K + i_t * K, (K,), (1,), (i_k * BK,), (BK,), (0,))
    p_vc = tl.make_block_ptr(vc + i_bh * N * V + i_t * V, (V,), (1,), (i_v * BV,), (BV,), (0,))

    tl.store(p_qc, b_qc.to(p_qc.dtype.element_ty), boundary_check=(0,))
    tl.store(p_kc, b_kc.to(p_kc.dtype.element_ty), boundary_check=(0,))
    tl.store(p_vc, b_vc.to(p_vc.dtype.element_ty), boundary_check=(0,))

    # Use manual pointers for atomic_add — tl.make_block_ptr + tl.atomic_add
    # triggers 'tt.atomic_rmw ptr/value type mismatch' in Triton 3.3.0.
    h_base = h_out + i_bh * K * V + i_k * BK * V + i_v * BV
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    p_h = h_base + offs_k[:, None] * V + offs_v[None, :]
    mask_h = (i_k * BK + offs_k[:, None] < K) & (i_v * BV + offs_v[None, :] < V)
    tl.atomic_add(p_h, b_hc.to(h_out.dtype.element_ty), mask=mask_h)


def chunk_reduce(q, k, v, chunk_size):
    B, H, T, K, V = *k.shape, v.shape[-1]
    N = triton.cdiv(T, chunk_size)

    BK = min(128, triton.next_power_of_2(K))
    BV = min(128, triton.next_power_of_2(V))

    qc = torch.empty(B, H, N, K, device=k.device, dtype=k.dtype)
    kc = torch.empty(B, H, N, K, device=k.device, dtype=k.dtype)
    vc = torch.empty(B, H, N, V, device=v.device, dtype=v.dtype)

    grid = (triton.cdiv(K, BK) * triton.cdiv(V, BV), N, B * H)
    chunk_reduce_kernel[grid](
        q=q,
        k=k,
        v=v,
        qc=qc,
        kc=kc,
        vc=vc,
        T=T, H=H,
        stride_qb=q.stride(0), stride_qh=q.stride(1), stride_qt=q.stride(2), stride_qd=q.stride(3),
        stride_kb=k.stride(0), stride_kh=k.stride(1), stride_kt=k.stride(2), stride_kd=k.stride(3),
        stride_vb=v.stride(0), stride_vh=v.stride(1), stride_vt=v.stride(2), stride_vd=v.stride(3),
        N=N,
        K=K,
        V=V,
        BT=chunk_size,
        BK=BK,
        BV=BV,
        **_safe_kernel_config(),
    )
    return qc, kc, vc


def fused_chunk_reduce_pytorch(q, k, v, chunk_size, use_bias=False, p=2):
    B, H, T, K, V = *k.shape, v.shape[-1]
    N = triton.cdiv(T, chunk_size)
    pad = N * chunk_size - T

    if pad > 0:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))

    q_chunks = q.reshape(B, H, N, chunk_size, K)
    k_chunks = k.reshape(B, H, N, chunk_size, K)
    v_chunks = v.reshape(B, H, N, chunk_size, V)

    lens = torch.full((N,), chunk_size, device=q.device, dtype=q.dtype)
    if pad > 0:
        lens[-1] = chunk_size - pad
        mask = torch.ones(N, chunk_size, device=q.device, dtype=q.dtype)
        mask[-1, chunk_size - pad :] = 0.0
        mask3d = mask.unsqueeze(-1)
        q_chunks = q_chunks * mask3d
        k_chunks = k_chunks * mask3d
        v_chunks = v_chunks * mask3d

    qc = q_chunks.sum(dim=3) / lens[None, None, :, None]
    kc = k_chunks.sum(dim=3) / lens[None, None, :, None]
    vc = v_chunks.sum(dim=3)

    k_centered = k_chunks - kc.unsqueeze(3)
    hc = torch.einsum("bhnsk,bhnsv->bhnkv", k_centered.float(), v_chunks.float()).to(q.dtype)

    h = hc.sum(dim=2)

    if use_bias:
        h_norm = torch.norm(hc.flatten(-2, -1), p=p, dim=-1)
        return qc, kc, vc, h, h_norm[..., None, :]
    return qc, kc, vc, h, None


def fused_chunk_reduce(q, k, v, chunk_size, use_bias=False, p=2):
    # Triton kernel for this op hangs during MLIR compilation on Triton 3.3.0
    # (tl.sum inlining causes infinite loop in MLIR canonicalizer).
    # Use PyTorch — this is just mean/sum/outer-product, not the bottleneck.
    return fused_chunk_reduce_pytorch(q, k, v, chunk_size, use_bias, p)


@triton.jit
def piecewise_sparse_attention_fwd_kernel(
    q,
    k,
    v,
    kc,
    vc,
    h,
    o,
    indices,
    scale,
    T, H,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    NS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    i_b = i_bh // H
    i_h = i_bh % H

    q_base = q + i_b * stride_qb + i_h * stride_qh
    k_base = k + i_b * stride_kb + i_h * stride_kh
    v_base = v + i_b * stride_vb + i_h * stride_vh
    o_base = o + i_b * stride_ob + i_h * stride_oh

    p_q = tl.make_block_ptr(q_base, (T, K), (stride_qt, stride_qd), (i_t * BT, 0), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))

    acc = tl.zeros([BT, BV], dtype=tl.float32)
    l_i = tl.zeros((BT,), dtype=tl.float32)
    m_i = tl.zeros((BT,), dtype=tl.float32) - float("inf")

    sm_scale = scale * 1.44269504

    for i in range(NS):
        i_n = tl.load(indices + i_bh * NT * NS + i_t * NS + i).to(tl.int32)
        bos = i_n * BT

        p_k = tl.make_block_ptr(k_base, (T, K), (stride_kt, stride_kd), (bos, 0), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v_base, (T, V), (stride_vt, stride_vd), (bos, i_v * BV), (BT, BV), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))

        b_s = tl.dot(b_q, tl.trans(b_k))
        b_s *= sm_scale
        b_s += tl.where((bos + tl.arange(0, BT))[None, :] < T, 0, float("-inf"))

        new_m_i = tl.maximum(m_i, tl.max(b_s, -1))
        alpha = tl.math.exp2(m_i - new_m_i)
        score = tl.math.exp2(b_s - new_m_i[:, None])

        l_i = l_i * alpha + tl.sum(score, -1)
        acc = acc * alpha[:, None] + tl.dot(score.to(b_v.dtype), b_v)
        m_i = new_m_i

    last_chunk_len = T - (NT - 1) * BT

    offs_n_idx = tl.arange(0, triton.next_power_of_2(NS))
    loaded_indices = tl.load(indices + i_bh * NT * NS + i_t * NS + offs_n_idx, mask=offs_n_idx < NS, other=-1)

    g_l = tl.zeros([BT], dtype=tl.float32)

    for start_n in range(0, NT, GROUP_SIZE):
        p_kc = tl.make_block_ptr(kc + i_bh * NT * K, (NT, K), (K, 1), (start_n, 0), (GROUP_SIZE, BK), (1, 0))
        b_kc = tl.load(p_kc, boundary_check=(0, 1))

        b_s_mean = tl.dot(b_q, tl.trans(b_kc))
        b_s_mean = b_s_mean * sm_scale

        chunk_indices = start_n + tl.arange(0, GROUP_SIZE)
        current_lens = tl.where(chunk_indices == NT - 1, last_chunk_len, BT)
        current_lens = current_lens.to(tl.float32)

        is_in_indices = chunk_indices[:, None] == loaded_indices[None, :]
        mask_is_selected = tl.max(is_in_indices, axis=1)
        valid_mask = (chunk_indices < NT) & (mask_is_selected == 0)

        b_s_mean = tl.where(valid_mask[None, :], b_s_mean, float("-inf"))

        new_m_i = tl.maximum(m_i, tl.max(b_s_mean, 1))
        alpha = tl.math.exp2(m_i - new_m_i)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha
        m_i = new_m_i

        prob_chunk = tl.math.exp2(b_s_mean - m_i[:, None])

        p_vc = tl.make_block_ptr(vc + i_bh * NT * V, (NT, V), (V, 1), (start_n, i_v * BV), (GROUP_SIZE, BV), (1, 0))
        b_vc = tl.load(p_vc, boundary_check=(0, 1))

        acc += tl.dot(prob_chunk.to(b_vc.dtype), b_vc)
        weighted_prob = prob_chunk * current_lens[None, :]
        g_l += tl.sum(weighted_prob, axis=1)

    p_h = tl.make_block_ptr(h + i_bh * K * V, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0))
    b_h = tl.load(p_h, boundary_check=(0, 1))

    b_r = tl.dot(b_q, b_h.to(b_q.dtype))
    correction_scale = g_l * (1.0 / T) * scale

    acc += b_r * correction_scale[:, None]

    l_i += g_l
    acc /= l_i[:, None]

    p_o = tl.make_block_ptr(o_base, (T, V), (stride_ot, stride_od), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, acc.to(b_q.dtype), boundary_check=(0, 1))


def piecewise_sparse_attention(q, k, v, o=None, density=0.1, block_size=64, scale=None, use_bias=False):
    B, H, T, K, V = *k.shape, v.shape[-1]
    scale = K ** -0.5 if scale is None else scale

    if q.shape[2] != T:
        raise RuntimeError(
            f"[PISA] Cross-attention is not supported by the config-driven PISA backend "
            f"(T_q={q.shape[2]}, T_kv={T})."
        )

    # Triton make_block_ptr requires power-of-2 block shapes.
    block_size_po2 = triton.next_power_of_2(block_size)
    if block_size_po2 != block_size:
        warnings.warn(
            f"[PISA] block_size={block_size} rounded to {block_size_po2} "
            f"(Triton requires power of 2)",
            stacklevel=2,
        )
        block_size = block_size_po2

    NT = triton.cdiv(T, block_size)
    BK = min(128, triton.next_power_of_2(K))
    BV = min(128, triton.next_power_of_2(V))
    group_size = 64

    qc, kc, vc, h, bias = fused_chunk_reduce(q, k, v, block_size, use_bias=use_bias)

    score = torch.einsum("bhid, bhjd -> bhij", qc, kc * scale)
    if bias is not None:
        score = torch.softmax(score + torch.log(bias + 1e-5), dim=-1)

    top_k = max(1, int(density * NT))
    indices = torch.topk(score, k=top_k, dim=-1).indices

    if o is None:
        o = torch.empty(B, H, T, V, device=v.device, dtype=v.dtype)

    grid = (triton.cdiv(V, BV), NT, B * H)
    piecewise_sparse_attention_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        kc=kc,
        vc=vc,
        h=h,
        o=o,
        indices=indices,
        scale=scale,
        T=T, H=H,
        stride_qb=q.stride(0), stride_qh=q.stride(1), stride_qt=q.stride(2), stride_qd=q.stride(3),
        stride_kb=k.stride(0), stride_kh=k.stride(1), stride_kt=k.stride(2), stride_kd=k.stride(3),
        stride_vb=v.stride(0), stride_vh=v.stride(1), stride_vt=v.stride(2), stride_vd=v.stride(3),
        stride_ob=o.stride(0), stride_oh=o.stride(1), stride_ot=o.stride(2), stride_od=o.stride(3),
        K=K,
        V=V,
        BT=block_size,
        BK=BK,
        BV=BV,
        NT=NT,
        NS=top_k,
        GROUP_SIZE=group_size,
        **_safe_kernel_config(),
    )

    return o
