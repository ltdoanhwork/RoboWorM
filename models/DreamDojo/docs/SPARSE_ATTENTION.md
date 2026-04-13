# Sparse Attention in DreamDojo

This guide explains how to apply sparse attention patterns to DreamDojo's DiT backbone for inference acceleration. The codebase provides three built-in mechanisms for sparse attention injection.

## Background

DreamDojo's DiT uses full bidirectional self-attention over flattened video tokens after patchify. For 14B at 480x640 with 13 frames:
- Latent shape before patchify: `(4, 60, 80)`
- DiT token grid after `patch_spatial=2`: `(4, 30, 40)` = 4,800 tokens per sequence
- Full attention: O(4800^2) = ~23M attention pairs per layer per head
- 36 layers x 40 heads = significant compute

Sparse attention reduces this by restricting which tokens can attend to each other.

## Method 1: NATTEN (Neighborhood Attention)

**Best for**: Replacing dense spatial attention with local windows in middle layers.

### How It Works

NATTEN restricts each token to attend only to its spatial neighbors within a kernel window. The existing function `replace_selfattn_op_with_sparse_attn_op()` handles injection.

**File**: `cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py` (line 1919+)

### Usage

```python
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import (
    replace_selfattn_op_with_sparse_attn_op
)

# After model loading
model.net = replace_selfattn_op_with_sparse_attn_op(
    model.net,
    n_dense_blocks=4,  # Keep 4 blocks with full attention (evenly distributed)
    natten_parameters={
        "kernel_size": (3, 7, 7),   # (temporal, height, width) window
        "dilation": (1, 1, 1),       # Dilation for larger receptive field
    }
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_dense_blocks` | int | Number of blocks that keep full dense attention. Distributed evenly across all blocks. Set to 0 for all sparse, -1 for no modification. |
| `natten_parameters` | dict | `kernel_size`: (T, H, W) local window size. `dilation`: stride for dilated attention. |

### Recommended Configurations for 14B (36 blocks)

| Config | Dense Blocks | Sparse Blocks | Speed vs Quality |
|--------|-------------|---------------|------------------|
| Conservative | 8 (every 4th) | 28 | Good quality, moderate speedup |
| Balanced | 4 (every 9th) | 32 | Slight quality drop, good speedup |
| Aggressive | 2 (first+last) | 34 | Noticeable quality drop, max speedup |

### Implementation Details

The replacement function (`minimal_v4_dit.py:1919-1987`):
1. Iterates over all transformer blocks in the model
2. For non-dense blocks, replaces `block.self_attn.attn_op` with `NattenA2AAttnOp`
3. `NattenA2AAttnOp` (from `a2a_cp.py:222`) wraps NVIDIA's `NeighborhoodAttention` with context parallelism compatibility
4. Requires the `natten` package to be installed

## Method 2: FlexAttention (BlockMask)

**Best for**: Custom sparse patterns (temporal causal, local windows, strided, etc.) with optimal kernel performance.

### How It Works

PyTorch's FlexAttention uses compiled block masks to define which tokens can attend to which. The mask is compiled once and reused across denoising steps.

**File**: `cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py` (line 280-360, `flex_attention_op`)

### Enabling FlexAttention

Set the attention backend to `torch-flex` in the network config:

```python
# In config or after model loading:
model.net.atten_backend = "torch-flex"
# Or per-block:
for block in model.net.blocks:
    block.self_attn.atten_backend = "torch-flex"
```

### Custom Mask Examples

Define mask functions for `create_block_mask()`:

```python
from torch.nn.attention.flex_attention import create_block_mask

# 1. Temporal causal mask (each frame sees only current + past frames)
def temporal_causal_mask(b, h, q_idx, kv_idx):
    H, W = 60, 80  # latent spatial dims
    spatial_size = H * W
    q_t = q_idx // spatial_size
    kv_t = kv_idx // spatial_size
    return kv_t <= q_t

# 2. Local spatial window (attend to neighbors within radius R)
def local_spatial_mask(b, h, q_idx, kv_idx):
    H, W = 60, 80
    R = 7  # radius
    q_h = (q_idx % (H * W)) // W
    q_w = q_idx % W
    kv_h = (kv_idx % (H * W)) // W
    kv_w = kv_idx % W
    return (torch.abs(q_h - kv_h) <= R) & (torch.abs(q_w - kv_w) <= R)

# 3. Temporal causal + spatial local (combined)
def causal_local_mask(b, h, q_idx, kv_idx):
    H, W = 60, 80
    R = 7
    spatial_size = H * W
    q_t = q_idx // spatial_size
    kv_t = kv_idx // spatial_size
    q_h = (q_idx % spatial_size) // W
    q_w = q_idx % W
    kv_h = (kv_idx % spatial_size) // W
    kv_w = kv_idx % W
    temporal_ok = kv_t <= q_t
    spatial_ok = (torch.abs(q_h - kv_h) <= R) & (torch.abs(q_w - kv_w) <= R)
    # Same frame: local spatial; past frames: all tokens
    same_frame = (q_t == kv_t)
    return temporal_ok & (spatial_ok | ~same_frame)

# 4. Strided attention (attend to every k-th token globally)
def strided_mask(b, h, q_idx, kv_idx):
    stride = 4
    return (kv_idx % stride == 0) | (q_idx == kv_idx)

# Create block mask
block_mask = create_block_mask(
    temporal_causal_mask,
    B=1, H=40,  # batch, num_heads
    Q_LEN=4800, KV_LEN=4800,
    device="cuda",
    _compile=True,
)
```

### Integration Point

To inject custom masks, modify `flex_attention_op()` in `minimal_v4_dit.py:330-354`:

```python
# Current: only validity mask (padding)
def allow_valid(b, h, q_idx, kv_idx):
    return (q_idx < S_q) & (kv_idx < S_kv)

# Replace with your custom mask:
def custom_mask(b, h, q_idx, kv_idx):
    return your_sparse_pattern(b, h, q_idx, kv_idx) & (q_idx < S_q) & (kv_idx < S_kv)
```

### Performance Notes

- FlexAttention pads Q/K/V to multiples of 128 for optimal CUDA kernels
- Block mask is compiled once per shape, cached for reuse
- Best with `torch.compile` for full kernel fusion
- Supports per-head different patterns (head-dependent sparsity)

## Method 3: Direct Attention Op Replacement

**Best for**: Completely custom attention implementations (e.g., linear attention, sparse transformers).

### How It Works

Each transformer block's self-attention module has a pluggable `attn_op` attribute. You can replace it with any `nn.Module` that implements the same interface.

### Interface

```python
class CustomAttnOp(nn.Module):
    def forward(
        self,
        query: torch.Tensor,    # (B, S_q, n_heads, head_dim)
        key: torch.Tensor,      # (B, S_kv, n_heads, head_dim)
        value: torch.Tensor,    # (B, S_kv, n_heads, head_dim)
        video_size: tuple = None,  # (T, H, W) for spatial structure
        **kwargs,
    ) -> torch.Tensor:          # (B, S_q, n_heads * head_dim)
        # Your implementation
        ...
```

### Injection

```python
# Replace attention in specific blocks
custom_op = CustomAttnOp()
for i, block in enumerate(model.net.blocks):
    if i in target_layers:
        block.self_attn.register_module("attn_op", custom_op)
```

This is exactly what `replace_selfattn_op_with_sparse_attn_op` does internally (line 1987).

## Layer-Selective Sparsity Strategy

Not all layers benefit equally from sparse attention. Based on analysis of similar DiT models:

### Recommended Pattern for 14B (36 blocks)

```
Layer 0-3:    Dense  (early layers: local feature extraction, need full context)
Layer 4-8:    NATTEN (kernel=7x7, spatial locality peaks here)
Layer 9:      Dense  (transition layer)
Layer 10-17:  NATTEN (kernel=13x13, middle layers: global features)
Layer 18:     Dense  (mid-point anchor)
Layer 19-26:  NATTEN (kernel=7x7, spatial locality returns)
Layer 27:     Dense  (transition layer)
Layer 28-34:  NATTEN (kernel=7x7, refinement layers)
Layer 35:     Dense  (final layer: full context for output)
```

### Implementation

```python
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import (
    replace_selfattn_op_with_sparse_attn_op
)

# Simple: evenly distribute dense blocks
model.net = replace_selfattn_op_with_sparse_attn_op(
    model.net,
    n_dense_blocks=5,  # Dense at layers ~0, 7, 14, 21, 28, 35
    natten_parameters={"kernel_size": (3, 7, 7), "dilation": (1, 1, 1)}
)
```

For per-layer configuration, modify `replace_selfattn_op_with_sparse_attn_op()` to accept a list of `natten_parameters` (one per block, `None` for dense).

## Measuring Speedup

After applying sparse attention, measure the impact:

```python
import time
import torch

# Warmup
for _ in range(3):
    _ = video2world.step_inference(img, actions, guidance=0, seed=1)
    torch.cuda.synchronize()

# Benchmark
times = []
for i in range(10):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = video2world.step_inference(img, actions, guidance=0, seed=i)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append(t1 - t0)

print(f"Mean: {sum(times)/len(times):.3f}s, Std: {(sum((t-sum(times)/len(times))**2 for t in times)/len(times))**0.5:.3f}s")
```

See `docs/TIMING.md` for more detailed profiling approaches.

---

<= Previous: [[Inference](INFERENCE.md)]

=> Next: [[Timing & Profiling](TIMING.md)]
