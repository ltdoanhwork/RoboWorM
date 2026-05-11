# Sparse-VideoGen (SVG) Adapter for Ctrl-World

## Overview

Modular, non-invasive integration of **SVG1** (Stripe Sparse + MSE Profiling) and
**SVG2/SAP** (Semantic Aware Permutation) into the Ctrl-World SVD UNet.

Follows the same adapter pattern as the PISA integration:
- Core model files (`ctrl_world.py`, `unet_spatio_temporal_condition.py`) are **not modified**.
- Injection via `unet.set_attn_processor()` after checkpoint load.
- `disable_svg()` cleanly restores default diffusers SDPA.

---

## Architecture

```
methods/sparse_attention/
├── Sparse-VideoGen/          ← upstream, unmodified
└── svg/                      ← Ctrl-World adapter
    ├── __init__.py           ← enable_svg / disable_svg
    ├── adapter.py            ← injection logic
    ├── processor_svg1.py     ← SVG1: stripe sparse + online MSE profiling
    ├── processor_svg2.py     ← SVG2: K-Means + block sparse (SAP)
    ├── mask_utils.py         ← SVD-specific attention masks
    ├── config.py             ← SVGConfig dataclass + add_svg_args()
    ├── benchmark.py          ← standalone speedup benchmark
    └── README.md
```

---

## SVD UNet Attention Shapes

Ctrl-World uses a 3-view stacked latent: `H=72, W=40 → T=2880` at L0.

| Layer | T | SVG applies? |
|-------|---|-------------|
| Temporal self-attn (F=11) | 11 | ❌ SDPA fallback (T < 128) |
| Mid-block (9×5) | 45 | ❌ SDPA fallback |
| L2 spatial (18×10) | 180 | ✅ SVG sparse |
| L1 spatial (36×20) | 720 | ✅ SVG sparse |
| **L0 spatial (72×40)** | **2880** | ✅ SVG sparse ← bottleneck |

Cross-attention layers are always skipped (encoder_hidden_states ≠ None).

---

## Attention Patterns

### SVG1 — Online MSE Profiling

Two candidate masks are compared at every attention call using sampled MSE:

- **Pattern 0 – Diagonal band**: token `i` attends `j` if `|i-j| ≤ bandwidth/2`.
  Captures local spatial correlation within a view.
- **Pattern 1 – View stripe**: tokens at the same `(row%24, col)` position
  attend each other across all 3 views. Captures cross-view semantic alignment.

The processor samples `num_sampled_rows` query rows, computes MSE for both
patterns, and applies the winning one via masked SDPA.

### SVG2 / SAP — Semantic Aware Permutation

1. Run K-Means on Q and K tokens to discover semantic clusters.
2. Build a dynamic sparse block map (which cluster pairs to attend).
3. Permute Q/K/V by cluster labels → block-contiguous in memory.
4. Apply block-sparse attention (flashinfer kernel if available, else SDPA).
5. Inverse permutation to restore token order.

---

## Usage

### Enable SVG in rollout scripts

```bash
# SVG1 (no extra deps needed)
python models/Ctrl-World/scripts/rollout_replay_traj.py \
  --ckpt_path ... \
  --use_svg --svg_pattern SVG1 --svg_sparsity 0.25

# SVG2 (requires: pip install flashinfer-cu12 cuvs-cu12)
python models/Ctrl-World/scripts/rollout_replay_traj.py \
  --ckpt_path ... \
  --use_svg --svg_pattern SVG2 \
  --svg_num_q_centroids 50 --svg_num_k_centroids 200
```

### Benchmark (no checkpoint needed)

```bash
cd video_gen_physics
python methods/sparse_attention/svg/benchmark.py --pattern SVG1 --sparsity 0.25
```

---

## Dependencies

### SVG1 — minimal
- `torch >= 2.0` (masked SDPA, no FlexAttention needed)
- SVG package itself (for future kmeans_utils): `pip install -e methods/sparse_attention/Sparse-VideoGen`

### SVG2 / SAP — additional
- `flashinfer-cu12`: `pip install flashinfer-cu12`
- `cuvs-cu12`: `pip install cuvs-cu12 --extra-index-url=https://pypi.nvidia.com`
- SVG Triton kernels: `cd methods/sparse_attention/Sparse-VideoGen && bash svg/kernels/setup.sh`

> SVG2 gracefully falls back to full SDPA if flashinfer is not installed.

---

## Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `svg_pattern` | `"SVG1"` | `"SVG1"` (stripe) or `"SVG2"` (SAP) |
| `svg_sparsity` | `0.25` | Fraction of tokens attended (SVG1) |
| `svg_first_layers_fp` | `2` | First N layers use full SDPA |
| `svg_num_sampled_rows` | `64` | MSE profiling rows per call (SVG1) |
| `svg_num_q_centroids` | `50` | K-Means Q clusters (SVG2) |
| `svg_num_k_centroids` | `200` | K-Means K clusters (SVG2) |
| `svg_top_p_kmeans` | `0.9` | Block selection threshold (SVG2) |
