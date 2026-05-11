# PISA Adapter for Ctrl-World (SVD UNet)

This folder contains the **Ctrl-World-specific adapter** for PISA
(Piecewise Sparse Attention). The upstream PISA source lives in
`../piecewise-sparse-attention/`.

## File layout

```
pisa/
├── __init__.py       — re-exports enable_pisa / disable_pisa
├── adapter.py        — enable_pisa() / disable_pisa()  ← main entry point
├── processor.py      — SVDPISAAttnProcessor (drop-in diffusers processor)
├── config.py         — PISAConfig dataclass + add_pisa_args() for argparse
├── benchmark.py      — standalone speed benchmark (no checkpoint needed)
└── README.md         — this file
```

## Quick start

### 1. Install PISA dependencies

```bash
cd ../piecewise-sparse-attention
pip install -e .
# Requirements: torch >= 2.7.1, triton >= 3.5.1
```

### 2. Use in a rollout script

```python
import sys, os
# From inside models/Ctrl-World/scripts/:
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../methods/sparse_attention/pisa"))
from adapter import enable_pisa, disable_pisa

# After loading the model:
enable_pisa(ctrl_world.unet, density=0.5, block_size=16, start_layer_idx=2)

# To restore:
disable_pisa(ctrl_world.unet)
```

### 3. Use via argparse helper

```python
import argparse
from methods.sparse_attention.pisa.config import add_pisa_args, PISAConfig

parser = argparse.ArgumentParser()
add_pisa_args(parser)
args = parser.parse_args()

if args.use_pisa:
    from methods.sparse_attention.pisa import enable_pisa
    enable_pisa(unet, args.pisa_density, args.pisa_block_size, args.pisa_start_layer_idx)
```

CLI example:
```bash
python rollout_replay_traj.py [...] --use_pisa --pisa_density 0.5
```

### 4. Benchmark (no checkpoint needed)

```bash
cd video_gen_physics
python methods/sparse_attention/pisa/benchmark.py --density 0.5 --block_size 16
```

## Tuning guide

| Parameter | Effect | Recommended |
|-----------|--------|-------------|
| `density` | % blocks computed exactly. Lower = faster, less precise | 0.3–0.7 |
| `block_size` | Token block size. Larger = more coarse approximation | 16 |
| `start_layer_idx` | Skip first N layers (keep full attention) | 2–4 |

## Design principles (no-conflict guarantee)

1. **All code stays in this folder** — no edits to `ctrl_world.py`, `unet_*.py`, etc.
2. **Inject via `unet.set_attn_processor()`** — standard diffusers hook, reversible.
3. **Automatic fallback** — cross-attention and short-sequence layers use SDPA silently.
4. **Future methods** (e.g. `svg/`, `radial/`) follow the same pattern in sibling folders.

## Hardware note

PISA is optimized for **NVIDIA Hopper** (H100/H800, compute capability ≥ 9.0).
On A100 (cc=8.0) a `UserWarning` is shown but the kernel still runs correctly,
with some performance degradation. On other architectures it falls back to a
pure-PyTorch path automatically.
