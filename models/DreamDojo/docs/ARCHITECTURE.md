# DreamDojo Architecture

DreamDojo is an action-conditioned video world model built on the **Diffusion Transformer (DiT)** backbone with **Rectified Flow** formulation. It generates future video frames conditioned on an input image/video, text prompt, and robot action sequences.

## Model Variants

| Variant | Blocks | Heads | Channels | Head Dim | Params |
|---------|--------|-------|----------|----------|--------|
| 2B      | 28     | 16    | 2048     | 128      | ~2B    |
| 7B      | 28     | 32    | 4096     | 128      | ~7B    |
| 14B     | 36     | 40    | 5120     | 128      | ~14B   |

Config definitions: `cosmos_predict2/_src/predict2/action/configs/action_conditioned/net.py`

## Core Architecture

### DiT Backbone: `MiniTrainDIT`

**File**: `cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py`

The backbone is a standard Vision Transformer adapted for video:

```
Input (B, C_in, T, H, W)
  --> Patchify (patch_spatial=2, patch_temporal=1)
  --> Linear projection to model_channels
  --> + 3D RoPE positional embeddings
  --> N x TransformerBlock:
      |-- AdaLN (adaptive layer norm from timestep embedding)
      |-- Self-Attention (Q/K/V with RMSNorm + RoPE)
      |-- Cross-Attention (to text embeddings)
      |-- FFN (GPT-2 style: Linear -> GELU -> Linear)
  --> Unpatchify
  --> Output (B, C_out, T, H, W)
```

**Key classes:**
- `MiniTrainDIT` (line 700+): Main model with `forward()` and `forward_with_cfg()`
- `Attention` (line 389+): Multi-head attention with configurable backends
- `GPT2FeedForward` (line 238+): MLP block
- `RMSNorm` (line 220+): Root mean square normalization

### Action Conditioning

**File**: `cosmos_predict2/_src/predict2/action/networks/action_conditioned_minimal_v1_lvg_dit.py`

Actions are injected into the model via the timestep embedding pathway:

```
action: (B, T-1, action_dim=384) 
  --> ActionEmbedder (Linear projection)
  --> Add to timestep embedding (t_emb)
  --> t_emb conditions AdaLN in each transformer block
```

**Action dimension layout (384-dim):**

| Range       | Robot/Type           |
|-------------|---------------------|
| `[0, 29)`   | Fourier GR-1        |
| `[29, 58)`  | Retargeted GR-1     |
| `[58, 101)` | Unitree G1          |
| `[101, 147)`| Bimanual YAM        |
| `[147, 169)`| AgiBot              |
| `[169, 220)`| Reserved            |
| `[220, 352)`| MANO hand actions   |
| `[352, 384)`| Latent actions      |

Each action frame contains: `[delta_xyz(3), delta_rotation(3), gripper_state(1)]` for the relevant robot, with unused dimensions zeroed.

### Attention Mechanisms

**File**: `cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py` (line 389+)

The `Attention` class supports multiple backends:

| Backend | Description | Default For |
|---------|-------------|-------------|
| `torch` | `F.scaled_dot_product_attention` | General |
| `torch-flex` | FlexAttention with BlockMask | Sparse patterns |
| `minimal_a2a` | Custom A2A attention | 14B model |
| `i4` | Imaginaire4 attention | Alternative |
| `transformer_engine` | NVIDIA TE attention | Training |

**Attention flow:**
```
Input x: (B, S, D)  where S = T*H*W (flattened video tokens)
  --> Q = q_proj(x): (B, S, n_heads*head_dim)
  --> K = k_proj(x): (B, S, n_heads*head_dim)  [self-attn]
       K = k_proj(context): (B, M, n_heads*head_dim)  [cross-attn]
  --> V = v_proj(...)
  --> Reshape: (B, S, n_heads, head_dim)
  --> Q, K = RMSNorm(Q), RMSNorm(K)
  --> Q, K = apply_RoPE_3D(Q, K)  [self-attn only]
  --> output = attn_op(Q, K, V)
  --> output = output_proj(output)
```

### Rectified Flow Diffusion

**File**: `cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py`

DreamDojo uses Rectified Flow (RF), a straight-path ODE formulation:

- **Forward process**: `x_t = (1 - t) * x_0 + t * noise`, where `t in [0, 1]`
- **Training objective**: Predict velocity `v = noise - x_0`
- **Inference**: Solve ODE from `t=1` (noise) to `t=0` (clean) using Euler steps
- **Scheduler**: `FlowUniPCMultistepScheduler` (2nd-order predictor-corrector)
- **Default steps**: ~35 Euler steps with `shift=5.0`
- **CFG**: Classifier-free guidance with `guidance_scale` parameter

### VAE Tokenizer

**File**: `cosmos_predict2/_src/predict2/tokenizers/cosmos.py`

The Cosmos VAE compresses video to latent space:

| Property | Value |
|----------|-------|
| Spatial compression | 8x (via patch_spatial=2 in model) |
| Temporal compression | 4x |
| Latent channels | 16 |
| Input format | `(B, 3, T, H, W)` uint8 [0, 255] |
| Latent format | `(B, 16, T/4, H/8, W/8)` bf16 |

For 480x640 input with 13 frames: latent shape = `(1, 16, 4, 60, 80)`

### Text Encoder

Uses T5-based text encoder for computing text embeddings from prompts. The embeddings condition the model through cross-attention in each transformer block.

For distilled models, pre-computed CR1 (empty-string) embeddings can be used for efficiency.

## Config System

DreamDojo uses a layered config system:

1. **Base configs** (`cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py`): Define model, net, optimizer, scheduler defaults
2. **Network configs** (`net.py`): Register model architectures (2B, 7B, 14B)
3. **Experiment configs** (`cosmos_predict2/experiments/base/action.py`): Auto-register from YAML files
4. **YAML overrides** (`configs/*.yaml`): Per-experiment overrides (e.g., `14b_480_640_gr1.yaml`)

Config loading: `load_model_from_checkpoint()` in `cosmos_predict2/_src/predict2/utils/model_loader.py`

## Distilled Model (Student)

**File**: `cosmos_predict2/_src/predict2/interactive/networks/dit_action_causal.py`

The distilled model differs from the teacher:

| Aspect | Teacher | Student (Distilled) |
|--------|---------|---------------------|
| Denoising steps | ~35 | 4 (DMD2) |
| Attention | Bidirectional | Temporal causal |
| KV cache | No | Yes (frame-indexed) |
| torch.compile | Optional | Recommended |
| Inference mode | Batch | Streaming (per-frame) |

**Distillation pipeline** (3 stages):
1. **Teacher generation** (`launch_teacher_gen.sh`): Generate training data with teacher
2. **Warmup** (`launch_warmup.sh`): Train student to match teacher outputs
3. **Self-forcing** (`launch_self_forcing.sh`): Finetune with autoregressive self-predictions

## Key File Map

```
DreamDojo/
  cosmos_predict2/_src/predict2/
    networks/
      minimal_v4_dit.py          # Core DiT backbone + Attention class
      a2a_cp.py                  # A2A and NATTEN attention ops
    action/
      networks/                  # Action-conditioned model wrappers
      inference/
        inference.py             # Standard inference script
        inference_batch.py       # Batch inference (custom benchmark)
        inference_pipeline.py    # ActionVideo2WorldInference class
      configs/                   # Action-conditioned configs
    interactive/
      networks/dit_action_causal.py  # Distilled causal DiT
      inference/action_video2world.py # Distilled streaming inference
    models/
      text2world_model_rectified_flow.py  # Rectified flow model
    tokenizers/                  # VAE tokenizer
    schedulers/                  # ODE solvers
    utils/
      model_loader.py            # Checkpoint loading
      kv_cache.py                # KV cache utilities
  configs/                       # YAML experiment configs
  docs/                          # Documentation
  scripts/                       # Utilities (checkpoint conversion, etc.)
```
