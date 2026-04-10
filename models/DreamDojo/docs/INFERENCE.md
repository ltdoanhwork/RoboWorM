# DreamDojo Inference Guide

## Checkpoint Formats

DreamDojo supports two checkpoint formats:

### DCP (Distributed CheckPointer)
- Native PyTorch distributed format (`.distcp` files)
- Produced by distributed training (FSDP)
- Structure: `iter_XXXXXXXXX/model/__0_0.distcp, __1_0.distcp, ...`

### PT (Single File)
- Standard PyTorch checkpoint (`.pt` file)
- Required for inference
- Convert from DCP:
  ```bash
  python scripts/convert_distcp_to_pt.py \
    /path/to/iter_000050000/model \
    /path/to/iter_000050000/
  ```
  Output: `model_ema_bf16.pt` (EMA weights in bf16)

The `load_model_from_checkpoint()` function auto-detects format. When using `load_ema_to_reg=True`, EMA weights are loaded as the primary model weights.

## Teacher Inference Pipeline

### Overview
```
Input image + text prompt + actions
  --> VAE encode (image -> latent)
  --> T5 text encoder (prompt -> embeddings)
  --> Rectified Flow sampling (~35 Euler steps):
      For each step t:
        noise_pred = DiT(x_t, t, text_emb, action)
        x_{t-1} = euler_step(x_t, noise_pred)
  --> VAE decode (latent -> video)
  --> Output: (T, H, W, 3) uint8 video
```

### Quick Start

**Single GPU:**
```bash
PYTHONPATH=. python -m cosmos_predict2._src.predict2.action.inference.inference_batch \
  --experiment=dreamdojo_14b_480_640_gr1 \
  --ckpt_path /path/to/checkpoint/ \
  --batch_input /path/to/batch_input.json \
  --output_dir /path/to/output/ \
  --resolution 480,640 \
  --guidance 0
```

**2-GPU context parallel:**
```bash
torchrun --nproc_per_node=2 --master_port=12341 \
  -m cosmos_predict2._src.predict2.action.inference.inference_batch \
  --experiment=dreamdojo_14b_480_640_gr1 \
  --ckpt_path /path/to/checkpoint/ \
  --batch_input /path/to/batch_input.json \
  --output_dir /path/to/output/ \
  --resolution 480,640 \
  --guidance 0 \
  --context_parallel_size 2
```

Or use the provided launcher:
```bash
bash launch_teacher_inference_14b_benchmark.sh
```

### Batch Input Format

```json
[
    {
        "input_video": "/path/to/image.png",
        "prompt": "Use the right hand to close lunch box",
        "output_video": "/path/to/output.mp4"
    }
]
```

Fields:
- `input_video`: Path to input image (PNG/JPG) or video (MP4)
- `prompt`: Text description for conditioning
- `output_video`: (optional) Overridden by `--output_dir`

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--experiment` | required | Config name (e.g., `dreamdojo_14b_480_640_gr1`) |
| `--ckpt_path` | required | Checkpoint directory path |
| `--resolution` | `480,640` | Output resolution `H,W` |
| `--guidance` | `0` | CFG scale (0=none, 3-7=moderate, 10+=strong) |
| `--chunk_size` | `12` | Actions per generation chunk |
| `--action_dim` | `384` | Action space dimension |
| `--context_parallel_size` | `1` | GPUs for context parallelism |
| `--use_neg_prompt` | false | Enable negative prompt for CFG |
| `--save_fps` | `5` | Output video FPS |

### Available Experiments

| Experiment Name | Model | Resolution | Robot |
|----------------|-------|------------|-------|
| `dreamdojo_2b_480_640_gr1` | 2B | 480x640 | GR-1 |
| `dreamdojo_14b_480_640_gr1` | 14B | 480x640 | GR-1 |
| `dreamdojo_2b_480_640_g1` | 2B | 480x640 | Unitree G1 |
| `dreamdojo_14b_480_640_g1` | 14B | 480x640 | Unitree G1 |

## Distilled Inference Pipeline

The distilled model uses DMD2 (Distribution Matching Distillation) for faster inference:

### Overview
```
Input image + actions
  --> VAE encode
  --> 4 DMD2 denoising steps (vs ~35 for teacher):
      For each frame (streaming with KV cache):
        cached_kv = read_cache(frame_idx)
        noise_pred = CausalDiT(x_t, t, cached_kv, action)
        update_cache(frame_idx, new_kv)
  --> VAE decode
  --> Output: per-frame video generation at ~10 FPS
```

### Key Differences from Teacher

| Aspect | Teacher | Distilled |
|--------|---------|-----------|
| Steps | ~35 | 4 |
| Latency | ~10-30s/sample | ~100ms/frame |
| Attention | Full bidirectional | Temporal causal |
| Generation | Full video at once | Frame-by-frame streaming |
| KV cache | No | Yes (rolling window) |
| torch.compile | Optional | Recommended (10-30% speedup) |

### Launch

```bash
bash launch_distilled_inference_benchmark.sh
```

Note: Only 2B distilled checkpoints are currently available. For 14B distillation, run the 3-stage pipeline described in `DISTILL.md`.

## Multi-GPU Context Parallelism

Context parallelism splits the **temporal dimension** of video tokens across GPUs. Each GPU holds the full model but processes a subset of temporal frames.

```bash
torchrun --nproc_per_node=N ...
  --context_parallel_size N
```

**How it works:**
1. `distributed.init()` initializes NCCL backend
2. `parallel_state.initialize_model_parallel(context_parallel_size=N)` creates process groups
3. `model.net.enable_context_parallel(process_group)` enables temporal splitting in attention
4. Each GPU processes `T/N` temporal frames, with all-to-all communication for attention

**Memory**: Each GPU still holds the full model weights (~28GB for 14B bf16). Activation memory is reduced by ~N times.

## torch.compile Optimization

The distilled inference path (`action_video2world.py`) implements a 4-step compilation pipeline:

1. **Strip activation checkpointing** (training artifact, not needed for inference)
2. **Replace TransformerEngine ops** with PyTorch equivalents (eliminates ~170 graph breaks)
3. **Pre-install tensor-based KV cache** (enables zero graph breaks per DiT block)
4. **Compile blocks + VAE**:
   - Each DiT block: `torch.compile(block)`
   - VAE decoder: `mode="reduce-overhead"` (CUDA graphs)

Result: 10-30% speedup with full fusion per block.

## Output

### Generated Videos
- Format: MP4 with specified FPS
- Shape: `(T, H, W, 3)` uint8
- Location: `--output_dir`

### Timing Report
- Format: JSON (`timing.json` in output directory)
- Contents: model load time, per-sample inference time, average FPS
- See `docs/TIMING.md` for detailed profiling

## Environment Variables

```bash
export PYTHONPATH=/path/to/DreamDojo:$PYTHONPATH
export OMP_NUM_THREADS=8
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
```

---

<= Previous: [[Post-Training](POSTTRAIN.md)]

=> Next: [[Sparse Attention](SPARSE_ATTENTION.md)]
