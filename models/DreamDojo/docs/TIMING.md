# Timing & Profiling DreamDojo Inference

This guide covers how to measure and profile DreamDojo inference performance at different granularities.

## Quick Start: Batch Timing

The `inference_batch.py` script automatically produces timing reports:

```bash
bash launch_teacher_inference_14b_benchmark.sh
# Output: output_dir/timing.json
```

Example `timing.json`:
```json
{
  "model": "dreamdojo_14b_480_640_gr1",
  "checkpoint": "/path/to/checkpoint",
  "context_parallel_size": 2,
  "resolution": "480,640",
  "model_load_time_s": 45.2,
  "num_samples": 10,
  "total_inference_time_s": 182.5,
  "avg_inference_time_s": 18.25,
  "total_frames_generated": 130,
  "avg_fps": 0.71,
  "per_sample": [...]
}
```

## Level 1: Per-Sample Wall-Clock Timing

The simplest approach using `torch.cuda.synchronize()`:

```python
import time
import torch

torch.cuda.synchronize()
t_start = time.perf_counter()

next_frame, video = video2world.step_inference(
    img_array=img, action=actions, guidance=0, seed=1, prompt=prompt
)

torch.cuda.synchronize()
t_elapsed = time.perf_counter() - t_start
print(f"Inference time: {t_elapsed:.3f}s")
```

**Important**: Always call `torch.cuda.synchronize()` before and after timing. CUDA operations are asynchronous -- without synchronization, you'll measure CPU launch time, not GPU execution time.

## Level 2: Component-Level Timing

Break down inference into its three main stages:

```python
import time
import torch
import numpy as np
import torchvision

# Setup
img_tensor = torchvision.transforms.functional.to_tensor(img_array).unsqueeze(0)
vid_input = torch.cat([img_tensor, torch.zeros_like(img_tensor).repeat(12, 1, 1, 1)], dim=0)
vid_input = (vid_input * 255.0).to(torch.uint8).unsqueeze(0).permute(0, 2, 1, 3, 4)

# --- VAE Encode ---
torch.cuda.synchronize()
t0 = time.perf_counter()
data_batch = video2world._get_data_batch_input(vid_input, prompt="", use_neg_prompt=False)
data_batch["action"] = torch.zeros(1, 12, 384, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
t_encode = time.perf_counter() - t0

# --- Diffusion Sampling ---
torch.cuda.synchronize()
t0 = time.perf_counter()
sample = video2world.model.generate_samples_from_batch(
    data_batch, n_sample=1, guidance=0, seed=1, is_negative_prompt=True
)
torch.cuda.synchronize()
t_diffusion = time.perf_counter() - t0

# --- VAE Decode ---
torch.cuda.synchronize()
t0 = time.perf_counter()
video = video2world.model.decode(sample)
torch.cuda.synchronize()
t_decode = time.perf_counter() - t0

print(f"VAE Encode + Text: {t_encode:.3f}s")
print(f"Diffusion ({N_STEPS} steps): {t_diffusion:.3f}s")
print(f"VAE Decode: {t_decode:.3f}s")
print(f"Total: {t_encode + t_diffusion + t_decode:.3f}s")
```

Typical breakdown (14B, 480x640):
- VAE Encode: ~0.5-1s
- Diffusion (35 steps): ~15-25s (dominates!)
- VAE Decode: ~0.5-1s

## Level 3: Per-Denoising-Step Timing

Instrument individual denoising steps using CUDA events:

```python
import torch

class StepTimer:
    """Wraps the DiT forward to time each denoising step."""
    
    def __init__(self, model):
        self.model = model
        self.original_net_forward = model.net.forward
        self.step_times = []
    
    def timed_forward(self, *args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = self.original_net_forward(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        self.step_times.append(start.elapsed_time(end))  # milliseconds
        return output
    
    def install(self):
        self.model.net.forward = self.timed_forward
    
    def uninstall(self):
        self.model.net.forward = self.original_net_forward
    
    def report(self):
        if not self.step_times:
            return
        print(f"Steps: {len(self.step_times)}")
        print(f"Total: {sum(self.step_times):.1f}ms")
        print(f"Avg: {sum(self.step_times)/len(self.step_times):.1f}ms/step")
        print(f"Min: {min(self.step_times):.1f}ms, Max: {max(self.step_times):.1f}ms")

# Usage
timer = StepTimer(video2world.model)
timer.install()

next_frame, video = video2world.step_inference(img, actions, guidance=0, seed=1)

timer.report()
timer.uninstall()
```

## Level 4: Per-Layer Attention Profiling

Monkey-patch `Attention.compute_attention` to measure per-layer time:

```python
import torch
from collections import defaultdict

class AttentionProfiler:
    """Profile attention time per layer."""
    
    def __init__(self, model):
        self.model = model
        self.times = defaultdict(list)  # layer_idx -> [time_ms, ...]
        self._originals = {}
    
    def install(self):
        for i, block in enumerate(self.model.net.blocks):
            attn = block.self_attn
            original = attn.compute_attention
            self._originals[i] = original
            
            def make_wrapper(layer_idx, orig_fn):
                def wrapper(*args, **kwargs):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    result = orig_fn(*args, **kwargs)
                    end.record()
                    torch.cuda.synchronize()
                    self.times[layer_idx].append(start.elapsed_time(end))
                    return result
                return wrapper
            
            attn.compute_attention = make_wrapper(i, original)
    
    def uninstall(self):
        for i, block in enumerate(self.model.net.blocks):
            if i in self._originals:
                block.self_attn.compute_attention = self._originals[i]
        self._originals.clear()
    
    def report(self):
        print(f"{'Layer':>6} {'Calls':>6} {'Total(ms)':>10} {'Avg(ms)':>10}")
        print("-" * 36)
        for layer_idx in sorted(self.times.keys()):
            t = self.times[layer_idx]
            print(f"{layer_idx:>6} {len(t):>6} {sum(t):>10.1f} {sum(t)/len(t):>10.2f}")

# Usage
profiler = AttentionProfiler(video2world.model)
profiler.install()
next_frame, video = video2world.step_inference(img, actions, guidance=0, seed=1)
profiler.report()
profiler.uninstall()
```

**Note**: Per-layer profiling adds synchronization overhead at each layer. Use for analysis only, not for production timing.

## Level 5: PyTorch Profiler

Full profiling with CUDA kernel-level detail:

```python
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler("./profiler_logs"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step in range(5):
        next_frame, video = video2world.step_inference(
            img, actions, guidance=0, seed=step
        )
        prof.step()

# Print summary
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

View in TensorBoard:
```bash
tensorboard --logdir=./profiler_logs
```

## Level 6: NVTX Ranges (for NSight Systems)

Add NVTX markers for profiling with NVIDIA NSight Systems:

```python
import torch

# Wrap key sections
with torch.cuda.nvtx.range("vae_encode"):
    data_batch = video2world._get_data_batch_input(vid_input, prompt="")

with torch.cuda.nvtx.range("diffusion_sampling"):
    sample = video2world.model.generate_samples_from_batch(data_batch, ...)

with torch.cuda.nvtx.range("vae_decode"):
    video = video2world.model.decode(sample)
```

Run with NSight:
```bash
nsys profile -t cuda,nvtx -o dreamdojo_profile \
  python -m cosmos_predict2._src.predict2.action.inference.inference_batch ...
```

## GPU Memory Monitoring

```python
# Current allocation
mem_alloc = torch.cuda.memory_allocated() / (1024**3)
mem_reserved = torch.cuda.memory_reserved() / (1024**3)
mem_max = torch.cuda.max_memory_allocated() / (1024**3)
print(f"Allocated: {mem_alloc:.2f}GB, Reserved: {mem_reserved:.2f}GB, Peak: {mem_max:.2f}GB")

# Reset peak stats
torch.cuda.reset_peak_memory_stats()
```

## Comparing Configurations

Template for A/B comparison:

```python
import json

configs = [
    {"name": "baseline", "guidance": 0, "sparse": False},
    {"name": "natten_k7", "guidance": 0, "sparse": True, "kernel": 7},
    {"name": "natten_k13", "guidance": 0, "sparse": True, "kernel": 13},
]

results = []
for cfg in configs:
    if cfg.get("sparse"):
        # Apply sparse attention
        model.net = replace_selfattn_op_with_sparse_attn_op(
            model.net, n_dense_blocks=4,
            natten_parameters={"kernel_size": cfg["kernel"]}
        )
    
    times = []
    for i in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = video2world.step_inference(img, actions, guidance=cfg["guidance"], seed=i)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    
    results.append({
        "config": cfg["name"],
        "mean_s": round(sum(times)/len(times), 3),
        "std_s": round((sum((t-sum(times)/len(times))**2 for t in times)/len(times))**0.5, 3),
    })
    print(f"{cfg['name']}: {results[-1]['mean_s']}s +/- {results[-1]['std_s']}s")

with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

<= Previous: [[Sparse Attention](SPARSE_ATTENTION.md)]
