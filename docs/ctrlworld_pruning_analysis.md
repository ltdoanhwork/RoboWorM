# Ctrl-World Token-Pruning Analysis (SiTo / ITM)

Empirical study of inference-time token pruning (SiTo, Importance Token Merge)
applied to the Ctrl-World action-conditioned video world model. Records the
profiling methodology, measurements, and conclusions for later write-up.

All numbers below are on `single_arm` multiview, `makovian` subset,
`episode_000255`, `num_inference_steps=25`, bf16, one H100-class GPU. Absolute
wall times were collected while the GPU was shared with other jobs, so treat
them as *relative* comparisons (dense baseline re-measured in every batch).

---

## 1. Model / workload

- Ctrl-World = SVD-based (Stable Video Diffusion) UNet, action-conditioned,
  autoregressive replay. 3-view multiview, 320x192-per-view stacked (height×3).
- Per episode: `interact_num` autoregressive segments × `num_inference_steps`
  denoise steps × UNet forward. `guidance_scale=1` by default (NO CFG).
- Attention layers: 64 diffusers `AttnProcessor` slots. Spatial self-attention
  sequence lengths observed: **2880 (72×40), 720 (36×20), 180 (18×10)**; temporal
  attention sequence length **11** (num frames), with batch = H·W.

## 2. Profiling methodology

- Wrapped `unet.forward`, `vae.encode/decode` with CUDA-synced timers.
- Registered forward pre/post hooks on `BasicTransformerBlock`,
  `TemporalBasicTransformerBlock`, `ResnetBlock2D`/`TemporalResnetBlock` to get
  per-module-type time.
- Instrumented `SVDSiToAttnProcessor._should_use_sito` to count fired seq_lens.
- Standalone micro-benchmarks: isolated `to_q/k/v + SDPA + to_out`, full
  `BasicTransformerBlock`, and full `TransformerSpatioTemporalModel` forwards,
  dense vs pruned, with plan `prepare()` timed separately.

## 3. Where the time goes

Episode wall-time split:

| Component | Share |
|---|---|
| UNet forward | ~64% |
| VAE decode | ~8% |
| VAE encode | ~1% |
| Python/loop/mediapy | ~27% |

UNet-internal split (per-module-type hooks):

| UNet component | Share of UNet |
|---|---|
| Spatial `BasicTransformerBlock` | ~16% |
| `TemporalBasicTransformerBlock` | ~22% |
| ResNet (2D + temporal) | ~31% |
| proj_in/out, time embed, other | ~31% |

Attention-only split: temporal attn (seq_len=11) ≈57% of attention time; spatial
attn (2880/720/180) the remaining ≈43%. **SiTo/ITM only touch spatial
self-attention ≈ 7% of total wall time.**

## 4. Per-layer prune profitability (micro-bench, dense vs pruned)

Isolated self-attention (`to_q/k/v` + SDPA + `to_out`), plan precomputed:

| seq_len | C | prune | speedup (SDPA-only) |
|---|---|---|---|
| 2880 | 320 | 0.45 | **1.71x** |
| 720 | 640 | 0.35 | 0.72x (slower) |
| 180 | 1280 | 0.20 | 0.76x (slower) |
| 11 (temporal) | 320 | 0.30 | 0.75x (slower) |

Only the largest stage (2880) benefits. Smaller stages have few tokens / high
channels; the index_select + scatter bookkeeping outweighs the attention FLOPs.

## 5. The two overhead killers

1. **`prepare()` cost > attention it prunes.**
   - SiTo `prepare()` ≈ 0.67 ms/call (normalize, mean, score matmul, argmax,
     patch layout, similarity recover), vs a dense 2880 attention ≈ 0.42 ms.
   - ITM `prepare()` ≈ **11.7 ms/block** (randperm, full argsort, src×dst
     similarity matmul, scatter). Catastrophic.
   - Mitigation implemented: **plan caching** across denoise steps
     (`plan_recompute_every`; the plan is spatial-structure-driven and stable),
     plus a **single `index_select` recover** (precomputed `gather_from_kept`)
     replacing two scatters + a gather. Cache-once (per episode) lifts the 2880
     layer from 0.42x → **1.26x**; 720/180 stay <1x even cached.

2. **ITM forces `guidance_scale → 2`.** ITM's importance = |cond − uncond| heat
   map needs classifier-free guidance, which doubles the UNet batch. Dense runs
   at guidance_scale=1. This alone makes vanilla ITM ~1.5–2x *slower* than dense
   regardless of pruning. Mitigation: block-hold `self_importance` mode derives
   importance from hidden-state feature L2 magnitude → runs at CFG=1.

## 6. Approaches tried & results

Dense baseline (25 steps): ~156–267 s depending on GPU sharing; PSNR ≈ 21.3.

| Approach | Speedup | PSNR | Note |
|---|---|---|---|
| SiTo per-attn, cache=5 | 0.94x | 19.8 | prepare still too frequent |
| SiTo per-attn, cache-once, ratio1, patch4 | ~1.0x | **21.4** | quality kept, no speedup |
| ITM per-attn (CFG=2) | 0.44x | 20.5 | CFG doubling dominates |
| ITM block-hold, self-importance, CFG=1 | 0.9x | 16–20.7 | attn+FF only, temporal untouched |
| **ST-hold keep0.9, stage2880 only** | ~1.0x | **18.9** | quality-leaning default |
| ST-hold keep0.8, all stages | 1.04x | 16.4 | |
| ST-hold keep0.5, all stages | **1.16x** | 13.7 | fastest, lossy |

### Spatio-temporal prune-once-hold (the only >1.1x path)

Key structural fact: the temporal block reshapes to `(batch·H·W, T, C)`, i.e. its
effective batch = number of spatial tokens. Pruning spatial tokens ONCE at the
`TransformerSpatioTemporalModel` input and holding them compressed through BOTH
the spatial and temporal blocks (unmerge only before `proj_out`) shrinks ~38% of
UNet instead of ~16%. Implemented in
`methods/prunning/SiTo/spatiotemporal_hold.py` (`--sito_spatiotemporal_hold`).

- Real speedup achieved: up to **1.16x** at keep_ratio 0.5.
- **Quality collapses**: pruning before the temporal block corrupts temporal
  dynamics; even keep_ratio 0.9 (drop 10%) gives PSNR ~18.9 and no real speed.
- `similarity_recover=True` (recover pruned token from most-similar kept token,
  cosine) adds ≈ +3.7 PSNR at equal speed vs nearest-index recover — keep it on.
- No config is simultaneously >1.1x and PSNR>18. Clear speed↔quality Pareto,
  no free lunch.

## 7. Ceiling argument (why 16% caps it)

If spatial blocks are sped k× and are fraction f of UNet, UNet speedup =
1/((1−f) + f/k). With f≈0.16 and an optimistic per-block k≈1.2, UNet≈1.03x; with
k≈1.95 (aggressive prune), UNet≈1.08x → episode ≈1.05x (UNet is 64% of wall).
Extending to spatial+temporal (f≈0.38) is what unlocks >1.1x, but at the quality
cost documented above.

## 8. Recommendation

- Token pruning (SiTo/ITM) on Ctrl-World: **quality-preserving ⇒ ~break-even
  speed; real speedup ⇒ large quality loss.** Fundamental to the small SVD UNet.
- For speedup that preserves quality, use **whole-UNet step caching**
  (DiCache/WorldCache) which skips entire UNet forwards (attacks the 64%).
  DiCache @25 steps kept PSNR ≈ 21.2 (≈ dense) in preliminary tests.
- Keep SiTo/ITM as measured benchmark data points (per project policy: run every
  method even when its speedup is weak/negative).

## 9. Reproduce

```bash
# dense baseline (25 steps)
python scripts/infer_single_arm_multiview_ctrlworld.py ... --num_inference_steps 25

# SiTo per-attn, quality-preserving cache-once (break-even speed)
... --use_sito --sito_max_downsample_ratio 1 --sito_plan_recompute_every 999999 \
    --sito_patch_h 4 --sito_patch_w 4 --sito_prune_ratio 0.7

# Spatio-temporal hold (fast, lossy)
... --use_sito --sito_spatiotemporal_hold --sito_st_keep_ratio 0.5 --sito_max_downsample_ratio 4
```

Run scripts: `scripts/run_single_arm_multiview_ctrlworld_{sito,itm}.sh`.
Related rules: `.cursor/rules/ctrlworld-pruning-insights.mdc`,
`.cursor/rules/acceleration-benchmark.mdc`.
