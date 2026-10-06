# Sparse attention: PISA, Radial, SVG1, SVG2

Sparse-attention methods replace the dense **self-attention** kernel inside each denoiser
forward. They never touch cross-attention, MLPs, the VAE or the step count.

## Amdahl ceiling

The attainable speedup is bounded by the self-attention share of generation time, which
differs a lot between the three backbones (profiled with `DD_ATTN_PROFILE=1` /
`CW_ATTN_PROFILE=1` / DreamGen per-step logs):

| backbone | self-attention share | sequence length | ceiling |
|---|---|---|---|
| DreamGen 2B DiT | ≈ 73 % of DiT block time | multiview 2×2 grid / singleview 480p, long | ≈ 1.4–1.8× in practice |
| DreamDojo 2B DiT | ≈ 15–18 % of generation | S = 4800 (T = 4 latent frames), 16 heads, D = 128 | ≈ 1.2× |
| Ctrl-World SVD U-Net | ≈ 25–35 % of the U-Net | spatial T ≤ 2880 (3-view) / 3840 (4-view) | below 1× measured |

On DreamDojo and Ctrl-World, every sparse method is ≈ 1.0× or slower than dense flash
attention, because the sequences are too short for the sparse kernels and their per-call
overhead. These results are kept as benchmark data points, not removed.

## Integration per family

| family | how it is enabled | where the kernel runs | layer index meaning |
|---|---|---|---|
| DreamGen | `--use_pisa` / `--use_radial` / `--use_svg --svg_variant {svg1,svg2}` → `get_cosmos_predict2_video2world_pipeline()` builds per-block `config.net.{pisa,radial,svg}_parameters` (`models/dreamgen/cosmos_predict2/configs/base/config_{pisa,radial,svg}.py`) | `models/dreamgen/cosmos_predict2/module/a2a_cp.py` (`PISALocalAttnOp`, `RadialLocalAttnOp`, `SVGLocalAttnOp`), loading `methods/sparse_attention/pisa/piecewise_sparse_attn_tma.py`, `radial/dreamgen_runtime.py`, `svg/dreamgen_runtime.py` | `*_start_layer_idx` = first DiT block (of 28 / 36) that is sparsified |
| DreamDojo | `--use-pisa` / `--use-radial` / `--use-svg --svg-variant …` in the infer scripts → Hydra overrides `model.config.net.self_attention_backend={pisa,radial,svg}` + `model.config.net.<knob>=…` | `networks/minimal_v4_dit.py` (`build_dreamdojo_{pisa,radial,svg}_parameters`) and `networks/a2a_cp.py`, same `methods/` kernels/runtimes as DreamGen | same as DreamGen |
| Ctrl-World | `--use_pisa` / `--use_radial` / `--use_svg --svg_pattern {SVG1,SVG2}` → `enable_pisa` / `enable_radial_attention` / `enable_svg` call `unet.set_attn_processor` (`scripts/infer/infer_*_ctrlworld.py::_enable_backend`) | `methods/sparse_attention/{pisa/processor.py, radial/processor.py, svg/processor_svg1.py, svg/processor_svg2.py}` | `pisa_start_layer_idx` / `*_first_layers_fp` = index into the U-Net's attention processors; cross-attention, temporal (T = 11) and short (T < 128) layers always fall back to SDPA |

Ctrl-World geometry: the U-Net sees one stacked latent canvas. The 3-view strip is
72 × 40 (T = 2880 / 720 / 180 per stage), the 4-view grid 48 × 80 (T = 3840 / 960 / 240); the
infer scripts pass the canvas as a single view (`num_views = 1`) to the SVG/Radial masks.

## PISA

Piecewise sparse attention ("exact-or-approximate" instead of "keep-or-drop"): for each query
block, the `density` fraction of critical key blocks is computed exactly and the remaining
blocks are approximated with a block-wise Taylor expansion, so the full attention span is
still covered. Triton kernel vendored in
`methods/sparse_attention/pisa/piecewise_sparse_attn_tma.py` (upstream:
`methods/sparse_attention/piecewise-sparse-attention/`).

| knob | DreamGen / Ctrl-World flag | DreamDojo flag | env | default (CLI) | effect |
|---|---|---|---|---|---|
| density | `--pisa_density` | `--pisa-density` | `PISA_DENSITY` | 0.5 (DD 0.2) | fraction of exact blocks; lower → faster, lossier |
| block size | `--pisa_block_size` | `--pisa-block-size` | `PISA_BLOCK_SIZE` | 16 (DD 64) | token block size; all launchers use 64 |
| start layer | `--pisa_start_layer_idx` | `--pisa-start-layer-idx` | `PISA_START_LAYER_IDX` (DiT) / `PISA_START_LAYER` (Ctrl-World) | 2 | first sparsified layer |
| bias | `--pisa_use_bias` | `--pisa-use-bias` | `PISA_USE_BIAS` | off | bias-enhanced kernel variant; unused |

PISA needs no special environment (it does not require the editable `piecewise_attn` package).

## Radial

Radial attention: a static block mask whose band width decays with temporal distance
(O(n log n) attention), executed with FlashInfer block-sparse kernels. Mask construction is
held across calls (`_RadialPlan` on DiTs; MaskMap + wrapper cached per padded length on Ctrl-World).

| knob | DreamGen flag | DreamDojo flag | Ctrl-World flag | env | effect |
|---|---|---|---|---|---|
| decay | `--radial_decay_factor` (1.0) | `--radial-decay-factor` | `--radial_decay_factor` | `RADIAL_DECAY_FACTOR` (Ctrl-World: `RADIAL_DECAY`) | lower → narrower band, sparser |
| block | `--radial_block_size` (64; 64/128 only) | `--radial-block-size` | `--radial_block_size` | `RADIAL_BLOCK_SIZE` | FlashInfer block size |
| first layer | `--radial_start_layer_idx` (4) | `--radial-start-layer-idx` | `--radial_first_layers_fp` (2) | `RADIAL_START_LAYER_IDX` / `RADIAL_FIRST_LAYERS_FP` | layers kept dense |
| mask profile | `--radial_model_type` (`hunyuan`) | `--radial-model-type` | `--radial_model_type` | `RADIAL_MODEL_TYPE` | `wan` uses a larger sink/window |

Keep the output cast in `radial/dreamgen_runtime.py` (`output.to(query.dtype)`): FlashInfer
can return fp16 for bf16 inputs, which otherwise crashes the following projection.

## SVG1 (Sparse VideoGen)

Each attention head is routed to a spatial or a temporal sparse mask, chosen by **online MSE
profiling** of both masks against dense attention on a few sampled query rows; the selected
mask runs as `flex_attention` on the DiTs. The Ctrl-World adapter uses a diagonal band and a
cross-view stripe mask instead and runs them as masked SDPA.

| knob | DreamGen flag | DreamDojo flag | Ctrl-World flag | env | effect |
|---|---|---|---|---|---|
| sparsity | `--svg_sparsity` (0.25) | `--svg-sparsity` | `--svg_sparsity` (0.25) | `SVG_SPARSITY` | attention window width as a fraction of the sequence; lower → sparser |
| profiled rows | `--svg_num_sampled_rows` (64) | `--svg-num-sampled-rows` | `--svg_num_sampled_rows` (64) | `SVG_NUM_SAMPLED_ROWS` | cost of the profiling step |
| profile range | `--svg_sample_mse_max_row` (3000) | `--svg-sample-mse-max-row` | — | `SVG_SAMPLE_MSE_MAX_ROW` | max row index sampled |
| warm-up | `--svg_dense_step_frac` (0.1) | `--svg-dense-step-frac` | — | `SVG_DENSE_STEP_FRAC` | leading fraction of steps kept dense |
| first layer | `--svg_start_layer_idx` (4) | `--svg-start-layer-idx` | `--svg_first_layers_fp` (2) | `SVG_START_LAYER_IDX` / `SVG_FIRST_LAYERS_FP` | |
| profile cache | — | `--svg-profile-recompute-every` | held per (layer, T) in the processor | `SVG_PROFILE_RECOMPUTE_EVERY` | DreamDojo launchers use 999999 (profile once) |

DreamDojo resolves unset SVG knobs from `methods/sparse_attention/svg/config.py::_dreamdojo_svg_defaults`;
the launchers set them all explicitly.

## SVG2 (Sparse VideoGen2)

Semantic-aware permutation: queries and keys are clustered with k-means, tokens are permuted
so each cluster is contiguous, and only the key-centroid blocks covering the `top_p` mass of
each query centroid's attention are computed (FlashInfer variable block-sparse). Upstream code:
`methods/sparse_attention/Sparse-VideoGen/` (`svg.kmeans_utils`).

| knob | DreamGen flag | DreamDojo flag | Ctrl-World flag | env | effect |
|---|---|---|---|---|---|
| centroids | `--svg_num_q_centroids` / `--svg_num_k_centroids` (50 / 200) | `--svg-num-{q,k}-centroids` | `--svg_num_{q,k}_centroids` | `SVG_NUM_Q_CENTROIDS`, `SVG_NUM_K_CENTROIDS` | all launchers use 32 / 64; fewer (e.g. 8 / 16) corrupts |
| top-p | `--svg_top_p_kmeans` (0.9) | `--svg-top-p-kmeans` | `--svg_top_p_kmeans` | `SVG_TOP_P_KMEANS` | **the speed lever**: fraction of key-centroid blocks kept |
| min keys | `--svg_min_kc_ratio` (0.0) | `--svg-min-kc-ratio` | — | `SVG_MIN_KC_RATIO` | floor on key centroids kept per query centroid |
| k-means iters | `--svg_kmeans_iter_init` / `--svg_kmeans_iter_step` (5 / 2) | `--svg-kmeans-iter-{init,step}` | `--svg_kmeans_iter_{init,step}` | `SVG_KMEANS_ITER_INIT`, `SVG_KMEANS_ITER_STEP` | first-step / later-step iterations |
| warm-up, first layer | as SVG1 | as SVG1 | `--svg_first_layers_fp` | as SVG1 | |

## Pitfalls

- **SVG2 on DreamGen needs `models/dreamgen/.venv_svg`.** The compiled SVG kernel targets
  Python 3.11 and cannot load; with FlashInfer 0.2.5 (`.venv`) SVG2 falls back to a dense-mask
  SDPA path that builds a `(B·H, S, S)` mask and OOMs (≈ 57 GiB) on multiview.
  FlashInfer 0.3.1 provides the variable block-sparse wrapper. The DreamGen `*_svg2.sh`
  launchers default `VENV` to `.venv_svg`; DreamDojo and Ctrl-World already ship 0.3.1.
- **SVG1 profiling is per call on DreamGen.** `DreamGenSVG1Runtime(profile_recompute_every=1)`
  is not exposed on the DreamGen CLI, so every sparse call re-profiles. DreamDojo exposes
  `--svg-profile-recompute-every` (launchers: 999999), which removed ≈ 47 % of a dense call of
  overhead there. Reduce `SVG_NUM_SAMPLED_ROWS` / `SVG_DENSE_STEP_FRAC` to cut DreamGen profiling.
- **SVG sparsity is a window width, not a keep ratio.** On short sequences (DreamDojo T = 4)
  lowering it barely changes block density; on DreamGen single_arm SV the 0.25 window was
  nearly as wide as the sequence (no speedup) and 0.15 is the shipped value.
- **Ctrl-World SVG1 runs masked SDPA**, which drops off the flash kernel: ≈ 4× slower than
  dense at T = 3840. Masks/plans are cached per layer; do not re-enable per-call profiling.
- **Radial and SVG2 kernels lose to dense flash at S = 4800** (DreamDojo); diagnose with
  `scripts/tools/diag/_diag_dreamdojo_svg.py` before re-tuning.
- No context parallelism: DreamDojo raises if CP is combined with pisa/radial/svg; DreamGen
  and Ctrl-World launchers force one GPU.
- Judge quality with SSIM + LPIPS + a last-frame check, not PSNR alone: several configurations
  keep PSNR at dense level while SSIM/LPIPS and the frame show smearing.
