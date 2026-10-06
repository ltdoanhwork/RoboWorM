# Token reduction: SiTo, ITM

Token-reduction methods shrink the number of tokens that pass through (part of) the
denoiser and restore the full token grid afterwards. What they actually do differs per
backbone, so read the Ctrl-World section before comparing numbers across families.

## DreamGen (Cosmos-Predict2 DiT)

Both run per DiT block, around self-attention, and are built as per-block parameter lists by
`get_cosmos_predict2_video2world_pipeline()` (`models/dreamgen/cosmos_predict2/configs/base/config_video2world.py`
→ `config_{sito,itm}.py` → `methods/prunning/{SiTo,importance_token_merge}/runtime.py::build_*_parameters`).
The DiT block (`cosmos_predict2/models/text2image_dit.py`) holds a `SiToTokenPruner` /
`ITMTokenMerger` per eligible block. Blocks `[start_layer_idx, num_blocks − keep_last_n_dense)` are eligible.

**SiTo** (`SiToTokenPruner`): in `per_frame` / `full_2d` mode, tokens in each
`patch_h × patch_w` group are scored by similarity to the group mean (`sim_beta`, plus
optional Gaussian noise `noise_alpha`) and the redundant ones are pruned before self-attention;
in `temporal` mode the tokens most similar to the previous frame at the same position are
pruned. Pruned tokens are recovered with a single `index_select`. `prune_ratio = None`
resolves an automatic ratio from the token count.

**ITM** (`ITMTokenMerger`): an importance heat map `|cond − uncond|` (computed in
`pipelines/video2world.py` with `compute_itm_heat_map`) drives a ToMe-style merge of
unimportant tokens. Phase schedule: dense before `prune_from_step`, prune-only until
`merge_from_step`, then merge. Needs CFG (DreamGen runs guidance 7). `group_mode` is fixed to
`per_frame`.

| knob | flag | env | default (CLI) | effect |
|---|---|---|---|---|
| SiTo enable | `--use_sito` | — | | |
| prune ratio | `--sito_prune_ratio` | `SITO_PRUNE_RATIO` | auto | fraction of tokens pruned |
| first / last dense | `--sito_start_layer_idx`, `--sito_keep_last_n_dense` | `SITO_START_LAYER_IDX`, `SITO_KEEP_LAST_N_DENSE` | 4 (6 for ≥ 36 blocks) / 2 | |
| grouping | `--sito_group_mode {per_frame,temporal,full_2d}` | `SITO_GROUP_MODE` | `per_frame` | see pitfall below |
| anchor noise / sim | `--sito_noise_alpha`, `--sito_sim_beta`, `--sito_patch_{h,w}` | `SITO_NOISE_ALPHA` | 0.1, 1.0, 2×2 | |
| ITM enable | `--use_itm` | — | | |
| compress ratio | `--itm_compress_ratio` | `ITM_COMPRESS_RATIO` | auto (must be < 0.75) | fraction of tokens merged |
| first / last dense | `--itm_start_layer_idx`, `--itm_keep_last_n_dense` | `ITM_START_LAYER_IDX`, `ITM_KEEP_LAST_N_DENSE` | 4 (6) / 2 | |
| schedule | `--itm_prune_from_step`, `--itm_merge_from_step` | `ITM_PRUNE_FROM_STEP`, `ITM_MERGE_FROM_STEP` | 1 / 2 | |
| where | `--itm_merge_attn` / `--no_itm_merge_attn`, `--itm_merge_crossattn`, `--itm_merge_mlp` | — | attn only | |

## DreamDojo

SiTo/ITM are wired only into `models/DreamDojo/examples/action_conditioned.py`
(`--use-sito` / `--use-itm` → `model.config.net.use_{sito,itm}=true` overrides in
`cosmos_predict2/action_conditioned.py`; runtime in `networks/minimal_v4_dit.py`). The
canonical DreamDojo infer scripts (`scripts/infer/infer_*_singleview_*.py`) have no SiTo/ITM
flags, so these launchers use a **different entrypoint from dense**. ITM requires
`guidance > 0`; `action_conditioned.py` forces `guidance = 1.5` when ITM is on (override with
`FORCE_GUIDANCE`). SiTo/ITM on DreamDojo are not in the released coverage (see [configs.md](configs.md)).

## Ctrl-World (SVD U-Net)

Profiling showed per-attention pruning cannot pay off on this U-Net: spatial self-attention is
≈ 7 % of wall time and `prepare()` costs more than it saves. The shipped launchers therefore
use **spatio-temporal prune-once-hold** (`methods/prunning/SiTo/spatiotemporal_hold.py::enable_spatiotemporal_hold`):
every `TransformerSpatioTemporalModel.forward` is patched so spatial tokens are pruned once
after `proj_in`, held compressed through both the spatial and the temporal block (whose batch
is `H × W`), and restored before `proj_out`.

- Kept tokens: top-`keep_ratio` by batch-averaged feature L2 norm.
- Restored tokens copy their most cosine-similar kept token (`similarity_recover=True`).
- Eligible stages: `max_downsample_ratio = 1` → only the largest spatial stage; `4` → all three.
- The keep plan is computed on the first call and then held (`recompute_every = 999999`).

**Both `--sito_spatiotemporal_hold` and `--itm_spatiotemporal_hold` call this same function
with `similarity_recover=True`.** On Ctrl-World, "SiTo" and "ITM" are therefore the same
algorithm and differ only in their shipped `keep_ratio` / `max_downsample_ratio`. Neither uses
SiTo's anchor similarity or ITM's CFG heat map.

| knob | SiTo flag / env | ITM flag / env |
|---|---|---|
| mode | `--use_sito --sito_spatiotemporal_hold` | `--use_itm --itm_spatiotemporal_hold` |
| keep ratio | `--sito_st_keep_ratio` / `SITO_ST_KEEP_RATIO` (CLI 0.5) | `--itm_st_keep_ratio` / `ITM_ST_KEEP_RATIO` (CLI 0.8) |
| stages | `--sito_max_downsample_ratio` / `SITO_MAX_DOWNSAMPLE_RATIO` (4) | `--itm_max_downsample_ratio` / `ITM_MAX_DOWNSAMPLE_RATIO` (4) |
| plan cache | `--sito_plan_recompute_every` (0 → 999999) | `--itm_plan_recompute_every` (0 → 999999) |

Other Ctrl-World modes (implemented, not shipped): per-attention SiTo (`SiTo/adapter.py::enable_sito`),
per-attention ITM (`importance_token_merge/adapter.py::enable_itm`, which **forces
`guidance_scale` to 2.0** because the heat map needs CFG, doubling the U-Net batch), and ITM
block-hold (`--itm_block_hold`, `importance_token_merge/block_hold.py`, feature-L2 importance so it
runs at CFG = 1).

## Pitfalls

- **`--sito_group_mode` is dropped on 5 of 6 DreamGen cells.** Only
  `examples/video2world_gr00t.py` (single_arm MV 14B) forwards `sito_group_mode` to
  `get_cosmos_predict2_video2world_pipeline()`. `examples/video2world.py` (used by the
  multiview rollout `infer_bimanual_multiview_rollout.py`), `video2world_lvg.py` and
  `video2world_gr00t_lvg.py` do not, so `SITO_GROUP_MODE=temporal` in those launchers runs as
  the default `per_frame`.
- **Ctrl-World single_arm SiTo/ITM launchers default to 25 steps** (`NUM_INFERENCE_STEPS=25`)
  while dense runs the model default of 50. Time them with `NUM_INFERENCE_STEPS=50`
  (the timing harness pins this).
- **Pruning before the temporal block hurts temporal dynamics.** Ctrl-World ST-hold below
  `keep_ratio ≈ 0.7` corrupts (PSNR −3 to −4); on the bimanual 4-view grid even 0.8 costs ≈ 2.4 dB.
- **DreamGen ITM is net-negative** on the 2B DiTs (its per-block prepare/merge cost exceeds the
  attention it saves); SiTo gives a real 1.2–1.3× there.
- Judge quality on the full multi-view prediction video, not PSNR alone: a static background
  hides per-view smearing.
