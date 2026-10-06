# Configuration matrix (cell × method)

Every value below is the **effective** command line of the launcher with its shipped defaults.
It was captured by running each launcher with `python`/`torchrun` replaced by a logging stub
(no GPU work, no files written), then hand-checked against the scripts. Reading the
`${VAR:-default}` lines alone is not enough: the Ctrl-World 4-view-grid wrappers set their own
defaults and pass them into the 3-view base script, overriding the base defaults.

Any knob can be overridden with the env var named in [caching.md](caching.md),
[sparse_attention.md](sparse_attention.md) or [pruning.md](pruning.md).

## Cells

Launcher of method `m` = `scripts/run/<family>/<prefix>_<m>.sh`. Output of method `m` =
the dense output path with the leading `sampling_dataset/dense/` replaced by
`sampling_dataset/<m>/`. All method launchers run on **one GPU**.

| cell | prefix (`scripts/run/…/`) | dense launcher | entrypoint (dense = methods unless noted) | checkpoint | output under `sampling_dataset/<method>/` |
|---|---|---|---|---|---|
| CW single_arm MV (3-view) | `ctrlworld/run_single_arm_multiview_ctrlworld` | same prefix `.sh` | `infer_single_arm_multiview_ctrlworld.py` (positional subset `makovian`/`non_makovian`/`both`) | `base/ctrl_world/Ctrl-World/checkpoint-10000.pt` | `single_arm/output/multiview/ctrlworld/{makovian,non_makovian}/episode_*/` |
| CW bimanual MV (4-view grid) | `ctrlworld/run_bimanual_multiview_ctrlworld_4view_grid` | same prefix `.sh` | `infer_bimanual_multiview_ctrlworld.py --config bimanual_grid` (wrapper → 3-view base script) | `finetuned/ctrlworld/bimanual_multiview_4view_grid/checkpoint-110000.pt` | `bimanual/output/multiview/ctrlworld_4view_grid/{makovian,non_makovian}/episode_*/` |
| CW humanoid MV (4-view grid) | `ctrlworld/run_humanoid_multiview_ctrlworld_4view_grid` | same prefix `.sh` | `infer_bimanual_multiview_ctrlworld.py --config humanoid_grid` | `finetuned/ctrlworld/humanoid_multiview_4view_grid/checkpoint-140000.pt` | `humanoid/output/multiview/ctrlworld_4view_grid/…` |
| CW bimanual MV (3-view strip, not in paper) | `ctrlworld/run_bimanual_multiview_ctrlworld` | same prefix `.sh` | `infer_bimanual_multiview_ctrlworld.py --config bimanual` | methods: `finetuned/ctrlworld/bimanual_multiview_5fps/checkpoint-280000.pt` (dense defaults differ, see below) | `bimanual/output/multiview/ctrlworld/…` |
| CW humanoid MV (3-view strip, not in paper) | `ctrlworld/run_humanoid_multiview_ctrlworld` | same prefix `.sh` | `infer_bimanual_multiview_ctrlworld.py --config humanoid` | methods: `finetuned/ctrlworld/humanoid_multiview/checkpoint-200000.pt` | `humanoid/output/multiview/ctrlworld/…` |
| DG single_arm MV (14B) | `dreamgen/run_single_arm_multiview_dreamgen` | `…_all.sh` → `…_batch.sh` (2 GPUs) | `-m examples.video2world_gr00t --gr00t_variant droid --fps 10` | `base/nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID` (resolved by the entrypoint) | `single_arm/output/multiview/dreamgen/<cat>/episode_*.mp4` |
| DG bimanual MV | `dreamgen/run_bimanual_multiview_dreamgen` | `…_full.sh` | `scripts/infer/infer_bimanual_multiview_rollout.py` (→ `examples.video2world.setup_pipeline`) | `finetuned/dreamgen/posttraining/multiview/bimanual_multiview_seg16/checkpoints_40_per` (latest = `iter_000120000`) | `bimanual/output/multiview/dreamgen/<cat>/iter_000120000/<task>/episode_*/` |
| DG humanoid MV | `dreamgen/run_humanoid_multiview_dreamgen` | `…_full.sh` | same rollout script | `…/humanoid_multiview_seg16/checkpoints_40_per` (latest = `iter_000100000`) | `humanoid/output/multiview/dreamgen/<cat>/iter_000100000/<task>/episode_*/` |
| DG single_arm SV | `dreamgen/run_single_arm_singleview_dreamgen` | `…_static16fps.sh` (2 GPUs) | `-m models.dreamgen.examples.video2world_gr00t_lvg --gr00t_variant droid` | `…/singleview/single_arm_static_singleview_16fps_v2/checkpoints_40_per` (latest = `iter_000090000`) | `single_arm/output/singleview/dreamgen/<split>/iter_000090000_static16fps/global/` |
| DG bimanual SV | `dreamgen/run_bimanual_singleview_dreamgen` | `…_16fps.sh` (2 GPUs) | `-m models.dreamgen.examples.video2world_gr00t_lvg --gr00t_variant droid` | `…/singleview/bimanual_singleview_16fps/…/iter_000110000.pt` | `bimanual/output/singleview/dreamgen/<split>/iter_000110000_bimanual16fps/` |
| DG humanoid SV | `dreamgen/run_humanoid_singleview_dreamgen` | `…_static16fps.sh` (4 GPUs) | `-m models.dreamgen.examples.video2world_lvg`; **DiCache, FasterCache, SiTo, ITM use `video2world_gr00t_lvg --gr00t_variant gr1`** | `…/singleview/humanoid_singleview_16fps/…/iter_000090000.pt` | `humanoid/output/singleview/dreamgen/<split>/iter_000090000_static16fps/` |
| DD single_arm SV | `dreamdojo/run_single_arm_singleview_dreamdojo` | `…_dense.sh` | `scripts/infer/infer_single_arm_singleview_autoregressive.py`, experiment `dreamdojo_2b_480_640_single_arm_sv`; **SiTo/ITM use `-m models.DreamDojo.examples.action_conditioned`** | `finetuned/dreamdojo/dreamdojo/exp_single_arm_sv/single_arm_singleview/checkpoints` (`iter_000030000`) | `single_arm/output/singleview/dreamdojo/<split>/iter_000030000/` |
| DD bimanual SV | `dreamdojo/run_bimanual_singleview_dreamdojo` | `…_full_episode.sh` | `scripts/infer/infer_bimanual_singleview_single_segment.py`, experiment `dreamdojo_2b_480_640_aloha_resume70k_bimanual_sv` | `finetuned/dreamdojo/dreamdojo/exp_aloha_resume70k/aloha_bimanual_singleview_full/checkpoints` (`iter_000100000`) | `bimanual/output/singleview/dreamdojo/<split>/iter_000100000/` |
| DD humanoid SV | `dreamdojo/run_humanoid_singleview_dreamdojo` | `…_full_episode.sh` | `scripts/infer/infer_humanoid_singleview_full_episode.py`, experiment `dreamdojo_2b_480_640_gr1`; **SiTo/ITM use `action_conditioned`** | `base/dreamdojo/2B_GR1_post-train/iter_000050000` | `humanoid/output/singleview/dreamdojo/<split>/iter_000050000/` |

Checkpoint paths are relative to `checkpoints/`; see [../../checkpoints/CHECKPOINTS.md](../../checkpoints/CHECKPOINTS.md).
Ctrl-World entrypoints live in `scripts/infer/`.
DreamDojo launchers also pass `--output-dir output/dreamdojo_<method>_<emb>_sv_<split>`
(Cosmos log/config directory, outside `sampling_dataset/`).

**Environments.** Ctrl-World → `models/Ctrl-World/.venv`, DreamDojo → `models/DreamDojo/.venv`,
DreamGen → `models/dreamgen/.venv`, except every DreamGen `*_svg2.sh` → `models/dreamgen/.venv_svg`.

**Inputs.** Most DreamGen launchers read the shared batch JSONs under
`sampling_dataset/dense/<emb>/input/…`. Exceptions that prepare inputs under their own method
tree: every Ctrl-World launcher (`INPUT_ROOT=sampling_dataset/<method>/<emb>/input/…`),
DreamGen single_arm MV PISA/Radial/SVG1/SVG2, and DreamGen single_arm SV
WorldCache/DiCache/FasterCache/PISA/Radial/SVG1/SVG2.

## Effective method configurations

Abbreviations: `thresh` rel-L1 threshold, `ret` warm-up ratio, `probe` probe depth, `motion`
motion sensitivity, `decay` (WorldCache) dynamic threshold decay on; `model_int`/`block_int`
FasterCache intervals; `start` first sparsified/pruned DiT block (Ctrl-World PISA: attention
layer), `fp` Ctrl-World layers kept dense; `dense_frac` dense warm-up fraction, `rows`
profiled rows, `mse_rows` max profiled row, `profile_every` SVG1 profile cache, `q`/`k`
centroids, `min_kc` min key-centroid ratio, `km_init`/`km_step` k-means iterations;
`last_dense` trailing dense blocks, `group` SiTo grouping, `prune_from`/`merge_from` ITM
schedule, `keep`/`down` ST-hold keep ratio / max downsample ratio. `—` = no launcher.
† 3-view strips are not reported in the paper (superseded by the 4-view grid).

### Ctrl-World (SVD U-Net, 50 steps, CFG off)

| method | single_arm MV 3-view | bimanual MV 4-view | humanoid MV 4-view | bimanual MV 3-view† | humanoid MV 3-view† |
|---|---|---|---|---|---|
| WorldCache | thresh 0.08, ret 0.30, probe 2, motion 2.5 | thresh 0.04, ret 0.0, probe 1, motion 2.5 | thresh 0.03, ret 0.0, probe 1, motion 2.5 | thresh 0.04, ret 0.4, probe 2, motion 2.5 | thresh 0.04, ret 0.4, probe 2, motion 2.5 |
| DiCache | thresh 0.06, ret 0.3, probe 3 | thresh 0.03, ret 0.0, probe 1 | thresh 0.03, ret 0.0, probe 1 | thresh 0.04, ret 0.4, probe 3 | thresh 0.03, ret 0.4, probe 3 |
| FasterCache | start 0, model_int 5, block_int 3 | start 0, model_int 5, block_int 3 | — | start 0, model_int 5, block_int 3 | — |
| PISA | density 0.5, block 64, start 2 | density 0.2, block 64, **start 0** | density 0.2, block 64, **start 0** | — | density 0.2, block 64, start 2 |
| Radial | decay 1.0, block 64, fp 4, hunyuan | decay 1.0, block 64, fp 2, hunyuan | decay 1.0, block 64, fp 2, hunyuan | — | decay 1.0, block 64, fp 4, hunyuan |
| SVG1 | sparsity 0.25, rows 32, fp 2 | sparsity 0.25, rows 32, fp 2 | sparsity 0.25, rows 32, fp 2 | — | sparsity 0.15, rows 32, fp 2 |
| SVG2 | q 32, k 64, top_p 0.7, fp 2 | q 32, k 64, top_p 0.7, fp 2 | q 32, k 64, top_p 0.7, fp 2 | — | q 32, k 64, top_p 0.7, fp 2 |
| SiTo | ST-hold, keep 0.9, down 1, **25 steps** | ST-hold, keep 0.8, down 4 | ST-hold, keep 0.8, down 4 | — | ST-hold, keep 0.8, down 4 |
| ITM | ST-hold, keep 0.8, down 4, **25 steps** | ST-hold, keep 0.85, down 4 | ST-hold, keep 0.85, down 4 | — | ST-hold, keep 0.8, down 4 |

The bimanual 4-view PISA/Radial/SVG1/SVG2/SiTo/ITM wrappers delegate to the *humanoid* 3-view
base scripts with `CONFIG=bimanual_grid` (no bimanual 3-view base exists for these methods).

### DreamGen (Cosmos-Predict2 DiT, 35 steps, guidance 7)

| method | single_arm MV (14B) | bimanual MV | humanoid MV | single_arm SV | bimanual SV | humanoid SV |
|---|---|---|---|---|---|---|
| WorldCache | thresh 0.08, ret 0.5, probe 4, motion 4.0, decay | thresh 0.20, ret 0.2, probe 2, motion 5.0 | thresh 0.20, ret 0.2, probe 2, motion 5.0 | thresh 0.08, ret 0.5, probe 4, motion 4.0, decay | thresh 0.20, ret 0.2, probe 2, motion 4.0 | thresh 0.12, ret 0.2, probe 4, motion 4.0, decay |
| DiCache | thresh 0.08, ret 0.2, probe 2 | thresh 0.20, ret 0.2, probe 2 | thresh 0.20, ret 0.2, probe 2 | thresh 0.08, ret 0.2, probe 2 | thresh 0.20, ret 0.2, probe 2 | thresh 0.15, ret 0.2, probe 2 ‡ |
| FasterCache | start 0, model_int 5, block_int 3 | start 0, model_int 2, block_int 2 | start 0, model_int 5, block_int 5 | start 0, model_int 5, block_int 3 | start 0, model_int 2, block_int 2 | start 0, model_int 5, block_int 3 ‡ |
| PISA | density 0.2, block 64, start 2 | density 0.2, block 64, start 2 | density 0.2, block 64, start 2 | density 0.2, block 64, start 2 | density 0.1, block 64, start 0 | density 0.2, block 64, start 2 |
| Radial | decay 1.0, start 4, hunyuan | decay 0.5, start 8, wan | decay 1.0, start 4, hunyuan | decay 1.0, start 4, hunyuan | decay 0.2, start 1, hunyuan | decay 1.0, start 4, hunyuan |
| SVG1 | sparsity 0.25, start 4, dense_frac 0.1, rows 64, mse_rows 3000 | sparsity 0.25, start 4, dense_frac 0.05, rows 16, mse_rows 3000 | sparsity 0.25, start 4, dense_frac 0.1, rows 64, mse_rows 3000 | sparsity 0.15, start 4, dense_frac 0.05, rows 16, mse_rows 1500 | sparsity 0.25, start 4, dense_frac 0.1, rows 64, mse_rows 3000 | sparsity 0.25, start 4, dense_frac 0.1, rows 64, mse_rows 3000 |
| SVG2 | top_p 0.7, min_kc 0.15 | top_p 0.6, min_kc 0.15 | top_p 0.7, min_kc 0.15 | top_p 0.7, min_kc 0.15 | top_p 0.9, min_kc 0.10 | top_p 0.7, min_kc 0.15 |
| SiTo | prune 0.20, start 6, last_dense 2, group temporal, noise 0 | prune 0.20, start 6, last_dense 2, group temporal§, noise 0 | prune 0.20, … group temporal§ | prune 0.30, … group temporal§ | prune 0.30, … group temporal§ | prune 0.30, … group temporal§ ‡ |
| ITM | compress 0.2, start 14, last_dense 4, prune_from 0, merge_from 0 | compress 0.2, start 11, last_dense 4, prune_from 1, merge_from 2 | compress 0.2, start 11, last_dense 4, prune_from 1, merge_from 2 | compress 0.2, start 11, last_dense 4, prune_from 0, merge_from 0 | compress 0.2, start 11, last_dense 4, prune_from 1, merge_from 2 | compress 0.2, start 11, last_dense 4, prune_from 0, merge_from 0 ‡ |

All DreamGen Radial rows use block 64. All SVG2 rows use `start 1, dense_frac 0.3, q 32, k 64,
km_init 10, km_step 1` and `models/dreamgen/.venv_svg`.
‡ entrypoint `video2world_gr00t_lvg --gr00t_variant gr1`, not the dense `video2world_lvg`.
§ passed on the command line but **ignored** by the entrypoint (runs as `per_frame`), see
[pruning.md](pruning.md#pitfalls).

### DreamDojo (2B DiT, 35 steps, 12-frame chunks)

| method | single_arm SV | bimanual SV | humanoid SV |
|---|---|---|---|
| WorldCache | thresh 0.15, ret 0.2, probe 4, motion 2.0 | thresh 0.03, ret 0.4, probe 4, motion 5.0 | thresh 0.03, ret 0.4, probe 4, motion 5.0 |
| DiCache | thresh 0.08, ret 0.2, probe 2 | thresh 0.08, ret 0.2, probe 2 | thresh 0.08, ret 0.2, probe 2 |
| FasterCache | start 3, model_int 7, block_int 4 | start 0, model_int 5, block_int 3 | start 0, model_int 5, block_int 3 |
| PISA | density 0.2, block 64, start 2 | density 0.2, block 64, start 2 | density 0.2, block 64, start 2 |
| Radial | decay 0.5, block 64, start 2, hunyuan | decay 0.5, block 64, start 2, hunyuan | decay 1.0, block 64, start 2, hunyuan |
| SVG1 | sparsity 0.25, start 2, dense_frac 0.05, rows 16, mse_rows 1200, profile_every 999999 | same as single_arm | sparsity 0.25, **start 4**, dense_frac 0.05, rows 16, mse_rows 1200, profile_every 999999 |
| SVG2 | top_p 0.6, min_kc 0.15, q 32, k 64, start 4, dense_frac 0.05, km_init 8, km_step 2 | same | same |
| SiTo | start 4, last_dense 2 (prune ratio auto) ¶ | — | start 4, last_dense 2 ¶ |
| ITM | start 4, last_dense 2, prune_from 1, merge_from 2 ¶ | — | same, output to `tmp_2/` ¶ |

Guidance passed by the launchers: **1.5** for every single_arm and humanoid method and for
bimanual PISA/Radial/SVG1/SVG2; **0** for bimanual WorldCache/DiCache/FasterCache. The dense
launchers pass none, so dense runs at the infer-script default **0** (see discrepancy 2).
¶ entrypoint `-m models.DreamDojo.examples.action_conditioned`, not the dense infer script.

## Coverage gaps

Paper grid = 12 cells × 10 (dense + 9 methods) = 120 pairs; 117 have a launcher.

| status | (cell, method) |
|---|---|
| no launcher | CW humanoid MV 4-view × FasterCache; DD bimanual SV × SiTo, ITM |
| launcher exists, **no released rollouts** (`results/reports/impl_accel_configs.json::script_without_coverage`) | CW bimanual MV 4-view × FasterCache; CW single_arm MV × FasterCache; DD single_arm SV × SiTo, ITM; DD humanoid SV × SiTo, ITM |
| non-paper strips | CW bimanual 3-view has only dense/WorldCache/DiCache/FasterCache; CW humanoid 3-view has no FasterCache |

Released rollout counts per (cell, method): [../datasets/cell_matrix.md](../datasets/cell_matrix.md).
App. C hyperparameter tables (`assets/paper/appendix/tables/accel_hyperparams_{cache,sparse,token}.tex`)
are generated from the launchers by `assets/scripts/make_impl_config_tables.py`.

## Known discrepancies (found while building this matrix; scripts unchanged)

1. **Ctrl-World 4-view PISA start layer.** The bimanual/humanoid 4-view wrappers set
   `PISA_START_LAYER=0` and pass it to the base script, so the effective flag is
   `--pisa_start_layer_idx 0`. `accel_hyperparams_sparse.tex` prints `start 2` for both cells:
   `make_impl_config_tables.py` (`DEFAULT_RE`) only reads the first `VAR=` on a line, and the
   wrapper declares density, block size and start layer on one line, so the start layer is
   taken from the base script's default (2).
2. **DreamDojo guidance differs between dense and methods.** The three dense launchers
   (`…_dense.sh`, `…_full_episode.sh`) pass no `--guidance`, so dense runs at the infer-script
   default 0 (`run_config.json` of the released humanoid dense rollouts: `"guidance": 0`; every
   humanoid method: 1.5). Method launchers pass 1.5 except the bimanual caches (0). DreamDojo
   always evaluates both CFG branches, so timing is comparable, but fidelity-vs-dense compares
   different guidance scales on single_arm and humanoid (all methods) and bimanual (sparse).
3. **DreamGen humanoid SV entrypoint split.** DiCache, FasterCache, SiTo and ITM run
   `video2world_gr00t_lvg --gr00t_variant gr1`; dense and the other five methods run
   `video2world_lvg` (same checkpoint).
4. **`SITO_GROUP_MODE=temporal` is ignored** on DreamGen bimanual MV, humanoid MV and all three
   singleview cells (see [pruning.md](pruning.md#pitfalls)); App. C lists "grouping temporal"
   for all six DreamGen cells. Only single_arm MV (14B) applies it.
5. **DreamDojo SiTo/ITM launchers** use `examples/action_conditioned.py` instead of the dense
   infer script; the humanoid ITM launcher writes to `<repo>/tmp_2/` instead of
   `sampling_dataset/itm/…`; in the uniform timing pass both humanoid runs failed while
   downloading the gated `nvidia/Cosmos-Predict2.5-2B` checkpoint.
6. **Ctrl-World single_arm SiTo/ITM run 25 steps** (`NUM_INFERENCE_STEPS=25`) vs 50 for dense.
7. **Ctrl-World 3-view dense defaults ≠ method defaults.** `run_bimanual_multiview_ctrlworld.sh`
   defaults to the base checkpoint, `DATASET_NAME=bimanual_multiview`, `DOWN_SAMPLE=1`;
   `run_humanoid_multiview_ctrlworld.sh` defaults to the base checkpoint. Their method launchers
   use the fine-tuned checkpoints (`bimanual_multiview_5fps/checkpoint-280000.pt` with
   `bimanual_multiview_5fps` / `DOWN_SAMPLE=6`; `humanoid_multiview/checkpoint-200000.pt`). Pass
   the same `CKPT` / `DATASET_NAME` / `DOWN_SAMPLE` to dense (see [../inference/README.md](../inference/README.md)).
8. **FasterCache on Ctrl-World 4-view is a no-op** by construction (CFG off; self-attention
   lengths 3840/960/240 are outside the processor's `{180, 720, 2880}` whitelist).
9. **Stale insight rules.** `.cursor/rules/ctrlworld-{bimanual,humanoid}-multiview-4view-grid-sparse-attention-insights.mdc`
   list SVG2 `top_p 0.5` as the script default; the scripts and App. C use 0.7.
   `dreamdojo-humanoid-singleview-sparse-attention-insights.mdc` labels the uniform-pass DiCache
   run `t0.15/ret0.2/probe2` and WorldCache `t0.15/ret0.3/probe2`; the run logs
   (`logs/exp_logs/timing_pass/dist/dreamdojo_humanoid_sv/`) show `0.08/0.2/2` and `0.03/0.4/4`,
   i.e. the shipped defaults.
10. **Other docs.** `scripts/README.md` lists the DreamGen bimanual SV dense entrypoint as
    `video2world_lvg`; it is `video2world_gr00t_lvg --gr00t_variant droid`. The top-level
    README's timing commands write to `results/reports/` (`speedup_perchunk.csv` from
    `merge_timing_shards.py`, `timing_pass.{json,csv}` from `make_timing_pass_table.py`), not
    `metrics/reports/`. Method trees do contain `input/` directories (see "Inputs" above).
