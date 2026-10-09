# Configuration matrix (cell × method)

Every value below is the **effective** command line of the launcher with its shipped defaults.
It was captured by running each launcher with `python`/`torchrun` replaced by a logging stub
(no GPU work, no files written), then hand-checked against the scripts. Reading the
`${VAR:-default}` lines alone is not enough: the Ctrl-World 4-view-grid wrappers set their own
defaults and pass them into the shared base script, overriding the base defaults.

Any knob can be overridden with the env var named in [caching.md](caching.md),
[sparse_attention.md](sparse_attention.md) or [pruning.md](pruning.md).

## Cells

Launcher of method `m` = `scripts/run/<family>/<prefix>_<m>.sh`. Output of method `m` =
the dense output path with the leading `sampling_dataset/dense/` replaced by
`sampling_dataset/<m>/`. All method launchers run on **one GPU**.

| cell | prefix (`scripts/run/…/`) | dense launcher | entrypoint (dense = methods unless noted) | checkpoint | output under `sampling_dataset/<method>/` |
|---|---|---|---|---|---|
| CW single_arm MV (3-view) | `ctrlworld/run_single_arm_multiview_ctrlworld` | same prefix `.sh` | `infer_single_arm_multiview_ctrlworld.py` (positional subset `makovian`/`non_makovian`/`both`) | `base/ctrl_world/Ctrl-World/checkpoint-10000.pt` | `single_arm/output/multiview/ctrlworld/{makovian,non_makovian}/episode_*/` |
| CW bimanual MV (4-view grid) | `ctrlworld/run_bimanual_multiview_ctrlworld_4view_grid` | same prefix `.sh` | `infer_bimanual_multiview_ctrlworld.py --config bimanual_grid` (wrapper → shared base script) | `finetuned/ctrlworld/bimanual_multiview_4view_grid/checkpoint-110000.pt` | `bimanual/output/multiview/ctrlworld_4view_grid/{makovian,non_makovian}/episode_*/` |
| CW humanoid MV (4-view grid) | `ctrlworld/run_humanoid_multiview_ctrlworld_4view_grid` | same prefix `.sh` | `infer_bimanual_multiview_ctrlworld.py --config humanoid_grid` | `finetuned/ctrlworld/humanoid_multiview_4view_grid/checkpoint-140000.pt` | `humanoid/output/multiview/ctrlworld_4view_grid/…` |
| DG single_arm MV (14B) | `dreamgen/run_single_arm_multiview_dreamgen` | `…_all.sh` → `…_batch.sh` (2 GPUs) | `-m examples.video2world_gr00t --gr00t_variant droid --fps 10` | `base/nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID` (resolved by the entrypoint) | `single_arm/output/multiview/dreamgen/<cat>/episode_*.mp4` |
| DG bimanual MV | `dreamgen/run_bimanual_multiview_dreamgen` | `…_full.sh` | `scripts/infer/infer_bimanual_multiview_rollout.py` (→ `examples.video2world.setup_pipeline`) | `finetuned/dreamgen/posttraining/multiview/bimanual_multiview_seg16/checkpoints_40_per` (latest = `iter_000120000`) | `bimanual/output/multiview/dreamgen/<cat>/iter_000120000/<task>/episode_*/` |
| DG humanoid MV | `dreamgen/run_humanoid_multiview_dreamgen` | `…_full.sh` | same rollout script | `…/humanoid_multiview_seg16/checkpoints_40_per` (latest = `iter_000100000`) | `humanoid/output/multiview/dreamgen/<cat>/iter_000100000/<task>/episode_*/` |
| DG single_arm SV | `dreamgen/run_single_arm_singleview_dreamgen` | `…_static16fps.sh` (2 GPUs) | `-m models.dreamgen.examples.video2world_gr00t_lvg --gr00t_variant droid` | `…/singleview/single_arm_static_singleview_16fps_v2/checkpoints_40_per` (latest = `iter_000090000`) | `single_arm/output/singleview/dreamgen/<split>/iter_000090000_static16fps/global/` |
| DG bimanual SV | `dreamgen/run_bimanual_singleview_dreamgen` | `…_16fps.sh` (2 GPUs) | `-m models.dreamgen.examples.video2world_gr00t_lvg --gr00t_variant droid` | `…/singleview/bimanual_singleview_16fps/…/iter_000110000.pt` | `bimanual/output/singleview/dreamgen/<split>/iter_000110000_bimanual16fps/` |
| DG humanoid SV | `dreamgen/run_humanoid_singleview_dreamgen` | `…_static16fps.sh` (4 GPUs) | `-m models.dreamgen.examples.video2world_lvg` (all methods since 2026-10-09; see "Fixed launcher mismatches") | `…/singleview/humanoid_singleview_16fps/…/iter_000090000.pt` | `humanoid/output/singleview/dreamgen/<split>/iter_000090000_static16fps/` |
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

### Ctrl-World (SVD U-Net, 50 steps, CFG off)

| method | single_arm MV 3-view | bimanual MV 4-view | humanoid MV 4-view |
|---|---|---|---|
| WorldCache | thresh 0.08, ret 0.30, probe 2, motion 2.5 | thresh 0.04, ret 0.0, probe 1, motion 2.5 | thresh 0.03, ret 0.0, probe 1, motion 2.5 |
| DiCache | thresh 0.06, ret 0.3, probe 3 | thresh 0.03, ret 0.0, probe 1 | thresh 0.03, ret 0.0, probe 1 |
| FasterCache | start 0, model_int 5, block_int 3 | start 0, model_int 5, block_int 3 | — |
| PISA | density 0.5, block 64, start 2 | density 0.2, block 64, **start 0** | density 0.2, block 64, **start 0** |
| Radial | decay 1.0, block 64, fp 4, hunyuan | decay 1.0, block 64, fp 2, hunyuan | decay 1.0, block 64, fp 2, hunyuan |
| SVG1 | sparsity 0.25, rows 32, fp 2 | sparsity 0.25, rows 32, fp 2 | sparsity 0.25, rows 32, fp 2 |
| SVG2 | q 32, k 64, top_p 0.7, fp 2 | q 32, k 64, top_p 0.7, fp 2 | q 32, k 64, top_p 0.7, fp 2 |
| SiTo | ST-hold, keep 0.9, down 1 | ST-hold, keep 0.8, down 4 | ST-hold, keep 0.8, down 4 |
| ITM | ST-hold, keep 0.8, down 4 | ST-hold, keep 0.85, down 4 | ST-hold, keep 0.85, down 4 |

The 4-view wrappers delegate to the shared base scripts
`run_{bimanual,humanoid}_multiview_ctrlworld_<method>.sh` with `CONFIG=*_grid` (the 3-view
bimanual / humanoid checkpoints these bases default to were removed on 2026-10-08). The bimanual
4-view PISA/Radial/SVG1/SVG2/SiTo/ITM wrappers use the *humanoid* base scripts with
`CONFIG=bimanual_grid` (no bimanual base exists for these methods).

### DreamGen (Cosmos-Predict2 DiT, 35 steps, guidance 7)

| method | single_arm MV (14B) | bimanual MV | humanoid MV | single_arm SV | bimanual SV | humanoid SV |
|---|---|---|---|---|---|---|
| WorldCache | thresh 0.08, ret 0.5, probe 4, motion 4.0, decay | thresh 0.20, ret 0.2, probe 2, motion 5.0 | thresh 0.20, ret 0.2, probe 2, motion 5.0 | thresh 0.08, ret 0.5, probe 4, motion 4.0, decay | thresh 0.20, ret 0.2, probe 2, motion 4.0 | thresh 0.12, ret 0.2, probe 4, motion 4.0, decay |
| DiCache | thresh 0.08, ret 0.2, probe 2 | thresh 0.20, ret 0.2, probe 2 | thresh 0.20, ret 0.2, probe 2 | thresh 0.08, ret 0.2, probe 2 | thresh 0.20, ret 0.2, probe 2 | thresh 0.15, ret 0.2, probe 2 |
| FasterCache | start 0, model_int 5, block_int 3 | start 0, model_int 2, block_int 2 | start 0, model_int 5, block_int 5 | start 0, model_int 5, block_int 3 | start 0, model_int 2, block_int 2 | start 0, model_int 5, block_int 3 |
| PISA | density 0.2, block 64, start 2 | density 0.2, block 64, start 2 | density 0.2, block 64, start 2 | density 0.2, block 64, start 2 | density 0.1, block 64, start 0 | density 0.2, block 64, start 2 |
| Radial | decay 1.0, start 4, hunyuan | decay 0.5, start 8, wan | decay 1.0, start 4, hunyuan | decay 1.0, start 4, hunyuan | decay 0.2, start 1, hunyuan | decay 1.0, start 4, hunyuan |
| SVG1 | sparsity 0.25, start 4, dense_frac 0.1, rows 64, mse_rows 3000 | sparsity 0.25, start 4, dense_frac 0.05, rows 16, mse_rows 3000 | sparsity 0.25, start 4, dense_frac 0.1, rows 64, mse_rows 3000 | sparsity 0.15, start 4, dense_frac 0.05, rows 16, mse_rows 1500 | sparsity 0.25, start 4, dense_frac 0.1, rows 64, mse_rows 3000 | sparsity 0.25, start 4, dense_frac 0.1, rows 64, mse_rows 3000 |
| SVG2 | top_p 0.7, min_kc 0.15 | top_p 0.6, min_kc 0.15 | top_p 0.7, min_kc 0.15 | top_p 0.7, min_kc 0.15 | top_p 0.9, min_kc 0.10 | top_p 0.7, min_kc 0.15 |
| SiTo | prune 0.20, start 6, last_dense 2, group temporal, noise 0 | prune 0.20, start 6, last_dense 2, group temporal§, noise 0 | prune 0.20, … group temporal§ | prune 0.30, … group temporal§ | prune 0.30, … group temporal§ | prune 0.30, … group temporal§ |
| ITM | compress 0.2, start 14, last_dense 4, prune_from 0, merge_from 0 | compress 0.2, start 11, last_dense 4, prune_from 1, merge_from 2 | compress 0.2, start 11, last_dense 4, prune_from 1, merge_from 2 | compress 0.2, start 11, last_dense 4, prune_from 0, merge_from 0 | compress 0.2, start 11, last_dense 4, prune_from 1, merge_from 2 | compress 0.2, start 11, last_dense 4, prune_from 0, merge_from 0 |

All DreamGen Radial rows use block 64. All SVG2 rows use `start 1, dense_frac 0.3, q 32, k 64,
km_init 10, km_step 1` and `models/dreamgen/.venv_svg`.
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
| ITM | start 4, last_dense 2, prune_from 1, merge_from 2 ¶ | — | same ¶ |

Guidance: every dense launcher now passes `--guidance "${GUIDANCE}"` with `GUIDANCE` default **0**
(the infer-script default it always ran at), and every method launcher of the cell reads the
same variable with the same default (fixed 2026-10-09; before, most methods passed 1.5).
Exception: ITM via `action_conditioned` raises guidance to 1.5 internally (it needs CFG).
¶ entrypoint `-m models.DreamDojo.examples.action_conditioned`, not the dense infer script.

## Coverage gaps

Paper grid = 12 cells × 10 (dense + 9 methods) = 120 pairs; 117 have a launcher.

| status | (cell, method) |
|---|---|
| no launcher | CW humanoid MV 4-view × FasterCache; DD bimanual SV × SiTo, ITM |
| launcher exists, **no released rollouts** (`script_without_coverage` in `results/reports/impl_accel_configs.json`, written by `assets/scripts/make_impl_config_tables.py`) | CW bimanual MV 4-view × FasterCache; CW single_arm MV × FasterCache; DD single_arm SV × SiTo, ITM; DD humanoid SV × SiTo, ITM |

Released rollout counts per (cell, method): [../datasets/cell_matrix.md](../datasets/cell_matrix.md).
App. C hyperparameter tables (`assets/paper/appendix/tables/accel_hyperparams_{cache,sparse,token}.tex`)
are generated from the launchers by `assets/scripts/make_impl_config_tables.py`.

## Fixed launcher mismatches (2026-10-09)

Launcher-level only (no method hyperparameter changed, nothing re-run). Validated by running
every touched launcher with `python`/`torchrun` stubbed and diffing its argv against the
cell's dense launcher: after the fix only method flags, the method output path and the
Cosmos `--output-dir` name differ (the DreamDojo bimanual method launchers also pass
`--chunk-size 12`, which equals the infer-script default the dense launcher uses).

| # | cell × methods | before | after | files |
|---|---|---|---|---|
| F1 | DD single_arm SV, humanoid SV × WorldCache, DiCache, FasterCache, PISA, Radial, SVG1, SVG2; DD bimanual SV × PISA, Radial, SVG1, SVG2 (also `run_humanoid_singleview_dreamdojo_all_methods.sh`) | `--guidance 1.5`; dense passed none (= 0) | `GUIDANCE` default 0, passed explicitly by dense and methods | 3 dense + 19 method launchers |
| F2 | CW single_arm MV × SiTo, ITM | `--num_inference_steps 25` forced; SiTo ckpt not overridable | no step flag unless `NUM_INFERENCE_STEPS` is set (model default 50, as dense); SiTo honours `CKPT` like dense | `run_single_arm_multiview_ctrlworld_{sito,itm}.sh` |
| F3 | DG humanoid SV × DiCache, FasterCache, SiTo, ITM | `video2world_gr00t_lvg --gr00t_variant gr1 --prompt_prefix "" --no_skip_existing`, no `--resolution/--fps/--disable_prompt_refiner` | `video2world_lvg` with the dense single-GPU argument list + method flags (the entrypoint already parses all four methods' flags; `--help` and `build_cache_runtime_config` checked) | `run_humanoid_singleview_dreamgen_{dicache,fastercache,sito,itm}.sh` |
| F4 | DD humanoid SV × ITM | output to `<repo>/tmp_2/` | `sampling_dataset/itm/humanoid/output/singleview/dreamdojo/` | `run_humanoid_singleview_dreamdojo_itm.sh` |

Other Ctrl-World launchers were checked the same way: all 3-view single_arm and 4-view grid
method launchers already match their dense argv except method flags. The 3-view
bimanual / humanoid *base* scripts (`run_{bimanual,humanoid}_multiview_ctrlworld_<method>.sh`)
default to retired 3-view checkpoints/datasets that differ from the base dense script, but they
are only used through the 4-view wrappers (which pass `CKPT`/`DATASET_NAME`/`DOWN_SAMPLE`), so
they were left unchanged.

### Released rollouts produced with the old settings (not re-run)

| rollouts under `sampling_dataset/` | old setting | count |
|---|---|---|
| `{worldcache,dicache,fastercache,pisa,radial,svg1,svg2}/humanoid/output/singleview/dreamdojo/` | guidance 1.5 vs dense 0 (`run_config.json` confirms 1.5) | 216 each |
| `{worldcache,dicache,fastercache,pisa,radial,svg1,svg2}/single_arm/output/singleview/dreamdojo/` | guidance 1.5 vs dense 0 (launcher default since the 2026-10-06 history start; no sidecar records it) | 240 each |
| `{pisa,radial,svg1,svg2}/bimanual/output/singleview/dreamdojo/` | guidance 1.5 vs dense 0 (same caveat) | 300 each |
| `{sito,itm}/single_arm/output/multiview/ctrlworld/` | 25 denoising steps vs dense 50 | 257 each |
| `{dicache,fastercache,sito,itm}/humanoid/output/singleview/dreamgen/` | `video2world_gr00t_lvg --gr00t_variant gr1` entrypoint | 520 files each (260 episodes) |

DreamDojo bimanual WorldCache/DiCache/FasterCache already ran at 0 and are unaffected. The
fidelity-vs-dense and speedup numbers of the rows above compare different settings until
those rollouts are regenerated.

**Paper/docs vs dense launcher.** `generation_config.tex` (App. C, `tab:supp_generation_cfg`),
`docs/inference/dense_generation_config.md` and `scripts/tables/make_dense_generation_table.py`
list **g = 1.5** for all three DreamDojo dense cells, but the dense launchers ran at 0 (humanoid
`run_config.json`: `"guidance": 0`; the other two cells record no guidance, and their launchers
never passed one). The methods were aligned to the dense launcher, the fidelity reference;
the paper table still needs correcting (not done here).

## Known discrepancies (open; scripts unchanged)

1. **Ctrl-World 4-view PISA start layer.** The bimanual/humanoid 4-view wrappers set
   `PISA_START_LAYER=0` and pass it to the base script, so the effective flag is
   `--pisa_start_layer_idx 0`. `accel_hyperparams_sparse.tex` prints `start 2` for both cells:
   `make_impl_config_tables.py` (`DEFAULT_RE`) only reads the first `VAR=` on a line, and the
   wrapper declares density, block size and start layer on one line, so the start layer is
   taken from the base script's default (2). Method hyperparameter left at 0; App. C needs the fix.
2. **`SITO_GROUP_MODE=temporal` is ignored** on DreamGen bimanual MV, humanoid MV and all three
   singleview cells (see [pruning.md](pruning.md#pitfalls)); App. C lists "grouping temporal"
   for all six DreamGen cells. Only single_arm MV (14B) applies it.
3. **DreamDojo SiTo/ITM launchers** use `examples/action_conditioned.py` instead of the dense
   infer script, because the dense infer scripts have no SiTo/ITM flags (migration would need
   new wiring in `scripts/infer/infer_*_singleview_*.py`). Consequences: different episode
   sampling (`--num-samples`, `--single-base-index`), single_arm save-fps default, and ITM is
   forced to guidance 1.5. In the uniform timing pass both humanoid runs failed while
   downloading the gated `nvidia/Cosmos-Predict2.5-2B` checkpoint.
4. **FasterCache on Ctrl-World 4-view is a no-op** by construction (CFG off; self-attention
   lengths 3840/960/240 are outside the processor's `{180, 720, 2880}` whitelist).
5. **Stale tuning notes.** Some development notes list SVG2 `top_p 0.5` as the Ctrl-World
   4-view-grid script default; the scripts and App. C use 0.7. For DreamDojo humanoid SV
   the uniform timing pass ran DiCache at `0.08/0.2/2` and WorldCache at `0.03/0.4/4`
   (thresh/ret/probe), i.e. the shipped defaults. The tuning numbers in the
   `.cursor/rules/*dreamdojo*` notes were measured at guidance 1.5 (dense control included),
   and the DreamGen humanoid SV DiCache/FasterCache tuning on `video2world_gr00t_lvg`.
6. **Other docs.** The DreamGen bimanual SV dense entrypoint is
   `video2world_gr00t_lvg --gr00t_variant droid` (fixed in `scripts/README.md`, which listed
   `video2world_lvg`). The timing commands write to `results/reports/`
   (`speedup_perchunk.csv` from `merge_timing_shards.py`, `timing_pass.{json,csv}` from
   `make_timing_pass_table.py`). Method trees do contain `input/` directories (see "Inputs" above).
