# `sampling_dataset/` — benchmark rollouts

`sampling_dataset/` holds the GT inputs of every benchmark episode and the predictions of
every (world model × acceleration method) pair: **3,166 dense rollouts** (DreamGen 1,605,
DreamDojo 756, Ctrl-World 805) and **29,343 rollouts** including the nine accelerators
(counts audited by `assets/scripts/audit_sampling_distribution.py`, see
[cell_matrix.md](cell_matrix.md)). All metrics read from here.

## Download (HF: `doanh25032004/video_gen_physics`, 178 GB, ~136k files)

<https://huggingface.co/datasets/doanh25032004/video_gen_physics> mirrors the tree file
by file, so any sub-tree can be fetched with `--include`.

```bash
pip install -U "huggingface_hub[cli]"
export HF_HUB_DISABLE_XET=1     # required on this cluster's parallel filesystem

# everything (178 GB)
hf download doanh25032004/video_gen_physics --repo-type dataset --local-dir sampling_dataset

# one method, all embodiments (e.g. the dense reference + DiCache)
hf download doanh25032004/video_gen_physics --repo-type dataset --local-dir sampling_dataset \
    --include "dense/*" "dicache/*"

# one cell, all methods: DreamGen bimanual multiview
hf download doanh25032004/video_gen_physics --repo-type dataset --local-dir sampling_dataset \
    --include "*/bimanual/input/multiview/dreamgen/*" "*/bimanual/output/multiview/dreamgen/*"

# one embodiment of one method: Ctrl-World humanoid 4-view, WorldCache
hf download doanh25032004/video_gen_physics --repo-type dataset --local-dir sampling_dataset \
    --include "worldcache/humanoid/output/multiview/ctrlworld_4view_grid/*"

# human labels and web catalog only (RQ3 / RQ5 human-alignment inputs)
hf download doanh25032004/video_gen_physics --repo-type dataset --local-dir sampling_dataset \
    --include "data_for_web/*.json" "data_for_web/*.csv"

rm -rf sampling_dataset/.cache   # regenerable HF metadata
```

`hf download` is resumable: re-run the same command after an interruption. The older
tar-shard backup `doanh25032004/vgp_sampling_dataset_tar` (104 GB, all-or-nothing) is
described in [../setup/hf_backup_restore.md](../setup/hf_backup_restore.md#33-tar-shard-repos-datasets-sampling_dataset-models-misc).

## Layout

```
sampling_dataset/
  {method}/{embodiment}/{input,output}/{view}/{model}/{split}/...

  method     = dense | worldcache | dicache | fastercache | pisa | radial | svg1 | svg2 | sito | itm
  embodiment = single_arm | bimanual | humanoid
  view       = singleview | multiview
  model      = dreamgen | dreamdojo | ctrlworld | ctrlworld_4view_grid
  split      = makovian | non_makovian

  data_for_web/            human labels (geometry_final_labels.json, hallucinate_final_labels.json),
                           catalog, benchmark workbooks used by RQ3 / RQ5 and the speedup figure
  multiview_consistency/   extracted per-view frames for RIGIS
  augment/                 augmentation track (not in the paper tables)
```

`input/` exists only under `dense/`: GT clips, batch-input JSONs and Ctrl-World
annotations are shared by every method, and each method launcher reads
`sampling_dataset/dense/<emb>/input/...` and only rewrites the output root. The method
is always the top-level directory; there are no `{method}_*` siblings under `dense/`.

### Per-model output conventions

| model / view | output path below `{split}/` | files |
|---|---|---|
| DreamGen singleview | `iter_000090000_static16fps/[global/]episode_XXXXXX.mp4` (bimanual: `iter_000110000_bimanual16fps/<task>__episode_XXXXXX.mp4`) | `.mp4` + `.txt` prompt sidecar |
| DreamGen multiview | `iter_<N>/<task>/episode_XXXXXX/` (single_arm MV 14B: flat `episode_XXXXXX.mp4`) | `full_pred.mp4 full_gt.mp4 full_merged.mp4 metrics.json seg_*.mp4` |
| DreamDojo singleview | `iter_<N>/episode_XXXXXX/` + `all_summary.json` | `full_pred.mp4 full_gt.mp4 metrics.json` |
| Ctrl-World multiview | `episode_XXXXXX/` + `all_summary.json` | `full_pred.mp4 pred_all_views.mp4 gt_all_views.mp4 metrics.json` |

`ctrlworld` = 3-view strip model, `ctrlworld_4view_grid` = 4-view 2×2 grid model
(paper cells use the 4-view grid on bimanual / humanoid and the 3-view strip on single-arm).
The `{bimanual,humanoid}/.../multiview/ctrlworld/` trees are legacy 3-view rollouts; their
checkpoints and launchers were removed on 2026-10-08 (the checkpoint ablation still reads
the dense ones).

## How the inputs are built

Inputs under `sampling_dataset/dense/*/input/` are produced from `datasets/` by the
prepare step of each cell (most dense launchers call it themselves when the input is
missing):

| cell | input builder | writes |
|---|---|---|
| DreamGen single_arm SV | `scripts/run/dreamgen/run_single_arm_singleview_dreamgen_static16fps.sh` (`prepare_batch_inputs`, length-matched chunk grouping; prompt mirrors `prepare_single_arm_robochallenge_dreamgen_16fps.py::build_caption`) | `dense/single_arm/input/singleview/dreamgen/` |
| DreamGen bimanual SV | `scripts/prepare/prepare_singleview_dreamgen_batch.py` | `dense/bimanual/input/singleview/dreamgen/{split}/batch_input_<n>chunks.json` |
| DreamGen humanoid SV | `scripts/prepare/prepare_humanoid_singleview_dreamgen_short.py` (called by `run_humanoid_singleview_dreamgen_static16fps.sh`) | `dense/humanoid/input/singleview/dreamgen/{split}/batch_input_short_{2,3}chunks.json` |
| DreamGen single_arm MV (14B) | `scripts/prepare/prepare_single_arm_multiview_dreamgen_batch.py` (called by `run_single_arm_multiview_dreamgen_batch.sh`) | `dense/single_arm/input/multiview/dreamgen/` |
| DreamGen bimanual / humanoid MV | built on the fly by `scripts/infer/infer_bimanual_multiview_rollout.py` from `datasets/{bimanual,humanoid}/multiview` (GT 2×2 grid via ffmpeg); batch builders `scripts/prepare/prepare_bimanual_multiview_dreamgen_batch.py`, `scripts/prepare/prepare_multiview_humanoid_grid_batch.py` | `dense/{bimanual,humanoid}/input/multiview/dreamgen*/` |
| DreamDojo (all SV) | read straight from `datasets/<emb>/singleview` by `scripts/infer/infer_*_singleview_*.py` | GT saved next to the prediction |
| Ctrl-World single_arm MV | `scripts/prepare/prepare_ctrlworld_single_arm_multiview.py` (called by `run_single_arm_multiview_ctrlworld.sh`) | `dense/single_arm/input/multiview/ctrlworld/` |
| Ctrl-World bimanual / humanoid MV (legacy 3-view, removed) | `prepare_ctrlworld_{bimanual,humanoid}_multiview.py` (removed 2026-10-08, see git history) → `models/Ctrl-World/dataset_example/<name>/` | `dense/<emb>/input/multiview/ctrlworld/` |
| Ctrl-World bimanual / humanoid MV (4-view) | `scripts/prepare/prepare_ctrlworld_{bimanual,humanoid}_multiview_grid.py` | `dense/<emb>/input/multiview/ctrlworld_4view_grid/` |
