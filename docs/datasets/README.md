# Datasets

RoboWorM evaluates on **1,605 real-world episodes** curated from **5 source corpora**,
covering **3 embodiments** (single-arm, bimanual, humanoid), **1–4 synchronized views**
and two task regimes (**Markovian** short-horizon, **non-Markovian** long-horizon,
directory name `makovian` / `non_makovian`). See App. A of the paper
(`docs/paper/RoboWM_ICLR2027_check.pdf`) and the measured counts in
[cell_matrix.md](cell_matrix.md) / [benchmark_stats.md](benchmark_stats.md).

| embodiment | view | corpus | robot | views/ep | fps | episodes (M + nM) |
|---|---|---|---|---|---|---|
| single_arm | SV | RoboChallenge | Galaxea A1X (global static view) | 1 | 15 | 120 + 120 |
| single_arm | MV | DROID | Franka Panda | 3 | 15 | 128 + 129 |
| bimanual | SV | AIST Bimanual | ALOHA (cam_high) | 1 | 30 | 150 + 150 |
| bimanual | MV | AIST Bimanual | ALOHA | 4 | 30 | 150 + 150 |
| humanoid | SV | GR00T Teleop GR1 | Fourier GR-1 (ego view) | 1 | 20 | 130 + 130 |
| humanoid | MV | RoboCOIN | AIRBOT MMK2 | 4 | 30 | 98 + 150 |

There are two data trees:

- `datasets/` — the **source / training** data (LeRobot format) plus the prepared
  DreamGen training corpora. This page.
- `sampling_dataset/` — the **benchmark rollouts** (GT inputs + every model × method
  prediction). See [sampling_dataset.md](sampling_dataset.md).

## Download `datasets/` (HF: `doanh25032004/vgp_datasets`, 36.4 GB)

<https://huggingface.co/datasets/doanh25032004/vgp_datasets> mirrors `datasets/` file
by file (LeRobot parquet + mp4, plus the prepared DreamGen corpora).

```bash
pip install -U "huggingface_hub[cli]"
export HF_HUB_DISABLE_XET=1          # required on this cluster (Xet backend fails on the parallel FS)

# everything the paper uses (~26 GB; skips the superseded prototype variants)
hf download doanh25032004/vgp_datasets --repo-type dataset --local-dir datasets \
    --include "single_arm/*" "bimanual/*" "humanoid/*" \
              "single_arm_static_dreamgen_16fps/*" "bimanual_dreamgen_16fps/*" \
              "humanoid_dreamgen_16fps/*" \
              "bimanual_multiview_grid_seg16_dreamgen/*" "humanoid_multiview_grid_seg16_dreamgen/*"

# only the LeRobot source data of one embodiment / view (enough for inference)
hf download doanh25032004/vgp_datasets --repo-type dataset --local-dir datasets \
    --include "bimanual/singleview/*"

# the whole repo (also contains archived prototype variants, see below)
hf download doanh25032004/vgp_datasets --repo-type dataset --local-dir datasets
rm -rf datasets/.cache               # regenerable HF metadata
```

## Layout of `datasets/`

```
datasets/
  {single_arm,bimanual,humanoid}/{singleview,multiview}/{makovian,non_makovian}/
      [<task>/]                       bimanual/* and humanoid/multiview are split per task
        meta/{info.json,episodes.jsonl,tasks.jsonl,...}
        data/chunk-XXX/episode_XXXXXX.parquet         states / actions
        videos/chunk-XXX/<camera>/episode_XXXXXX.mp4  one dir per camera
        captions/                                     humanoid/singleview only (LLM captions)

  # prepared DreamGen (Cosmos-Predict2) training corpora: videos/ metas/ t5_xxl/
  single_arm_static_dreamgen_16fps/            single_arm SV  (RoboChallenge, 16 fps)
  bimanual_dreamgen_16fps/                     bimanual SV    (ALOHA cam_high, 16 fps)
  humanoid_dreamgen_16fps/                     humanoid SV    (GR1, 20 -> 16 fps)
  bimanual_multiview_grid_seg16_dreamgen/      bimanual MV    (2x2 grid, 10 s segments, true 16 fps)
  humanoid_multiview_grid_seg16_dreamgen/      humanoid MV    (2x2 grid, 10 s segments, true 16 fps)
```

Ctrl-World and DreamDojo train directly from the LeRobot trees (Ctrl-World first
encodes SVD latents into `models/Ctrl-World/dataset_example/<name>/`, see
[../training/README.md](../training/README.md)).

### Which variants are kept (and why)

Only the corpora that the paper checkpoints were trained on are kept in `datasets/`.
Evidence = the `dataset_dir` / `dataset_path` recorded in each checkpoint's
`config.yaml` under `checkpoints/finetuned/`:

| kept dir | read by checkpoint config |
|---|---|
| `single_arm/singleview` | `dreamdojo/.../exp_single_arm_sv/single_arm_singleview/config.yaml` |
| `bimanual/singleview` | `dreamdojo/.../exp_aloha_resume70k/aloha_bimanual_singleview_full/config.yaml` |
| `bimanual/multiview`, `humanoid/multiview` | Ctrl-World train launchers (`scripts/train/*.sh` → `scripts/prepare/prepare_ctrlworld_*.py`) |
| `single_arm_static_dreamgen_16fps` | `dreamgen/posttraining/singleview/single_arm_static_singleview_16fps_v2/config.yaml` |
| `bimanual_dreamgen_16fps` | `dreamgen/posttraining/singleview/bimanual_singleview_16fps/config.yaml` |
| `humanoid_dreamgen_16fps` | `dreamgen/posttraining/singleview/humanoid_singleview_16fps/config.yaml` |
| `bimanual_multiview_grid_seg16_dreamgen` | `dreamgen/posttraining/multiview/bimanual_multiview_seg16/config.yaml` |
| `humanoid_multiview_grid_seg16_dreamgen` | `dreamgen/posttraining/multiview/humanoid_multiview_seg16/config.yaml` |

Archived to `tmp/20261005_refactor/misc/datasets/` (restore commands in
`tmp/20261005_refactor/MANIFEST.md`; still present on the HF repo):

- `bimanual_dreamgen`, `humanoid_dreamgen`, `bimanual_multiview_dreamgen` — early
  prototypes, superseded by the `*_16fps` / `*_seg16` corpora; no paper checkpoint reads them.
- `{bimanual,humanoid}_multiview_grid_dreamgen`, `{bimanual,humanoid}_multiview_grid_seg_dreamgen`
  — intermediate stages of the multiview pipeline (grid → 10 s segments → 16 fps re-encode).
  The `*_seg16_dreamgen` corpora already carry their own videos, metas and T5 embeddings,
  so training does not need them. They are rebuilt automatically if you re-run the
  pipeline from the LeRobot source (see [../training/README.md](../training/README.md)).

## Rebuilding the source data from the original corpora (optional)

The HF mirror is the supported path. To rebuild from upstream:

| corpus | how |
|---|---|
| DROID | `python scripts/tools/data/download_droid_dataset.py` (HF `cadene/droid_1.0.1`, 20 chunks → `datasets/droid_1.0.1_20chunks`), then Markovian labelling below and sampling into `datasets/single_arm/multiview` |
| RoboChallenge | `python scripts/tools/data/prepare_single_arm_robochallenge.py` (HF `Traly/RoboChallenge-lerobot` → `datasets/single_arm/singleview`) |
| AIST Bimanual (ALOHA) | <https://aistairc.github.io/aist_bimanip_site/>; `bash scripts/tools/data/download_bimanual_multiview_dropbox.sh`; singleview = `python scripts/tools/data/export_bimanual_singleview_cam_high.py` |
| RoboCOIN (AIRBOT MMK2) | `git clone https://huggingface.co/datasets/RoboCOIN/<task>` into `datasets/humanoid/multiview/{makovian,non_makovian}/<task>`; tasks used: makovian `AIRBOT_MMK2_mobile_car`, `Airbot_MMK2_move_pan`; non-Markovian `AIRBOT_MMK2_boxs_storage`, `Airbot_MMK2_Airbot_MMK2_storage_mango_pomegranate`, `Airbot_MMK2_doodled_line`, `Airbot_MMK2_storage_bottle_part` |
| GR00T Teleop GR1 | `python scripts/tools/data/split_humanoid_singleview.py`, captions with `bash scripts/tools/data/run_caption_humanoid_singleview.sh` |

### Markovian / non-Markovian labelling

Task strings are labelled by a local Qwen3-VL server, then joined back to episodes and
sampled per label:

```bash
bash scripts/tools/data/launch_qwen3vl_server.sh            # keep running (vLLM)
python scripts/tools/data/classify_markovian_qwen.py         # DROID tasks -> data/labels/task_labels.csv
python scripts/tools/data/join_labels_to_episodes.py         # -> data/labels/episode_markovian_split{,_full}.csv
python scripts/tools/data/classify_markovian_gr1.py          # GR1 -> data/labels/gr1_task_labels.csv, gr1_episode_sampled.csv
```

The label files used for the release are tracked in `data/labels/`.
