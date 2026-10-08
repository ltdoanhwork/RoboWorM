# Checkpoint index

Every checkpoint used by the RoboWorM benchmark lives under `checkpoints/`.
This file is the only part of `checkpoints/` that git tracks.

```
checkpoints/
  base/        upstream / released weights (Cosmos, SVD, CLIP, T5, DreamDojo pretrain, Ctrl-World release, ...)
  finetuned/   our post-trained runs, same inner layout as the old exp_logs/*_logs/
    ctrlworld/   (was exp_logs/ctrlworld_logs/)
    dreamdojo/   (was exp_logs/dreamdojo_logs/)
    dreamgen/    (was exp_logs/dreamgen_logs/)
  metrics/     weights used only by the metrics (FVD I3D, WorldScore probes, RIGIS VGGT-Omega, KIVA readers)
```

The layout was introduced on 2026-10-05. The old paths are kept as compat symlinks
(`checkpoints/nvidia -> base/nvidia`, `checkpoints/fvd -> metrics/fvd`,
`exp_logs/ctrlworld_logs -> ../checkpoints/finetuned/ctrlworld`, ...), so old
commands keep working. New code should use the new paths.

## Paper cells (Table C1)

One row per generation cell. "Launcher" is the dense script that loads the
checkpoint. Every method launcher of that cell loads the same checkpoint.
Sizes are on-disk sizes of the weights that are actually loaded.

| # | model | embodiment / view | size | checkpoint path | iter | launcher (dense) |
|---|---|---|---|---|---|---|
| 1 | Ctrl-World (SVD U-Net) | single_arm MV (DROID), **released** | 8.7G | `checkpoints/base/ctrl_world/Ctrl-World/checkpoint-10000.pt` | 10000 | `scripts/run/ctrlworld/run_single_arm_multiview_ctrlworld.sh` |
| 2 | Ctrl-World (4-view 2x2 grid) | bimanual MV, 4 views | 8.7G | `checkpoints/finetuned/ctrlworld/bimanual_multiview_4view_grid/checkpoint-110000.pt` | 110000 | `scripts/run/ctrlworld/run_bimanual_multiview_ctrlworld_4view_grid.sh` |
| 3 | Ctrl-World (4-view 2x2 grid) | humanoid MV, 4 views | 8.7G | `checkpoints/finetuned/ctrlworld/humanoid_multiview_4view_grid/checkpoint-140000.pt` | 140000 | `scripts/run/ctrlworld/run_humanoid_multiview_ctrlworld_4view_grid.sh` |
| 4 | DreamGen = Cosmos-Predict2 **14B** GR00T-Dreams-DROID | single_arm MV, **released** | 55G (dir) | `checkpoints/base/nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID/model-480p-16fps.pt` | released | `scripts/run/dreamgen/run_single_arm_multiview_dreamgen_all.sh` (per split: `..._batch.sh`, `video2world_gr00t --gr00t_variant droid`) |
| 5 | DreamGen = Cosmos-Predict2 2B (post-trained) | bimanual MV (2x2 grid) | 3.7G | `checkpoints/finetuned/dreamgen/posttraining/multiview/bimanual_multiview_seg16/checkpoints_40_per/model/iter_000120000.pt` | 120000 | `scripts/run/dreamgen/run_bimanual_multiview_dreamgen_full.sh` |
| 6 | DreamGen 2B (post-trained) | humanoid MV (2x2 grid) | 3.7G | `checkpoints/finetuned/dreamgen/posttraining/multiview/humanoid_multiview_seg16/checkpoints_40_per/model/iter_000100000.pt` | 100000 | `scripts/run/dreamgen/run_humanoid_multiview_dreamgen_full.sh` |
| 7 | DreamGen 2B (post-trained, static 16fps) | single_arm SV | 3.7G | `checkpoints/finetuned/dreamgen/posttraining/singleview/single_arm_static_singleview_16fps_v2/checkpoints_40_per/model/iter_000090000.pt` | 90000 | `scripts/run/dreamgen/run_single_arm_singleview_dreamgen_static16fps.sh` |
| 8 | DreamGen 2B (post-trained, 16fps) | bimanual SV | 3.7G | `checkpoints/finetuned/dreamgen/posttraining/singleview/bimanual_singleview_16fps/checkpoints_40_per/model/iter_000110000.pt` | 110000 | `scripts/run/dreamgen/run_bimanual_singleview_dreamgen_16fps.sh` |
| 9 | DreamGen 2B (post-trained, static 16fps) | humanoid SV | 3.7G | `checkpoints/finetuned/dreamgen/posttraining/singleview/humanoid_singleview_16fps/checkpoints_40_per/model/iter_000090000.pt` | 90000 | `scripts/run/dreamgen/run_humanoid_singleview_dreamgen_static16fps.sh` |
| 10 | DreamDojo 2B action-cond. DiT (`dreamdojo_2b_480_640_single_arm_sv`) | single_arm SV | 21G (iter dir) | `checkpoints/finetuned/dreamdojo/dreamdojo/exp_single_arm_sv/single_arm_singleview/checkpoints/iter_000030000` | 30000 | `scripts/run/dreamdojo/run_single_arm_singleview_dreamdojo_dense.sh` |
| 11 | DreamDojo 2B (`dreamdojo_2b_480_640_aloha_resume70k_bimanual_sv`) | bimanual SV | 21G (iter dir) | `checkpoints/finetuned/dreamdojo/dreamdojo/exp_aloha_resume70k/aloha_bimanual_singleview_full/checkpoints/iter_000100000` | 100000 | `scripts/run/dreamdojo/run_bimanual_singleview_dreamdojo_full_episode.sh` |
| 12 | DreamDojo 2B (`dreamdojo_2b_480_640_gr1`) | humanoid SV, **released** | 21G (iter dir) | `checkpoints/base/dreamdojo/2B_GR1_post-train/iter_000050000` | 50000 | `scripts/run/dreamdojo/run_humanoid_singleview_dreamdojo_full_episode.sh` |

The table covers the 12 cells of Table C1. Ctrl-World uses 3 views on single_arm (row 1)
and a 4-view 2x2 grid on bimanual and humanoid (rows 2 and 3). DreamDojo launchers take
the parent `checkpoints/` dir and read `latest_checkpoint.txt`. DreamGen launchers take
`checkpoints_40_per/` and an `ITER_NAME`.

## Base models (`checkpoints/base/`)

| path | size | used by |
|---|---|---|
| `nvidia/Cosmos-Predict2-2B-Video2World` | 72G | DreamGen 2B init + tokenizer (all DreamGen 2B cells) |
| `nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID` | 55G | DreamGen single_arm MV (cell 4) |
| `nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1` | 28G | DreamGen `--gr00t_variant gr1` (older humanoid entrypoint) |
| `nvidia/Cosmos-Predict2-14B-Video2World` | 28G | DreamGen 14B video2world (non-paper runs) |
| `nvidia/Cosmos-Predict2.5-2B` | 4.0G | DreamDojo: Wan2.1 VAE `tokenizer.pth` + `robot/action-cond` default, resolved through the local-mirror lookup in `models/DreamDojo/.../checkpoint_db.py` |
| `nvidia/Cosmos-Reason1-7B` | 31G | DreamGen prompt refiner (`prompt_refiner_config.enabled=True` by default) and DreamDojo checkpoint registry |
| `nvidia/Cosmos-Guardrail1` | 11G | DreamGen guardrail (on by default; most launchers pass `--disable_guardrail`) |
| `meta-llama/Llama-Guard-3-8B` | 52K (stub) | DreamGen guardrail |
| `google-t5/t5-11b` | 85G | DreamGen / DreamDojo text embeddings |
| `dreamdojo/2B_pretrain/iter_000140000` | 21G | DreamDojo init for post-training (`experiments/base/action.py`) |
| `dreamdojo/2B_GR1_post-train/iter_000050000` | 21G | DreamDojo humanoid SV (cell 12) |
| `dreamdojo/LAM_400k.ckpt` | 8.0G | DreamDojo latent action model |
| `ctrl_world/Ctrl-World/checkpoint-10000.pt` | 8.7G | Ctrl-World release (cell 1; root of the warm-start chain of cells 2-3) |
| `stabilityai/stable-video-diffusion-img2vid` | 61G | Ctrl-World SVD backbone |
| `openai/clip-vit-base-patch32` | 3.4G | Ctrl-World text encoder |
| `pi05_droid` | 12G | Ctrl-World policy-in-the-loop rollouts (not in paper tables) |
| `droid` | 934M | Ctrl-World DROID example data |
| `mjcf`, `mjcf_robosuite` | 72M / 115M | robot MJCF assets for `metrics/embodied_exec` |

DreamGen resolves base weights through `--checkpoints` (default now `checkpoints/base`,
see `models/dreamgen/imaginaire/constants.py`). DreamDojo looks for HF mirrors in
`checkpoints/base/nvidia` and then `checkpoints/nvidia`.

## Metric checkpoints (`checkpoints/metrics/`)

| path | size | metric |
|---|---|---|
| `fvd/i3d_torchscript.pt` | 49M | FVD (I3D, `metrics/fvd/`, setup in `metrics/fvd/README.md`) |
| `worldscore/{groundingdino_swint_ogc.pth, sam2.1_hiera_base_plus.pt, VFIMamba.pkl, Tartan-C-T-TSKH-spring540x960-M.pth}` | 1.3G | RQ3 baselines: MAS (GroundingDINO + SAM2 + SEA-RAFT, `metrics/motion_amplitude.py`), MS (VFIMamba, `metrics/motion_smoothness.py`); paths in `metrics/third_party.py` |
| `vggt_omega/vggt_omega_1b_512.pt` | 4.3G | RIGIS (symlink to `thanhnv/VideoWorldModelBenchmark/checkpoint/`) |
| `kiva/` | 1.1G | KIVA readers (symlink to `triquang/kinescore-store/ckpt`) |

## Not released (not used by any paper cell)

Intermediate or superseded weights that no paper cell loads:
DreamDojo `exp_aloha_resume40k`
(training-only init of cell 11), DreamDojo `exp_single_arm_sv` iters 10000/20000,
DreamGen `single_arm_static_singleview_16fps_v1_90k`, `nvidia/PhysicalAI-Robotics-GR00T-Eval`.

### Removed: 3-view Ctrl-World bimanual / humanoid (2026-10-08)

Ctrl-World bimanual and humanoid are benchmarked only on the 4-view grid. The 3-view
fine-tunes were deleted from the server. They live on in the HF repo
`doanh25032004/vgp_exp_logs` (sha256 checked against the local copy before deletion):

| removed path (`checkpoints/finetuned/ctrlworld/`) | size | HF file (`vgp_exp_logs`) | note |
|---|---|---|---|
| `bimanual_multiview_5fps/` (`checkpoint-280000.pt` + `samples/`) | 8.7G | `ctrlworld_logs/bimanual_multiview_5fps/` | warm start of row 2 |
| `humanoid_multiview/` (`checkpoint-200000.pt` + `samples/`) | 8.7G | `ctrlworld_logs/humanoid_multiview/` | warm start of row 3 |
| `humanoid_multiview/checkpoint-150000.pt` (was archived under `tmp/`) | 8.7G | `ctrlworld_logs/humanoid_multiview/checkpoint-150000.pt` | intermediate |

You only need them to re-train the 4-view grids from their warm start
(`scripts/train/run_*_multiview_ctrlworld_train_4view_grid.sh`, `WARM_CKPT`). To restore one:

```bash
export HF_HUB_DISABLE_XET=1
hf download doanh25032004/vgp_exp_logs --repo-type model \
  --include 'ctrlworld_logs/humanoid_multiview/checkpoint-200000.pt' --local-dir /tmp/vgp_exp_logs
mkdir -p checkpoints/finetuned/ctrlworld/humanoid_multiview
mv /tmp/vgp_exp_logs/ctrlworld_logs/humanoid_multiview/checkpoint-200000.pt checkpoints/finetuned/ctrlworld/humanoid_multiview/
```

Their rollouts under `sampling_dataset/*/{bimanual,humanoid}/output/multiview/ctrlworld/`
are kept; the checkpoint ablation (`assets/scripts/make_checkpoint_finetune_comparison.py`)
reads them.

## Restore from HuggingFace

See [docs/setup/hf_backup_restore.md](../docs/setup/hf_backup_restore.md) for prerequisites
(`HF_HUB_DISABLE_XET=1`, token). The repos are `doanh25032004/<name>`:

| HF repo | restores |
|---|---|
| `vgp_checkpoints_nvidia` | `checkpoints/base/nvidia` |
| `vgp_checkpoints_dreamdojo` | `checkpoints/base/dreamdojo` |
| `vgp_checkpoints_google_t5` | `checkpoints/base/google-t5` |
| `vgp_checkpoints_own` | `checkpoints/{ctrl_world,pi05_droid,stabilityai,openai,worldscore,droid,mjcf,meta-llama,...}` (old layout) |
| `vgp_exp_logs` | `exp_logs/{ctrlworld_logs,dreamdojo_logs,dreamgen_logs,...}` (old layout) |

```bash
export HF_HUB_DISABLE_XET=1
hf download doanh25032004/vgp_checkpoints_nvidia    --repo-type model --local-dir checkpoints/base/nvidia
hf download doanh25032004/vgp_checkpoints_dreamdojo --repo-type model --local-dir checkpoints/base/dreamdojo
hf download doanh25032004/vgp_checkpoints_google_t5 --repo-type model --local-dir checkpoints/base/google-t5
hf download doanh25032004/vgp_checkpoints_own       --repo-type model --local-dir checkpoints
hf download doanh25032004/vgp_exp_logs              --repo-type model --local-dir exp_logs
```

`vgp_checkpoints_own` and `vgp_exp_logs` were uploaded with the old layout. Re-layout
them after download, from the repo root (idempotent):

```bash
cd checkpoints
mkdir -p base finetuned metrics
for n in nvidia stabilityai google-t5 openai dreamdojo ctrl_world pi05_droid meta-llama mjcf mjcf_robosuite droid; do
  [ -d "$n" ] && [ ! -L "$n" ] && [ ! -e "base/$n" ] && mv "$n" "base/$n"
  [ -e "base/$n" ] && [ ! -e "$n" ] && ln -s "base/$n" "$n"
done
for n in fvd worldscore vggt_omega; do
  [ -d "$n" ] && [ ! -L "$n" ] && [ ! -e "metrics/$n" ] && mv "$n" "metrics/$n"
  [ -e "metrics/$n" ] && [ ! -e "$n" ] && ln -s "metrics/$n" "$n"
done
cd ../exp_logs
for k in ctrlworld dreamdojo dreamgen; do
  [ -d "${k}_logs" ] && [ ! -L "${k}_logs" ] && mv "${k}_logs" "../checkpoints/finetuned/$k"
  [ -e "../checkpoints/finetuned/$k" ] && [ ! -e "${k}_logs" ] && ln -s "../checkpoints/finetuned/$k" "${k}_logs"
done
mkdir -p ../logs/exp_logs
for n in *; do [ -L "$n" ] || mv "$n" ../logs/exp_logs/; done
cd ..
```

Non-checkpoint content of the old `exp_logs/` (timing passes, tmux logs, sparse-attention
queue logs, augment runs, HF-upload state) now lives in `logs/exp_logs/`.
