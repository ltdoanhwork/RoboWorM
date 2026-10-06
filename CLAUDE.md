# RoboWorM: video world model physics benchmark

RoboWorM (ICLR 2027 submission) benchmarks robot video world models and training-free
acceleration methods. The paper is at `docs/paper/RoboWM_ICLR2027_check.pdf`.

- **Data**: 5 corpora (DROID, RoboChallenge, AIST Bimanual, RoboCOIN, GR00T Teleop GR1),
  3 embodiments, 1–4 views, splits `makovian` / `non_makovian`.
- **World models**: DreamGen (Cosmos-Predict2 2B/14B), DreamDojo (2B, action-conditioned),
  Ctrl-World (SVD 1.5B).
- **Acceleration methods**: 9, plus a `dense` baseline.
- **Metrics**: KIVA, RIGIS, Visual Integrity, Text Alignment, FVD, and PSNR/SSIM/LPIPS
  vs the dense rollout.

Start with `README.md`. The full docs index is `docs/README.md`.

## Layout

```
datasets/{single_arm,bimanual,humanoid}/{singleview,multiview}/{makovian,non_makovian}/[<task>/]
        meta/ data/chunk-XXX/*.parquet videos/chunk-XXX/<camera>/*.mp4   (LeRobot)
datasets/*_dreamgen_16fps, *_multiview_grid_seg16_dreamgen               (DreamGen training corpora)
sampling_dataset/{method}/{embodiment}/{input,output}/{view}/{model}/{split}/...  (input/ only under dense/)
checkpoints/{base,finetuned,metrics}/        index: checkpoints/CHECKPOINTS.md
models/{dreamgen,DreamDojo,Ctrl-World}/      model code, each with its own .venv
methods/{cache_strategy,sparse_attention,prunning}/
finetuning/{dreamgen,dreamdojo,ctrlworld}/   training launchers + data prep
scripts/{prepare,infer,run/{dreamgen,dreamdojo,ctrlworld},train,metrics,tables,tools,augment}/
metrics/                                     `python -m metrics.evaluate`, kiva/, rigis/, reports/
assets/{scripts,paper}/                      paper figure/table producers + LaTeX
results/                                     metric outputs read by paper producers
docs/{setup,datasets,training,inference,metrics,reproduce,paper}/
tmp/20261005_refactor/                       archive of moved items (MANIFEST.md has restore cmds)
```

## Conventions

- **Pipeline**: prepare (`scripts/prepare/`, `finetuning/*/scripts/`) → infer
  (`scripts/infer/`, `models/*/examples/`) → run wrapper
  (`scripts/run/<family>/run_<emb>_<view>_<model>[_4view_grid]_<method>.sh`).
- **Method launchers** mirror the dense launcher of their cell exactly. They only add the
  method flags and swap `sampling_dataset/dense/` → `sampling_dataset/<method>/`. Never
  delete or swap a method because it is slow; a weak result is still a valid data point.
- **Episode discovery**: launchers find episodes from the dataset/annotation dirs (never
  hard-code episode lists) and are resumable.
- **Scratch runs**: send sweeps and trials to a scratch `OUTPUT_ROOT` / `SAVE_ROOT`, never
  to the canonical tree.
- **Speedup** = internal per-segment timer vs a paired dense run on the same GPU. Never use
  wall clock.
- **Fidelity reference**: PSNR/SSIM/LPIPS use the dense prediction of the same episode as
  the reference, not GT.
- **Environments**: never mix them. See `docs/setup/README.md`. SVG2 uses
  `models/dreamgen/.venv_svg`.
- **Ctrl-World**: runs at 5 fps, 320x192, pred_step 5, overlap 1. Stats come from
  `models/Ctrl-World/dataset_meta_info/<name>/stat.json`.
- **DreamDojo**: always read data through `MultiVideoActionDataset` (`dataset[i]`).
  Chunked autoregressive, chunk_size 12.
- **GPU**: default is `CUDA_VISIBLE_DEVICES=1` unless told otherwise. Set
  `export HF_HUB_DISABLE_XET=1` for HF transfers on this cluster.
- **Archiving**: move unneeded files into `tmp/20261005_refactor/` and add a row to its
  `MANIFEST.md`. Don't delete them.
