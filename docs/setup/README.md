# Environment setup

RoboWorM uses **six isolated Python environments** (plus one optional env for data
augmentation). Never mix them: each world model pins its own torch / CUDA /
attention-kernel stack, and the two physical audits (RIGIS, KIVA) have their own.
Launchers in `scripts/run/**`, `metrics/rigis/run.sh` and `metrics/kiva/run.sh` activate
the right interpreter themselves, so you only have to build them.

Every venv on the cluster was built with [uv](https://docs.astral.sh/uv/) (≥ 0.9), so
the commands below use uv too. All paths are relative to the repo root, and all commands
are run from the repo root.

## Overview (versions checked on the cluster venvs, 2026-10-08)

| env | python | torch (CUDA) | attention kernels and other key deps | used by | how it is built |
|---|---|---|---|---|---|
| `models/dreamgen/.venv` | 3.12.13 | 2.6.0 (cu124) | flash-attn 2.8.3, flashinfer 0.2.5, transformer-engine 2.14.1, megatron-core 0.12.1, diffusers 0.33.1, transformers 4.51.3; metric deps piq 0.8.0, pyiqa 0.1.16, torchmetrics 1.9.0, openai-clip | DreamGen (Cosmos-Predict2 2B/14B) dense + every method except SVG2; DreamGen training; `python -m metrics.evaluate` (PSNR/SSIM/LPIPS, FVD, CLIP-IQA+, CLIPScore, motion metrics); `assets/scripts/`, `scripts/tables/` | [freeze](env_freeze/dreamgen.txt) |
| `models/dreamgen/.venv_svg` | 3.12.13 | 2.6.0 (cu124) | same stack, but **flashinfer 0.3.1** (block-sparse `VariableBlockSparse` wrapper), cuvs-cu12 26.4 | DreamGen **SVG2** only (`run_*_dreamgen_svg2.sh` default `VENV=models/dreamgen/.venv_svg`) | [freeze](env_freeze/dreamgen__svg.txt) |
| `models/DreamDojo/.venv` | 3.10.19 | 2.7.0 (cu128) | flash-attn 2.7.3, natten 0.21.0, transformer-engine 2.2, xformers 0.0.30, megatron-core 0.14.0, diffusers 0.35.2; flashinfer 0.3.1 + apache-tvm-ffi + cuvs-cu12 (Radial / SVG2); pytorch3d | DreamDojo (2B action-conditioned DiT) inference + training, all 9 methods | `uv sync` on `models/DreamDojo/pyproject.toml` + `uv.lock`; [freeze](env_freeze/DreamDojo.txt) |
| `models/Ctrl-World/.venv` | 3.10.19 | 2.7.1 (cu126) | diffusers 0.34.0, transformers 4.48.1, jax 0.5.3 (π0.5 policy-in-the-loop); flashinfer 0.3.1 + cuvs-cu12 (Radial / SVG2) | Ctrl-World (SVD 1.5B U-Net) inference + training, all methods | `models/Ctrl-World/requirements.txt`; [freeze](env_freeze/Ctrl-World.txt) |
| `metrics/rigis/.venv` | 3.13.14 | 2.14.1 (cu126) | VGGT-Omega code (`metrics/rigis/vggt_omega`), numba injected at run time | RIGIS (`metrics/rigis/run.sh`, `--metrics rigis`) | `uv sync` on `metrics/rigis/pyproject.toml` + `uv.lock` |
| `hdmi` conda env (`HAL_PYTHON`) | 3.10.19 | 2.7.0 (cu128) | transformers 5.2.0 (DINOv3 reader backbone), pytorch-kinematics 0.10, robot-descriptions, numpy 1.26.4 | KIVA (`metrics/kiva/run.sh`, `--metrics kiva`) | conda + `pip install -e "metrics/kiva[dino,video,bench]"` |
| `.venv_edit` (optional) | 3.12.13 | 2.6.0 (cu124) | diffusers 0.39.0, transformers 5.15.0 (Qwen-Image-Edit) | only the appearance-augmentation launchers (`scripts/augment/`, `*_augment.sh`, override `EDIT_VENV`) | not frozen; any env with `diffusers>=0.35` |

The freeze files in [env_freeze/](env_freeze/) are `uv pip freeze` snapshots of the
cluster venvs. Their first lines name the wheel indexes that the CUDA builds come from.
The editable lines (`-e ./models/...`) are relative to the repo root, so install them
from the repo root.

## System requirements

- Linux x86-64 and an NVIDIA driver recent enough for CUDA 12.8 (DreamDojo's cu128
  wheels). The cluster runs driver 610 on A100-80GB.
- `uv` ≥ 0.9 (`curl -LsSf https://astral.sh/uv/install.sh | sh`), and conda only for
  the KIVA env.
- **ffmpeg / ffprobe on `PATH`.** Prepare scripts, GT rendering
  (`scripts/metrics/render_missing_gt.py`, `gt_cells.py`), RIGIS frame extraction and KIVA
  clip probing call the CLI. The cluster has `/usr/bin/ffmpeg` 4.4.2. `metrics/kiva/run.sh`
  prefers a static build (`HAL_FFMPEG_BIN`), because the system ffmpeg crashes when it is
  launched from a process that has already imported torch (liblapack symbol clash).
- A CUDA toolkit (`nvcc`) only if flash-attn, transformer-engine or pytorch3d have to be
  built from source (the case when no matching wheel exists).
- `export HF_HUB_DISABLE_XET=1` for every Hugging Face download on a parallel filesystem
  (see [Secrets](#secrets)).

## Build the model environments

Install order matters: torch from its CUDA index first, then the source-built kernels
with `--no-build-isolation` (they compile against the installed torch), then the rest
of the freeze file. Without the explicit CUDA index, the resolver picks a different torch
build.

```bash
# --- DreamGen: models/dreamgen/.venv (torch 2.6.0+cu124, flashinfer 0.2.5) ---
uv venv --python 3.12 models/dreamgen/.venv
PY=models/dreamgen/.venv
uv pip install --python $PY torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python $PY --no-build-isolation flash-attn==2.8.3 "transformer-engine[pytorch]==2.14.1"
uv pip install --python $PY --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://flashinfer.ai/whl/cu124/torch2.6/ \
    --extra-index-url https://pypi.nvidia.com \
    -r docs/setup/env_freeze/dreamgen.txt

# --- DreamGen SVG2: models/dreamgen/.venv_svg (same base, flashinfer 0.3.1) ---
uv venv --python 3.12 models/dreamgen/.venv_svg
PY=models/dreamgen/.venv_svg
uv pip install --python $PY torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python $PY --no-build-isolation flash-attn==2.8.3 "transformer-engine[pytorch]==2.14.1"
uv pip install --python $PY --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.nvidia.com \
    -r docs/setup/env_freeze/dreamgen__svg.txt

# --- DreamDojo: models/DreamDojo/.venv (upstream install.sh, torch 2.7.0+cu128) ---
(cd models/DreamDojo && uv sync --extra=cu128 --python 3.10)   # torch, flash-attn, natten, TE from the cosmos-dependencies index
PY=models/DreamDojo/.venv
uv pip install --python $PY openai tyro numpydantic albumentations tianshou pyarrow fastparquet piq lightning \
    moviepy==2.2.1 "jsonargparse[signatures]>=4.27.7"
uv pip install --python $PY --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@b6a77ad7aaf41ed90fca80ce6a2bac3c462a7881"
uv pip install --python $PY torchcodec==0.5 --index-url https://download.pytorch.org/whl/cu128
# Radial / SVG2 extras (the cluster venv has these; numpy must stay < 2)
uv pip install --python $PY flashinfer-python==0.3.1 apache-tvm-ffi cuvs-cu12 "numpy<2" --extra-index-url https://pypi.nvidia.com

# --- Ctrl-World: models/Ctrl-World/.venv (torch 2.7.1+cu126 from PyPI) ---
uv venv --python 3.10 models/Ctrl-World/.venv
PY=models/Ctrl-World/.venv
uv pip install --python $PY -r models/Ctrl-World/requirements.txt
# Radial / SVG2 extras, plus the exact versions of everything else:
uv pip install --python $PY --index-strategy unsafe-best-match --extra-index-url https://pypi.nvidia.com \
    -r docs/setup/env_freeze/Ctrl-World.txt
```

Notes:

- `.venv_svg` exists only for SVG2. With the flashinfer 0.2.5 of `.venv`, SVG2 falls
  back to a dense-mask SDPA path that builds a `(BH, S, S)` mask and OOMs (~57 GiB) on
  multiview sequences. The compiled SVG kernels in
  `methods/sparse_attention/Sparse-VideoGen/svg/kernels` are built for Python 3.11 and do
  not load in these 3.12 venvs; SVG2 uses the flashinfer block-sparse path instead.
- flashinfer versions: 0.2.5 in `models/dreamgen/.venv` (enough for Radial), 0.3.1 in
  `.venv_svg`, `models/DreamDojo/.venv` and `models/Ctrl-World/.venv`.
- DreamDojo can also be rebuilt with the upstream helper: `cd models/DreamDojo && bash install.sh`.
  Its first two lines install uv into `$HOME`; skip them if uv is already installed.
- Ctrl-World's upstream README uses a conda env with Python 3.11. The benchmark venv uses
  3.10 (that is what the freeze was taken from). The π0.5 policy-in-the-loop launcher
  (`run_single_arm_multiview_ctrlworld_pi.sh`) also needs the `openpi` checkout in
  `models/Ctrl-World/openpi` (see [THIRD_PARTY.md](../THIRD_PARTY.md)). It is put on
  `PYTHONPATH`, not installed.
- `scripts/tools/ops/bootstrap_svg_env.sh` and `bootstrap_radial_env.sh` are **legacy**.
  They build a torch 2.5.1 venv outside the repo (`../DreamDojo-svg/.venv`,
  `../DreamDojo-radial/.venv`) and are not how `.venv_svg` or any env above was built.
- Sparse attention and token reduction have no context parallelism, and their launchers
  force a single GPU.

## Build the metric environments

```bash
# RIGIS: uv project (metrics/rigis/run.sh calls `uv run --project metrics/rigis --with numba`)
(cd metrics/rigis && uv sync)            # ~7 GB; set UV_CACHE_DIR to a big disk if ~/.cache/uv is quota-limited

# KIVA: python 3.10 env with the kinescore deps (do not use uv here, see docs/metrics/kiva.md)
conda create -n hdmi python=3.10 -y
conda run -n hdmi pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
conda run -n hdmi pip install -e "metrics/kiva[dino,video,bench]"
export HAL_PYTHON="$(conda run -n hdmi which python)"   # read by metrics/kiva/run.sh
```

The other metrics (PSNR/SSIM/LPIPS, FVD, Visual Integrity, Text Alignment, motion
amplitude / smoothness) run in `models/dreamgen/.venv`. The motion metrics also need
the `metrics/WorldScore` checkout (see [THIRD_PARTY.md](../THIRD_PARTY.md)). On the
cluster, `run.sh` does not install kinescore. It sets `PYTHONPATH=metrics/kiva/src` and
takes `pytorch_kinematics` from `PYTHONUSERBASE` (`HAL_PYTHONUSERBASE`). The editable
install above makes both settings unnecessary.

Metric checkpoints (I3D, VGGT-Omega, KIVA readers, WorldScore probes) are listed in
[checkpoints/CHECKPOINTS.md](../../checkpoints/CHECKPOINTS.md#metric-checkpoints-checkpointsmetrics).

## Verify

Each command prints the torch version, `True` for CUDA, and the key kernel versions from
the table above.

```bash
for v in models/dreamgen/.venv models/dreamgen/.venv_svg models/DreamDojo/.venv; do
  $v/bin/python -c "import torch, flash_attn, flashinfer; print('$v', torch.__version__, torch.cuda.is_available(), flash_attn.__version__, flashinfer.__version__)"
done
models/Ctrl-World/.venv/bin/python -c "import torch, diffusers, flashinfer; print(torch.__version__, torch.cuda.is_available(), diffusers.__version__, flashinfer.__version__)"
(cd metrics/rigis && .venv/bin/python -c "import torch, vggt_omega, multiview_consistency; print(torch.__version__, torch.cuda.is_available())")
$HAL_PYTHON -c "import torch, kinescore, pytorch_kinematics, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
models/dreamgen/.venv/bin/python -m metrics.evaluate --help | head -3
ffmpeg -version | head -1 && ffprobe -version | head -1
```

Expected output on the cluster:

```
models/dreamgen/.venv      2.6.0+cu124 True 2.8.3 0.2.5+cu124torch2.6
models/dreamgen/.venv_svg  2.6.0+cu124 True 2.8.3 0.3.1
models/DreamDojo/.venv     2.7.0+cu128 True 2.7.3 0.3.1
Ctrl-World                 2.7.1+cu126 True 0.34.0 0.3.1
RIGIS                      2.14.1+cu126 True
KIVA                       2.7.0+cu128 True 5.2.0
```

## Secrets

Put credentials in a local `.env` (git-ignored) or export them; never hard-code them in
scripts. The HF token is only needed for private repos or uploads:

```bash
export HF_TOKEN=<token>
export HF_HUB_DISABLE_XET=1   # required on this cluster's parallel filesystem
```

## Data, checkpoints, backups

- Source datasets and benchmark rollouts: [../datasets/README.md](../datasets/README.md).
- Checkpoints: [checkpoints/CHECKPOINTS.md](../../checkpoints/CHECKPOINTS.md).
- Full workspace backup / restore (checkpoint repos `vgp_checkpoints_*`, tar-shard
  mirrors, gotchas): [hf_backup_restore.md](hf_backup_restore.md).
- Third-party code vendored into the repo and its licenses: [../THIRD_PARTY.md](../THIRD_PARTY.md).
