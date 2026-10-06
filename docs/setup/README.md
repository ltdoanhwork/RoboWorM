# Environment setup

RoboWorM uses **six isolated Python environments**. They are never mixed: each world
model pins its own torch / CUDA / attention-kernel stack, and the two vendored metrics
(RIGIS, KIVA) have their own. Launchers in `scripts/run/**`, `metrics/rigis/run.sh` and
`metrics/kiva/run.sh` activate the right one themselves; you only have to build them.

| env | python | used for | freeze file |
|---|---|---|---|
| `models/dreamgen/.venv` | 3.12 | DreamGen (Cosmos-Predict2 2B/14B) dense, caches, PISA, SVG1, Radial, SiTo, ITM; DreamGen training; FVD / CLIPScore / PSNR-SSIM-LPIPS (`python -m metrics.evaluate`) | [env_freeze/dreamgen.txt](env_freeze/dreamgen.txt) |
| `models/dreamgen/.venv_svg` | 3.12 | DreamGen **SVG2** only (needs FlashInfer 0.3.1 block-sparse wrapper) | [env_freeze/dreamgen__svg.txt](env_freeze/dreamgen__svg.txt) |
| `models/DreamDojo/.venv` | 3.10 | DreamDojo (2B action-conditioned DiT) inference + training, all 9 methods | [env_freeze/DreamDojo.txt](env_freeze/DreamDojo.txt) |
| `models/Ctrl-World/.venv` | 3.10 | Ctrl-World (SVD 1.5B U-Net) inference + training, all methods | [env_freeze/Ctrl-World.txt](env_freeze/Ctrl-World.txt) |
| `metrics/rigis/.venv` | 3.13 (uv project) | RIGIS (VGGT-Omega); `metrics/rigis/pyproject.toml` + `uv.lock` | — (`uv sync`) |
| `hdmi` conda env | 3.10 | KIVA (kinescore readers); override with `HAL_PYTHON` | — |

## Build the model environments

The freeze files are `pip freeze` snapshots taken on the cluster. Install torch first
from the matching CUDA wheel index, then the rest; otherwise pip resolves a CPU build.
Lines pointing at local editable installs (`-e /pfss/...`) must be re-pointed at your
checkout.

```bash
# DreamGen (torch 2.6 + cu124)
python3.12 -m venv models/dreamgen/.venv
models/dreamgen/.venv/bin/pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124
models/dreamgen/.venv/bin/pip install -r docs/setup/env_freeze/dreamgen.txt

# DreamGen SVG2 env (same base + flashinfer 0.3.1)
python3.12 -m venv models/dreamgen/.venv_svg
models/dreamgen/.venv_svg/bin/pip install -r docs/setup/env_freeze/dreamgen__svg.txt
bash scripts/tools/ops/bootstrap_svg_env.sh        # optional helper used on the cluster

# DreamDojo (torch 2.7 + cu128)
python3.10 -m venv models/DreamDojo/.venv
models/DreamDojo/.venv/bin/pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
models/DreamDojo/.venv/bin/pip install -r docs/setup/env_freeze/DreamDojo.txt

# Ctrl-World (torch 2.7.1)
python3.10 -m venv models/Ctrl-World/.venv
models/Ctrl-World/.venv/bin/pip install torch==2.7.1 torchvision
models/Ctrl-World/.venv/bin/pip install -r docs/setup/env_freeze/Ctrl-World.txt
```

Notes:

- `.venv_svg` exists only for SVG2. With the FlashInfer 0.2.5 of `.venv`, SVG2 falls
  back to a dense-mask SDPA path that OOMs on multiview sequences. All SVG2 launchers
  already default to `VENV=models/dreamgen/.venv_svg`.
- Radial and SVG2 need `flashinfer` (frozen versions: 0.2.5 in `models/dreamgen/.venv`,
  0.3.1 in `.venv_svg`, 0.6.12 in `models/DreamDojo/.venv`). DreamDojo SVG2/Radial also
  pull `apache-tvm-ffi` and `cuvs-cu12`; `scripts/tools/ops/bootstrap_radial_env.sh`
  installs the Radial extras.
- Sparse attention and token reduction have no context parallelism; their launchers
  force a single GPU.

## Build the metric environments

```bash
# RIGIS: uv project (metrics/rigis/run.sh calls `uv run --project metrics/rigis`)
cd metrics/rigis && uv sync && cd -

# KIVA: any python 3.10 env with the kinescore deps; the cluster uses a conda env
conda create -n hdmi python=3.10 -y
conda run -n hdmi pip install -e metrics/kiva
export HAL_PYTHON="$(conda run -n hdmi which python)"   # read by metrics/kiva/run.sh
```

Metric checkpoints (I3D, VGGT-Omega, KIVA readers, WorldScore probes) are listed in
[checkpoints/CHECKPOINTS.md](../../checkpoints/CHECKPOINTS.md#metric-checkpoints-checkpointsmetrics).

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
