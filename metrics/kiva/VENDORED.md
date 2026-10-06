# Vendored: KIVA (kinescore embodiment-hallucination metric)

| field | value |
|---|---|
| upstream repo | branch `origin/quang/hallucination-official` (unrelated git history) |
| commit | `5225ce75769df27bba6626fe0a4218aee2bcbcd3` (2026-10-01, "docs: README gives the exact download command for each benchmark checkpoint") |
| kinescore version | 0.2.0 |
| vendored on | 2026-10-05 (paper-aligned refactor, branch `refactor/paper-cleanup`) |
| command | `git archive 5225ce7 src/kinescore configs docs scripts README.md pyproject.toml .env.example \| tar -x -C metrics/kiva` |
| not vendored | `tests/` (golden `.npz`, ~440 KB), `data/` (only `.gitkeep` + README), `.gitignore`, `uv.lock` |

The previous checkout (`external/hallucination`, a detached worktree) had no local changes.

## What is ours (not upstream)

| file | role |
|---|---|
| `run.sh` | launcher: interpreter, store, ffmpeg workaround, variant sweep (`HAL_*` env) |
| `aggregate.py` | cross-variant table from the launcher's scored dirs |
| `probe_clips.py` | pre-flight: frame size / count per clip |
| `map_to_dr.py` | join KIVA rows onto the DR table |
| `metric.py`, `__init__.py` | `Hallucination` `BaseMetric` wrapper, registered as `kiva` (alias `hallucination`) |
| `VENDORED.md` | this file |

## Local patches to upstream files

1. `src/kinescore/registry/provenance.py::git_state` -- the tree now lives inside
   this repo, so `git -C metrics/kiva` resolves to the HOST repo. The dirty check is
   scoped to this package (`git status --porcelain -- .`, also avoids a full-repo
   status on every run) and a constant `upstream_sha = 5225ce7...` is added to every
   `run_manifest.json`. `git_sha` now records the host repo HEAD. Scores are unaffected.

## Environment (nothing is pip-installed)

`run.sh` runs the `hdmi` conda env
(`/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/miniconda3/envs/hdmi/bin/python`, override
`HAL_PYTHON`) with `PYTHONPATH=metrics/kiva/src` and
`PYTHONUSERBASE=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local` (supplies
`pytorch_kinematics`). `uv sync` on `pyproject.toml` does not work on this cluster
(triton unpack exceeds the `~/.cache/uv` quota) -- see docs/metrics/hallucination_metric.md.
`configs/*.yaml` are found relative to the package (`src/kinescore/registry/*.py`
use `parents[3]` = `metrics/kiva`).

## Store / checkpoints

`KINESCORE_ASSETS`, `KINESCORE_DATA_ROOT`, `KINESCORE_CACHE_DIR` point into the kinescore
store (`$HAL_STORE`, default `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/triquang/kinescore-store`).
Reader checkpoints (`KINESCORE_CKPT_DIR`) default to `checkpoints/metrics/kiva/` when that
directory exists, else `$HAL_STORE/ckpt` (override `HAL_CKPT_DIR` or the per-embodiment
`HAL_CKPT_{SINGLE_ARM,BIMANUAL,HUMANOID}`). Benchmark readers:

| embodiment | reader |
|---|---|
| single_arm | `a1x_ee.a1x_sv.sv1_4x3.pt` (sha256 `5e463b70...`) |
| bimanual | `aloha_bimanual.bimanual_sv.sv1_16x9.pt` |
| humanoid | `fourier_gr1.humanoid_sv.sv1_16x9.huber_old.pt` (sha `c40fc006...`, K=12) |

## Updating

Re-run the `git archive` command with the new SHA, re-apply the provenance patch,
update this file and `UPSTREAM_SHA` in `provenance.py`.
