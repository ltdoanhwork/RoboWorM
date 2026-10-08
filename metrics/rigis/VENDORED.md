# Vendored: RIGIS (VGGT-Omega multi-view consistency)

| field | value |
|---|---|
| upstream repo | this repo, branch `origin/thanhnv/multiview-consistency-indep` |
| commit | `8015d97aa68eb9bbb36d5ae04cd3acd348dd6a62` (2026-09-14, "Update") |
| vendored on | 2026-10-05 (paper-aligned refactor, branch `refactor/paper-cleanup`) |
| command | `git archive 8015d97 multiview_consistency vggt_omega pyproject.toml requirements.txt README.md \| tar -x -C metrics/rigis` |
| not vendored | upstream `run.sh`, `run_human.sh`, `.gitignore` (replaced by our `run.sh`) |

The upstream files above are byte-identical to that commit. The previous checkout
(`external/multiview-consistency`, a detached worktree) had no local changes.

## What is ours (not upstream)

| file | role |
|---|---|
| `run.sh` | launcher: uv interpreter + model service + variant sweep (`MVC_*` env) |
| `extract_frames.py` | mosaic / per-view mp4 -> `<episode>/view1, view2` frames |
| `aggregate.py` | cross-variant table from the launcher's CSVs |
| `metric.py`, `__init__.py` | `MultiviewConsistency` `BaseMetric` wrapper, registered as `rigis` (alias `multiview_consistency`) |
| `uv.lock` | lock file `uv` produced for this project on 2026-10-01 (torch 2.14.1+cu126); kept so `uv run` does not re-resolve |
| `VENDORED.md` | this file |
| `vggt_omega/LICENSE` | added 2026-10-08: the FAIR Noncommercial Research License that `vggt_omega/*.py` headers refer to, copied from [facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega) `main` @ `48b23c8` (the upstream archive above did not include it) |

Local patches to upstream files: **none**.

`vggt_omega/` is Meta's VGGT-Omega inference code (FAIR Noncommercial Research License,
non-commercial use only). The rest of this directory is RoboWorM code.

## Environment

This directory is the `uv` project. Its venv is `metrics/rigis/.venv` (git-ignored;
Python 3.13, moved here from `external/multiview-consistency/.venv` and re-synced
with `uv run`, which rebuilt the editable `vggt` install against this path).
`numba` is imported by `multiview_consistency/geometric_consistency.py` but not
declared in `pyproject.toml`; `run.sh` injects it with `uv run --with numba`.

Recreate from scratch: `cd metrics/rigis && uv sync` (needs ~7 GB; mind the
`~/.cache/uv` quota).

## Checkpoint

`checkpoints/metrics/vggt_omega/vggt_omega_1b_512.pt` (override `MVC_CKPT`). `run.sh`
falls back to the legacy `checkpoints/vggt_omega/` path if the new one is absent.

## Updating

Re-run the `git archive` command with the new SHA, diff, update this file. Keep
`run.sh` / `metric.py` contract (CSV columns `name, geometric_consistency,
pose_stability, num_disjoint_frames, disjoint_frame_ratio`).
