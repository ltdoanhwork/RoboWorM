# Evaluating a new world model (path A)

Paper App. E.1. A new model enters the benchmark as **one more row**: fine-tune it on
the released corpus of each cell it covers, roll it out over that cell's episodes with
the existing conditioning, write the rollouts into the `sampling_dataset/` tree with the
same layout as the existing models, and score them with the same metrics.

The three existing families are the templates: **Ctrl-World** (SVD U-Net, multi-view,
action-conditioned), **DreamGen** (Cosmos-Predict2 2B/14B DiT, text + image), and
**DreamDojo** (2B action-conditioned DiT, chunked autoregressive). They differ on every
architectural axis and are compared only because each satisfies one interface.

## 1. The interface the benchmark requires

- **In**: a conditioning prefix drawn from a benchmark episode — any subset of
  - the observation prefix (conditioning frames at the model's declared resolution / fps),
  - the instruction string (verbatim from the source corpus; LeRobot `meta/tasks.jsonl`),
  - proprioception (per-frame state / action vectors, strided like the frames).
- **Out**: an RGB rollout **frame-aligned to the episode's ground truth** at a declared
  frame rate, generated to the ground-truth episode length (not a fixed horizon).
- Two hard constraints inside a cell: one fixed per-dimension action / state normaliser
  shared by every method, and an identical conditioning prefix for every method.
- A model that ignores text is scored on everything else; its Text Alignment is `None`
  (see `metrics/text_alignment.py::TEXT_CONDITIONED_MODELS`, which today lists
  `dreamgen`, `ctrlworld`, `ctrlworld_4view_grid`, `gt`).

Action truncation is a property of the consumer: existing checkpoints use 14-D
joint + gripper (bimanual), 8-D joint, or 7-D Cartesian interfaces. Declare yours.
Ctrl-World normalisation stats live in `models/Ctrl-World/dataset_meta_info/<name>/stat.json`
(1st / 99th percentile per dimension, built by `models/Ctrl-World/dataset_meta_info/create_meta_info*.py`).

## 2. Put the model in the repo

```
models/<YourModel>/            # model code (vendored), with its OWN venv
  .venv/                       # never share an env with another model
```

- Environments are isolated; see [../setup/README.md](../setup/README.md) and add your
  frozen package list to `docs/setup/env_freeze/<YourModel>.txt`.
- Base weights go under `checkpoints/base/<org>/...`, fine-tuned weights under
  `checkpoints/finetuned/<yourmodel>/<cell>/...`. Add one row per cell to the
  "Paper cells" table of [`checkpoints/CHECKPOINTS.md`](../../checkpoints/CHECKPOINTS.md)
  (path, size, iter, dense launcher).

## 3. Fine-tune (expected, not optional)

A zero-shot checkpoint measures domain shift, not acceleration: fine-tuning moved PSNR by
+4 to +7 dB on the existing backbones (App. C.1), and the un-finetuned failure is the
object leaving the scene. Fine-tune on the released training corpus of each cell:

| source | where |
|---|---|
| LeRobot source episodes | `datasets/<embodiment>/<view>/<split>/[<task>/]` ([../datasets/README.md](../datasets/README.md)) |
| existing training launchers to copy from | `finetuning/{dreamgen,dreamdojo,ctrlworld}/`, `scripts/train/` |
| which corpus trained which checkpoint | [../training/README.md](../training/README.md) |

Put your launcher in `finetuning/<yourmodel>/` and report dense quality per cell before
any accelerator is applied. An unmatched recipe is fine, but it must be disclosed.

## 4. The three stages: prepare → infer → run

### 4.1 `scripts/prepare/prepare_<embodiment>_<view>_<yourmodel>*.py`

Converts curated episodes into your model's native input. Existing examples:
`prepare_singleview_dreamgen_batch.py` (DreamGen SV batch JSON),
`prepare_ctrlworld_single_arm_multiview.py` (Ctrl-World annotations + latents),
`prepare_single_arm_multiview_dreamgen_batch.py`. DreamDojo reads `datasets/` directly
through `MultiVideoActionDataset` (`dataset[i]`).

Rules: discover episodes from the dataset / annotation dirs (never a hard-coded list);
write shared inputs to `sampling_dataset/dense/<embodiment>/input/<view>/<yourmodel>/<split>/`
(only `dense/` has `input/`); stride actions by the same factor as frames.

### 4.2 `scripts/infer/infer_<embodiment>_<view>_<yourmodel>.py`

Performs the rollout and writes per-episode artefacts. Follow the existing scripts:

- argparse with `--save_dir` / `--input_save_dir` (or equivalent) so the launcher sets
  the output tree, and `--num_episodes` (or `MAX_EPISODES`) for trials;
- **skip episodes whose output already exists** (resumable);
- roll out to the GT episode length;
- write the GT clip **next to** the prediction at the same resolution / fps / length
  (KIVA's real anchor and the NR wave's GT rows read it from there);
- print one authoritative per-segment / per-episode timing line and persist it, e.g.
  `wall_time_s` in `metrics.json` + `mean_wall_time_s` in `all_summary.json`
  (`infer_single_arm_multiview_ctrlworld.py`), `[Timing] Generated ... s total`
  (`infer_humanoid_singleview_full_episode.py`), `Total inference time` /
  `LVG Total inference time` (DreamGen). Speedup is computed from this line, never from
  wall clock;
- register every acceleration flag group so methods can be enabled by flags alone
  (see [new_acceleration_method.md](new_acceleration_method.md)): e.g.
  `add_worldcache_args(parser)`, `add_pisa_args(parser)`, ... and one mutual-exclusion
  check (`methods/cache_strategy/ctrl_world_utils.py::validate_ctrl_world_backend_args`,
  `methods/cache_strategy/config.py::build_cache_runtime_config`).

Per-episode output convention (pick the closest existing one,
[../datasets/sampling_dataset.md](../datasets/sampling_dataset.md#per-model-output-conventions)):

| style | files |
|---|---|
| per-episode dir (Ctrl-World / DreamDojo / DreamGen MV) | `episode_<id>/full_pred.mp4`, GT (`full_gt.mp4` or `gt_all_views.mp4`), `metrics.json` (`instruction`, `per_view` for multi-view), split-level `all_summary.json` |
| flat (DreamGen SV) | `episode_<id>.mp4` + `episode_<id>_gt.mp4` + `episode_<id>.txt` prompt sidecar (`[Prompt]` section) |
| multi-view mosaic | `pred_all_views.mp4` as a 1×N strip or a 2×2 grid; record the view order in `metrics.json["per_view"]` |

Name episode dirs by the real trajectory id. Multi-view models must keep camera count
fixed within a cell; unused canvas slots are black, declared inactive in the prompt, and
excluded from every metric.

### 4.3 `scripts/run/<family>/run_<embodiment>_<view>_<yourmodel>.sh`

The dense launcher pins checkpoint, split and output tree. Model it on
`scripts/run/ctrlworld/run_single_arm_multiview_ctrlworld.sh`:

```bash
#!/bin/bash
# Usage: bash scripts/run/<family>/run_<emb>_<view>_<yourmodel>.sh [makovian|non_makovian|both]
# Env: CKPT, NUM_EPISODES, OUTPUT_ROOT, INPUT_ROOT, SKIP_PREPARE
set -euo pipefail
SUBSET=${1:-both}
PROJECT_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/models/<YourModel>/.venv/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/models/<YourModel>:${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CKPT_PATH="${CKPT:-./checkpoints/finetuned/<yourmodel>/<cell>/<iter>.pt}"
OUTPUT_BASE="${OUTPUT_ROOT:-${PROJECT_ROOT}/sampling_dataset/dense/<emb>/output/<view>/<yourmodel>}"
INPUT_BASE="${INPUT_ROOT:-${PROJECT_ROOT}/sampling_dataset/dense/<emb>/input/<view>/<yourmodel>}"
EXTRA=""
[ -n "${NUM_EPISODES:-}" ] && EXTRA="${EXTRA} --num_episodes ${NUM_EPISODES}"

if [ "${SKIP_PREPARE:-0}" != "1" ]; then
  python scripts/prepare/prepare_<emb>_<view>_<yourmodel>.py --subset "${SUBSET}" \
      --dataset-base "${PROJECT_ROOT}/datasets/<emb>/<view>" --output-dir "${INPUT_BASE}"
fi
for SPLIT in makovian non_makovian; do
  { [ "${SUBSET}" = both ] || [ "${SUBSET}" = "${SPLIT}" ]; } || continue
  python scripts/infer/infer_<emb>_<view>_<yourmodel>.py --ckpt_path "${CKPT_PATH}" \
      --subset "${SPLIT}" --input_save_dir "${INPUT_BASE}/${SPLIT}" \
      --save_dir "${OUTPUT_BASE}/${SPLIT}" ${EXTRA}
done
```

The flags of the prepare / infer scripts above are placeholders for your own scripts;
the structure (repo root via git, own venv, `CKPT` / `OUTPUT_ROOT` / `INPUT_ROOT` /
`NUM_EPISODES` / `SKIP_PREPARE` env, both splits) is what existing launchers share.
Every method launcher of the cell will later be a copy of this file plus the method's
flags, with `sampling_dataset/dense/` swapped for `sampling_dataset/<method>/`.

Trial runs: `NUM_EPISODES=2 OUTPUT_ROOT=/tmp/<yourmodel>_trial/output INPUT_ROOT=/tmp/<yourmodel>_trial/input bash scripts/run/...`.

## 5. Score it

No per-model decision is needed. Physical audits and reference-free columns need no dense
counterpart, so a new dense model is scored on KIVA, RIGIS, Visual Integrity (CLIP-IQA+)
and Text Alignment (CLIPScore) before any accelerator exists. PSNR / SSIM / LPIPS are
always against the model's **own** dense rollout, so a dense-only entrant has no value in
them.

```bash
PY=models/dreamgen/.venv/bin/python; export HF_HUB_OFFLINE=1
CELL=sampling_dataset/dense/<emb>/output/<view>/<yourmodel>/makovian

# reference-free columns on the paper grid (one sample per clip, layout from Sample.meta)
$PY -m metrics.evaluate --pred-root $CELL --metrics visual_integrity text_alignment --out /tmp/<yourmodel>_nr

# KIVA (single-view): clip must match the reader's input size, see docs/metrics/kiva.md
VIDEOS_DIR=$CELL EMBODIMENTS=<emb> HAL_PATTERN='full_pred.mp4' OUT_DIR=/tmp/<yourmodel>_kiva \
  bash metrics/kiva/run.sh

# RIGIS (multi-view): extract two static views, then score, see docs/metrics/rigis.md
python3 metrics/rigis/extract_frames.py --videos-root sampling_dataset/dense/<emb>/output/multiview/<yourmodel> \
  --out-root /tmp/<yourmodel>_frames/<yourmodel> --video-name pred_all_views.mp4 --layout 1x4 --fps 2
MVC_DATA_ROOT=/tmp/<yourmodel>_frames VARIANTS=<yourmodel> MVC_OUT_ROOT=/tmp/<yourmodel>_rigis \
  bash metrics/rigis/run.sh
```

For the full-benchmark waves the inventory code must know your model's output layout:

- `scripts/metrics/run_dense_nr.py::_pred_rows` decides which mp4 is a prediction and how
  its views are laid out (`single` / `strip` / `grid`); `episode_key` decides how a clip
  pairs with its GT / dense counterpart. Add a branch for a new layout there.
- `metrics/kiva/run.sh` sweeps only `.../output/singleview/dreamgen/`; for another model
  use `VIDEOS_DIR` mode or extend the launcher.
- FVD: multi-view rows are scored against the shared per-camera real pool (keyed by
  embodiment, camera, regime) and are comparable to existing rows. A single-view model
  reports FVD against **its own** ground truth and must not be FVD-ranked against the
  other single-view rows.

## 6. Register the cell for tables and timing

- `assets/scripts/benchmark_cells.py` — which (model, embodiment, view) cells the paper
  reports; producers import `is_excluded` / `filter_rows`.
- `scripts/tools/tune/_bench_uniform_timing.sh` — `CELLS=(...)` rows
  `id|embodiment|view|model|dense_s|unit|dense_launcher|env_pins`; add yours to get a
  matched dense control in the uniform timing pass.
- Model-name maps in the producers (e.g. `MODEL_FAMILY` in
  `scripts/tables/make_speedup_summary.py`) — grep for an existing model name such as
  `dreamdojo` and add yours next to it.

## Checklist (paper App. E.4, items vii–xi + shared)

- [ ] Own venv under `models/<YourModel>/`; freeze in `docs/setup/env_freeze/`.
- [ ] Checkpoint per cell + fine-tuning recipe + corpus; row in `checkpoints/CHECKPOINTS.md`; label / fps defects disclosed.
- [ ] Declared fps, resolution, conditioning-prefix length, action truncation, fixed normaliser.
- [ ] `prepare` / `infer` / `run` scripts; episodes discovered, resumable, GT-length rollouts, GT written next to pred.
- [ ] One authoritative timing line persisted per episode / segment.
- [ ] Every covered cell complete: both regimes, every camera, every curated episode.
- [ ] Dense quality per cell reported before any accelerator.
- [ ] Seeding stated; single-sample values marked.
- [ ] Single-view FVD reported against own GT, not ranked across configurations.
- [ ] Inventory / table / timing code knows the new model name and layout.
