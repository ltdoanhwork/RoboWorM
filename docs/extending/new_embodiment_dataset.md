# Adding a new embodiment or dataset (path B)

Paper App. E.2. Adding a world model or an accelerator *consumes* the benchmark's
invariants; adding an embodiment *supplies* them. KIVA asks whether this robot's links
kept their length and whether its motion stayed inside a plausible jerk budget; RIGIS
asks whether the simultaneous views are explained by one scene under one rig. Both need
knowledge only the new domain can give, so it is **declared** up front, not inferred at
scoring time. Expect the effort to sit in registration and calibration, not training.

Flow: **register → prepare data → fine-tune each backbone → instantiate the audits →
roll out → score**.

## 1. What the source corpus must provide

- Synchronised RGB for every camera of the rig, with a stated fps and resolution.
- Per-frame proprioception (state, and action if a backbone uses it) time-aligned to the
  frames.
- One instruction string per episode, verbatim as shown to the teleoperator.
- Consistent camera count and stability within a cell.
- Episode lengths that support training windows (a DreamGen example needs 93 consecutive
  frames; short clips need the overlapping-window pipeline).
- Redistributable, or reproducibly obtainable. No success or minimum-length filter
  (current episodes span 4–3,000 frames).

## 2. Put it in `datasets/` (LeRobot layout)

```
datasets/<embodiment>/<singleview|multiview>/<makovian|non_makovian>/[<task>/]
  meta/info.json            fps, features (observation.state, action, video keys), paths
  meta/episodes.jsonl       one line per episode (index, tasks, length)
  meta/tasks.jsonl          task_index -> instruction
  data/chunk-XXX/episode_XXXXXX.parquet
  videos/chunk-XXX/<camera>/episode_XXXXXX.mp4
```

`<embodiment>` is one of `single_arm`, `bimanual`, `humanoid` today. A new morphology
can reuse one of these names if it fits the axis (e.g. another dual-arm robot under
`bimanual`); a genuinely new axis name has to be added to every place that enumerates
embodiments (the `sampling_dataset` layout, `metrics/kiva/run.sh` `EMBODIMENTS`,
`metrics/kiva/metric.py::EMBODIMENTS`, `scripts/metrics/run_dense_nr.py::GT_SOURCES`,
`metrics/motion_amplitude.py::camera_mount`, the producers in `assets/scripts/`).
Existing corpora and how they were rebuilt are in [../datasets/README.md](../datasets/README.md);
DreamDojo always reads data through `MultiVideoActionDataset` (`dataset[i]`).

Keep the split name exactly `makovian` / `non_makovian` (the on-disk spelling).

## 3. Assign the labels before any rollout exists

**Regime** (benchmark axis): a task is non-Markovian when continuing it correctly needs
history that is not visible in the current frame. Label the *task* from its definition,
never from rollout behaviour. Existing tooling (Qwen3-VL served locally):

```bash
bash scripts/tools/data/launch_qwen3vl_server.sh            # keep running (vLLM)
python scripts/tools/data/classify_markovian_qwen.py        # tasks -> <labels>/task_labels.csv (+ task_labels_cache.jsonl)
python scripts/tools/data/join_labels_to_episodes.py        # -> <labels>/episode_markovian_split.csv
```

(`classify_markovian_gr1.py` / `classify_markovian_gr1robot.py` are the GR1 variants.)
`<labels>` defaults to `tmp/20261005_refactor/misc/data_labels/`, a local, untracked
directory (created on first run; the paper's label files are not shipped, their splits are
already baked into the HF dataset). For a new corpus, pass your own location with `--cache-file` / `--output-csv` (classifier) and
`--task-labels-csv` / `--output-csv` (join), and keep the label files with the release.

**Task taxonomy**: each instruction's leading verb maps to a canonical primitive via the
`SKILL` lexicon in `assets/scripts/make_benchmark_stats.py`; unmatched verbs are reported
as `unmapped`, not silently bucketed. Extend the lexicon for new verbs and keep the
unmapped list in the report.

## 4. Declare the camera setup

Per camera: **mount class** (`static`, `head`, `wrist`), extrinsic relation to the body,
and whether it is world-rigid. This is the most load-bearing field:

- the reference-free motion columns use a different estimator per mount, and
  aggregation is only allowed within one mount class; the mapping today is
  `metrics/motion_amplitude.py::camera_mount(embodiment, view)` (wrist if `"wrist"` in
  the view name; humanoid head-camera names are listed explicitly; everything else
  `static`). Add your camera names there.
- RIGIS consumes the camera set: pick the two views it scores (they should be static,
  overlapping cameras). A rig whose cameras are all wrist-mounted supports no cross-view
  claim; say so instead of discovering it in the results. Providing calibrated camera
  parameters is recommended as an extra reference, but RIGIS itself predicts them.
- Multi-view generators compose one canvas (Ctrl-World 1×N strip, DreamGen 2×2 grid).
  When the rig has fewer cameras than slots, fill the slot with black, declare it inactive
  in the prompt and exclude it from every metric. Record the view order in each episode's
  `metrics.json["per_view"]`.

## 5. Prepare data for each backbone

Preprocessing follows paper App. A.2, with four inherited decisions: bilinear resize to
the checkpoint's training resolution without letterboxing; integer frame striding to the
model's training fps (measure the realised fps on the outputs); one composed canvas for
multi-view; proprioception truncated to the consumer's dimensionality and normalised with
per-dimension 1st/99th-percentile stats over the prepared episodes (one fixed normaliser
per cell). Drop episodes missing a required stream rather than padding.

Templates per backbone:

| backbone | prepare (benchmark inputs) | training corpus | notes |
|---|---|---|---|
| Ctrl-World | `scripts/prepare/prepare_ctrlworld_{single_arm,bimanual,humanoid}_multiview[_grid].py` → `models/Ctrl-World/dataset_example/<name>/` + `sampling_dataset/dense/<emb>/input/multiview/ctrlworld*/` | same latents | stats in `models/Ctrl-World/dataset_meta_info/<name>/stat.json` (`create_meta_info*.py`); 5 fps, 320×192 per view, pred_step 5 |
| DreamGen SV | `scripts/prepare/prepare_singleview_dreamgen_batch.py`, `prepare_humanoid_singleview_dreamgen_short.py` → `batch_input_*chunks.json` | `datasets/*_dreamgen_16fps/` (`videos/ metas/ t5_xxl/`) via `finetuning/dreamgen/train_*_16fps.sh` | 16 fps |
| DreamGen MV | built by `scripts/infer/infer_bimanual_multiview_rollout.py` (`detect_layout()` picks the camera set per task); batch builders `prepare_bimanual_multiview_dreamgen_batch.py`, `prepare_multiview_humanoid_grid_batch.py` | `datasets/*_multiview_grid_seg16_dreamgen/` via `finetuning/dreamgen/train_*_multiview_grid_16fps.sh` | 2×2 grid; add a layout dict next to `LAYOUT_BIMANUAL` / `LAYOUT_HUMANOID` / `LAYOUT_HUMANOID_A` and a branch in `detect_layout()` |
| DreamDojo | none: `scripts/infer/infer_*_singleview_*.py` read `datasets/<emb>/singleview` | `finetuning/dreamdojo/launch.sh <experiment>` | chunk_size 12 |

Full training mapping: [../training/README.md](../training/README.md). Fine-tune every
backbone you cover (all three recipes are full fine-tunes) and add the checkpoints to
[`checkpoints/CHECKPOINTS.md`](../../checkpoints/CHECKPOINTS.md).

## 6. Instantiate the physical audits

### 6.1 KIVA: keypoint reader + real-segment calibration

A new morphology needs its own reader `g_theta` (DINOv3 + diffusion head) and its own
thresholds. Reusing another robot's thresholds would report the difference between two
robots as hallucination. Follow
[`metrics/kiva/docs/ADDING_A_ROBOT.md`](../../metrics/kiva/docs/ADDING_A_ROBOT.md)
(eight steps, worked Franka and ALOHA examples):

1. URDF (xacro-expanded, one tree) under `$KINESCORE_ASSETS/<robot>/urdf/`.
2. `metrics/kiva/src/kinescore/robots/<robot>/constants.py`: predicted joint names in
   corpus order, keypoint links, `ACTUATED_LINKS`, `EE_LINK`.
3. `fk.py` (forward kinematics with `pytorch_kinematics`, helpers in `robots/base.py`).
4. `spec.py` with the rigid-bone mask (`structural_rigid_bone_mask`, check
   `warn_dropped_bones` output: gripper-driven and < 1 mm bones are dropped).
5. Register the factory in `robots/__init__.py` (`_REGISTRY.register("<robot>", _build_<robot>)`).
6. `configs/robots.yaml`: `embodiment`, `keypoints` (K is checked against FK).
7. `configs/cells.yaml`: a corpus anchor (`adapter: lerobot`, `root`, `cameras`,
   `joint_field`, `joint_columns`, `gripper_columns`), a reader
   `<robot>.<corpus>.<view_id>`, and the cells you score; the view packing (panel size)
   in `configs/views.yaml`, measured from real clips.
8. `kinescore data / cache / train --reader <id>`; thresholds are then fitted
   automatically at every `kinescore score` from the reader's own val split
   (`$KINESCORE_DATA_ROOT/trees/<reader_id>/videos/val/`).

Then point `metrics/kiva/run.sh` at it with `HAL_CELL_<EMB>` / `HAL_CKPT_<EMB>` (or extend
its `cell_for` / `ckpt_for` / `tag_for` tables for a new embodiment name). Choose which
detectors enter `s_S` for this morphology (today: rigidity + jerk on single-arm, rigidity
only on bimanual / humanoid), measure the real-video flag rate on held-out real clips, and
**do not fit the cuts to your human labels** (App. E.2). Details: [../metrics/kiva.md](../metrics/kiva.md).

### 6.2 RIGIS: camera-set declaration + real anchor

Decide the two scored views (Sec. 4), extract frames from real and generated clips with
the same `--layout` / `--fps` (`metrics/rigis/extract_frames.py`), and score the
co-located real-video anchor: tables report `S / S_real` per morphology. RIGIS has no
threshold to calibrate. Details: [../metrics/rigis.md](../metrics/rigis.md).

### 6.3 FVD: a real pool

Multi-view FVD is scored against a real pool keyed by (embodiment, camera, regime) and
shared by every model and method in the setting. Build it and run the split-half
exchangeability check before pooling: `assets/scripts/fvd_camera_match.py`,
`extract_perview_fvd_feats.py`, `pool_perview_fvd.py`, `verify_shared_pool_fairness.py`
(code in `metrics/fvd/`, `metrics.fvd.SharedRealPool`, `split_half`). Single-view FVD is
referenced to each configuration's own ground truth.

### 6.4 Reference-free columns

Visual Integrity (CLIP-IQA+) and Text Alignment (CLIPScore) need no calibration, but the
NR wave's inventory must know the GT views of the new embodiment
(`scripts/metrics/run_dense_nr.py::GT_SOURCES`) and the mount of each camera (Sec. 4).

## 7. Roll out, register the cell, score

- Write dense / method launchers per [new_world_model.md](new_world_model.md) Sec. 4
  and [new_acceleration_method.md](new_acceleration_method.md) Sec. 4.
- Outputs under `sampling_dataset/<method>/<embodiment>/output/<view>/<model>/<split>/`.
- Add the cell to `assets/scripts/benchmark_cells.py` decisions, the `CELLS=` list of
  `scripts/tools/tune/_bench_uniform_timing.sh`, and the dataset-section producers
  (`make_benchmark_stats.py`, `make_dataset_appendix_tables.py`, `make_cell_matrix.py`);
  re-run `check_dataset_appendix_claims.py`.

## Checklist (paper App. E.4, items xii–xix + shared)

- [ ] Corpus provenance, licence, reproducible acquisition path.
- [ ] LeRobot tree under `datasets/<embodiment>/<view>/<split>/[<task>/]`.
- [ ] Regime label per task, from the task definition, before any rollout; label files kept with your release (the paper's splits are baked into `vgp_datasets`).
- [ ] Taxonomy mapping extended; unmapped verbs reported.
- [ ] Per-camera mount class, body-relative extrinsics, world-rigidity; `camera_mount` updated.
- [ ] Preprocessing as applied; realised fps / frame count / duration measured on generated clips.
- [ ] Every covered backbone fine-tuned; checkpoints in `CHECKPOINTS.md`.
- [ ] KIVA reader (robot module, `robots.yaml`, `cells.yaml`, `views.yaml`), trained; real-segment thresholds, not fitted to human labels; real-video flag rate reported.
- [ ] RIGIS view pair declared (or "no cross-view claim"); real anchor scored.
- [ ] FVD real pool + split-half exchangeability check.
- [ ] Inventory / cell lists / producers updated for the new cell.
