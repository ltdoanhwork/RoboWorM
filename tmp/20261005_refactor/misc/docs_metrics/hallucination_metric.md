# Hallucination metric (KIVA; kinescore embodiment-integrity, rigidity + jerk)

> **2026-10-05 refactor:** the metric code is now **vendored** into `metrics/kiva` (`git archive 5225ce7`, see `metrics/kiva/VENDORED.md`); the old `external/hallucination` git worktree was removed after a parity test. Registry name `kiva` (old name `hallucination` kept as an alias). Historical sections below that mention a worktree describe the pre-refactor setup; the code is byte-identical.

How to measure whether our generated robot videos **hallucinate the robot's own
body** — an arm that stretches, snaps, folds through itself or teleports — and
how to turn that into a one-command comparison across our method variants.

**TL;DR**

```bash
# 0. pre-flight: will the reader even accept these clips?
python metrics/kiva/probe_clips.py --variant dense pisa radial svg1 svg2

# 1. score the real-video anchor FIRST (it is what makes the numbers mean anything)
HAL_OUT_ROOT=/tmp/hal_results ROLE=gt VARIANTS="dense pisa radial svg1 svg2" \
  EMBODIMENTS=single_arm SPLITS=makovian bash metrics/kiva/run.sh --tmux

# 2. score every variant in one invocation
HAL_OUT_ROOT=/tmp/hal_results VARIANTS="dense pisa radial svg1 svg2" \
  EMBODIMENTS=single_arm SPLITS=makovian bash metrics/kiva/run.sh --tmux

# 3. leaderboard, anchored on real video
python metrics/kiva/aggregate.py --results-root /tmp/hal_results \
  --variants dense pisa radial svg1 svg2 --embodiment single_arm \
  --anchor-role gt --baseline real
```

Both headline numbers are **violation rates and severities: LOWER IS BETTER**,
and neither has a meaningful zero — see §7 for why you read them as an *excess
over the real-video anchor*, not against 0.

---

## 1. Which metric is this, and why it is the right hallucination metric

### 1.1 What `kinescore` is

`kinescore` is **not a second metric next to "the hallucination metric" — it is
the hallucination metric's implementation**. It is the Python package / CLI name
of the self-contained benchmark repo that lives on branch
`origin/quang/hallucination-official` of *this* repo:

| | |
|---|---|
| package | `kinescore` (`metrics/kiva/src/kinescore/`) |
| CLI entry point | `kinescore = kinescore.cli.main:main` (`pyproject.toml`) |
| self-description | "Physics-plausibility benchmark for AI-generated robot video." (`README.md:1-3`) |
| author / history | 22 commits, all `triquang26`, 2026-08-29 → 2026-10-01 |
| our pin | `metrics/kiva` = vendored copy of `5225ce7` (`VENDORED.md`) |

So "run the hallucination metric" == "run `kinescore score`". The word
*hallucination* appears in the **branch** name and the **HF bucket** name
(`twanghcmut/hallucinate-bench`); the word *kinescore* is what the code calls
itself. They are the same thing. There is no separate `hallucination/` package
anywhere on the cluster (`find` over both metric branches returns only
`src/kinescore/`).

### 1.2 What it actually measures

The pipeline is **frozen DINOv3 → trained diffusion keypoint reader → analytic
detectors**. There is **no simulator and no VLM judge** (`README.md:5-8`):

```
mp4 frames  --DINOv3 ViT-L/16 (frozen)-->  patch tokens
            --diffusion keypoint head-->   P (T, K, 3) in metres, robot base frame
            --analytic detectors-->        one per-frame score series per error type
            --16-frame segments-->         one verdict per segment per detector
```

Five detectors are computed; **two are the headline**
(`violations/scorer.py:51`, `HEADLINE = ("rigidity", "jerk")`):

| detector | units | direction | what body-hallucination it catches | source |
|---|---|---|---|---|
| **`rigidity`** | mm | higher worse | A rigid link **stretches or shrinks** away from its true rest length. The single most direct "that is not a robot body any more" signal. | `violations/detectors.py:139-187` |
| **`jerk`** | mm/s³ | higher worse | The arm **snaps / jitters / teleports between poses** — 3rd time derivative of keypoint position. | `violations/detectors.py:189-204` |
| `teleport` | mm/s | higher worse | Worst keypoint *speed* (1st derivative). Companion diagnostic. | `violations/detectors.py:206-219` |
| `joint_limit` | deg | higher worse | A bend angle leaves the envelope real motion ever shows. | `violations/detectors.py:222-263` |
| `self_collision` | mm (min dist) | **lower worse** | Two non-adjacent keypoints get closer than real motion ever gets — the arm folds through itself. | `violations/detectors.py:265-297` |

**Why this is the right hallucination metric for our question.** The failure
mode we care about is a world model that renders a *plausible-looking* robot
whose body is not a body: links that change length frame to frame, a gripper
that pops across the workspace, an elbow that inverts. That is invisible to
every metric we already compute:

- **PSNR / SSIM / LPIPS** are reference-based appearance metrics; a smoothly
  stretched arm can have *better* PSNR than a correct one.
- **FVD** is a distribution distance over a Kinetics-pretrained feature space;
  our own audit (`metrics/reports/fvd_kinetics_domain_audit.json`) shows the
  domain gap already dominates it.
- **WorldScore** has, in our own words, **"no true physics-law metric"**
  (`tmp/20261005_refactor/docs/WORLDSCORE_PHYSICS_REPORT.md:120-122`) — its physics signal is
  indirect, via motion magnitude and smoothness.
- **Multi-view consistency** (`docs/metrics/multiview_consistency_metric.md`) asks
  whether two *views* agree, which is a camera/scene question, and it needs a
  multiview generation to exist at all.

`rigidity` and `jerk` are the only numbers we have that are a function of the
**robot's own 3-D kinematics**, with a threshold set by **real teleop from the
same robot**. That is what makes them a hallucination metric rather than a
quality metric.

**What it explicitly does not measure** (`docs/METRICS.md:97-101`): task
success, visual fidelity, object dynamics, contact. A clip that passes both
detectors "is not physically correct; it is a clip in which the arm's own
kinematics show no violation this reader can see."

### 1.3 The variant chosen, and why

There are four axes of choice. Ours, and the reasoning:

| axis | options | our choice | why |
|---|---|---|---|
| **detector set** | 5 computed; `HEADLINE` declares 2 | **`rigidity` + `jerk`** | It is the upstream-declared headline (`scorer.py:51`), it is what `docs/METRICS.md` documents, and it is the pair that is *structural* (rest length) + *temporal* (3rd derivative). The other three are stored and available via `--detectors`. |
| **granularity** | per-frame flag vs **16-frame segment verdict** | **segment** | `SEGMENT_LEN = 16` (`violations/segments.py:8`) is one window of the reader's trained prediction horizon, so a segment is one complete unit of what the reader can see. A per-frame rate is dominated by reader noise. |
| **reduction** | `worst` vs `median` within a segment | **upstream per-detector default**: `rigidity` → `median`, `jerk` → `worst` | A rigid link cannot stretch, so a lone stretched frame is measurement noise, but a segment whose *typical* frame is stretched is not (`detectors.py:164-166`). Jerk is a genuine single-frame event, so `worst` (`Detector.segment_reduce = "worst"`, `detectors.py:49`). |
| **calibration** | p95 of real motion vs fixed thresholds | **p95, per reader, refit per run** | There is no default threshold; scoring before calibration raises (`detectors.py:98-101`). See §7 and the drift gotcha in §8.3. |

We report **both** the segment **violation rate** and the **severity ratio**
(`value / segment_threshold`), because a rate alone cannot distinguish "barely
over the line everywhere" from "catastrophically over the line once"
(`metrics/kiva/aggregate.py:166-188`).

Trade-offs of this choice:

- **Pro**: a hard geometric quantity in millimetres, no VLM, no prompt, no
  judge model to pin, fully deterministic given the reader output.
- **Pro**: the real-video anchor is free — a paired `*_gt.mp4` already sits
  next to every prediction, so the anchor is a `--pattern` switch (§4.3).
- **Con**: it is **gated by the reader's keypoint error**. The A1X reader is
  24.4 mm RMS; the ALOHA reader is 84.9 mm. At 84.9 mm the thresholds are so
  loose that nothing is flagged — which is exactly what we measured (§6.3).
- **Con**: **unpublished.** No arXiv entry, no BibTeX, no external numbers.
  `grep -ri 'arxiv|cite|bibtex' README.md docs/` over the metric branch returns
  exactly one hit, and it refers to *our own* in-progress paper
  (`README.md:15`). See §9 for the SOTA consequence.

---

## 2. Exact metric definition, with citations

All citations are to the metric checkout, i.e.
`metrics/kiva/src/kinescore/...` == `origin/quang/hallucination-official`
@ `5225ce7`.

### 2.1 Pixels to keypoints

`cmd_score.py:121-129` is the whole read path:

```python
clip = resolve_timebase(path, view_layout=layout)      # width, height, dt
cell.view.check_frame_size(clip.width, clip.height)    # HARD REJECT on mismatch
frames = load_rgb(clip, max_frames=args.max_frames)
out = reader.read(frames.to(device), frame_chunk=args.frame_chunk)
ClipContext(dt=clip.dt, P=out.P.float().cpu(), robot=robot, ...)
```

The reader is `DINOv3 ViT-L/16 @ 768 px, 2x2 patch pool` →
`dinov3_vitl16@768:p2` (`backbones/default.py:12-15`,
`backbones/dino.py:20-22`), frozen, followed by a 19.8 M-parameter diffusion
head sampling `P (T, K, 3)` in metres with **DDIM 10 steps x 5 samples**
(`docs/MODEL.md §1 step 10`). Steps 1-9 have no learned parameters.

`K` is per robot:

| robot | `K` | keypoints | source |
|---|---|---|---|
| `a1x_ee` (Galaxea A1X) | **4** | `arm_link6`, `gripper_link`, `gripper_finger_link1`, `gripper_finger_link2` | `robots/a1x_ee/spec.py:32-34` |
| `aloha_bimanual` (ALOHA 2xvx300s) | **18** | 9 per arm | `robots/aloha/spec.py:63` |
| `fourier_gr1` (Fourier GR-1) | **22** default, **12** arms-only | 6 arm + 5 fingertip per side, or 6 arm per side | `robots/gr1/spec.py:75-95` |

`cmd_score.py:117-119` is the reconciliation: if the loaded checkpoint predicts
fewer keypoints than the robot spec builds, the robot is **rebuilt at the
reader's keypoint count**. This is the fix in commit `ef760d1` and it is why the
GR-1 benchmark reader works at all — see §3.2.

### 2.2 Per-frame detector scores

Let `P` be `(T, K, 3)` in metres and `dt` the clip timebase in seconds.

**`rigidity`** (`detectors.py:176-187`) — for each pair `(a, b)` in
`robot.rigid_bone_pairs`:

```
L[t, i] = 1000 * || P[t, a_i] - P[t, b_i] ||          # mm
rest[i] = 1000 * robot.rigid_bone_lengths[i]          # mm, from the URDF
score[t] = max_i | L[t, i] - rest[i] |                # mm, worst bone this frame
```

`rigid_bone_pairs` already drops degenerate (near-zero rest length) bones. For
`a1x_ee` the four bones are `[(0,1), (1,2), (1,3), (2,3)]` and the rest lengths
are **not read from a URDF at all** — they are derived from the hard-coded
`EE_OFFSETS_M` constant (`robots/a1x_ee/spec.py:19-46`). That matters: the A1X
"rigidity" is the gripper-triangle geometry only.

**`jerk`** (`detectors.py:195-204`) — third central difference:

```
j[t] = P[t] - 3 P[t-1] + 3 P[t-2] - P[t-3]            # (K, 3), t >= 3
score[t+2] = 1000 * max_k || j[t, k] || / dt**3        # mm/s^3
score[0..1] = 0                                        # and all-zero if T < 4
```

Division by `dt**3` is deliberate: thresholds calibrated at one frame rate
apply to clips at another, so a per-frame difference "would rank the
slowest-sampled generator as the jerkiest" (`docs/METRICS.md:81-84`).

**`teleport`** (`detectors.py:212-219`): `1000 * max_k ||P[t]-P[t-1]|| / dt`,
`score[0] = 0`.

**`joint_limit`** (`detectors.py:244-262`): bend angle at each interior
keypoint, `acos` of the normalised dot of successive bone vectors, in degrees;
score is how far outside the GT `[q0.01, q0.99]` per-joint envelope it falls.
The envelope is fitted in `fit()` from the calibration clips
(`detectors.py:251-254`).

**`self_collision`** (`detectors.py:288-297`): minimum distance in mm over all
pairs `(i, j)` with `j >= i + 2`. **`higher_is_worse = False`**
(`detectors.py:282`) — a *smaller* value is the violation.

### 2.3 Segments, and the verdict

`bounds` (`segments.py:11-18`) cuts the clip into **consecutive, non-overlapping
16-frame windows**, end inclusive, last window short:

```python
SEGMENT_LEN = 16                                        # segments.py:8
[(i, min(i + length, n_frames) - 1) for i in range(0, n_frames, length)]
```

So `n_segments = ceil(T / 16)`. Each window reduces to one number
(`Detector.reduce_window`, `detectors.py:63-67`):

```python
"median" -> float(np.median(window))                    # rigidity
otherwise -> max(window) if higher_is_worse else min(window)   # jerk, teleport, ...
```

and the verdict is that reduced value against the **segment** threshold
(`segments.py:40-42`):

```python
value    = detector.reduce_window(window)
violated = value > threshold if detector.higher_is_worse else value < threshold
```

### 2.4 Calibration — where the threshold comes from

`cmd_score.py:131-139`:

```python
real_root = cell.reader.train_tree / "videos" / "val"   # the reader's OWN val split
real      = _clips(str(real_root), args.calibration_clips)   # default 24
scorer.calibrate([read(p) for p in real], pct=args.percentile)   # default 95.0
```

`Detector.calibrate` (`detectors.py:81-96`) sets **two** thresholds from the
same pooled real clips:

```python
self.threshold         = self._cut(pooled_per_frame_scores,  pct, floor)
self.segment_threshold = self._cut(pooled_segment_scores,    pct, 0.0)
```

with `_cut` (`detectors.py:76-79`):

```python
higher_is_worse  ->  max(floor, percentile(scores, pct))
otherwise        ->  percentile(scores, 100 - pct)
```

The **per-detector floor** (`scorer.py:26`) is
`{"rigidity": 18.0, "joint_limit": 3.0}` and applies only to the *per-frame*
threshold: "Real motion is rigid up to float error, so without a floor the
rigidity threshold would collapse onto the reader's own jitter and flag
everything" (`docs/METRICS.md:33-35`). **For the A1X reader this floor is
active** — every run we measured reports a per-frame rigidity threshold of
*exactly* `18.0` (§6.1), i.e. the data-driven p95 is below the floor. The
segment threshold takes **no floor** (`detectors.py:96`), so it is purely
data-driven (8.34 mm for A1X).

Two consequences you must carry:

1. **A threshold is not a physical constant.** It is the p95 of *this reader's*
   output on *this reader's* val clips. A reader with worse `val_mm` gets looser
   thresholds, so numbers compare across cells **only when the readers are
   comparable** (`docs/METRICS.md:37-40`). They are not, across our three
   embodiments.
2. **Calibration is re-run on every invocation** and is not bit-reproducible
   (fp16 autocast + diffusion sampling). We measured 3-4 % drift on the
   identical 12-clip set — see §8.3. This is the single biggest threat to a
   cross-variant comparison.

### 2.5 Output files

`cmd_score.py:181-209` writes, per `(variant, embodiment, split)`:

| file | one row per | key columns |
|---|---|---|
| `summary.json` | run | `cell_id`, `reader_id`, `checkpoint`, `checkpoint_sha256`, `n_calibration_clips`, `percentile`, `thresholds{frame, segment, reduce}` |
| `segments.csv` | **16-frame segment** | `id`, `segment`, `start_frame`, `end_frame`, and per detector `_reduce`, `_value`, `_threshold` (the **segment** threshold), `_violated` (`table.py:54-71`) |
| `metrics.csv` | clip | per detector `n_segments`, `n_violated`, `worst_segment`, `frame_fraction`, `severity_median` (`table.py:14-31`) |
| `results.jsonl` | clip | full per-frame series + flagged intervals; **this is the big file** |
| `render/` | clip | overlay mp4s — large and slow, always produced, so our launcher deletes them unless `HAL_RENDER=1` (`metrics/kiva/run.sh:240-243`; there is no `--no-render` on this branch) |

`metrics/kiva/aggregate.py` reads **`segments.csv`** only.

---

## 3. Inputs

### 3.1 `-official` vs `-humanoid` — resolution

Both remotes exist. `git fetch --all` then:

```
$ git rev-list --left-right --count \
    origin/quang/hallucination-official...origin/quang/hallucination-humanoid
12      0
```

**`-official` is 12 commits ahead of `-humanoid` and 0 behind.** It is a strict
superset: `-humanoid` is an ancestor. So this is not a fork to choose between —
`-official` *is* `-humanoid` plus 12 commits. Diff summary (34 files,
+2266/-688):

| change on `-official` | why it matters to us |
|---|---|
| `ef760d1` **fix: score GR-1 with the 12-keypoint benchmark reader** | **Required.** Without it, humanoid scoring dies with an `IndexError`: the GR-1 spec now builds 22 keypoints (fingertips were added) but the benchmark reader predicts 12, so the rigidity detector indexed past the reader's output. Adds `GR1Spec(n_keypoints=...)` + the `cmd_score.py:117-119` reconciliation. |
| `violations/{detectors,scorer,segments}.py` rewritten | The whole segment-verdict + per-detector-floor machinery of §2.3-2.4 is `-official` only. `-humanoid` has a different, per-frame-centric scoring path. |
| docs consolidated (`METRICS.md` trimmed to rigidity+jerk, `RUNBOOK.md`, `MODEL.md`, `ADDING_A_ROBOT.md` added) | `HEADLINE = (rigidity, jerk)` is documented only on `-official`. |
| `scripts/build_method_val.py`, `scripts/score_agreement.py` added | Not used by us, but they are the upstream human-agreement hooks. |
| `robots/gr1/{fk,spec}.py` reworked (+245 lines) | Fingertip keypoints. |
| **removed** on `-official`: `docs/MULTIVIEW_STATIC.md`, `docs/BIMANUAL.md`, `docs/SINGLEARM.md` | Folded into `RUNBOOK.md`. Nothing functional lost. |

**We use `origin/quang/hallucination-official`, pinned at `5225ce7`.** It is vendored into
`metrics/kiva/` (`src/kinescore`, `configs/`, `docs/`, `scripts/`); to re-pin, re-run the
`git archive` command in `metrics/kiva/VENDORED.md` with the new SHA.

### 3.2 On-disk input format — mp4 **directly**

This is the biggest practical difference from the multiview metric: **there is
no frame-extraction step.** `--videos` is searched recursively for the
`--pattern` glob and each mp4 is decoded in-process
(`cmd_score.py:44-49, 121-129`).

The hard constraint is **frame size**. `ViewSpec.check_frame_size`
(`registry/views.py:67-90`) raises on any mismatch, and `cmd_score.py:166-174`
catches it and records the clip as `error` in `results.jsonl` instead of scoring
it — so a wrong-sized tree produces an *empty-looking* but *successful* run.
Required sizes, from `configs/views.yaml`:

| embodiment | view | required frame size | reader |
|---|---|---|---|
| `single_arm` | `sv1_4x3` | **640x480** | `a1x_ee.a1x_sv.sv1_4x3` |
| `bimanual` | `sv1_16x9` | **768x432** | `aloha_bimanual.bimanual_sv.sv1_16x9` |
| `humanoid` | `sv1_16x9` | **768x432** | `fourier_gr1.humanoid_sv.sv1_16x9` |

`metrics/kiva/probe_clips.py` exists exactly to check this before you
burn GPU hours. It `ffprobe`s every clip and reports scorable / wrong-size /
too-short counts per role, plus the estimated segment count. **Run it first.**

Our variant trees are laid out as:

```
sampling_dataset/<variant>/<embodiment>/output/singleview/dreamgen/<split>/
    <iter_tag>/<episode>.mp4        <- prediction
    <iter_tag>/<episode>_gt.mp4     <- paired real ground truth
```

with `<split>` in `{makovian, non_makovian}`. The launcher composes exactly this
path (`metrics/kiva/run.sh:267`).

### 3.3 The two roles that share every directory

A prediction and its real ground truth sit **side by side in the same directory
at the same resolution**. So the real-video anchor is a `--pattern` switch, not
a second tree (`metrics/kiva/run.sh:189-195`):

| `ROLE` | `--pattern` | selects |
|---|---|---|
| `pred` (default) | `*[0-9].mp4` | `<episode>.mp4`, the generated clip |
| `gt` | `*_gt.mp4` | `<episode>_gt.mp4`, the real clip |

`*[0-9].mp4` rather than `*.mp4` is load-bearing: `*.mp4` would sweep the `_gt`
clips into the prediction run and silently halve the measured violation rate.

Minor asymmetry, measured: `single_arm/makovian` has **120 predictions but 119
`_gt`** in every variant. One episode has no paired GT. Every other
`(embodiment, split)` cell is exactly paired.

---

## 4. Checkpoints

### 4.1 The reader inventory — present vs missing

Store root `$HAL_STORE` (default
`$KINESCORE_STORE`, else
`/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/triquang/kinescore-store`), with
`ckpt/ assets/ data/ cache/`. All three benchmark readers are **present**:

| embodiment | file in `ckpt/` | sha256 (first 16) | size | `val_mm` | status |
|---|---|---|---|---|---|
| `single_arm` | `a1x_ee.a1x_sv.sv1_4x3.pt` | `5e463b70358e53b5` | 79 244 723 B | **24.44** | **present, benchmark** |
| `bimanual` | `aloha_bimanual.bimanual_sv.sv1_16x9.pt` | `663a39a11e0f0ef0` | 79 275 063 B | **84.92** | **present, benchmark** |
| `humanoid` | `fourier_gr1.humanoid_sv.sv1_16x9.huber_old.pt` | `c40fc00688a53179` | 79 262 931 B | *no train log* | **present, benchmark** |

`val_mm` is the final `val_mm` in the sibling `*.train_log.jsonl` (measured by
us from the store, not quoted from a doc). Compare `val_mm` only **within** a
reader (`README.md:119-120`).

**The humanoid checkpoint is the trap.** There are five GR-1 `sv1_16x9`
checkpoints in the same directory:

| file | sha256 (16) | `val_mm` | K | use it? |
|---|---|---|---|---|
| `...huber_old.pt` | `c40fc00688a53179` | — (no log emitted) | **12** | **YES — the benchmark reader** |
| `...huber_newsampler.pt` | — | 44.45 | 22 | no — different training |
| `...pseudohuber_newsampler.pt` | — | 44.55 | 22 | no |
| `...l2norm_newsampler.pt` | — | 47.78 | 22 | no |
| `...mse.pt` | — | 67.86 | 22 | no |

The four `newsampler`/`mse` checkpoints have **better `val_mm` and are still the
wrong choice**: they predict `K = 22` (arms + fingertips) and are separate
trainings, so they are not benchmark-consistent with the published tables. Our
launcher defaults to `huber_old.pt` (`metrics/kiva/run.sh:177`) and the commit
message of `ef760d1` is the authority: "The humanoid benchmark reader predicts
the 12 arm keypoints." Override with `HAL_CKPT_HUMANOID` if you deliberately
want a K=22 ablation — but then you cannot compare to anything else.

**Missing / not applicable** for our three cells: nothing. Also present in the
store but irrelevant here: `fourier_gr1.humanoid_sv.sv1_4x3.pt`,
`franka_panda.single_arm_mv.{mv3_row,mv4_grid_br_blank,mv4_grid_static,sv1_4x3}.pt`
(the multiview-static Franka readers, `val_mm` 157.8-161.4 — far too coarse to
be useful).

### 4.2 The other three things the store must have

`metrics/kiva/run.sh:132-138` hard-fails if any of these is absent:

| subdir | holds | notes |
|---|---|---|
| `ckpt/` | the readers above | |
| `assets/` | URDF + mesh trees | GR-1 alone is ~285 MB upstream; the store carries the 29 MB `gr1t1` subtree. `assets/MANIFEST.json` pins `urdf_sha256` per robot (GR-1: `fc4a03e2...`, upstream `FFTAI/Wiki-GRx-Models` @ `7d96c758`). **Rigidity rest lengths come from here**, so a URDF swap silently changes every rigidity number. |
| `data/trees/<reader_id>/videos/val/` | **the real calibration clips** | Not optional: with no clips here, `score` exits (`cmd_score.py:132-136`). Present counts: A1X **12**, ALOHA **29**, GR-1 **26**. |
| `cache/hf/` | the DINOv3 weights | `models--facebook--dinov3-vitl16-pretrain-lvd1689m`. We run with `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` (`metrics/kiva/run.sh:158-159`), so this cache must be warm. |

**Note the A1X calibration set is only 12 clips** (158-180 frames each, 640x480,
verified by `ffprobe`), so `CALIB_CLIPS=24` silently calibrates on 12. That is
the number that appears in `summary.json`.

### 4.3 Is there a judge / VLM model? No.

**There is no VLM judge and no judge-model version to pin.** Checked three ways:

1. `README.md:5-8` states it outright: "with no simulator and no VLM judge".
2. `find metrics/kiva -name '*vlm*'` → **zero hits**. There is no
   `cmd_vlm.py`, despite `.env.example` reserving a `KINESCORE_MODEL_DIR` for
   "Vision-language model weights for `kinescore vlm`" — that subcommand does
   not exist on this branch.
3. The full CLI surface is `cache, data, export, ledger, models, pull, push,
   readers, render, report, score, train` (`cli/cmd_*.py`). No judge.

The only *model* in the loop is the frozen backbone, and it **is** pinned:

| | |
|---|---|
| backbone | `facebook/dinov3-vitl16-pretrain-lvd1689m` (`backbones/dino.py:21`, `backbones/default.py:13`) |
| input / pooling | 768 px, 2x2 patch pool → identity string `dinov3_vitl16@768:p2` (`backbones/default.py:42-45`) |
| frozen | `requires_grad_(False)`, eval, fp16 autocast (`docs/MODEL.md §2`) |

That identity string is written into both the feature cache and the checkpoint,
"so tokens produced by a different configuration cannot be mixed up"
(`docs/MODEL.md §2`). Incidentally, a `models--google--gemma-4-31B-it` tree
exists in the same HF cache — it belongs to **other** work in this workspace and
is **not** used by this metric.

### 4.4 How to obtain each

From the branch README (`README.md:49-76`), token-free public bucket
`hf://buckets/twanghcmut/hallucinate-bench`:

```bash
# shared URDF + mesh trees
hf sync hf://buckets/twanghcmut/hallucinate-bench/asset $KINESCORE_ASSETS

# per embodiment: the reader, then the real clips its thresholds are fitted on
for R in a1x_ee.a1x_sv.sv1_4x3 \
         aloha_bimanual.bimanual_sv.sv1_16x9 \
         fourier_gr1.humanoid_sv.sv1_16x9; do
  hf buckets cp hf://buckets/twanghcmut/hallucinate-bench/train/$R/diffusion/checkpoint.pt \
      $KINESCORE_CKPT_DIR/$R.pt
  hf sync hf://buckets/twanghcmut/hallucinate-bench/train/$R/val \
      $KINESCORE_DATA_ROOT/trees/$R/videos/val
done
```

**Caveat we could not resolve offline:** the bucket path for the humanoid is
`.../fourier_gr1.humanoid_sv.sv1_16x9/diffusion/checkpoint.pt` → `$R.pt`, i.e.
it lands as `fourier_gr1.humanoid_sv.sv1_16x9.pt` with **no `.huber_old`
suffix**. Our store has the suffixed file and no unsuffixed one. Whether the
bucket's unsuffixed artifact is byte-identical to our `huber_old.pt`
(`c40fc00688a53179`) is **UNVERIFIED** — we ran fully offline. If you re-download,
sha256 it against `c40fc00688a53179` before trusting a comparison. Listed in §10.

To train a reader from scratch (only for a robot/view with no checkpoint):

```bash
kinescore pull  --what train
kinescore data  --reader $R
kinescore cache --reader $R --device cuda
kinescore train --reader $R --device cuda --steps 8000
```

---

## 5. Running it

### 5.1 One-time setup

Nothing to install. The code is vendored in `metrics/kiva/` and imported via
`PYTHONPATH=metrics/kiva/src` (`metrics/kiva/run.sh`), using an existing conda env — see §8.1 for
why `uv sync` is not an option here.

```bash
# optional: point at a different store / interpreter / output root
export KINESCORE_STORE=/pfss/.../triquang/kinescore-store   # default
export HAL_PYTHON=/pfss/.../miniconda3/envs/hdmi/bin/python  # default
export HAL_OUT_ROOT=/tmp/hal_results                         # see §8.2 (quota!)
```

### 5.2 Pre-flight the clips (cheap, CPU-only, do this first)

```bash
python metrics/kiva/probe_clips.py \
  --variant dense pisa radial svg1 svg2 --json /tmp/hal_ready.json
```

### 5.3 Score the real-video anchor — **do this before the predictions**

```bash
HAL_OUT_ROOT=/tmp/hal_results ROLE=gt \
  VARIANTS="dense pisa radial svg1 svg2" \
  EMBODIMENTS=single_arm SPLITS=makovian MAX_CLIPS=6 \
  bash metrics/kiva/run.sh --tmux
```

### 5.4 The multi-variant comparison — one invocation

```bash
HAL_OUT_ROOT=/tmp/hal_results \
  VARIANTS="dense pisa radial svg1 svg2" \
  EMBODIMENTS=single_arm SPLITS=makovian \
  bash metrics/kiva/run.sh --tmux
# attach: tmux attach -t hal_<HHMMSS>
# log:    logs/exp_logs/tmux_logs/hal_<HHMMSS>.log
```

Full sweep, all ten variants, all three embodiments, both splits:

```bash
HAL_OUT_ROOT=/tmp/hal_results \
  VARIANTS="dense pisa radial svg1 svg2 itm sito dicache fastercache worldcache" \
  bash metrics/kiva/run.sh --tmux
```

Single explicit directory (bypasses the variant-tree layout):

```bash
VIDEOS_DIR=sampling_dataset/dense/humanoid/output/singleview/dreamgen/makovian \
  EMBODIMENTS=humanoid OUT_DIR=/tmp/hal_results/single/dense.humanoid \
  bash metrics/kiva/run.sh
```

### 5.5 Leaderboard

```bash
python metrics/kiva/aggregate.py \
  --results-root /tmp/hal_results \
  --variants dense pisa radial svg1 svg2 \
  --embodiment single_arm \
  --anchor-role gt --baseline real \
  --json /tmp/hal_results/summary.json
```

All five detectors instead of the two headline ones:

```bash
python metrics/kiva/aggregate.py --results-root /tmp/hal_results \
  --variants dense svg1 --embodiment single_arm \
  --detectors rigidity jerk teleport joint_limit self_collision \
  --anchor-role gt --baseline real
```

`--embodiment` is effectively mandatory. Pooling embodiments mixes three
different readers with thresholds that differ by **two orders of magnitude**
(A1X segment jerk 355 k vs ALOHA 3.47 M mm/s³, §6.1), so a pooled mean is
meaningless. The aggregator warns `(pooled -- mixes readers!)` when you omit it.

`--intersect` is **on by default**: only clips present in *every* requested
variant are compared, so `clips` is identical across rows.

### 5.6 Cost, measured

On one idle A100-SXM4-80GB, `FRAME_CHUNK=16`, ~5.2 GB VRAM at peak:

| stage | measured |
|---|---|
| calibration, 12 A1X clips (~170 frames each) | **~70 s** |
| scoring, per A1X clip (~190 frames) | **~11 s** |
| one `(variant, embodiment, split)` cell, 8 clips **incl.** calibration | **2 min 43 s** – **3 min 12 s** (three separate runs, §6) |
| the `render/` overlay pass we delete | a significant share of the above; `HAL_RENDER=1` to keep it |

**Calibration is re-paid on every cell**, so cost is roughly
`n_cells x (70 s + 11 s x n_clips)`. A full 10-variant x 3-embodiment x 2-split
sweep at ~130 clips/cell is 60 cells x ~25 min ≈ **25 GPU-hours**. Use tmux.

---

## 6. Smoke-test results (actually measured)

Date: 2026-10-04 / 2026-10-05. GPU: one idle A100-SXM4-80GB, ~5.2 GB peak.
Metric checkout `metrics/kiva` @ `5225ce7`. Interpreter: the `hdmi`
conda env (`torch 2.7.0+cu128`, `transformers 5.2.0`, `numpy 1.26.4`).
`FRAME_CHUNK=16`. Outputs under `/tmp/hal_results`, `/tmp/hal_agg`,
`/tmp/hal_smoke` (**not** under the repo — §8.2).

> **Every number below is n ≤ 8 clips. They are a plumbing proof and a scale
> anchor, NOT a result. Do not quote them as a comparison.** What they do
> establish is listed at the end of each subsection.

### 6.1 Calibrated thresholds, per reader

From `summary.json`. These are the quantities everything else is judged against.

| reader | `n_cal` | rigidity frame / **segment** | jerk frame / **segment** (mm/s³) | teleport seg | joint_limit seg | self_coll seg |
|---|---|---|---|---|---|---|
| `a1x_ee...sv1_4x3` (K=4) | 12 | **18.00** / 8.29–8.71 mm | 247 k–249 k / **359 k–390 k** | 646–669 | 3.90–4.97 | 29.5–30.1 |
| `aloha_bimanual...sv1_16x9` (K=18) | 24 | 56.0–56.5 / **54.6–56.2 mm** | 1.27 M–1.28 M / **3.47 M–3.50 M** | 1890–1907 | 7.99–8.49 | 41.4–41.9 |
| `fourier_gr1...sv1_16x9.huber_old` (K=12) | 24 | 43.4–43.5 / **41.9–42.8 mm** | 456 k–458 k / **731 k–732 k** | 857–864 | 11.5–12.6 | 159.2–159.5 |

Two things to read off this table:

1. **The A1X per-frame rigidity threshold is exactly `18.00` in every single
   run** — that is `_CALIBRATION_FLOOR["rigidity"]` (`scorer.py:26`), so the
   data-driven p95 is *below* the floor and the per-frame rigidity threshold is
   not data-driven at all for A1X. The segment threshold (no floor) is ~8.4 mm.
2. **Thresholds scale with reader error, over an enormous range.** The ALOHA
   segment rigidity threshold (55 mm) is **6.6x** the A1X one (8.4 mm), and its
   jerk threshold is **9.5x**. That is `val_mm` 84.9 vs 24.4 showing up directly.
   This is why §5.5 forbids pooling embodiments.

### 6.2 The discriminative cell — single_arm (A1X), 5 variants, anchored on real video

`EMBODIMENTS=single_arm SPLITS=makovian MAX_CLIPS=6 CALIB_CLIPS=12`,
`VARIANTS="dense pisa radial svg1 svg2"`, one `metrics/kiva/run.sh` invocation
per role. **6 clips intersected, 72 segments each.** Wall time for the
5-variant prediction sweep: **13 min 37 s**.

```
variant         clips    segs   rigidity_rate    ±sem   rigidity_sev       jerk_rate    ±sem       jerk_sev
----------------------------------------------------------------------------------------------------------
svg1                6      72          0.0139  0.0139         0.4219          0.4861  0.0545         0.9991
svg2                6      72          0.0139  0.0139         0.4246          0.4861  0.0624         0.9922
dense               6      72          0.0278  0.0176         0.4916          0.4861  0.0586         0.9841
pisa                6      72          0.0556  0.0412         0.4209          0.4444  0.0556         0.9514
radial              6      72          0.0972  0.0817         0.4649          0.3056  0.0465         0.8146
```

A second, independent 8-clip run from 2026-10-04 (`/tmp/hal_agg`, 88–96
segments) with the **real-video anchor in the same table**:

```
variant         clips    segs   rigidity_rate    ±sem        Δ   rigidity_sev       jerk_rate    ±sem        Δ       jerk_sev
-----------------------------------------------------------------------------------------------------------------------------
real *              8      88          0.0000  0.0000  +0.0000         0.3604          0.0909  0.0298  +0.0000         0.6793
dense               8      96          0.0208  0.0136  +0.0208         0.4826          0.5208  0.0516  +0.4299         1.0175
svg1                8      96          0.0208  0.0208  +0.0208         0.4421          0.4896  0.0429  +0.3987         0.9850
```

and all five detectors on the same 8 clips:

```
variant       rigidity_rate   jerk_rate   teleport_rate   joint_lim_rate   self_coll_rate
-----------------------------------------------------------------------------------------
real *               0.0000      0.0909          0.0568           0.0568           0.0000
dense                0.0208      0.5208          0.2500           0.1250           0.0312
svg1                 0.0208      0.4896          0.3125           0.0729           0.0104
```

**What this establishes.** The metric is discriminative on A1X: generated video
shows a jerk violation rate of **0.44–0.52** against the real anchor's
**0.0909** — a 5x separation, far outside the ±0.06 sem. `teleport` separates
too (0.25–0.31 vs 0.057). `rigidity` is near the floor for everything
(0.014–0.097 vs 0.0) and its ±sem is as large as its mean, so **`rigidity` is
not usable as a discriminator at this n** — on A1X the only "rigid" structure is
the 4-keypoint gripper triangle (§2.2), which the reader gets nearly right.

### 6.3 The non-discriminative cells — bimanual and humanoid

Same pipeline, `MAX_CLIPS=2`, `CALIB_CLIPS=24`, from `/tmp/hal_smoke` and
`/tmp/hal_agg`. Rates are per-clip means of the segment violation fraction:

| cell | role | clips | segs | rigidity | jerk | teleport | joint_limit | self_coll |
|---|---|---|---|---|---|---|---|---|
| `single_arm` A1X | pred | 2 | 24 | 0.0417 | **0.6667** | 0.2917 | 0.0833 | 0.0000 |
| `single_arm` A1X | **gt (real)** | 2 | 22 | 0.0000 | **0.0000** | 0.0000 | 0.0000 | 0.0000 |
| `bimanual` ALOHA | pred | 2 | 41 | 0.0000 | **0.0000** | 0.0000 | 0.0862 | 0.0000 |
| `bimanual` ALOHA | **gt (real)** | 2 | 36 | 0.0000 | **0.0000** | 0.0556 | 0.0370 | 0.0370 |
| `humanoid` GR-1 | pred | 2 | 24 | 0.0000 | **0.0000** | 0.0833 | 0.0000 | 0.0000 |
| `humanoid` GR-1 | **gt (real)** | 2 | 15 | 0.0000 | **0.0000** | 0.0000 | 0.0000 | 0.0000 |

**Both headline detectors read exactly 0.0000 for bimanual and humanoid, on
generated *and* real video alike.** That is the §6.1 threshold table biting: an
84.9 mm reader produces a 55 mm rigidity threshold and a 3.5 M mm/s³ jerk
threshold, which nothing clears. **As configured, only the `single_arm` (A1X)
cell can answer a comparison question.** Raising `n` may recover some signal in
the tail, but the right fix is a better bimanual/humanoid reader, not more clips.

> A stale `/tmp/hal_agg/humanoid.json` written on 2026-10-04 14:39 shows
> `rate = 1.0` for both humanoid pred and gt. **That file is wrong** — it was
> produced by an earlier revision of `metrics/kiva/aggregate.py` that
> truthiness-tested the `*_violated` CSV column, and the string `"False"` is
> truthy in Python. Re-aggregating the identical directories with the committed
> script gives `0.0000`, which matches a direct count of the CSV
> (`{('False','False'): 24}`). The file is left in place only as the audit
> trail; do not read numbers out of it.

### 6.4 Negative control — deliberately broken video

To check the metric fires on obvious body hallucination, two A1X clips were
corrupted (`/tmp/hal_broken`): `episode_000000_shuffled.mp4` (frames randomly
permuted) and `episode_000000_pingpong.mp4` (forward-then-reverse). Scored
through the same launcher, `n=2`, 32 segments:

```
variant         clips    segs   rigidity_rate   rigidity_sev       jerk_rate       jerk_sev
-------------------------------------------------------------------------------------------
real *              8      88          0.0000         0.3604          0.0909         0.6793
dense               8      96          0.0208         0.4826          0.5208         1.0175
broken              2      32          0.0000         0.3513          0.6905         3.8467
```

**`jerk_severity` is the number that moves: 0.68 (real) → 1.02 (generated) →
3.85 (broken), a 5.7x span.** The rate saturates (0.69) but the severity keeps
scaling, which is the argument for reporting both. Note `rigidity` stays at
0.0000 even for shuffled frames — frame order does not change bone lengths, so
this is correct behaviour, and it is another demonstration that `rigidity` and
`jerk` are measuring genuinely different things.

### 6.5 Calibration drift — measured, and it changes the ranking

The five prediction cells in §6.2 each re-calibrated on the **identical** 12
A1X val clips. The resulting segment thresholds:

| cell | rigidity | jerk | teleport | joint_limit | self_coll |
|---|---|---|---|---|---|
| `dense` | 8.29 | 389 558.17 | 668.57 | 4.80 | 29.88 |
| `pisa` | 7.90 | 370 646.96 | 665.08 | 4.20 | 29.70 |
| `radial` | 8.56 | 377 170.48 | 655.84 | 4.11 | 29.91 |
| `svg1` | 8.71 | 359 166.84 | 647.99 | 3.90 | 30.06 |
| `svg2` | 8.44 | 360 240.89 | 645.99 | 4.97 | 29.46 |
| **spread (max−min)/mean** | **9.67 %** | **8.18 %** | 3.44 % | **24.34 %** | 2.01 % |

The branch RUNBOOK says "calibration drifts ~1% across GPUs"
(`docs/RUNBOOK.md:53-55`). **On one GPU, same clips, same seed-free code path,
we measure 8–10 % on the two headline detectors and 24 % on `joint_limit`.**
Each variant is therefore judged against a *different* ruler.

Re-scoring every variant's `segments.csv` values against **one frozen
threshold** (the `dense` run's: rigidity 8.29, jerk 389 558.17) on the same 6
intersected clips:

| variant | rigidity own → frozen | jerk own → frozen |
|---|---|---|
| `dense` | 0.0278 → 0.0278 | 0.4861 → 0.4861 |
| `pisa` | 0.0556 → 0.0556 | 0.4444 → **0.3889** |
| `radial` | 0.0972 → 0.0972 | 0.3056 → **0.2639** |
| `svg1` | 0.0139 → **0.0278** | 0.4861 → **0.3889** |
| `svg2` | 0.0139 → 0.0139 | 0.4861 → **0.3889** |

**The ranking changes.**

```
rigidity  own    : svg1 < svg2 < dense < pisa < radial
rigidity  frozen : svg2 < dense < svg1 < pisa < radial      <- svg1 drops 1 -> 3
jerk      own    : radial < pisa < dense < svg1 < svg2
jerk      frozen : radial < svg1 < pisa < svg2 < dense      <- dense drops 3 -> 5
```

**This is the most important measured result in this document.** It means the
harness as it stands cannot be trusted for a cross-variant ranking: a ~10 %
threshold wobble is the same size as the between-variant differences. §8.3
states the fix.

### 6.6 The anchored 5-variant leaderboard, and the metric's own noise floor

The completed run (`/tmp/hal_results`, `pred` + `gt` roles, 6 clips/variant,
13 min 37 s + 13 min 00 s) with the real-video anchor in the table. The anchor
row pools the same 6 `_gt.mp4` clips scored once per variant run, so **n = 30**:

```
variant         clips    segs   rigidity_rate    ±sem        Δ   rigidity_sev       jerk_rate    ±sem        Δ       jerk_sev
-----------------------------------------------------------------------------------------------------------------------------
real *             30     330          0.0000  0.0000  +0.0000         0.3804          0.0939  0.0154  +0.0000         0.6672
svg1                6      72          0.0139  0.0139  +0.0139         0.4219          0.4861  0.0545  +0.3922         0.9991
svg2                6      72          0.0139  0.0139  +0.0139         0.4246          0.4861  0.0624  +0.3922         0.9922
dense               6      72          0.0278  0.0176  +0.0278         0.4916          0.4861  0.0586  +0.3922         0.9841
pisa                6      72          0.0556  0.0412  +0.0556         0.4209          0.4444  0.0556  +0.3505         0.9514
radial              6      72          0.0972  0.0817  +0.0972         0.4649          0.3056  0.0465  +0.2116         0.8146
```

**This independently reproduces the §6.2 anchor**: `jerk_rate` 0.0939 ±0.0154
at n=30 here vs 0.0909 ±0.0298 at n=8 in the 2026-10-04 run, on different clips
and a different day. The real-video jerk floor of **≈0.09** is the most
replicated number in this document.

**And it gives the metric's own noise floor for free.** Those 30 anchor
observations are the *same 6 real clips* put through 5 independent invocations —
identical input, so all spread is the metric's own nondeterminism:

| run (which variant's invocation) | segs | rigidity_rate | **jerk_rate** | jerk_sev | rig segment thr | jerk segment thr |
|---|---|---|---|---|---|---|
| `dense` | 66 | 0.0000 | 0.1061 | 0.6683 | 8.21 | 365 844.70 |
| `pisa` | 66 | 0.0000 | 0.1212 | 0.6982 | 8.23 | 356 806.26 |
| `radial` | 66 | 0.0000 | 0.0758 | 0.6250 | 8.29 | 389 320.92 |
| `svg1` | 66 | 0.0000 | 0.0909 | 0.6680 | 8.17 | 367 225.69 |
| `svg2` | 66 | 0.0000 | 0.0758 | 0.6766 | 8.31 | 363 872.88 |
| **mean / min / max / std** | | **0.0000 / 0 / 0 / 0** | **0.0939 / 0.0758 / 0.1212 / 0.0198** | 0.6672 / 0.6250 / 0.6982 / 0.0266 | | |

**So the metric's run-to-run noise on identical input is `jerk_rate`
std = 0.0198 (±21 % relative), and `rigidity_rate` std = 0.0000.** Now the
§6.2 leaderboard can be read properly:

| comparison | observed gap | noise floor (1σ) | verdict |
|---|---|---|---|
| `radial` vs `dense` jerk (0.3056 vs 0.4861) | **0.1805** | 0.0198 | **~9σ — real** |
| `pisa` vs `dense` jerk (0.4444 vs 0.4861) | 0.0417 | 0.0198 | ~2σ — **suggestive only** |
| `svg1`/`svg2` vs `dense` jerk (all 0.4861) | 0.0000 | 0.0198 | **indistinguishable** |
| any `rigidity` ordering (0.0139–0.0972) | ≤0.0833 | 0 on real, but ±sem ≈ mean on pred | **not resolvable at n=6** |

The one genuinely robust signal at this sample size: **`radial` has a
substantially lower jerk-violation rate than the dense baseline** (0.31 vs 0.49,
~9σ against the identical-input noise floor), while simultaneously having the
*highest* rigidity rate (0.0972). That trade is worth a real run to confirm.

---

## 7. How to interpret the numbers

### 7.1 Direction and scale

- **Every number this harness prints is LOWER IS BETTER.** They are violation
  rates and violation severities, not quality scores. This is the opposite
  convention from `docs/metrics/multiview_consistency_metric.md`, where both numbers are
  in `[0,1]` higher-better. Do not put them in the same table without saying so.
- **`rate` is in `[0, 1]`**: the fraction of a clip's 16-frame segments whose
  reduced value crossed the segment threshold, averaged over clips.
- **`severity` is a ratio, unbounded above, with a meaningful 1.0**:
  `value / segment_threshold`, median over segments, inverted for
  `self_collision` so that `> 1` always means "worse than the real-motion p95"
  (`metrics/kiva/aggregate.py:180-184`). `severity = 1.0` is exactly the p95 of
  real motion. `severity = 3.85` (the broken control, §6.4) means the typical
  segment is nearly 4x past where real motion's 95th percentile sits.
- **`rate` has no meaningful zero.** The reader's own keypoint error (24–85 mm)
  puts a nonzero floor under every rate. Read a variant's rate as its **excess
  over the real-video anchor**, which is what `--anchor-role gt --baseline real`
  computes and what the `Δ` column shows.

### 7.2 The real-video anchor — this metric's "practical ceiling"

For the multiview metric the equivalent exercise found the ceiling was ~0.89
rather than 1.0. Here the **floor is not 0**:

| cell | detector | real (`_gt.mp4`) rate | generated rate | separation |
|---|---|---|---|---|
| `single_arm` A1X, **n=30** | **jerk** | **0.0939** ±0.0154 | 0.3056–0.4861 | **3.3–5.2x** |
| `single_arm` A1X, n=8 | **jerk** | **0.0909** ±0.0298 | 0.4896–0.5208 | **5.4–5.7x** |
| `single_arm` A1X, **n=30** | **rigidity** | **0.0000** ±0.0000 | 0.0139–0.0972 | floor-bound |
| `single_arm` A1X, n=8 | teleport | 0.0568 ±0.0166 | 0.2500–0.3125 | 4.4–5.5x |
| `single_arm` A1X, n=8 | joint_limit | 0.0568 ±0.0239 | 0.0729–0.1250 | 1.3–2.2x |
| `single_arm` A1X, n=8 | self_collision | 0.0000 | 0.0104–0.0312 | floor-bound |
| `bimanual` ALOHA, n=2 | jerk / rigidity | 0.0000 | 0.0000 | **none** |
| `humanoid` GR-1, n=2 | jerk / rigidity | 0.0000 | 0.0000 | **none** |

The two independent jerk-anchor measurements (0.0939 ±0.0154 at n=30;
0.0909 ±0.0298 at n=8, different clips, different day) agree, so **≈0.09 is the
A1X jerk floor**.

And in severity terms, the full scale we have measured end to end:

```
jerk_severity :  0.67  (real video)  ->  0.81-1.00  (generated)  ->  3.85  (broken control)
```

**Crucially, the metric's own noise floor is `jerk_rate` std = 0.0198** on
identical input (§6.6), so a gap below ~0.04 between two variants is not
interpretable.

**How to read the anchor.** Real video scores **jerk_rate ≈ 0.09, not 0.00**,
because ~9 % of real segments contain one frame whose reader-estimated jerk
exceeds the p95 of pooled real motion — that is the `worst`-reduce (§2.3)
interacting with reader noise, exactly as `docs/METRICS.md:44-48` predicts
("`worst` (jerk) needs one frame and lands far above" 5 %). So:

- **0.09 is the practical floor for A1X jerk, not 0.** A variant at 0.48 is not
  "48 % bad"; it is **0.39 above an achievable 0.09**.
- **rigidity's real-video anchor *is* 0.0000**, so for that detector the floor
  genuinely is zero — but every generated variant also sits within 0.01–0.10 of
  it with ±sem of the same size, so there is no headroom to measure in.

The branch's own `docs/METRICS.md:57-62` publishes one segment-level real-motion
baseline, measured by upstream on the reader's own calibration clips
(**in-sample** with the thresholds, so not directly comparable to our
held-out-ish `_gt` anchor):

| reader | segments | rigidity | jerk |
|---|---|---|---|
| `fourier_gr1.humanoid_sv.sv1_16x9` | 524 | 4.0 % | **31.3 %** |
| `aloha_bimanual.bimanual_sv.sv1_16x9` | — | — | — |
| `a1x_ee.a1x_sv.sv1_4x3` | — | — | — |

Upstream's humanoid jerk baseline of **31.3 %** is wildly inconsistent with our
measured **0.0 %** on the same reader (§6.3). Two candidate explanations — their
524-segment in-sample set vs our 15, or a different checkpoint — and we did not
resolve it. Flagged in §10.

### 7.3 What a defensible claim looks like

Given the above, the only statements this metric currently supports:

- **Supportable**: "On the A1X single-arm cell, variant X shows a jerk violation
  rate of `a` ± `s` against the dense baseline's `b` ± `s` and a real-video
  anchor of 0.09, on the same N clips, scored with reader
  `5e463b70358e53b5` under a single frozen threshold set."
- **Not supportable**: any statement about bimanual or humanoid (no signal,
  §6.3); any ranking produced with per-run re-calibration at N ≈ 6 (§6.5); any
  absolute statement like "variant X hallucinates 2 % of the time"; anything
  about external SOTA (§9).

### 7.4 Reading the aggregator output

```
variant         clips    segs   rigidity_rate    ±sem        Δ   rigidity_sev   jerk_rate ...
```

`clips` = clips in the comparison set (identical across rows under the default
`--intersect`). `segs` = total segments pooled. `±sem` = standard error of the
mean over clips. `Δ` = this variant's mean **minus the baseline's**, so
**negative = fewer violations = better**. Rows are sorted ascending by the first
detector's rate, so **the top row is the best**. `*` marks the baseline. The
trailing `reader[...]` lines report the reader id, checkpoint sha256 prefix and
calibration setting per variant — **check these match across rows** before
believing a comparison.

---

## 8. Known limitations and gotchas

1. **`uv sync` does not work on this cluster — use the conda env.** The branch's
   `pyproject.toml` pins `torch>=2.5,<3`, which `uv` resolves to torch 2.14.1 →
   triton 3.8.0, whose unpack exceeds the shared `~/.cache/uv` quota
   (`OSError: [Errno 122] Disk quota exceeded`). The launcher therefore defaults
   `HAL_PYTHON` to the `hdmi` conda env (torch 2.7.0+cu128, transformers 5.2.0,
   numpy 1.26.4) and supplies `pytorch_kinematics` via `PYTHONUSERBASE`
   (`metrics/kiva/run.sh:114-127`). This is the analogue of the multiview
   branch's missing `numba` — see §8.9 for the full undeclared-dependency list.
2. **The pfss workspace quota is exhausted; writes silently truncate.** A 20 MiB
   `dd` into the repo wrote only 4 MiB. The first smoke attempt died in
   `cmd_score.py:163` (`open(results_path, "w")` → `Errno 122`) **after
   completing all four clips** — i.e. the GPU work succeeded and the output was
   lost. `exp_logs/` alone is 245 GB. **Always set `HAL_OUT_ROOT` outside the
   workspace** (we used `/tmp/hal_results`). `results.jsonl` is the largest
   output by far since it carries every per-frame series.
3. **Calibration drift (8–24 %) can invert the ranking — §6.5.** The harness
   re-calibrates per cell, so each variant gets a different ruler. **The fix is
   a `--thresholds <summary.json>` flag that loads a frozen threshold set
   instead of refitting; it does not exist on this branch.** Until it does,
   either (a) score every variant in one process, or (b) post-process
   `segments.csv` against one frozen threshold as §6.5 does. Our aggregator
   reports the per-variant thresholds in `--json` provenance so the drift is at
   least visible.
4. **Wrong frame size is a silent near-no-op.** `check_frame_size`
   (`views.py:67-90`) raises, `cmd_score.py:166-174` catches it and records
   `error` in `results.jsonl`, and the run exits **0-ish** with an empty
   `segments.csv`. Always `metrics/kiva/probe_clips.py` first.
5. **`--pattern` defaults to `*.mp4`, which mixes predictions and GT.** Since
   `<episode>.mp4` and `<episode>_gt.mp4` share a directory, the upstream default
   scores both and halves the apparent violation rate. Our launcher forces
   `*[0-9].mp4` / `*_gt.mp4` (`metrics/kiva/run.sh:189-195`).
6. **`render/` is always produced and there is no `--no-render`.**
   `cmd_score.py:209` calls `render_results` unconditionally. The launcher
   `rm -rf`s it unless `HAL_RENDER=1`. On a quota-bound filesystem this matters.
7. **Two of five detectors are floor-bound on our data** (`rigidity`,
   `self_collision` — §7.2), and **`rigidity` is one of the two HEADLINE
   detectors.** For A1X, `rigid_bone_pairs` is just the 4-keypoint gripper
   triangle built from the hard-coded `EE_OFFSETS_M` constant, **not from a
   URDF** (`robots/a1x_ee/spec.py:19-46`), so "rigidity" there means much less
   than it does for an 18- or 22-keypoint arm.
8. **Rigidity rest lengths come from `$KINESCORE_ASSETS`.** A URDF change
   silently rescales every rigidity number. `assets/MANIFEST.json` pins
   `urdf_sha256` per robot (GR-1 `fc4a03e2...`); diff against it if numbers
   drift. The Franka URDF additionally resolves through the
   `robot_descriptions` package cache, which will **re-clone from GitHub** if
   that cache is ever cleared — not used by our three cells, but a trap nearby.
9. **Undeclared / unresolvable dependencies.** `pyproject.toml` declares
   `numpy, torch, pytorch-kinematics, robot-descriptions` + extras
   `dino/video/bench`. In practice:
   - `pytorch_kinematics` is **not** in the conda env and is injected via
     `PYTHONUSERBASE=/pfss/.../.local` (`metrics/kiva/run.sh:125-151`).
   - `ffmpeg`/`ffprobe` must be on `PATH` (`README.md:36`) but are **not** a
     declared dependency. Worse, **`/usr/bin/ffprobe` segfaults on a
     gotoblas/liblapack symbol clash once `torch` is imported in the same
     process tree.** The launcher prepends a pfss ffmpeg build **and** its
     `lib/` to `LD_LIBRARY_PATH`, without which the binary cannot resolve
     `libavdevice.so.62` (`metrics/kiva/run.sh:140-148`).
   - The DINOv3 weights are a hard requirement but are not a package; they must
     be warm in `HF_HOME` because we force `HF_HUB_OFFLINE=1`.
10. **No determinism guarantee.** fp16 autocast + a DDIM-10 x 5-sample
    diffusion head mean the keypoints — and therefore both thresholds and scores
    — vary run to run. This is the root cause of §6.5. No seed flag is exposed.
11. **`joint_limit` is calibration-dependent twice over**: its envelope is
    *fitted* on the calibration clips (`detectors.py:251-254`) **and** its
    threshold is a percentile of that fit, which is why it drifts 24 % (§6.5).
12. **Segments are non-overlapping and the last one is short**
    (`segments.py:11-18`), so a clip of 190 frames yields 12 segments of which
    the last covers 14 frames. Short final segments are weighted equally.
13. **`metrics.csv`'s per-detector `threshold` column is the PER-FRAME
    threshold**, while `segments.csv`'s is the **segment** threshold
    (`table.py:22` vs `table.py:68`). Mixing them up changes rigidity by 2x.
14. **The clip `id` is the path below `--videos` with separators replaced by
    `__`** (`cmd_score.py:52-57`), so it embeds the `iter_000090000_static16fps`
    tag. Two variants generated at different iteration tags produce
    **non-intersecting ids** and `--intersect` will find 0 shared clips. All our
    variants share tags, so this did not bite, but it will across generations.
15. **`n_clips` in `summary.json` counts *attempted*, `n_failed` counts
    rejected** — a cell can report `n_clips: 120` and still have scored nothing.
16. **The upstream RUNBOOK's reference `val_mm` table does not match the
    store.** `docs/RUNBOOK.md:41-45` lists GR-1 `K=22, val_mm 51.2` and ALOHA
    `83.5`; the store's benchmark GR-1 reader is `K=12` and its ALOHA train log
    ends at `84.92`. The RUNBOOK numbers describe different trainings.
17. **Upstream's published humanoid real-motion baseline (31.3 % jerk) conflicts
    with our measurement (0.0 %)** on the same reader — §7.2, unresolved.

---

## 9. Are we SOTA on this metric? No — and the question is not externally answerable

This is the direct answer to the driving question, and it is the same verdict as
the multiview metric, for the same structural reason plus a worse one.

### 9.1 Is the metric published? No.

| check | result |
|---|---|
| arXiv / DOI / BibTeX in the metric repo | **none.** `grep -ri 'arxiv\|cite\|bibtex\|@inproceedings' README.md docs/` returns **one** hit: `README.md:15` "the checkpoints the paper's tables were scored with" — referring to **our own unpublished paper**. |
| external leaderboard | **none found.** |
| git history | 22 commits, 2026-08-29 → 2026-10-01, single author `triquang26`, on a branch of **this** repo. |
| is the name "kinescore" a known benchmark? | No. It is this package's own name (§1.1). |

**The metric is internal and unpublished.** There is therefore **no external
number to be SOTA against.**

### 9.2 Does it derive from a published benchmark? No — not PhyGenEval, not WorldScore

This was checked explicitly because both were plausible lineages:

- **PhyGenEval**: `grep -rn -i 'phygeneval|videophy|physbench|vbench'` across the
  metric branch's `*.py`, `*.md`, `*.yaml` → **zero hits**. No shared code, no
  shared metric definition, no citation. PhyGenEval is additionally a
  **VLM-judge** benchmark, whereas this metric has **no VLM at all** (§4.3), so
  they are not even the same family. **No lineage.**
- **WorldScore**: vendored in *this* repo at `metrics/WorldScore/`, but **zero
  references from the metric branch**. And our own assessment of it concludes
  WorldScore has **"no true physics-law metric"**
  (`tmp/20261005_refactor/docs/WORLDSCORE_PHYSICS_REPORT.md:120-122`) — its physics signal is
  indirect, via motion magnitude and smoothness. So WorldScore could not supply
  a baseline for rigidity/jerk even if it were wired up. **No lineage.**
- A sibling doc, `docs/metrics/phyworld_nr_adapted_metrics.md`, covers the PhyWorld
  adaptation. That is a **different** metric family (object-dynamics based) and
  shares no code with `kinescore`.

So the hallucination metric is **not** a re-implementation of anything
published, and **published numbers from PhyGenEval, WorldScore or PhyWorld are
not usable here**: different inputs, different scales, different directions
(this one is lower-better violation rates; WorldScore's aggregates are
higher-better `[0,1]`), and no shared definition to convert through.

### 9.3 Second, independent blocker: the metric has no signal on 2 of 3 cells

Even restricted to **internal** comparison, §6.3 showed both headline detectors
read exactly `0.0000` on bimanual and humanoid for generated **and** real video.
So an internal claim can only ever be made on the A1X single-arm cell today —
one third of the benchmark.

### 9.4 Third blocker: the ranking is not yet stable

§6.5 measured 8–24 % threshold drift across runs on identical calibration
clips, which **reorders the leaderboard**. Even the A1X-only internal claim is
not safe until thresholds are frozen across variants.

### 9.5 What we can and cannot say

- ❌ **"We are SOTA on hallucination."** Not achievable with this metric. There
  is no published definition, no external baseline, and no leaderboard.
- ❌ **"We beat PhyGenEval/WorldScore numbers."** Those numbers measure
  different quantities on different scales; there is no conversion.
- ⚠️ **"Variant X hallucinates less than variant Y."** Achievable **only** on
  the A1X cell, **only** with frozen thresholds, and **only** at an N far above
  6. Not yet done.
- ✅ **"Generated video shows a 3.3–5.4x higher jerk-violation rate than real
  video through the same reader (0.31–0.49 vs 0.094 ±0.015, n=30 real / 6 per
  variant, A1X)."** Measured twice independently, anchored, and defensible as a
  *characterisation of the gap to real* — arguably the more interesting
  scientific statement than a variant ranking.
- ✅ **"`radial` reduces the jerk-violation rate relative to dense (0.31 vs
  0.49), a gap ~9x the metric's own identical-input noise floor of 0.0198."**
  The one variant difference that survives §6.6's noise analysis at n=6 — but it
  still needs the full run and frozen thresholds before publication.
- ✅ **A controlled internal statement** of the form "variant X preserves N % of
  the dense baseline's jerk-violation excess over real, at Y x speedup", once
  §9.3 and §9.4 are addressed.

**To make an external SOTA claim possible** you would have to either publish
this metric with a reference table others can score against, or re-score our
variants under an already-published physics-plausibility benchmark. Neither is
in scope here.

---

## 10. Status — what is verified, what is not

### 10.1 Resolved questions

**`-official` vs `-humanoid`** — resolved, §3.1. `-official` is 12 commits ahead
and 0 behind; `-humanoid` is its ancestor. We pin `-official` @ `5225ce7`.
`-official` carries the mandatory GR-1 12-keypoint fix (`ef760d1`) and the whole
segment-verdict machinery.

**What `kinescore` is** — resolved, §1.1. It is the *implementation* of the
hallucination metric, not a separate metric. Same thing, two names.

**Is there a judge / VLM?** — resolved, §4.3. **No.** No VLM, no judge model, no
version to pin. The only model is the frozen DINOv3 backbone
`facebook/dinov3-vitl16-pretrain-lvd1689m` @ 768 px, p2 pool. The
`kinescore vlm` subcommand that `.env.example` alludes to does not exist on this
branch.

**Which humanoid checkpoint** — resolved, §4.1. `*.huber_old.pt`
(`c40fc00688a53179`), K=12, despite four same-directory alternatives with better
`val_mm`. Authority: the `ef760d1` commit message.

### 10.2 Which variants have generations ready to score

Measured 2026-10-05 by counting mp4s under
`sampling_dataset/<variant>/<embodiment>/output/singleview/dreamgen/<split>/`,
and validated for frame size with `metrics/kiva/probe_clips.py`.

| variant | single_arm mak / non_mak | bimanual mak / non_mak | humanoid mak / non_mak | paired `_gt`? | ready? |
|---|---|---|---|---|---|
| `dense` (baseline) | 120 / 120 | 150 / 150 | 130 / 130 | yes (119 in sa/mak) | **yes** |
| `pisa` | 120 / 120 | 150 / 150 | 130 / 130 | yes | **yes** |
| `radial` | 120 / 120 | 150 / 150 | 130 / 130 | yes | **yes** |
| `svg1` | 120 / 120 | 150 / 150 | 130 / 130 | yes | **yes** |
| `svg2` | 120 / 120 | 150 / 150 | 130 / 130 | yes | **yes** |
| `itm` | 120 / 120 | 150 / 150 | 130 / 130 | yes | yes (not requested) |
| `sito` | 120 / 120 | 150 / 150 | 130 / 130 | yes | yes (not requested) |
| `dicache` | 120 / 120 | 150 / 150 | 130 / 130 | yes | yes (not requested) |
| `fastercache` | 120 / 120 | 150 / 150 | 130 / 130 | yes | yes (not requested) |
| `worldcache` | 120 / 120 | 150 / 150 | 130 / 130 | yes | yes (not requested) |

**Nothing needs a new generation run.** All ten variants are complete across all
three embodiments and both splits, **and every one ships its paired `_gt.mp4`**,
so the real-video anchor is available for every cell at no extra generation
cost. Total 2400 prediction clips + 2390 GT clips. Frame sizes verified correct
for the sampled cells (`6/6` scorable for both roles in all three embodiments).

**The variant-naming trap, checked.** On the multiview side `multiview_svg`
mapped to **two** directories, `svg1` and `svg2`, and both had to be scored. The
same applies here — `svg1` and `svg2` are separate variant dirs with separate,
complete generation trees, and §6.2/§6.5 score both. They are **not** duplicates:
they produce different numbers (frozen-threshold rigidity 0.0278 vs 0.0139).
Treat "svg" in any branch name as **two** variants. No other variant name in
this tree aliases to multiple directories — `dense, pisa, radial, itm, sito,
dicache, fastercache, worldcache` are each exactly one dir.

### 10.3 Undeclared dependencies hit

The multiview branch needed `numba` injected into `uv run`. Here the
equivalents, all worked around in `metrics/kiva/run.sh`:

| # | dependency | declared? | symptom | workaround |
|---|---|---|---|---|
| 1 | whole `uv` env | `pyproject.toml` pins `torch>=2.5,<3` | `uv` picks torch 2.14.1 → triton 3.8.0, unpack hits `OSError: [Errno 122] Disk quota exceeded` on the shared `~/.cache/uv` | **abandon `uv`**; use the `hdmi` conda env via `HAL_PYTHON` |
| 2 | `pytorch_kinematics` | declared, **not installed** in the conda env | `ModuleNotFoundError` | `PYTHONUSERBASE=/pfss/.../.local` |
| 3 | `ffprobe` / `ffmpeg` | **not declared** (only a README line) | needed for every clip's timebase | prepend a pfss ffmpeg build to `PATH` |
| 4 | `libavdevice.so.62` | **not declared** | the pfss ffmpeg binary cannot start | also prepend its `lib/` to `LD_LIBRARY_PATH` |
| 5 | `/usr/bin/ffprobe` vs `torch` | **not documented** | **segfault / symbol clash (gotoblas, liblapack) once `torch` is imported in the same process tree** | must use the pfss build, not the system one |
| 6 | DINOv3 weights | not a package | hard requirement, and we run `HF_HUB_OFFLINE=1` | pre-warmed in `$HAL_STORE/cache/hf` |

Item 5 is the nastiest and has no upstream mention at all.

### 10.4 Verified

- `origin/quang/hallucination-official` exists; vendored @ `5225ce7` (was a detached worktree);
  12 commits ahead of `-humanoid`, 0 behind.
- All three benchmark reader checkpoints load and score; sha256 recorded
  (`5e463b70358e53b5`, `663a39a11e0f0ef0`, `c40fc00688a53179`).
- Store completeness: `ckpt/ assets/ data/ cache/` all present; calibration
  clip counts A1X 12, ALOHA 29, GR-1 26.
- The GR-1 K=12 reconciliation path (`cmd_score.py:117-119`) works — humanoid
  scores without the `IndexError` that `-humanoid` would hit.
- End-to-end mp4 → `segments.csv` → leaderboard, for **all three embodiments**
  and **both roles** (`pred` and `gt`), via the committed launcher only.
- Multi-variant sweep in one invocation: 5 variants, 13 min 37 s, 0 failures.
- `metrics/kiva/probe_clips.py` on all three embodiments; both roles reported.
- `metrics/kiva/aggregate.py`: `--intersect`/`--no-intersect`,
  `--anchor-role gt`, `--baseline`, `--embodiment`, all 5 `--detectors`,
  `--json` provenance.
- **Real-video anchor measured** (§7.2) and **replicated**: A1X jerk
  **0.0939 ±0.0154 (n=30)** and **0.0909 ±0.0298 (n=8)** on different clips/days,
  vs generated 0.31–0.52; rigidity 0.0000; and `jerk_severity` 0.67 (real) →
  0.81–1.00 (generated) → 3.85 (broken).
- **Negative control fires** (§6.4): shuffled/ping-pong clips reach
  `jerk_severity` 3.85.
- **The metric's own noise floor is quantified** (§6.6): the same 6 real clips
  through 5 independent invocations give `jerk_rate` std **0.0198** and
  `rigidity_rate` std **0.0000** — so between-variant gaps below ~0.04 jerk are
  not interpretable, and `radial`'s 0.18 gap vs dense is ~9σ.
- Both roles complete for all 5 variants: `pred` sweep 13 min 37 s, `gt` anchor
  sweep 13 min 00 s, 60 clips scored, **0 failures**.
- `_CALIBRATION_FLOOR["rigidity"] = 18.0` is **active** for A1X — confirmed by
  every run reporting a per-frame rigidity threshold of exactly `18.00`.
- Calibration drift quantified (§6.5): 8.18 % jerk, 9.67 % rigidity, 24.34 %
  joint_limit on identical clips — **and shown to reorder the leaderboard**.
- No VLM / judge anywhere in the tree (three independent checks, §4.3).
- No PhyGenEval / WorldScore / VideoPhy / VBench lineage (grep, §9.2).

### 10.5 NOT verified / open

- **No full-scale run.** Everything here is n ≤ 8 clips of a possible 120–150
  per cell, one split, mostly one embodiment. ±sem on rigidity is as large as
  the mean. **Nothing in §6 is quotable as a result.**
- **No stable ranking.** Blocked on freezing thresholds across variants (§8.3).
  The needed `--thresholds <summary.json>` flag **does not exist** on this
  branch; our §6.5 post-processing is a workaround, not a fix.
- **bimanual and humanoid produce no signal** on either headline detector, on
  generated *or* real video (§6.3). Root cause is reader `val_mm` (84.9 mm for
  ALOHA) inflating the thresholds. Unresolved; needs a better reader, not more
  clips.
- **Upstream's humanoid real-motion baseline (31.3 % jerk,
  `docs/METRICS.md:57-62`) contradicts our 0.0 %** on the same reader id (§7.2).
  Not resolved — could be their 524-segment in-sample set vs our 15, or a
  different checkpoint behind the same reader id.
- **The HF bucket humanoid artifact is unverified.** The README downloads it to
  an unsuffixed `fourier_gr1.humanoid_sv.sv1_16x9.pt`; our store has only
  `.huber_old.pt`. Whether they are byte-identical is **unknown** (we ran fully
  offline). sha256 against `c40fc00688a53179` before trusting a re-download.
- **No `non_makovian` split scored** for any cell.
- **No human-agreement study for this metric.** `metrics/reports/hallucinate_human_vs_dr.csv`
  holds 291 human ratings mapped to 258 local clips, and
  `metrics/kiva/map_to_dr.py` / upstream's `scripts/score_agreement.py`
  are the hooks — but **the correlation between this metric and those human
  labels has not been computed.** This is the single highest-value next step: it
  would tell us whether `jerk` actually tracks perceived hallucination.
- **Determinism**: characterised for the **real-clip** path only (§6.6,
  `jerk_rate` std 0.0198 on identical input). Per-clip score variance on
  *generated* clips under re-runs was not measured separately.
- **`metrics/reports/` outputs were not refreshed** — the quota (§8.2) blocks
  writing into the repo, so all results live in `/tmp` and will not survive a
  reboot. Re-run or copy them off before relying on them.
- **The 2026-10-04 `/tmp/hal_agg/humanoid.json` contains wrong numbers**
  (`rate = 1.0`) from a pre-fix aggregator; superseded by §6.3 but kept as an
  audit trail.

---

## 11. Files

Created in this repo (sizes as of 2026-10-05):

| Path | Bytes | Role |
|---|---|---|
| `metrics/kiva/run.sh` | 13 921 | Launcher: vendored-code check, store/env/ffmpeg wiring, per-embodiment reader+cell+checkpoint resolution, `pred`/`gt` role switch, multi-variant sweep, tmux mode, single-directory mode. |
| `metrics/kiva/probe_clips.py` | 10 797 | Pre-flight: `ffprobe` every clip, report scorable / wrong-size / too-short per role, estimate segments. The analogue of the multiview frame extractor — this metric needs a *probe*, not extraction. |
| `metrics/kiva/aggregate.py` | 15 989 | `segments.csv` → variant leaderboard with rate + severity, real-video anchor as a pseudo-variant, `--intersect`, reader/threshold provenance, `--json`. |
| `docs/metrics/hallucination_metric.md` | this file | This document. |

Pre-existing in this repo (**prior work by others, not created here**):

| Path | Role |
|---|---|
| `scripts/metrics/build_hallucination_segment_catalog.py` | September, segment catalog builder. |
| `scripts/metrics/export_hallucinate_results.py` | September, result exporter. |
| `tmp/20261005_refactor/web/scripts/export_hallucinate_final_labels.js` | September, label exporter. |
| `metrics/kiva/map_to_dr.py` | September, maps rated web clips → local paths. |
| `metrics/reports/hallucinate_human_vs_dr.csv` | September, 291 human ratings + PSNR/SSIM/LPIPS. |
| `metrics/reports/hallucinate_source_map.json` | September, 258/291 resolved clip provenance. |

Created outside git tracking:

| Path | Role |
|---|---|
| `metrics/kiva/` | vendored kinescore @ `5225ce7` + our wrappers (`VENDORED.md`). |
| `/tmp/hal_results/` | the §6.2/§6.5 5-variant sweep + GT anchor. |
| `/tmp/hal_agg/` | the §6.2 8-clip anchored run + §6.4 broken control. |
| `/tmp/hal_smoke/` | the §6.3 three-embodiment pred/gt runs. |
| `/tmp/hal_broken/` | the two corrupted negative-control clips. |
| `logs/exp_logs/tmux_logs/hal_*.log` | launcher logs. |

Upstream, read-only references:

| Path | Role |
|---|---|
| `metrics/kiva/src/kinescore/violations/{detectors,scorer,segments,table}.py` | the metric itself. |
| `metrics/kiva/docs/{METRICS,MODEL,RUNBOOK,QUICKSTART}.md` | upstream definition, reader maths, reproduction. |
| `$HAL_STORE/{ckpt,assets,data,cache}/` | readers, URDFs, calibration clips, DINOv3 weights. |
| `hf://buckets/twanghcmut/hallucinate-bench` | public, token-free source for all of the above. |
