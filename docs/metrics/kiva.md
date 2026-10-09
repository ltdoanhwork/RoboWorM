# KIVA — setup, usage and score definitions

**KIVA** (Kinematic Invariant Violation Assessment, paper Sec. 3.4.1, App. B.1) is the
benchmark's intra-view embodiment-hallucination audit. A diagnostic reader `g_theta`
reads 3D robot keypoints from the pixels of a **single-view** rollout. Analytic
detectors then check two invariants on the keypoint trajectory: **rigidity** (do rigid
links keep their length?) and **jerk** (is the motion free of abrupt third-derivative
spikes?). Every threshold comes from real video of the same robot, so a score says how
far a clip sits outside real motion. No simulator and no VLM judge are involved.

The code is the `kinescore` package, vendored in [`metrics/kiva/`](../../metrics/kiva/)
(provenance and local patch in [`metrics/kiva/VENDORED.md`](../../metrics/kiva/VENDORED.md);
upstream docs in [`metrics/kiva/docs/`](../../metrics/kiva/docs/)). This page covers
environment, weights, input format, commands, outputs, and the setup pitfalls.

Contents: [1 What it computes](#1-what-it-computes) · [2 Files](#2-files) ·
[3 Environment](#3-environment) · [4 Weights and the store](#4-weights-and-the-store) ·
[5 Input format](#5-input-format) · [6 Running it](#6-running-it) ·
[7 Outputs](#7-outputs) · [8 From detector rates to the paper's KIVA](#8-from-detector-rates-to-the-papers-kiva) ·
[9 Common errors](#9-common-errors) · [10 Known caveats](#10-known-caveats)

---

## 1. What it computes

### 1.1 The reader `g_theta` (App. B.1)

- **Training** (real teleop only): robot states `s_1:T` go through forward kinematics,
  `X_FK = FK(s_1:T)` in R^(T×K×3), then a fixed workspace affine map `W` to
  `x_0 = W(X_FK)` in [-1, 1]. A **frozen DINOv3** encoder
  (`facebook/dinov3-vitl16-pretrain-lvd1689m`, `backbones/dino.py`) gives frame tokens
  `F_tau`. A diffusion head (`heads/diffusion.py::DiffusionKeypointHead`, cosine
  schedule, cross-attention to `F_tau` plus a temporal encoder) is trained to predict
  `x_0` with a Huber loss (`training/trainer.py`, `huber_beta = 0.05`).
- **Inference** (generated video): `N = 5` DDIM samples from independent noise with the
  same conditioning (`DiffusionKeypointHead(..., sample_steps=10, n_samples=5)`), averaged and
  mapped back with `W^-1` to a metric trajectory `P_tau,k`. Hallucinated frames are
  out-of-distribution for the reader, so its samples diverge and the averaged keypoints
  break the invariants.

### 1.2 Detectors, segments, calibration

With connected keypoint pairs `B`, nominal bone lengths `L_ab` (from the URDF rest
pose), frame interval `dt`, and `lambda` converting metres to millimetres:

- rigidity `r_tau = lambda * max_(a,b) | ||P_tau,a − P_tau,b|| − L_ab |` (mm)
- jerk `j_tau = lambda / dt^3 * max_k || P_tau,k − 3 P_tau−1,k + 3 P_tau−2,k − P_tau−3,k ||` (mm/s^3)

A clip is cut into consecutive, non-overlapping **16-frame segments**
(`violations/segments.py::SEGMENT_LEN = 16`; a short tail segment is kept). Per segment
`S`: `v_rig = median_tau r_tau`, `v_jerk = max_tau j_tau`.

Calibration (`Detector.calibrate`, run inside every `kinescore score` call): the reader
scores real clips from its own validation split; `theta_d = Q_0.95` of the
**segment-reduced** values is stored as `segment_threshold`. The normalised segment
ratio is `s_d = v_d / theta_d`, so `s_d = 1` is the upper range of real recordings. (A
separate per-frame p95 `threshold`, floored at 18 mm for rigidity, drives the per-frame
flags and intervals only.)

### 1.3 The KIVA number (App. B.1, eq. B.2)

- Segment score `s_S = max_d s_d`. Detectors: **rigidity + jerk for single-arm, rigidity
  alone for bimanual and humanoid**.
- A segment is flagged when `s_S > gamma_e`, with per-embodiment cuts. App. B.1 reports
  fitted cut pairs `(gamma1, gamma2)` = (1.70, 2.10) single-arm, (1.00, 1.50) bimanual,
  (0.32, 0.80) humanoid (low / medium / high bands). Sec. 4.1 instead states the
  real-video level `max(s_rig, s_jerk) > 1` as the flag. See Sec. 8 below.
- Per rollout with `N` segments: `KIVA = 100/N * sum_i 1[s_S_i > gamma_e]`, in % of
  segments, **lower is better**. The cell score is the mean over rollouts, reported next
  to the same rate on held-out real recordings of that embodiment (the real-video
  reference level: 5.05 / 4.98 / 10.12 % for bimanual / humanoid / single-arm, as
  transcribed in `assets/scripts/make_kiva_audit_data.py`, which writes
  `results/reports/kiva_audit.{csv,json}`).

---

## 2. Files

| path | role | upstream? |
|---|---|---|
| `metrics/kiva/run.sh` | launcher: interpreter, store, ffmpeg workaround, variant sweep, single-dir mode (`HAL_*` env) | ours |
| `metrics/kiva/aggregate.py` | leaderboard from `segments.csv`: per-detector rate + severity, anchored on real video | ours |
| `metrics/kiva/metric.py` | `Hallucination(ExternalMetric)`, registered as `kiva` (aliases `hallucination`, `hal`, `kinescore`); reuses `aggregate.py`'s reducers | ours |
| `metrics/kiva/probe_clips.py` | CPU pre-flight: frame size / count per clip | ours |
| `metrics/kiva/map_to_dr.py` | join KIVA rows onto the distortion-reference table | ours |
| `metrics/kiva/src/kinescore/` | the package (`cli/`, `readers/`, `heads/`, `backbones/`, `robots/`, `violations/`, `registry/`, `training/`, `video/`) | upstream |
| `metrics/kiva/configs/{robots,views,cells,sources,bench}.yaml` | readers, view packings, scored cells, corpora | upstream |
| `metrics/kiva/docs/` | ADDING_A_ROBOT, ARCHITECTURE, MODEL | upstream |

Upstream = branch `quang/hallucination-official` @ `5225ce7`, kinescore 0.2.0. One local
patch: `registry/provenance.py::git_state` (dirty check scoped to the package, constant
`upstream_sha` written to every `run_manifest.json`). Scores are unaffected.

---

## 3. Environment

KIVA needs Python 3.10 with torch, `transformers >= 4.55` (DINOv3), `pytorch-kinematics`,
`robot-descriptions` and the `dino`, `video`, `bench` extras of
`metrics/kiva/pyproject.toml`, plus `ffmpeg` / `ffprobe` on `PATH`.

**On this cluster (what `run.sh` does, nothing is pip-installed):**

| piece | value | override |
|---|---|---|
| interpreter | `hdmi` conda env, `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/miniconda3/envs/hdmi/bin/python` (torch 2.7.0+cu128, transformers 5.2.0, numpy 1.26.4) | `HAL_PYTHON` |
| package | `PYTHONPATH=metrics/kiva/src` | `HAL_ROOT` |
| `pytorch_kinematics` | `PYTHONUSERBASE=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local` | `HAL_PYTHONUSERBASE` |
| ffmpeg | `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/ffmpeg/ffmpeg-master-latest-linux64-gpl-shared/{bin,lib}` prepended to `PATH` / `LD_LIBRARY_PATH` | `HAL_FFMPEG_BIN` |

`uv sync` on `metrics/kiva/pyproject.toml` does **not** work here: it resolves torch
2.14 → triton 3.8, whose unpack exceeds the `~/.cache/uv` quota.

**On a fresh machine:**

```bash
conda create -n hdmi python=3.10 -y
conda run -n hdmi pip install torch --index-url https://download.pytorch.org/whl/cu128   # or your CUDA build
conda run -n hdmi pip install -e "metrics/kiva[dino,video,bench]"
export HAL_PYTHON="$(conda run -n hdmi which python)"
export HAL_PYTHONUSERBASE="$HOME/.local"      # or any dir; only needed if pytorch_kinematics lives there
# ffmpeg/ffprobe on PATH; a static build avoids the BLAS clash in Sec. 9
```

Smoke-check the CLI (CPU, no weights needed):

```bash
PYTHONPATH=metrics/kiva/src "$HAL_PYTHON" -m kinescore.cli.main score --help
```

`metric.py` (the `python -m metrics.evaluate` wrapper) runs from any Python with numpy
(e.g. `models/dreamgen/.venv`) and shells out to `run.sh`, which owns the `hdmi` env.

---

## 4. Weights and the store

`run.sh` points kinescore at a **store** directory, `HAL_STORE` (default
`$KINESCORE_STORE`, else `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/triquang/kinescore-store`).
It refuses to start unless `assets/` and `cache/` exist under it:

| store path | kinescore env | holds | needed for |
|---|---|---|---|
| `$HAL_STORE/assets/` | `KINESCORE_ASSETS` | URDF + mesh trees (`a1x/ aloha/ gr1/ franka/ airbot_mmk2/ ...`) | bone rest lengths, FK |
| `$HAL_STORE/data/trees/<reader_id>/videos/val/` | `KINESCORE_DATA_ROOT` | the reader's real validation clips | `kinescore calibrate` only (12 / 29 / 26 clips) |
| `$HAL_STORE/cache/hf/` | `HF_HOME` | HF cache incl. `facebook/dinov3-vitl16-pretrain-lvd1689m` | the frozen DINOv3 backbone |
| reader checkpoints | `KINESCORE_CKPT_DIR` = `HAL_CKPT_DIR` | `g_theta` per embodiment | reading keypoints |

`HAL_CKPT_DIR` defaults to `checkpoints/metrics/kiva/` when that directory exists (on
this cluster a symlink to `$HAL_STORE/ckpt`), else `$HAL_STORE/ckpt`.

**Benchmark readers** (the checkpoints whose sha256 is in the published `summary.json`):

| embodiment | robot | reader file (`HAL_CKPT_DIR/`) | input | thresholds (`metrics/kiva/thresholds/`) | override |
|---|---|---|---|---|---|
| single_arm | Galaxea A1X (RoboChallenge) | `a1x_ee.a1x_sv.sv1_4x3.pt` (sha256 `5e463b70…`) | 640×480 | `a1x_ee.a1x_sv.sv1_4x3.json` | `HAL_CKPT_SINGLE_ARM`, `HAL_THR_SINGLE_ARM` |
| bimanual | ALOHA (AIST) | `aloha_bimanual.bimanual_sv.sv1_16x9.pt` | 768×432 | `aloha_bimanual.bimanual_sv.sv1_16x9.json` | `HAL_CKPT_BIMANUAL`, `HAL_THR_BIMANUAL` |
| humanoid | Fourier GR-1 | `fourier_gr1.humanoid_sv.sv1_16x9.huber_old.pt` (sha256 `c40fc006…`, K = 12 arm keypoints) | 768×432 | `fourier_gr1.humanoid_sv.sv1_16x9.json` | `HAL_CKPT_HUMANOID`, `HAL_THR_HUMANOID` |

The other humanoid files in the same dir (`*_newsampler`, `*.mse`, K = 22) are different
trainings and are **not** benchmark-consistent.

**DINOv3** is gated on Hugging Face: accept the licence of
`facebook/dinov3-vitl16-pretrain-lvd1689m` once, then populate the cache online, since
`run.sh` exports `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`:

```bash
HF_HOME=$HAL_STORE/cache/hf HF_HUB_DISABLE_XET=1 \
  hf download facebook/dinov3-vitl16-pretrain-lvd1689m
```

**Building a store from the public bucket** (recipe from `metrics/kiva/README.md`; the
bucket is `twanghcmut/hallucinate-bench`, no token needed; needs a `huggingface_hub`
recent enough to have `hf buckets`):

```bash
export HAL_STORE=/path/to/kinescore-store
mkdir -p $HAL_STORE/{assets,data,cache,ckpt}
hf sync hf://buckets/twanghcmut/hallucinate-bench/asset $HAL_STORE/assets
for R in a1x_ee.a1x_sv.sv1_4x3 aloha_bimanual.bimanual_sv.sv1_16x9 fourier_gr1.humanoid_sv.sv1_16x9; do
  hf buckets cp hf://buckets/twanghcmut/hallucinate-bench/train/$R/diffusion/checkpoint.pt $HAL_STORE/ckpt/$R.pt
  hf sync hf://buckets/twanghcmut/hallucinate-bench/train/$R/val $HAL_STORE/data/trees/$R/videos/val
done
# run.sh expects the humanoid reader under its benchmark name:
export HAL_CKPT_DIR=$HAL_STORE/ckpt
export HAL_CKPT_HUMANOID=$HAL_STORE/ckpt/fourier_gr1.humanoid_sv.sv1_16x9.pt
sha256sum $HAL_STORE/ckpt/*.pt   # compare with the summary.json of a published run
```

The bucket names the humanoid reader `fourier_gr1.humanoid_sv.sv1_16x9`, while
`run.sh` defaults to `*.huber_old.pt`. Check the sha256 against `c40fc006…` before you
treat its numbers as benchmark-consistent.

---

## 5. Input format

- **mp4 files, scored directly** (no frame extraction), one clip per file, single view.
- Frame size must equal the reader's panel exactly (`registry/views.py::check_frame_size`
  rejects anything else): `sv1_4x3` = 640×480, `sv1_16x9` = 768×432. No implicit resize.
- Every frame is read; the fps of the file sets `dt` for jerk.
- `run.sh` sweep mode looks only at the DreamGen single-view trees,
  `$HAL_DATA_ROOT/<variant>/<embodiment>/output/singleview/dreamgen/<split>/` (searched
  recursively), with basename glob `*[0-9].mp4` for predictions and `*_gt.mp4` for the
  co-located real clip (`ROLE=gt`). `HAL_PATTERN` overrides the glob.
- Sizes on disk (checked 2026-10-06): DreamGen SV = 640×480 (single_arm, 16 fps) and
  768×432 (bimanual, humanoid, 16 fps), matching the three benchmark readers. DreamDojo
  SV `full_pred.mp4` is 640×480 for all three embodiments (10–15 fps): it fits the A1X
  reader on single_arm, but bimanual / humanoid need a `sv1_4x3` reader (Sec. 10).
- Pre-flight before a long run (CPU, `ffprobe` only; reports which clips the reader
  will accept and how many 16-frame segments each contributes):

```bash
python3 metrics/kiva/probe_clips.py --variant dense pisa --embodiment single_arm --split makovian
python3 metrics/kiva/probe_clips.py --videos /path/to/clips --embodiment bimanual
```

---

## 6. Running it

All commands from the repo root. Each `kinescore score` call loads DINOv3 + the reader
and calibrates on the real val clips first (about a minute), then scores.

### 6.1 One clip (smoke test)

```bash
CUDA_VISIBLE_DEVICES=0 MAX_CLIPS=1 VARIANTS=dense EMBODIMENTS=single_arm SPLITS=makovian \
HAL_OUT_ROOT=/tmp/kiva_smoke \
  bash metrics/kiva/run.sh
"$HAL_PYTHON" metrics/kiva/aggregate.py --results-root /tmp/kiva_smoke --variants dense
```

Measured on 2026-10-06 (idle A100-80GB, `episode_000000`, 185 frames = 12 segments):
**105 s wall**, `calibrating on 12 real clip(s)`, output dir
`/tmp/kiva_smoke/pred/dense.makovian.sv1_4x3.a1x/` with `segments.csv`, `metrics.csv`,
`results.jsonl`, `summary.json`, `run_manifest.json`; `summary.json` recorded reader
`a1x_ee.a1x_sv.sv1_4x3`, sha256 `5e463b70…`, rigidity `segment_threshold` 8.53 mm,
jerk `segment_threshold` 366053 mm/s^3.

### 6.2 One explicit directory

```bash
VIDEOS_DIR=/path/to/clips EMBODIMENTS=bimanual OUT_DIR=/tmp/kiva_one \
HAL_PATTERN='*.mp4' \
  bash metrics/kiva/run.sh
```

`VIDEOS_DIR` mode needs exactly one `EMBODIMENTS` value; the default `OUT_DIR` is
`$HAL_OUT_ROOT/single/<dir name>.<ROLE>`.

### 6.3 A whole sampling_dataset tree (variant sweep)

Score the **real-video anchor first**; rates are uninterpretable without it.

```bash
VARIANTS="dense pisa radial svg1 svg2 sito itm worldcache dicache fastercache"
ROLE=gt VARIANTS=dense bash metrics/kiva/run.sh --tmux     # co-located *_gt.mp4 = real video
VARIANTS="$VARIANTS"   bash metrics/kiva/run.sh --tmux     # predictions, every method
"$HAL_PYTHON" metrics/kiva/aggregate.py --results-root results/hallucination \
  --variants $VARIANTS --embodiment single_arm \
  --anchor-role gt --baseline real --json results/hallucination/summary_single_arm.json
```

Output dirs: `$HAL_OUT_ROOT/<ROLE>/<variant>.<split>.<tag>/` with tag `sv1_4x3.a1x`,
`bimanual_sv16.aloha` or `humanoid_sv16.gr1`. `--tmux` starts a session `hal_<HHMMSS>`,
log in `exp_logs/tmux_logs/`. Aggregate one embodiment at a time (`--embodiment`);
pooling mixes three different readers.

| env (run.sh) | default | meaning |
|---|---|---|
| `HAL_ROOT` | `metrics/kiva` | kinescore tree (`src/` goes on `PYTHONPATH`) |
| `HAL_STORE` | `$KINESCORE_STORE` or the triquang store | assets / data / cache |
| `HAL_CKPT_DIR` | `checkpoints/metrics/kiva` if present, else `$HAL_STORE/ckpt` | readers |
| `HAL_PYTHON` | `hdmi` env python | interpreter |
| `HAL_DATA_ROOT` | `sampling_dataset` | variant trees |
| `HAL_OUT_ROOT` | `results/hallucination` | output |
| `VARIANTS` | `dense` | method dirs under `HAL_DATA_ROOT` |
| `EMBODIMENTS` | `single_arm bimanual humanoid` | subset |
| `SPLITS` | `makovian non_makovian` | subset |
| `ROLE` | `pred` | `pred` (glob `*[0-9].mp4`) or `gt` (glob `*_gt.mp4`) |
| `HAL_PATTERN` | per `ROLE` | basename glob override |
| `MAX_CLIPS` | `0` (all) | cap per (variant, embodiment, split) |
| `FRAME_CHUNK` | `16` | frames per DINOv3 forward (lower = less VRAM) |
| `MAX_FRAMES` | `0` (all) | cap frames per clip |
| `VIDEOS_DIR`, `OUT_DIR` | — | single-directory mode |
| `HAL_CKPT_{SINGLE_ARM,BIMANUAL,HUMANOID}`, `HAL_THR_{...}` | benchmark readers / `thresholds/` | per-embodiment overrides |

### 6.4 Through the unified metrics CLI

```bash
PY=models/dreamgen/.venv/bin/python
CELL=sampling_dataset/dense/single_arm/output/singleview/dreamgen/makovian
$PY -m metrics.evaluate --pred-root $CELL --metrics kiva \
  --opt kiva.embodiment=single_arm --opt kiva.score_ref=true --limit 4 --out /tmp/kiva_eval
```

`discover_samples` pairs every `<stem>.mp4` with `<stem>_gt.mp4` next to it, so
`score_ref=true` scores the real clip in the **same** launcher call and adds
`<det>_rate_ref` and `<det>_rate_excess = pred − ref`. Constructor keys for `--opt kiva.*`:
`embodiment` (required: `single_arm | bimanual | humanoid`), `detectors` (default
`["rigidity","jerk"]`; also `teleport`, `joint_limit`, `self_collision`), `score_ref`,
`frame_chunk`, `max_frames`, `calib_clips`, `percentile`, `work_dir`, `log_path`.

---

## 7. Outputs

Per scored directory (`kinescore score --out`):

| file | one row per | holds |
|---|---|---|
| `summary.json` | run | model path, thresholds file, n clips / failed, per-detector rate, `threshold` (per-frame) and `segment_threshold` (= `theta_d`) |
| `segments.csv` | 16-frame segment | `<det>_value`, `<det>_threshold` (segment), `<det>_violated`, `<det>_reduce` per detector, `start_frame`, `end_frame` |
| `metrics.csv` | clip | `n_segments`, per detector `n_violated`, `violated`, `worst_segment`, `frame_fraction`, `severity_median`; `error` if the clip was rejected |
| `results.jsonl` | clip | per-frame series and flagged intervals |

`aggregate.py` / `metric.py` reduce `segments.csv` per clip (same functions, so there is
one definition in the repo):

| key | meaning | direction |
|---|---|---|
| `<det>_rate` | fraction of the clip's segments with `<det>_violated` (`s_d > 1`) | ↓ |
| `<det>_severity` | median over segments of `value / segment_threshold` = median `s_d` | ↓ (> 1 = worse than real p95) |
| `<det>_rate_ref`, `<det>_rate_excess` | the paired real clip's rate, and pred − ref (`score_ref=true` only) | ↓ |

`metrics.evaluate --metrics kiva` writes `kiva.json`, `per_sample.csv`, `summary.json`;
the headline keys are `rigidity_rate` and `jerk_rate`.

---

## 8. From detector rates to the paper's KIVA

What `metrics/kiva/` emits is the per-detector flag at the **real-video level**
(`s_d > 1`) and the ratio `s_d` itself (`value / segment_threshold` in `segments.csv`).
The paper's KIVA number is built from those per-segment ratios at table time:

1. per segment, `s_S = max_d s_d`, over rigidity + jerk (single-arm) or rigidity only
   (bimanual, humanoid);
2. flag `s_S > gamma_e` and report the % of flagged segments per rollout (eq. B.2),
   the mean over the cell, and the same rate on real recordings beside it;
3. the per-cell numbers used by the tables are the ones transcribed in
   `assets/scripts/make_kiva_audit_data.py` → `results/reports/kiva_audit.{csv,json}`.

Paper and code differ on `gamma_e`; state which you use:

- Sec. 4.1 and the KIVA box: a segment is a violation when `max(s_rig, s_jerk) > 1`
  (the real-video p95). This is exactly `rigidity_violated OR jerk_violated` in
  `segments.csv`.
- App. B.1 / eq. B.1: two cuts per embodiment fitted on a 9-rollout human calibration
  set — (1.70, 2.10) single-arm, (1.00, 1.50) bimanual, (0.32, 0.80) humanoid.
- `scripts/tables/compare_ours_vs_vlm_segment_overlap.py::calibrate_cuts` documents
  pipeline cuts 1.0 / 1.5 "from the real-segment p95 definition, not from human
  agreement" and refits them with GroupKFold for the human-agreement study.

For an **acceleration method**, report `ΔKIVA` in percentage points of invalid segments
against the method's own dense rollout on the same episodes (App. E.3).

---

## 9. Common errors

| symptom | cause | fix |
|---|---|---|
| `ERROR: HAL_PYTHON not executable` | no `hdmi` env | set `HAL_PYTHON` (Sec. 3) |
| `ERROR: <store>/assets missing.` (or `data`, `cache`) | wrong / empty `HAL_STORE` | point `HAL_STORE` at a populated store (Sec. 4) |
| `[hal] SKIP <emb>: no reader checkpoint at ...` | reader file missing under `HAL_CKPT_DIR` | download it, or set `HAL_CKPT_<EMB>` |
| `[hal] SKIP ...: no *[0-9].mp4 under ...` | wrong `ROLE`/pattern, or no DreamGen SV tree for that variant | check the tree; use `HAL_PATTERN` or `VIDEOS_DIR` |
| clip has `error` in `metrics.csv`, frame size message | clip not at the reader's panel size | resample upstream to 640×480 / 768×432, or use a reader trained for that view |
| `ffprobe` / `ffmpeg` segfaults, or `libavdevice.so.62` not found | `/usr/bin/ffmpeg` clashes with torch's BLAS in the same process tree; or the static build's `lib/` missing from `LD_LIBRARY_PATH` | use the standalone ffmpeg build with both `bin/` on `PATH` and `lib/` on `LD_LIBRARY_PATH` (`HAL_FFMPEG_BIN`) |
| `OSError: ... dinov3 ... offline` / gated-repo error | DINOv3 not in `$HAL_STORE/cache/hf` while offline | download it once online (Sec. 4) |
| `ModuleNotFoundError: pytorch_kinematics` | `PYTHONUSERBASE` not pointing at its install | set `HAL_PYTHONUSERBASE`, or `pip install pytorch-kinematics` into the env |
| `Disk quota exceeded (os error 122)` on `uv sync` | triton wheel vs `~/.cache/uv` quota | do not use uv for KIVA here; use the conda env |
| CUDA OOM | long clips | lower `FRAME_CHUNK` (e.g. 8) or cap `MAX_FRAMES` |
| `embodiment must be one of ...` (`metrics.evaluate`) | `--opt kiva.embodiment` missing / misspelt | pass `single_arm`, `bimanual` or `humanoid` |
| CLIP / other HF models later fail offline in the same tmux server | `run.sh` exports `HF_HOME` into the store; it leaks through the tmux global env | `unset HF_HOME` (or open a new tmux server) before other jobs |

---

## 10. Known caveats

1. **Single-view only.** KIVA scores one camera per clip; multi-view cells are audited
   by [RIGIS](rigis.md). `configs/cells.yaml` also declares multi-view readers
   (`mv4_row`, `mv4_grid`), but the benchmark tables use the three single-view readers.
2. **DreamDojo resolution.** The `run.sh` sweep only walks `.../singleview/dreamgen/`.
   DreamDojo SV rollouts are 640×480: single-arm fits the A1X reader, while bimanual /
   humanoid need the `*.sv1_4x3` readers (`aloha_bimanual.bimanual_sv.sv1_4x3`, cells
   `bimanual.sv1_4x3.dreamdojo` / `humanoid.sv1_4x3.dreamdojo` in `configs/cells.yaml`).
   On this cluster only `fourier_gr1.humanoid_sv.sv1_4x3.pt` is present in the store;
   there is no `aloha_bimanual.bimanual_sv.sv1_4x3.pt`. Use `VIDEOS_DIR` +
   `HAL_CKPT_*` / `HAL_THR_*` overrides to score these (calibrate the reader first).
3. **Read rates as excess over the real anchor.** The reader's own keypoint error
   (tens of mm) puts a non-zero floor under every rate; real video sits at 5–10 %.
4. **Thresholds are fixed per reader** in `metrics/kiva/thresholds/<reader>.json`, fitted
   once by `kinescore calibrate` at the 95th percentile of the reader's real val clips.
   Re-calibrating changes every rate; keep one thresholds file within a comparison.
5. Jerk is a worst-frame statistic, so its real-segment baseline flag rate is far above
   5 % (e.g. 31 % on the GR-1 reader); rigidity is a
   median and sits below 5 %.
6. Compare across embodiments only through the normalised ratios; raw mm values depend
   on the reader.
7. A new embodiment needs its own reader and its own real-segment calibration (paper
   App. E.2); see [`metrics/kiva/docs/ADDING_A_ROBOT.md`](../../metrics/kiva/docs/ADDING_A_ROBOT.md)
   and [../extending/new_embodiment_dataset.md](../extending/new_embodiment_dataset.md).

Related: [README.md](README.md) (metric index), [rigis.md](rigis.md), paper Sec. 3.4.1,
App. B.1, App. E.2 ("Calibration that must be redone rather than inherited").
