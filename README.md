# kinescore

Physics-plausibility benchmark for AI-generated robot video.

A frozen DINOv3 backbone and a trained diffusion head read 3-D robot keypoints
from the pixels. Analytic detectors then measure physics violations on those
keypoints, with no simulator and no VLM judge. Each threshold is the 95th
percentile of the same quantity on real teleop from the same robot, so a score
says how far a clip sits outside real motion. Keypoint error is tens of
millimetres, so compare systems through the same reader rather than reading
absolute values.

## Benchmark checkpoints

One reader per embodiment, the checkpoints the paper's tables were scored
with. They sit in the public bucket
[twanghcmut/hallucinate-bench](https://huggingface.co/buckets/twanghcmut/hallucinate-bench)
and download without a token (see [Download](#download)).

| embodiment | robot | reader | input |
|---|---|---|---|
| single arm | Galaxea A1X | `a1x_ee.a1x_sv.sv1_4x3` | 640×480 |
| bimanual | ALOHA | `aloha_bimanual.bimanual_sv.sv1_16x9` | 768×432 |
| humanoid | Fourier GR-1 | `fourier_gr1.humanoid_sv.sv1_16x9` | 768×432 |

## Install

```bash
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -e ".[dino,video,bench]"
cp .env.example .env
```

For a specific CUDA build, install torch first with
`uv pip install torch --index-url https://download.pytorch.org/whl/<cuXXX>`.
`ffmpeg` and `ffprobe` must be on `PATH`.

Set the five paths in `.env`, then load it once per shell with
`set -a; . .env; set +a`. None has a default.

```bash
KINESCORE_ASSETS=/data/kinescore/assets      # URDF and mesh trees
KINESCORE_CKPT_DIR=/data/kinescore/ckpt      # readers
KINESCORE_CACHE_DIR=/data/kinescore/cache    # backbone token cache
KINESCORE_DATA_ROOT=/data/kinescore/data     # clips and corpora
KINESCORE_OUTPUT_DIR=/data/kinescore/out     # scores
```

## Download

The `hf` command comes with the install above. Run the asset line once, then
the two lines of each embodiment you need: the checkpoint, and the real clips
its thresholds are fitted on.

```bash
# URDF and mesh trees, shared by every reader
hf sync hf://buckets/twanghcmut/hallucinate-bench/asset $KINESCORE_ASSETS

# single arm
hf buckets cp hf://buckets/twanghcmut/hallucinate-bench/train/a1x_ee.a1x_sv.sv1_4x3/diffusion/checkpoint.pt \
    $KINESCORE_CKPT_DIR/a1x_ee.a1x_sv.sv1_4x3.pt
hf sync hf://buckets/twanghcmut/hallucinate-bench/train/a1x_ee.a1x_sv.sv1_4x3/val \
    $KINESCORE_DATA_ROOT/trees/a1x_ee.a1x_sv.sv1_4x3/videos/val

# bimanual
hf buckets cp hf://buckets/twanghcmut/hallucinate-bench/train/aloha_bimanual.bimanual_sv.sv1_16x9/diffusion/checkpoint.pt \
    $KINESCORE_CKPT_DIR/aloha_bimanual.bimanual_sv.sv1_16x9.pt
hf sync hf://buckets/twanghcmut/hallucinate-bench/train/aloha_bimanual.bimanual_sv.sv1_16x9/val \
    $KINESCORE_DATA_ROOT/trees/aloha_bimanual.bimanual_sv.sv1_16x9/videos/val

# humanoid
hf buckets cp hf://buckets/twanghcmut/hallucinate-bench/train/fourier_gr1.humanoid_sv.sv1_16x9/diffusion/checkpoint.pt \
    $KINESCORE_CKPT_DIR/fourier_gr1.humanoid_sv.sv1_16x9.pt
hf sync hf://buckets/twanghcmut/hallucinate-bench/train/fourier_gr1.humanoid_sv.sv1_16x9/val \
    $KINESCORE_DATA_ROOT/trees/fourier_gr1.humanoid_sv.sv1_16x9/videos/val
```

## Score

```bash
kinescore score --cell bimanual.sv1_16x9.dreamgen --device cuda \
    --videos /path/to/clips --out $KINESCORE_OUTPUT_DIR/myrun
```

| embodiment | cell |
|---|---|
| single arm | `single_arm.sv1_4x3.radial_dreamgen` |
| bimanual | `bimanual.sv1_16x9.dreamgen` |
| humanoid | `humanoid.sv1_16x9.dreamgen` |

`--videos` is searched recursively for `*.mp4`, and each clip must match the
reader's input size. `--limit N` caps the clip count and `--frame-chunk 16`
lowers GPU memory.

## Output

| file | one row per | holds |
|---|---|---|
| `summary.json` | run | cell, reader, checkpoint, fitted thresholds |
| `segments.csv` | 16-frame segment | value, threshold and verdict per detector |
| `metrics.csv` | clip | flagged fraction and worst segment per detector |
| `results.jsonl` | clip | per-frame series and flagged intervals |
| `render/` | clip | the clip with keypoints and verdicts drawn on |

The headline numbers are read from `rigidity` and `jerk`. Rigidity takes the
median over a segment's 16 frames, and jerk takes the worst frame.

## Train a reader

Only needed for a robot or view with no checkpoint above.

```bash
kinescore pull  --what train
kinescore data  --reader $R
kinescore cache --reader $R --device cuda
kinescore train --reader $R --device cuda --steps 8000
```

`train` reports keypoint error in millimetres (`val_mm`) on a scene-disjoint
split. Compare it only within one reader.

## Docs

- [QUICKSTART](docs/QUICKSTART.md): every flag of data, train and score
- [RUNBOOK](docs/RUNBOOK.md): reproduce the full benchmark
- [METRICS](docs/METRICS.md): detectors, segments, calibration
- [ARCHITECTURE](docs/ARCHITECTURE.md): modules and the diffusion head
- [ADDING_A_ROBOT](docs/ADDING_A_ROBOT.md): add an embodiment
