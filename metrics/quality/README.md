# `metrics/quality` — quality & controllability metrics

WorldScore's non-dynamics aspects. These are all **per-frame and reference-free**, so —
unlike the dynamics metrics — they need no matched-grid resampling: each clip is scored
on its own frames. Needs the dreamgen venv (`models/dreamgen/.venv`).

## Files

| file | aspect | metric | model actually used by WorldScore |
|---|---|---|---|
| `quality_control_probe.py` | `subjective_quality` | `clip_iqa+` | pyiqa `clipiqa+` (CLIP **RN50** + learned prompts) — [0,1] ↑ |
| | `subjective_quality` | `clip_aesthetic` | pyiqa `laion_aes` (CLIP **ViT-L/14** + MLP head) — ~[1,10] ↑ |
| | `content_alignment` | `clip_score` | **torchmetrics** `CLIPScore(clip-vit-base-patch16)` — [0,100] ↑ |
| | `object_control` | `object_detection` | GroundingDINO SwinT-OGC (thr 0.4) + spaCy noun match — [0,1] ↑ |

Each raw score is also normalized to [0,1] with WorldScore's own constants
(`benchmark/utils/utils.py::aspect_info`) so numbers are comparable to published values.

## "Everything says CLIP" — but these are four different models

`clip_` names the **backbone (OpenAI CLIP), not the aspect**. What decides the aspect is
*what the image gets compared against*:

| metric | aspect | CLIP backbone | text side |
|---|---|---|---|
| `clip_iqa+` | subjective_quality | RN50 | **learned** prompt vectors (CoOp-style, init `"Good photo.."`/`"Bad photo.."`) — task-agnostic |
| `clip_aesthetic` (`laion_aes`) | subjective_quality | ViT-L/14 | **none** — 5-layer MLP head on the image embedding |
| `clip_score` | content_alignment | ViT-B/16 | **your task prompt** |
| `object_detection` | object_control | *not CLIP* — GroundingDINO SwinT + BERT | object phrases from the prompt |

`clip_iqa+` / `clip_aesthetic` never see your prompt → they only judge sharpness/prettiness
→ `subjective_quality` (WorldScore averages the two after normalizing each). `clip_score`
requires the prompt → it judges "is this what was asked" → `content_alignment`, which is
exactly why it saturates here (the prompt describes the *scene*, true for every clip).

## ⚠️ Trap: there are TWO CLIPScore classes and the obvious one is wrong

`WorldScore/worldscore/benchmark/metrics/__init__.py` exports pyiqa's CLIPScore as
`IQACLIPScoreMetric` — **it is never used**. The evaluator (`get_aspect_evaluator()`) uses
the **torchmetrics** one. They differ in three ways at once:

| | pyiqa `clipscore` (unused) | torchmetrics `CLIPScore` (**used**) |
|---|---|---|
| backbone | ViT-B/32 | **ViT-B/16** |
| scale | `2.5·max(cos,0)` → [0, 2.5] | `100·max(cos,0)` → [0, 100] |
| text preprocessing | prepends `"A photo depicts"` | none |

Swapping them gives ≈0.76 instead of ≈31, silently breaking the `avg=26.67` normalization
(everything floors at 0). Always check `get_aspect_evaluator()` for which class a metric
name actually resolves to.

## How each metric works

Shared preprocessing (WorldScore `IQAPytorchMetric._process_image`): frame → PIL RGB →
`Resize((512,512))` → `ToTensor()` → forward per frame → aspect score = **mean over frames**.
Our probe scores frames one at a time (WorldScore stacks them all, which OOMs on long clips)
— control the sampling with `STRIDE`.

**`clip_iqa+`** — an antonym-pair *classifier*, not a similarity:
CLIP **RN50** encodes the image; `PromptLearner` emits two learned text embeddings (a "good"
and a "bad" pole, context vectors optimized on IQA data, initialized from
`"Good photo.."`/`"Bad photo.."`); take `softmax` over the 2 poles and return the *good*
pole's probability → **[0,1]**. Not a raw cosine, so don't compare it to one.

**`clip_aesthetic` (`laion_aes`)** — regression head, no text:
CLIP **ViT-L/14** image embedding → **L2-normalize** → frozen **5-layer MLP**
(1024→128→64→16→1, `sac+logos+ava1-l14-linearMSE.pth`) trained to regress human aesthetic
ratings → scalar ≈[1,10]. Pure prettiness; no notion of the task or prompt.

**`clip_score`** — the only prompt-aware one:
encode frame with CLIP **ViT-B/16** → `E_I`, encode task prompt → `E_C`, then
`max(100·cos(E_I,E_C), 0)`. WorldScore feeds **uint8** images here (`(imgs*255).to(uint8)`),
unlike the pyiqa metrics which take floats — a real gotcha. A single global cosine, so it
sees scene-level content only: not ordering, not progress, not whether the box closed.

**`object_detection`** — a detection *rate* with two-pass matching:
caption = expected objects joined (`"robot arm, cardboard box"`) → GroundingDINO SwinT-OGC
forward → keep queries with `max(logit) > 0.4`, decode each phrase from token positions where
`logit > 0.4` → match against the expected list, **pass 1** exact string, **pass 2** fuzzy
(`##` wordpiece fragments or **spaCy noun overlap**), each expected phrase consumable once →
`rate = min(count/len(expected), 1)`. The only **spatially grounded** metric here (it must
localize a box), which is why it degrades when an object dissolves.

## Normalization to [0,1] (WorldScore constants)

Raw scores are mapped with `benchmark/utils/utils.py::aspect_info`. Two schemes:

- **z-score → range** (`clip_iqa+`, `clip_aesthetic`, `clip_score`):
  `z = (score − avg)/std`, then linearly map `[z_min, z_max]` → `range = [0.25, 0.75]`,
  clamped to [0,1]. `subjective_quality` = mean of its two normalized halves.
- **min-max** (`object_detection`): `(score − empirical_min)/(empirical_max − empirical_min)`,
  which is identity here since the range is already [0,1].

⚠️ Those `avg`/`std` constants were calibrated on generated **scenes/landscapes**, not robot
video — see the findings below for why that flattens `subjective_quality` on our data.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 STRIDE=10 bash scripts/metrics/run_quality_control_probe.sh
# -> <OUT>/quality_control_summary.json  (+ quality_control_perframe.json)
```

Overridable env: `GT`, `DG`, `DD`, `OUT`, `CAPTION`, `OBJECTS`, `STRIDE`.

(The box-overlay viewer `object_control_viz.py` was archived to
`tmp/20261005_refactor/misc/metrics/quality/` in the 2026-10-05 refactor.)

## Known findings (measured — see `docs/metrics/WORLDSCORE_PHYSICS_REPORT.md` §9)

- **`object_control` is the only one of the three that ranks models usefully**, and it
  agrees with visual inspection (GT 0.875 > DreamDojo 0.800 > DreamGen 0.543). It is a good
  proxy for **object persistence** — a physically plausible rollout must keep its objects
  identifiable. Recommended for promotion to a Tier-1 physics-adjacent metric.
- **`subjective_quality` is calibrated on the wrong image domain.** WorldScore's constants
  come from generated scenes/landscapes (`clip_aesthetic avg=5.5952`); robot-tabletop frames
  measure ≈3.9 (≈ −6.6σ), so every clip — **including real GT video** — normalizes to ≈0 and
  the aspect loses all ranking power. Fix: recalibrate avg/std from our own GT distribution
  per embodiment, or report `|pred − GT|` on the raw metric.
- **`content_alignment` saturates.** All clips normalize to exactly 1.0000 (raw spread only
  0.45 pts, all above `z_max`). The prompt describes the *scene*, which every clip satisfies
  — even one whose box dissolves. Use it as a catastrophic-failure guard, not a score, or
  make it time-aware with per-phase prompts ("box open" → "box closed").

## Dependency warning

`pyiqa` **must** be installed with `--no-deps`: its full dependency set upgrades
`numpy 1.26 → 2.4` and `transformers 4.51 → 5.14`, which breaks cosmos-predict2/DreamGen.
Required extras: `pyiqa` (no-deps), `openai-clip` (no-deps), `torchmetrics`, `spacy` +
`en_core_web_sm`. Verify after install: torch 2.6.0+cu124, numpy 1.26.4, transformers 4.51.3.
