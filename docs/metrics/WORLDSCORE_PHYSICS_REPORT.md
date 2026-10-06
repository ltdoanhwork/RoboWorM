# WorldScore for a Physics-Focused Video-World Benchmark

Report + metric recommendation for the `sampling_dataset` benchmark.

This document (1) explains what the WorldScore benchmark is and what each of its
metrics measures, (2) discusses why our setting differs from WorldScore's, and
(3) recommends a concrete, tiered set of metrics to adopt for a **physics-focused**
evaluation of our video-world models.

---

## 1. What is WorldScore?

**WorldScore: A Unified Evaluation Benchmark for World Generation** — Haoyi Duan,
Hong-Xing "Koven" Yu, Sirui Chen, Li Fei-Fei, Jiajun Wu (Stanford), arXiv:2504.00983,
**ICCV 2025**. Project page: `haoyi-duan.github.io/WorldScore`. The full code is
vendored at [metrics/WorldScore](../../metrics/WorldScore).

```
@article{duan2025worldscore,
  title={WorldScore: A Unified Evaluation Benchmark for World Generation},
  author={Duan, Haoyi and Yu, Hong-Xing and Chen, Sirui and Fei-Fei, Li and Wu, Jiajun},
  journal={arXiv preprint arXiv:2504.00983},
  year={2025}
}
```

### Core idea
WorldScore unifies the evaluation of **video generation, 3D-scene generation, and
4D generation** under one framework. Instead of rating a single clip's fidelity,
it treats "world generation" as producing a **sequence of scenes along a prescribed
camera path** (e.g. "pan left -> move left -> pull out") from a text/image prompt,
and then measures whether the model (a) generates a genuinely new, coherent scene
and (b) obeys the instructed camera movement — jointly across **controllability,
quality, and dynamics**.

### Two headline numbers
- **WorldScore-Static** = mean of 7 static aspects.
- **WorldScore-Dynamic** = mean of the 7 static + 3 dynamic aspects.

The aspect list is defined in
[worldscore/run_evaluate.py](../../metrics/WorldScore/worldscore/run_evaluate.py), normalization
constants live in
[worldscore/benchmark/utils/utils.py](../../metrics/WorldScore/worldscore/benchmark/utils/utils.py),
and metric dispatch happens in
[worldscore/benchmark/helpers/evaluator.py](../../metrics/WorldScore/worldscore/benchmark/helpers/evaluator.py).

### Input format
WorldScore metrics operate on **extracted PNG/JPG frames** (not raw mp4), plus per
instance:
- `frames/` — generated frames in temporal order,
- `input_image.png` — the reference/anchor frame,
- `image_data.json` — prompts, `total_frames`, scenes, and (dynamic) object
  `masks` + `objects`,
- `camera_data.json` — GT camera path (`[frames,4,4]` matrices + scale), used by
  camera/3D metrics.

Each raw metric score is empirically normalized to `[0,1]` and scaled by 100.

---

## 2. The 10 metrics — what each one measures

| Aspect | Metric | File | What it measures | Dependency | Needs GT video? |
|---|---|---|---|---|---|
| camera_control | `camera_error` | [camera_error_metrics.py](../../metrics/WorldScore/worldscore/benchmark/metrics/third_party/camera_error_metrics.py) | Rotation + translation error between the instructed camera path and the trajectory recovered from generated frames | DROID-SLAM, lietorch, cvxpy | No (uses GT **camera path**, not GT video) |
| object_control | `object_detection` | `object_detection_metrics.py` | Whether the newly-prompted object actually appears | GroundingDINO + spaCy | No |
| content_alignment | `clip_score` | torchmetrics `clip_score_metrics.py` | CLIP similarity between frame and its text prompt | CLIP (ViT-B/16) | No |
| 3d_consistency | `reprojection_error` | [reprojection_error_metrics.py](../../metrics/WorldScore/worldscore/benchmark/metrics/third_party/reprojection_error_metrics.py) | Multi-view geometric consistency via SLAM reprojection error over generated frames | DROID-SLAM | No |
| photometric_consistency | `optical_flow_aepe` | [flow_aepe_metrics.py](../../metrics/WorldScore/worldscore/benchmark/metrics/third_party/flow_aepe_metrics.py) | Forward-backward optical-flow end-point error between consecutive frames (temporal photometric stability) | SEA-RAFT | No |
| style_consistency | `gram_matrix` | [gram_matrix_metrics.py](../../metrics/WorldScore/worldscore/benchmark/metrics/third_party/gram_matrix_metrics.py) | Gram-matrix (style) distance of each frame vs. the reference frame | VGG19 | No (vs first/anchor frame) |
| subjective_quality | `clip_iqa+`, `clip_aesthetic` | `iqa_pytorch/clip_iqa_metrics.py`, `clip_aesthetic_metrics.py` | No-reference perceptual quality + LAION aesthetic score | pyiqa | No |
| **motion_accuracy** | `motion_accuracy` | [motion_accuracy_metrics.py](../../metrics/WorldScore/worldscore/benchmark/metrics/third_party/motion_accuracy_metrics.py) | Whether the intended object moves: object-mask flow magnitude minus background flow magnitude | SAM2 + RAFT + GroundingDINO + spaCy | No |
| **motion_magnitude** | `optical_flow` | [flow_metrics.py](../../metrics/WorldScore/worldscore/benchmark/metrics/third_party/flow_metrics.py) | Median optical-flow magnitude (frozen/static video is penalized) | SEA-RAFT | No |
| **motion_smoothness** | `motion_smoothness` | [motion_smoothness_metrics.py](../../metrics/WorldScore/worldscore/benchmark/metrics/third_party/motion_smoothness_metrics.py) | Interpolate odd frames from even frames; compare to real odd frames (MSE/SSIM/LPIPS) — temporal continuity/jerk | VFIMamba + SSIM/LPIPS | No |

The three **dynamics** aspects (`motion_accuracy`, `motion_magnitude`,
`motion_smoothness`) are the physics-relevant core. The rest are appearance,
instruction-following, and camera/geometry aspects.

---

## 3. Why our setting is different (and why that matters)

WorldScore is **reference-free / instruction-driven**: there is no ground-truth
video, so it checks that the generated video *obeys a camera path* and *looks
plausible*. It answers "did the model make a good-looking world that follows the
prompt/camera?"

Our benchmark ([sampling_dataset](../../sampling_dataset)) is **reference-based**. Layout:

```
{method}/{embodiment}/{input|output}/{view_type}/{model}/{split}/.../episode_*
```

where `method` in `{dense, sito, itm, dicache, worldcache, fastercache, pisa,
radial, svg1, svg2}`. For every prediction we have a **paired ground-truth video**
(`full_gt.mp4`, per-view GT). We currently compute only **PSNR / SSIM / LPIPS**
inline (via `piq`) in the infer scripts, e.g. `compute_metrics()` in
[scripts/infer/infer_single_arm_multiview_ctrlworld.py](../../scripts/infer/infer_single_arm_multiview_ctrlworld.py),
with standalone helpers in [metrics/video_metrics.py](../../metrics/video_metrics.py).
Results are written to per-episode `metrics.json` + `all_summary.json`.

Consequences for adopting WorldScore metrics:

1. **We have GT motion.** WorldScore's dynamics metrics are designed without a
   reference (e.g. "object should move more than background", "video shouldn't be
   frozen"). Because we have GT, we can convert these into much stronger
   **error-vs-GT** signals (does the pred move *the same way* GT does), which is a
   far better physics probe than "does it move at all".
2. **Fixed camera, robot manipulation.** Our clips are mostly fixed-camera robot
   episodes, not free camera fly-throughs. So `camera_error` and (for singleview)
   `reprojection_error` have little to measure, and `object_control` /
   `clip_score` / aesthetics test appearance/instruction, not physics.
3. **PSNR/SSIM/LPIPS are pixel/perceptual, not dynamic.** They can score a
   temporally-frozen or physically-implausible clip highly if each frame looks like
   GT. They do not penalize jitter, teleporting objects, or wrong motion. This is
   exactly the gap the WorldScore dynamics metrics fill.

**Key gap to state plainly:** WorldScore has **no true physics-law metric** — no
contact/penetration, no energy/momentum conservation, no collision checks.
"Physics-ness" is only captured *indirectly* via motion accuracy, motion
magnitude, motion smoothness, and (multi-view) 3D consistency.

---

## 4. Recommended metrics for the physics benchmark

Tiered by physics value. For each we give the WorldScore source and the concrete
adaptation to our paired-GT setting.

### Tier 1 — physics core (adopt; adapt to reference-based)

| Metric | Source | Physics adaptation for our benchmark |
|---|---|---|
| **Flow / motion-magnitude error vs GT** | [flow_metrics.py](../../metrics/WorldScore/worldscore/benchmark/metrics/third_party/flow_metrics.py) (SEA-RAFT) | Compute the optical-flow field for both pred and GT and compare them (per-frame flow **EPE** and a **motion-magnitude ratio** `|flow_pred| / |flow_gt|`). Directly catches frozen motion (ratio -> 0) and exaggerated/hallucinated motion (ratio >> 1). This is the single most useful physics signal we don't currently have. |
| **Motion smoothness** | [motion_smoothness_metrics.py](../../metrics/WorldScore/worldscore/benchmark/metrics/third_party/motion_smoothness_metrics.py) (VFIMamba) | Use as-is (reference-free): interpolate odd frames from even frames and measure MSE/SSIM/LPIPS to the real odd frames. Flags teleporting/jittery objects that violate continuity of motion. Optionally also report GT smoothness as a ceiling. |
| **Motion accuracy on the manipulated object** | [motion_accuracy_metrics.py](../../metrics/WorldScore/worldscore/benchmark/metrics/third_party/motion_accuracy_metrics.py) (SAM2 + RAFT) | Strongest physics probe for robot data. Adapt from "object flow − background flow" to **object-region flow error vs GT**: segment the manipulated object (SAM2, seeded from the instruction/first frame), then compare the object-region flow of pred vs GT (magnitude + direction/cosine). Answers "does the right thing move, in the right direction, by the right amount". |

### Tier 2 — temporal-consistency / geometry (adopt)

| Metric | Source | Notes |
|---|---|---|
| **Photometric consistency / flow-AEPE** | [flow_aepe_metrics.py](../../metrics/WorldScore/worldscore/benchmark/metrics/third_party/flow_aepe_metrics.py) (SEA-RAFT) | Reference-free temporal stability (forward-backward flow consistency). Cheap, catches flicker/warping. Good complement to smoothness. |
| **3D reprojection consistency** | [reprojection_error_metrics.py](../../metrics/WorldScore/worldscore/benchmark/metrics/third_party/reprojection_error_metrics.py) (DROID-SLAM) | Only meaningful for **multiview** / camera-moving clips. Heavy dependency (DROID-SLAM). Mark optional; enable for multiview embodiments only. |

### Tier 3 — keep for completeness (not physics)

`clip_score`, `clip_iqa+`, `clip_aesthetic`, `gram_matrix`, `camera_error`,
`object_detection`. These test appearance / instruction / camera control. Low
physics value in our fixed-camera robot setting; keep only if we want a full
WorldScore-style report card alongside the physics metrics.

### Not covered by WorldScore (future physics extensions)

To move from motion *proxies* to true physics we would later need bespoke metrics,
e.g. **object-mask penetration / non-intersection**, **gripper–object contact
timing** (does contact begin when GT contact begins), and **trajectory error of the
tracked object centroid vs GT**. These are out of scope for adopting WorldScore but
worth flagging as the natural next step.

---

## 5. Integration sketch

Recommended: a **standalone offline metrics pass** rather than re-running inference,
since predictions already exist on disk.

- **Where it hooks in.** Mirror the existing `compute_metrics()` contract in
  [scripts/infer/infer_single_arm_multiview_ctrlworld.py](../../scripts/infer/infer_single_arm_multiview_ctrlworld.py)
  and extend [metrics/video_metrics.py](../../metrics/video_metrics.py) with new
  functions (e.g. `flow_error_vs_gt`, `motion_smoothness`, `object_motion_error`).
  A new script (e.g. `scripts/compute_physics_metrics.py`) would walk
  `sampling_dataset/{method}/.../episode_*`, read `full_pred.mp4` + `full_gt.mp4`
  (decode to frames), compute the Tier 1/2 metrics, and **merge new keys into the
  existing `metrics.json`** (and a `physics_summary.json` alongside
  `all_summary.json`) so nothing is overwritten.
- **Input adaptation.** WorldScore reads PNG frames from disk; our data is mp4.
  Decode each mp4 to an in-memory frame list (or temp PNGs) and pass to the metric
  the same way WorldScore does (`cv2.imread` -> tensor).
- **Reuse vs port.** The metric math (SEA-RAFT flow, VFIMamba interpolation, SAM2
  masks) can be imported from `metrics/WorldScore/...` directly, or the small,
  self-contained pieces (flow magnitude, smoothness loop) ported into
  `metrics/video_metrics.py` to avoid the heavy WorldScore package import path.

### Dependencies, checkpoints, and cost

| Metric | Model / checkpoint | Approx. cost |
|---|---|---|
| Flow error vs GT, flow-AEPE, motion-magnitude | SEA-RAFT (`Tartan-C-T-TSKH-spring540x960-M.pth`) | Light — 1 forward per frame pair, GPU; fast |
| Motion smoothness | VFIMamba (`VFIMamba.pkl`, needs `mamba_ssm`/`causal_conv1d`) | Medium — 1 interpolation per frame triplet; extra install |
| Motion accuracy (object) | SAM2 (`sam2.1_hiera_base_plus.pt`) + RAFT (+ GroundingDINO + spaCy for auto object grounding) | Heavy — video segmentation per episode; slowest |
| 3D reprojection (multiview only) | DROID-SLAM (`droid.pth`, lietorch) | Heavy — SLAM per clip; optional |

Suggested rollout order: **Tier 1 flow-vs-GT + smoothness first** (light, no
grounding needed), then **object motion accuracy** (needs SAM2/grounding), then
**3D reprojection** for multiview only.

---

## 6. Mapping: WorldScore aspects -> physics relevance -> our benchmark

```mermaid
flowchart LR
    subgraph ws [WorldScore aspects]
        cam[camera_control]
        obj[object_control]
        content[content_alignment]
        threeD[3d_consistency]
        photo[photometric_consistency]
        style[style_consistency]
        subj[subjective_quality]
        macc[motion_accuracy]
        mmag[motion_magnitude]
        msmooth[motion_smoothness]
    end

    subgraph phys [Physics relevance]
        high[High: dynamics/motion]
        mid[Medium: temporal/3D consistency]
        low[Low: appearance/instruction/camera]
    end

    macc --> high
    mmag --> high
    msmooth --> high
    photo --> mid
    threeD --> mid
    cam --> low
    obj --> low
    content --> low
    style --> low
    subj --> low

    high -->|"adopt, adapt to error-vs-GT"| t1[Tier 1 physics core]
    mid -->|"adopt as-is / multiview only"| t2[Tier 2 consistency]
    low -->|"optional report card"| t3[Tier 3 keep for completeness]

    t1 --> bench["sampling_dataset metrics.json (new physics keys)"]
    t2 --> bench
    t3 -.optional.-> bench
```

---

## 7. Bottom line

- WorldScore is a reference-free, camera-path-driven world-generation benchmark;
  its **dynamics trio** (motion accuracy / magnitude / smoothness) is the part that
  matters for physics.
- Because we have **paired GT**, we should adopt those dynamics metrics but convert
  the flow/motion ones into **error-vs-GT** form (flow EPE, motion-magnitude ratio,
  object-region flow error) — a stronger physics signal than WorldScore's absolute
  versions and than our current PSNR/SSIM/LPIPS.
- Start with **Tier 1** (SEA-RAFT flow-vs-GT + VFIMamba smoothness), add
  **object motion accuracy** (SAM2), and reserve **3D reprojection** for multiview.
- No WorldScore metric checks physics laws directly; true contact/penetration/
  trajectory metrics are a future extension.

---

## 8. ⚠️ Known weakness discovered in practice: resolution / fps confound (TO FIX)

Empirical finding while probing `motion_accuracy` on a paired sample
(`close_cardboard_box__episode_000026`; GT vs DreamGen vs DreamDojo — full write-up in
`metrics/motion_probe_out/close_cardboard_box__ep000026/REPORT.md`, Part 3):

- Optical flow is **px between consecutive frames**, so it scales with both the raster
  **resolution** and the **fps** (px/frame ∝ width × 1/fps). Our predictions run at
  *different* res+fps than GT and each other (GT 640×480@30, DreamGen 768×432@16,
  DreamDojo 640×480@10) → raw flow magnitudes are **not comparable** and neither is a
  raw `motion_accuracy` / `motion_magnitude`.
- Concretely, DreamDojo's 10 fps triples its px/frame; that inflation coincidentally
  cancelled its genuinely slow motion, so **raw** numbers looked GT-like while the
  **res/fps-normalized** speed exposed it as ~4× too slow. The ranking flips depending
  on normalization.
- **Human observation contradicts the metric**: DreamDojo visually resembles the
  dense/GT clip more (framing, structure, calm motion), yet a pure flow-**magnitude**
  metric punishes it for being slower. Flow magnitude ignores whether the motion has the
  right **shape/direction** — the thing the eye actually judges.

**Fixes to apply before trusting flow-based physics metrics:**
1. Resample all clips to a common (resolution, fps), **or** always normalize
   `px/frame → (px/frame ÷ width × fps)`; never compare raw px across clips.
2. **Temporally align** pred↔GT (same timeline) before any per-frame error.
3. Add a **trajectory/structure term** (flow-direction cosine, warped-frame photometric
   error, or object-mask centroid/IoU trajectory) so magnitude-only speed does not
   dominate — a clip that reproduces GT's motion *shape* should not be penalized just for
   being slower/faster.

### 8.1 ✅ Correct protocol (validated): transform GT to each model's grid

The formula-normalize above still mixed in clip-length / temporal-alignment error and
*over*-penalized DreamDojo. The **correct, validated** fix is to **resample GT to each
prediction's own (resolution, fps)** and score each model against *its own matched GT*
(`scripts/metrics/run_motion_accuracy_matched.sh`; full numbers in the sample REPORT **Part 4**):

| pair (matched grid) | matched-GT fg flow | pred fg flow | **pred/GT ratio** | verdict |
|---|---|---|---|---|
| GT@768×432@16 vs **DreamGen** | 0.273 | 0.602 | **2.20×** | over-moves (\|log\|=0.79) |
| GT@640×480@10 vs **DreamDojo** | 0.296 | 0.176 | **0.59×** | under-moves (\|log\|=0.52) |

This **reverses both earlier verdicts**: raw px said DreamDojo best (fps artifact);
formula-normalize said DreamDojo far worst (~4× too slow, over-penalized). With GT put on
each model's own grid, **DreamDojo (|log|=0.52) is closer to GT than DreamGen (|log|=0.79)**
— matching the human observation that DreamDojo resembles dense/GT more. Note GT's own fg
speed changes with the grid (0.273 @16fps → 0.296 @10fps), which is exactly why *both*
sides must be on the same grid. **Recommended `motion_accuracy` protocol: resample GT →
prediction grid, report foreground flow ratio (or |log-ratio|) vs matched GT.**

### 8.2 Full dynamics trio on the matched grid (magnitude + smoothness)

Extending the matched-grid protocol to the other two WorldScore dynamics metrics
(`scripts/metrics/run_motion_mag_smooth_matched.sh`; sample REPORT **Part 5**):

| metric | DreamGen (768×432@16) | DreamDojo (640×480@10) | who's closer to GT |
|---|---|---|---|
| motion_magnitude ratio pred/GT | 1.49× (\|log\|=0.40) | 0.60× (\|log\|=0.51) | DreamGen (speed-only) |
| motion_smoothness SSIM (pred) | 0.9943 (GT 0.9970) | **0.9982 (GT 0.9970)** | **DreamDojo (smoother than GT)** |
| motion_smoothness MSE (pred) | 44.5 | **16.7** | **DreamDojo** |

Key point: **no single scalar wins** — DreamGen matches GT's *speed* better (it
over-moves less in log terms than DreamDojo under-moves), but **DreamDojo matches GT's
*temporal structure* better and is the smoothest of all** (SSIM even above GT). The
magnitude metric is speed-only and rewards the over-mover; the smoothness metric captures
the "calm, dense-like" quality the eye sees. **The physics score must combine
magnitude-distance + smoothness (+ a future trajectory/direction term)** rather than rely
on flow magnitude alone.

Env note: motion_smoothness needs VFIMamba (`mamba_ssm`). To keep the dreamgen `.venv` on
torch 2.6 (flashinfer/cosmos requirement) we vendored a **pure-PyTorch `mamba_ssm` shim**
(`selective_scan_ref`) instead of the CUDA kernel (which pulls torch 2.9 / triton 3.5).

---

## 9. Quality & Controllability aspects measured (subjective_quality, content_alignment, object_control)

Ran the three non-dynamics aspects on the same paired sample
(`close_cardboard_box · episode_000026`, GT vs DreamGen vs DreamDojo) with
`metrics/quality/quality_control_probe.py` / `scripts/metrics/run_quality_control_probe.sh`, stride 10.
Implementations mirror WorldScore's own class mapping in `helpers/evaluator.py`:

| aspect | metric | model actually used by WorldScore | raw range |
|---|---|---|---|
| subjective_quality | `clip_iqa+` | **pyiqa** `clipiqa+` | [0,1] ↑ |
| subjective_quality | `clip_aesthetic` | **pyiqa** `laion_aes` | [0,10] ↑ |
| content_alignment | `clip_score` | **torchmetrics** `CLIPScore(clip-vit-base-patch16)` | [0,100] ↑ |
| object_control | `object_detection` | GroundingDINO SwinT-OGC (box/text thr 0.4) + spaCy phrase match | [0,1] ↑ |

### 9.0 "Everything says CLIP" — but these are four different models

Three of the four metrics are prefixed `clip_`, which invites the assumption that they all
belong to `subjective_quality`. They do not: **`clip_` names the backbone (OpenAI CLIP), not
the aspect.** What decides the aspect is *what the image is compared against*:

| metric | aspect | CLIP backbone | text side | raw range |
|---|---|---|---|---|
| `clip_iqa+` | subjective_quality | **RN50** | **learned** prompt vectors (CoOp-style, init from `"Good photo.."` / `"Bad photo.."`) — task-agnostic | [0,1] |
| `clip_aesthetic` (`laion_aes`) | subjective_quality | **ViT-L/14** | **none** — 5-layer MLP regression head on the image embedding | ~[1,10] |
| `clip_score` | content_alignment | **ViT-B/16** | **your task prompt**, `100·max(cos,0)` | [0,100] |
| `object_detection` | object_control | *not CLIP* — GroundingDINO SwinT + BERT | object phrases parsed from the prompt | [0,1] |

- **`clip_iqa+` / `clip_aesthetic` never see your prompt** → they can only answer "is this
  image sharp / pretty", so they measure *perceptual quality* → `subjective_quality`.
  WorldScore averages the two after normalizing each.
- **`clip_score` needs the task prompt** → it answers "does this frame depict what was
  asked" → `content_alignment`. That prompt-dependence is exactly why it saturates here: the
  prompt describes the *scene*, and all three clips are a robot closing a box.
- Note `clip_iqa+` uses **learned** prompts, unlike plain `clipiqa` (unused here), which uses
  a fixed handcrafted antonym list (`'Good image'`/`'bad image'`, `'Sharp image'`/`'blurry
  image'`, …). Either way the text is fixed at training time and independent of our task.

⚠️ **Gotcha: `worldscore/benchmark/metrics/__init__.py` exports *two* CLIPScore classes.**
The pyiqa one is aliased to `IQACLIPScoreMetric` and is **NOT** used; the evaluator uses the
**torchmetrics** one. They are not interchangeable — they differ in three ways at once:

| | pyiqa `clipscore` (unused) | torchmetrics `CLIPScore` (**used**) |
|---|---|---|
| backbone | ViT-B/32 | **ViT-B/16** |
| scale | `2.5·max(cos,0)` → [0, 2.5] | `100·max(cos,0)` → [0, 100] |
| text preprocessing | prepends `"A photo depicts"` | none |

Swapping them yields ~0.76 instead of ~31, silently breaking the `avg=26.67` normalization
(everything floors at 0). Always cross-check `get_aspect_evaluator()` for which class a
metric name actually resolves to — the `__init__.py` alias is easy to misread.

### 9.0b How each one actually works (mechanism, verified from source)

Common preprocessing for all three CLIP metrics (WorldScore `IQAPytorchMetric._process_image`):
decode frame → PIL RGB → **`Resize((512,512))`** → `ToTensor()` → per-frame forward → the
aspect score is the **mean over frames**. (Our probe scores frames one at a time; WorldScore
stacks them, which OOMs on long clips — hence `--stride`.)

**`clip_iqa+` — an antonym-pair *classifier*, not a similarity score.**
1. Normalize the image with CLIP's mean/std, encode with **CLIP RN50**.
2. `PromptLearner` produces two learned text embeddings, a "good" and a "bad" pole (context
   vectors optimized on IQA data; initialized from `"Good photo.."` / `"Bad photo.."`).
3. Compute `logits_per_image` for the pair, then **`softmax` over the 2 poles** and take the
   probability of the *good* pole: `probs[..., 0].mean(dim=1)`.

   → Output is a **probability in [0,1]** ("how likely a human calls this a good photo"),
   which is why the range is exactly [0,1] and why it is *not* comparable to a raw cosine.

**`clip_aesthetic` (`laion_aes`) — a regression head, no text at all.**
1. CLIP-preprocess, encode with **CLIP ViT-L/14** → image embedding.
2. **L2-normalize** the embedding (`F.normalize(p=2, dim=-1)`) — this matters; the MLP was
   trained on unit-norm embeddings.
3. Feed to a frozen **5-layer MLP** (1024→128→64→16→1, weights
   `sac+logos+ava1-l14-linearMSE.pth`) trained to regress human aesthetic ratings (AVA /
   SAC / LAION-logos).

   → Output is a scalar ≈[1,10]. Pure "prettiness" — it has no notion of the task, the
   prompt, or even whether the image is a robot.

**`clip_score` — the only one that reads your prompt.**
1. Encode the frame with **CLIP ViT-B/16** → `E_I`; encode the task prompt → `E_C`.
2. `score = max(100 · cos(E_I, E_C), 0)`, i.e. cosine similarity rescaled to [0,100].
3. WorldScore feeds **uint8** images here (`(imgs*255).to(torch.uint8)`) because torchmetrics
   expects uint8 input — a different convention from the pyiqa metrics above, which take
   float tensors. Getting this wrong shifts the score.

   → Because it's a single global cosine, it answers "does this frame *look like* the caption"
   at scene level. It cannot see ordering, progress, or whether the box actually closed.

**`object_detection` — a detection *rate*, with a two-pass phrase match.**
1. Build the caption by joining the expected objects: `"robot arm, cardboard box"`.
2. GroundingDINO (SwinT-OGC) forward → per-query `pred_logits` / `pred_boxes`; keep queries
   with `max(logit) > box_threshold (0.4)`; decode each kept query's phrase from the token
   positions where `logit > text_threshold (0.4)`.
3. Match predicted phrases against the expected list in **two passes**:
   *pass 1* exact string match; *pass 2* fuzzy — sub-word (`##` wordpiece fragments) or
   **spaCy noun overlap** (`en_core_web_sm`), each expected phrase consumable only once.
4. `rate = min(count / len(expected), 1.0)`; aspect score = mean rate over frames.

   → This is the only metric here that is **spatially grounded** (it must localize a box), so
   it degrades when an object dissolves — which is exactly why it discriminates on our data
   while the two global CLIP scores do not.

Consequence for interpretation: three of the four are **global, whole-frame** judgments
(one probability, one regression, one cosine), so they are blind to *where* things are and
*what happened over time*. Only `object_detection` is localized. That asymmetry — not model
quality — explains the results in §9.1.

### 9.1 Results

| aspect / metric | GT (real) | DreamGen | DreamDojo |
|---|---|---|---|
| `clip_iqa+` | 0.4671 | 0.4081 | **0.4852** (> GT) |
| `clip_aesthetic` | 3.801 | 3.900 | 3.895 |
| **subjective_quality** normalized | 0.0696 | **0.0000** | **0.0972** |
| `clip_score` | 31.07 | **31.36** | 30.91 |
| **content_alignment** normalized | 1.0000 | 1.0000 | 1.0000 |
| `object_detection` rate | **0.8750** | 0.5426 | 0.8000 |
| **object_control** normalized | 0.8750 | 0.5426 | 0.8000 |

### 9.2 ⚠️ Two more metric weaknesses found (TO FIX)

**(a) `subjective_quality` normalization constants are calibrated on the wrong domain.**
WorldScore's constants (`clip_iqa+ avg=0.5842`, `clip_aesthetic avg=5.5952`) come from
generated *scenes/landscapes*. Our robot-tabletop frames sit far below that
(`clip_aesthetic ≈ 3.8–3.9` vs avg 5.60, i.e. ≈ −6.6σ), so **every** model — including
**real GT video** — gets crushed to ≈0 (0.00–0.10) and the aspect loses all discriminative
power. Note the raw signal is still fine and even *inverted vs expectation*: DreamDojo's
`clip_iqa+` (0.4852) is **higher than real GT** (0.4671). Fix: recalibrate `avg`/`std` from
our own GT distribution (per embodiment), or drop absolute normalization and report
`|pred − GT|` on the raw metric.

**(b) `content_alignment` is fully saturated — useless as a ranker here.**
All three clips normalize to exactly **1.0000** because raw `clip_score` differs by only
0.45 (30.91–31.36) yet all exceed `z_max`. Cause: the prompt
(`"A bimanual robot performing the task: Close cardboard box."`) matches *every* frame of
*every* clip — same scene, same robot, so CLIP can't separate a good rollout from a bad one.
It only catches catastrophic content failure. Fix: treat it as a **guard/sanity check**, not
a score; or make it discriminative with per-frame *phase* prompts (e.g. "box open" vs "box
closed") so alignment must track task progress over time.

**(c) `object_control` is the one that works — and it agrees with the eye.**
GT 0.875 > DreamDojo 0.800 > DreamGen 0.543 is exactly the intuitive ordering: real video is
always detectable, DreamDojo keeps object structure crisp (near-GT), DreamGen degrades the
box/arm enough that GroundingDINO misses it in ~46% of frames. This makes it a good proxy for
"does the object survive the rollout" — recommend promoting it to a **Tier-1 physics-adjacent
metric** (object persistence), since a physics-correct video must keep objects identifiable.

### 9.3 Cumulative verdict on DreamGen vs DreamDojo

This is now the **third independent aspect** where DreamDojo ≥ DreamGen (after
matched-grid `motion_accuracy` and `motion_smoothness`); DreamGen only leads on raw
`motion_magnitude` (speed). Consistent with the human observation that **DreamDojo looks
more like dense/GT**, and further evidence that a speed-only metric mis-ranks models.

### 9.4 Env note (deps installed safely)

`pyiqa`, `openai-clip`, `spacy`+`en_core_web_sm`, `torchmetrics` were added to
`models/dreamgen/.venv`. **`pyiqa` must be installed with `--no-deps`**: its full dependency
set would upgrade `numpy 1.26.4 → 2.4.6` and `transformers 4.51.3 → 5.14.1`, breaking
cosmos-predict2/DreamGen. Verified after install: torch 2.6.0+cu124, numpy 1.26.4,
transformers 4.51.3, flashinfer all intact.

```bash
CUDA_VISIBLE_DEVICES=0 STRIDE=10 bash scripts/metrics/run_quality_control_probe.sh
# -> metrics/motion_probe_out/close_cardboard_box__ep000026_quality/quality_control_summary.json
```
