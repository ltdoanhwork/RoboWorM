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
vendored at [metrics/WorldScore](WorldScore).

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
[worldscore/run_evaluate.py](WorldScore/worldscore/run_evaluate.py), normalization
constants live in
[worldscore/benchmark/utils/utils.py](WorldScore/worldscore/benchmark/utils/utils.py),
and metric dispatch happens in
[worldscore/benchmark/helpers/evaluator.py](WorldScore/worldscore/benchmark/helpers/evaluator.py).

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
| camera_control | `camera_error` | [camera_error_metrics.py](WorldScore/worldscore/benchmark/metrics/third_party/camera_error_metrics.py) | Rotation + translation error between the instructed camera path and the trajectory recovered from generated frames | DROID-SLAM, lietorch, cvxpy | No (uses GT **camera path**, not GT video) |
| object_control | `object_detection` | `object_detection_metrics.py` | Whether the newly-prompted object actually appears | GroundingDINO + spaCy | No |
| content_alignment | `clip_score` | torchmetrics `clip_score_metrics.py` | CLIP similarity between frame and its text prompt | CLIP (ViT-B/16) | No |
| 3d_consistency | `reprojection_error` | [reprojection_error_metrics.py](WorldScore/worldscore/benchmark/metrics/third_party/reprojection_error_metrics.py) | Multi-view geometric consistency via SLAM reprojection error over generated frames | DROID-SLAM | No |
| photometric_consistency | `optical_flow_aepe` | [flow_aepe_metrics.py](WorldScore/worldscore/benchmark/metrics/third_party/flow_aepe_metrics.py) | Forward-backward optical-flow end-point error between consecutive frames (temporal photometric stability) | SEA-RAFT | No |
| style_consistency | `gram_matrix` | [gram_matrix_metrics.py](WorldScore/worldscore/benchmark/metrics/third_party/gram_matrix_metrics.py) | Gram-matrix (style) distance of each frame vs. the reference frame | VGG19 | No (vs first/anchor frame) |
| subjective_quality | `clip_iqa+`, `clip_aesthetic` | `iqa_pytorch/clip_iqa_metrics.py`, `clip_aesthetic_metrics.py` | No-reference perceptual quality + LAION aesthetic score | pyiqa | No |
| **motion_accuracy** | `motion_accuracy` | [motion_accuracy_metrics.py](WorldScore/worldscore/benchmark/metrics/third_party/motion_accuracy_metrics.py) | Whether the intended object moves: object-mask flow magnitude minus background flow magnitude | SAM2 + RAFT + GroundingDINO + spaCy | No |
| **motion_magnitude** | `optical_flow` | [flow_metrics.py](WorldScore/worldscore/benchmark/metrics/third_party/flow_metrics.py) | Median optical-flow magnitude (frozen/static video is penalized) | SEA-RAFT | No |
| **motion_smoothness** | `motion_smoothness` | [motion_smoothness_metrics.py](WorldScore/worldscore/benchmark/metrics/third_party/motion_smoothness_metrics.py) | Interpolate odd frames from even frames; compare to real odd frames (MSE/SSIM/LPIPS) — temporal continuity/jerk | VFIMamba + SSIM/LPIPS | No |

The three **dynamics** aspects (`motion_accuracy`, `motion_magnitude`,
`motion_smoothness`) are the physics-relevant core. The rest are appearance,
instruction-following, and camera/geometry aspects.

---

## 3. Why our setting is different (and why that matters)

WorldScore is **reference-free / instruction-driven**: there is no ground-truth
video, so it checks that the generated video *obeys a camera path* and *looks
plausible*. It answers "did the model make a good-looking world that follows the
prompt/camera?"

Our benchmark ([sampling_dataset](sampling_dataset)) is **reference-based**. Layout:

```
{method}/{embodiment}/{input|output}/{view_type}/{model}/{split}/.../episode_*
```

where `method` in `{dense, sito, itm, dicache, worldcache, fastercache, pisa,
radial, svg1, svg2}`. For every prediction we have a **paired ground-truth video**
(`full_gt.mp4`, per-view GT). We currently compute only **PSNR / SSIM / LPIPS**
inline (via `piq`) in the infer scripts, e.g. `compute_metrics()` in
[scripts/infer_single_arm_multiview_ctrlworld.py](scripts/infer_single_arm_multiview_ctrlworld.py),
with standalone helpers in [metrics/video_metrics.py](metrics/video_metrics.py).
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
| **Flow / motion-magnitude error vs GT** | [flow_metrics.py](WorldScore/worldscore/benchmark/metrics/third_party/flow_metrics.py) (SEA-RAFT) | Compute the optical-flow field for both pred and GT and compare them (per-frame flow **EPE** and a **motion-magnitude ratio** `|flow_pred| / |flow_gt|`). Directly catches frozen motion (ratio -> 0) and exaggerated/hallucinated motion (ratio >> 1). This is the single most useful physics signal we don't currently have. |
| **Motion smoothness** | [motion_smoothness_metrics.py](WorldScore/worldscore/benchmark/metrics/third_party/motion_smoothness_metrics.py) (VFIMamba) | Use as-is (reference-free): interpolate odd frames from even frames and measure MSE/SSIM/LPIPS to the real odd frames. Flags teleporting/jittery objects that violate continuity of motion. Optionally also report GT smoothness as a ceiling. |
| **Motion accuracy on the manipulated object** | [motion_accuracy_metrics.py](WorldScore/worldscore/benchmark/metrics/third_party/motion_accuracy_metrics.py) (SAM2 + RAFT) | Strongest physics probe for robot data. Adapt from "object flow − background flow" to **object-region flow error vs GT**: segment the manipulated object (SAM2, seeded from the instruction/first frame), then compare the object-region flow of pred vs GT (magnitude + direction/cosine). Answers "does the right thing move, in the right direction, by the right amount". |

### Tier 2 — temporal-consistency / geometry (adopt)

| Metric | Source | Notes |
|---|---|---|
| **Photometric consistency / flow-AEPE** | [flow_aepe_metrics.py](WorldScore/worldscore/benchmark/metrics/third_party/flow_aepe_metrics.py) (SEA-RAFT) | Reference-free temporal stability (forward-backward flow consistency). Cheap, catches flicker/warping. Good complement to smoothness. |
| **3D reprojection consistency** | [reprojection_error_metrics.py](WorldScore/worldscore/benchmark/metrics/third_party/reprojection_error_metrics.py) (DROID-SLAM) | Only meaningful for **multiview** / camera-moving clips. Heavy dependency (DROID-SLAM). Mark optional; enable for multiview embodiments only. |

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
  [scripts/infer_single_arm_multiview_ctrlworld.py](scripts/infer_single_arm_multiview_ctrlworld.py)
  and extend [metrics/video_metrics.py](metrics/video_metrics.py) with new
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
