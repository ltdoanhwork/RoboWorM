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
(`scripts/run_motion_accuracy_matched.sh`; full numbers in the sample REPORT **Part 4**):

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
(`scripts/run_motion_mag_smooth_matched.sh`; sample REPORT **Part 5**):

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
