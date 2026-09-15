# PhyWorld NR metric catalog

Reference-free (NR) only. No dense comparison. Pure PhyWorld composites and
pure-ours signals are excluded: `phy_qual`, `phy_dyn`, and
`actor_motion_active_rate`. Every row below has lineage from WorldScore, RBench,
or RoboPhys-3D.

## Source papers

| Lineage | Paper | Used for |
|---|---|---|
| WorldScore | [Duan et al., *WorldScore: A Unified Evaluation Benchmark for World Generation*, ICCV 2025 / arXiv:2504.00983](https://arxiv.org/abs/2504.00983) | Frame quality, task/entity control, motion amount, motion continuity |
| RBench | [Deng et al., *Rethinking Video Generation Model for the Embodied World*, ICML 2026 / arXiv:2601.15282](https://arxiv.org/abs/2601.15282) | Subject-localized Motion-Aware Score concept |
| RoboPhys-3D | [Wang et al., *RoboPhys-3D*, arXiv:2608.28718](https://arxiv.org/html/2608.28718v1) | Local peak motion, active-motion continuity, flicker, appearance consistency, camera drift |

## 1. Frame, task, and entity integrity

| Key | Status | Source metric | Local function / planned backend | Mechanism | Direction |
|---|---|---|---|---|---|
| `scene_integrity` | Current | WorldScore `clip_iqa+` | `metrics.quality.quality_control_probe.IQA.frame_scores()` → `scene_integrity_score()` | CLIP-IQA+ per frame, temporal mean; visual artifact and clarity signal | higher |
| `task_lock` | Current | WorldScore `clip_score` | `IQA.frame_scores()` → `task_lock_score()` | CLIP ViT-B/16 frame–task-instruction similarity, temporal mean | higher |
| `entity_presence` | Current | WorldScore `object_detection` | `metrics.quality.quality_control_probe.Detector.rate()` → `entity_presence_score()` | GroundingDINO detection rate for arm/gripper plus task nouns; each frame independently, not identity tracking | higher |
| `look_score` | Current, diagnostic | WorldScore `clip_aesthetic` | `IQA.frame_scores()` | CLIP aesthetic predictor per frame; visual appeal only, never physics composite | higher |
| `actor_appearance_drift` | Proposed P1 | RoboPhys-3D Subject Consistency | SAM2/GDINO subject crop + DINOv2 embeddings | Adjacent and first-frame subject-embedding drift; catches arm/object identity, colour, or shape swaps that `entity_presence` misses | lower |

## 2. Motion amount and local action

| Key | Status | Source metric | Local function / planned backend | Mechanism | Direction |
|---|---|---|---|---|---|
| `kinetic_budget` | Current, diagnostic | WorldScore `motion_magnitude` | `metrics.dynamics.motion_mag_smooth_probe.motion_magnitude()` | SEA-RAFT; temporal mean of full-frame median `|flow|` | raw px/frame; no universal better direction |
| `actor_motion_raw` | Current, diagnostic | RBench Motion-Aware Score (MAS), adapted | `metrics.dynamics.motion_accuracy_gdino.probe()` → `ego_compensated_magnitude()` | Temporal mean of median `|flow_fg − median(flow_bg)|` inside `arm ∪ object` mask | raw px/source-frame; no universal better direction |
| `actor_dynamics` | Current | WorldScore `motion_accuracy`, adapted | `motion_accuracy_gdino.probe()` → `actor_dynamics_score()` | Subject-vs-background flow separation: temporal mean `median(|flow_fg|) − median(|flow_bg|)`, soft-mapped 0–100 | higher |
| `actor_motion_peak95_raw` | Proposed P0 | RoboPhys-3D Dynamic Degree | Reuse subject mask + ego-compensated SEA-RAFT flow | Per pair P95 subject `|flow|`, then temporal mean; sees strong local arm action hidden by a median | raw px/source-frame; diagnostic |

## 3. Temporal motion integrity

| Key | Status | Source metric | Local function / planned backend | Mechanism | Direction |
|---|---|---|---|---|---|
| `motion_continuity` | Current | WorldScore `motion_smoothness` | `metrics.dynamics.motion_mag_smooth_probe.motion_smoothness()` → `motion_continuity_score()` | VFIMamba reconstructs odd frame from adjacent even frames; SSIM against true odd frame, temporal mean | higher; read beside motion amount |
| `active_motion_continuity` | v3 | RoboPhys-3D motion-quality protocol | VFIMamba MSE vs frame-copy MSE + subject flow | clip(1 − MSE_VFI / MSE_copy) weighted by subject motion; copy error normalizes motion amount (v2 SSIM form correlated −0.68 with motion) | higher |
| `actor_flicker_raw` | Proposed P0 | RoboPhys-3D Temporal Flickering | Flow-warped frame residual inside subject mask | High-frequency temporal residual after compensating subject motion; catches texture/part flashing despite smooth optical flow | lower |

## 4. Camera stability

| Key | Status | Source metric | Local function / planned backend | Mechanism | Direction |
|---|---|---|---|---|---|
| `camera_drift_raw` | Dropped (v3) | RoboPhys-3D Camera Control | Background SEA-RAFT median flow | v2 audit: high values came from subject masks covering ~98% of frame, not camera motion | — |

