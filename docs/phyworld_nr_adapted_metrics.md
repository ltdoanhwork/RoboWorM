## PhyWorld NR — the metrics actually in use

Sources: [WorldScore](https://arxiv.org/abs/2504.00983),  
[VBench](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.pdf),  
[RBench](https://arxiv.org/abs/2601.15282),  
[RoboPhys-3D](https://arxiv.org/html/2608.28718v1).

## The four reported metrics

`digits` is the print precision the tables use; `direction` is the metric's polarity.  
Polarity is not the same thing as permission to rank — see the per-column  
restrictions below.

| Metric (JSON key) | Source | Direction | What it measures |
| --- | --- | --- | --- |
| **Visual Integrity** (`visual_integrity`) | WorldScore — CLIP-IQA+ | higher = better quality (WorldScore polarity; **may not rank models**, see `CEILING_FAILED`) | Per-frame perceptual quality, aggregated as the **10th percentile x100** (worst frames, not the mean — `visual_integrity_mean` keeps the mean for reference). Sensitive to blur, blocking and noise; says nothing about robot structure or physics. |
| **Text Alignment** (`text_condition_alignment`) | WorldScore — CLIP Score | higher = more on-task | CLIPScore \[0,100\] between frames and the task clause. **Paired only.** Text-conditioned models only (`text_condition_applicable`). |
| **Actor Activity** (`actor_motion_raw`) | RBench — Motion-Aware Score, adapted | **none** (descriptive) | Motion magnitude in px per _source_ frame inside the robot ∪ object mask (SAM2), after removing estimated camera motion. Per-mount statistic, see below. More motion is not better. |
| **Active Motion Continuity** (`active_motion_continuity`, AMC) | WorldScore — Motion Smoothness, adapted | higher = smoother | VFIMamba interpolates the middle frame of each triple; gain over a frame-copy baseline, **weighted by that triple's activity**, x100. The activity weighting is the adaptation: unweighted smoothness rewards frozen video. |

Backends: CLIP-IQA+ and CLIPScore (quality/text), GroundingDINO + SAM2 (robot and  
object masks), SEA-RAFT (optical flow), VFIMamba (interpolation).

**Active Rate** (`actor_motion_active_rate`, VBench Dynamic Degree adapted to a  
continuous duty cycle) was a fifth column until 2026-09-20. It is **still computed**  
— it is the AMC activity gate and the only way to audit why a clip got `AMC = None`  
— but it is no longer reported: over 2,428 valid clips it is ρ +0.875 with Actor  
Activity (+0.946 on wrist), and the "still, then a lurch" pattern it existed to  
separate occurs in 1 clip of 2,428. Recorded in `NOT_REPORTED` in  
`assets/scripts/table_dense_nr.py`.