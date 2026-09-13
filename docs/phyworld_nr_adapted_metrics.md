# PhyWorld (provisional name)

Sources: [WorldScore](https://arxiv.org/abs/2504.00983),
[RBench](https://arxiv.org/abs/2601.15282),
[RoboPhys-3D](https://arxiv.org/html/2608.28718v1).

The evaluation suite contains **six metrics organized into two groups**. It
assesses visual quality, alignment with input conditions, and motion over time
in robot videos. Here, no-reference (NR) means that neither ground-truth future
video nor dense-model output is required as a reference; condition-based metrics
may still use the task instruction.

The metric names below are used in this report. Backends and aggregation methods
are adapted to the robot dataset and should not be treated as exact reproductions
of the source papers' formulas.

## 1. Visual & Condition Integrity

| Metric | Source | What it measures |
|---|---|---|
| **VisualIntegrity** (`scene_integrity`) | WorldScore — CLIP-IQA+ | Evaluates perceptual quality per frame and aggregates over time. Sensitive to blur, noise, and artifacts; does not verify robot structure or physics. |
| **EntityPresence** (`entity_presence`) | WorldScore — Object Detection | Uses detection rates to measure the presence of the robot and required objects across sampled frames. A disappearing object may lower the score, but occlusion or missed detections can also do so. Does not track identity or require all objects to co-occur for a frame to receive credit. |
| **TaskLock** (`task_lock`) | WorldScore — CLIP Score | Measures similarity between each frame and the task instruction, then averages over time. For example, a robot scene containing a cup may align better with a cup-related instruction than one without it; a high score does not establish correct action execution. |

These metrics respectively assess **visual clarity and artifacts, the presence
of required entities, and alignment with the instruction**. VisualIntegrity is
the display name; the JSON key remains `scene_integrity`.

## 2. Motion & Temporal Integrity

| Metric | Source | What it measures |
|---|---|---|
| **ActorMotion** (`actor_motion_raw`) | RBench — Motion-Aware Score, adapted | Measures motion magnitude within the robot ∪ object mask after subtracting estimated background motion. The adaptation uses optical flow: a median within the mask for each frame pair, followed by a temporal mean. Static clips typically score near zero; stronger motion does not necessarily indicate better quality. |
| **MotionContinuity** (`motion_continuity`) | WorldScore — Motion Smoothness | Uses VFIMamba to interpolate a middle frame from its neighbors, then measures SSIM against the actual frame from the same clip. Assesses continuity and ease of interpolation; frozen clips can still score highly, so interpret it alongside ActorMotion. |
| **ActorFlicker** (`actor_flicker_raw`) | Proposed adaptation of RoboPhys-3D's Temporal Flickering approach | Measures rapid visual fluctuations on the robot or objects after flow-based motion alignment. Targets flickering textures, colors, or details across frames. Occluded regions and unreliable flow must be excluded to avoid mistaking genuine motion for flicker. |

These metrics respectively assess **how much the subject moves, how continuous
the motion is, and how stable visual details remain over time**. MotionContinuity
currently operates on the full frame; ActorMotion and ActorFlicker focus on the
subject.

## Reporting guidelines

- Report the six metrics separately by model, method, and view; do not yet combine
  them into an overall score.
- Use TaskLock only when a valid instruction is available. If the text is an
  annotation that was not supplied to the generator, describe the result as
  semantic alignment with the annotation, rather than instruction-following
  ability. Report N/A when the instruction is missing; do not substitute a generic
  prompt to produce a task-alignment score.
- Match resolution, FPS, and sampling interval when comparing motion across
  methods. Report ActorMotion in pixels per source frame. Flag masks that omit
  the robot as object-only or invalid; do not interpret them as robot inactivity.
- Exclude conditioning frames copied directly into the output when scoring newly
  generated frames. Do not score ground-truth panels in combined GT/pred videos.
- Low ActorMotion with high MotionContinuity may indicate a smooth but inactive
  clip. Treat this as a failure only when the task or phase requires motion.
- Camera Stability is excluded from the six-metric suite. Camera compensation
  within ActorMotion remains necessary; background-flow translation compensation
  has limitations for wrist cameras, rotation, and parallax, so analyze these views
  separately.
- ActorFlicker is an adaptation requiring implementation and validation. This
  report specifies the selected metrics; it does not claim that results for all
  six are already available. Validate on frozen clips, clean motion, deliberately
  introduced flicker, and occlusion before using it for ranking. None of these
  metrics alone establishes correct grasping, physical validity, or task success.
