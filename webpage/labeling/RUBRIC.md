# Scoring rubric (placeholders)

Fill these in before raters start. Examples can be a screenshot / clip id
once you pick them.

## Scale (1–2–3)

| Score | Label | Meaning (edit me) | Example (add later) |
|---|---|---|---|
| **1** | Fail | Physics broken, melted, wrong task, unusable. | _TBD_ |
| **2** | Mixed | Recognizable robot/task but artifacts, mild physics errors, or drift. | _TBD_ |
| **3** | Pass | Coherent motion, follows the prompt, physics look right. | _TBD_ |

## Mode A — Multiview triplets (first 300 videos)

Each **point** shows **3 videos**. Each video is **16 frames** of the two
**static** cameras playing side-by-side (flow heatmap overlay). Score **each
of the 3 videos** 1/2/3.

## Mode B — Sequential 16-frame segments (remaining 300)

Watch the video. After every **16 frames** (regardless of fps) the clip
pauses — score 1/2/3 — then play the next segment until the video ends.
Mix of multiview + singleview. Dataset for this mode is not locked yet.
