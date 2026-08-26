# Scoring rubric (placeholders)

Fill these in before raters start. Examples can be a screenshot / clip id
once you pick them.

## Scale (1–2–3)

| Score | Label | Meaning (edit me) | Example (add later) |
|---|---|---|---|
| **1** | Fail | Physics broken, melted, wrong task, unusable. | _TBD_ |
| **2** | Mixed | Recognizable robot/task but artifacts, mild physics errors, or drift. | _TBD_ |
| **3** | Pass | Coherent motion, follows the prompt, physics look right. | _TBD_ |

## Mode A — Multiview static pair (first 300 videos)

Full original length. Only the **two static cameras**. The page shows **raw**
and **optical-flow overlay** of the same clip, playing together. Score the
**whole video** 1 / 2 / 3 once.

## Mode B — Sequential 16-frame segments (remaining 300)

Dataset not locked yet. Will mix **multiview + singleview**. If the clip is
multiview, show **all views**. The video plays, pauses every **16 frames**
(regardless of fps), you score 1 / 2 / 3, then the next segment, until the
clip ends.
