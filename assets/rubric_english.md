You will evaluate the synchronization and consistency between two views of generated videos depicting a robot performing actions. In addition to spatial consistency, camera stability is also a factor (ideally, both camera views are completely static).

**Rating scale:** Good - Medium - Bad

## General Principles

### Evaluation Focus

Consistency in motion, shape, color, and relative spatial position of objects/robot within the overlapping field of view visible to both camera angles; along with the stability of both views (ideally, both cameras are completely static).

### Rule on Synchronized Anomalies

Synchronizations are evaluated **WITHOUT** regard to the semantic realism or physical plausibility of abnormal events (including, but not limited to: strange or non-physical behaviors; objects deforming, appearing/disappearing, moving on their own, etc). As long as these abnormal phenomena appear identically and simultaneously across both views, they are still considered consistent.

### Out-of-Scope Factors

Task completion rate, aesthetic appeal, and the accuracy or execution quality of the robot's actions are **not** considered.

## Evaluation Criteria

| Rating | Multiview Consistency & Camera Stability | Threshold Conditions (Area & Duration)* |
|--------|--------------------------------------------|-------------------------------------------|
| **Good** | • Motion, shape, color, and relative spatial positioning across both views are virtually identical.<br>• The cameras remain nearly static. | Affected region covers **< 10%** of the frame area **and** persists for **< 10%** of the video duration. |
| **Medium** | • Noticeable discrepancies appear between the two views (e.g., slight variations in motion or shape, minor relative position misalignments).<br>• Cameras may exhibit occasional jitter or small pans/drifts. | Affected region covers **≤ 50%** of the frame area **and** persists for **≤ 50%** of the video duration. |
| **Bad** | • Asynchronous issues are prominent, severe, and persistent.<br>• Cameras may shift, pan, or drift with large amplitudes. | Affected region covers **> 50%** of the frame area **or** persists for **> 50%** of the video duration. |

\* *Note: The combination of quantitative thresholds for frame area percentage (%) and video duration percentage (%) only serves as a reference guideline.*
