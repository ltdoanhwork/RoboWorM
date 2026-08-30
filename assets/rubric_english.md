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
| **Good** | • Motion, shape, color, and relative spatial positioning across both views are virtually identical.<br>• Camera is almost static. | Error region covers < 10% of the frame area and appears in < 10% of the video duration. |
| **Medium** | • Discrepancies appear between the two views (e.g., minor differences in motion, shape, or relative position), but it can still be concluded that the robots in both views are performing the same action.<br>• Cameras may exhibit occasional jitter or small pans/drifts. | Error region covers ≤ 30% of the frame area and appears in ≤ 30% of the video duration. |
| **Bad** | • Inconsistencies are pronounced, severe, and persistent, making it obvious that the robots in the two views are performing different actions.<br>• Cameras may shift, pan, or drift with large amplitudes. | Error region covers > 30% of the frame area or appears in > 30% of the video duration. |

\* *Note: The combination of quantitative thresholds for frame area percentage (%) and video duration percentage (%) only serves as a reference guideline.*
