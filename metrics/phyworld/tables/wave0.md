# PhyWorld Wave 0

Cell: `single_arm / multiview / ctrlworld / makovian`.
Methods: `dense`, `worldcache`.
Episodes: `001471` (good), `001533` (bad), `000255`, `000397`.
Date: 2026-08-30, ActorMotion/ActorMotionDrift columns added + verified 2026-09-10
via real `run_phyworld.sh --track dyn` then `--track fid` runs (GDINO+SAM2+RAFT+VFIMamba
on GPU, not backfilled from a formula). Source: `sampling_dataset/phyworld_tables.json`.

## PhyQual-NR (higher better)

| method | SceneIntegrity | TaskLock | EntityPresence | **PhyQual** |
|---|---:|---:|---:|---:|
| dense | 39.4 | 21.2 | 20.6 | **27.1** |
| worldcache | 38.0 | 21.3 | 19.6 | 26.3 |

## PhyDyn-NR

| method | KineticBudget (raw) | ActorMotion (raw) | ActorMotion (0-100) | ActorDynamics | MotionContinuity | **PhyDyn** |
|---|---:|---:|---:|---:|---:|---:|
| dense | 0.057 | 0.949 | 38.3 | 67.9 | 98.8 | 83.4 |
| worldcache | 0.094 | 1.370 | 45.0 | 71.0 | 98.8 | 84.9 |

ActorMotion (raw) beside PhyDyn, not inside its mean — same rule as KineticBudget.
It is mean(fg_flow_med) from the arm∪object mask already computed for ActorDynamics
(`_phyworld_actor/pred_motion_accuracy_gdino.json`), no extra pass. The 0-100 column
is mean-of-per-episode-scores (not tanh of the mean raw value) — matches how every
other table column is aggregated in `compute_phyworld.py`.

Per-episode ActorMotion (raw, dense): 001471=2.289, 001533=0.902, 000255=0.030, 000397=0.575.
Per-episode ActorMotion (raw, worldcache): 001471=3.610, 001533=1.280, 000255=0.066, 000397=0.522.

**Sanity check (smooth-but-inactive):** episode 000255 dense — Continuity=99.4,
ActorMotion(raw)=0.030 (score 1.5/100), yet PhyDyn=74.2 because Continuity alone
carries the mean. This is exactly the failure ActorMotion is meant to expose:
PhyDyn and Continuity both look fine; ActorMotion flags the robot as basically not
moving. Compare episode 001471 dense — Continuity=99.0, ActorMotion(raw)=2.289
(score 81.6) — actively moving, PhyDyn correctly high for the right reason.

## PhyFid-DR (worldcache vs dense; lower raw error, higher PhyFid)

| method | MotionFidelity | KineticDrift | ActorMotionDrift | ContinuityGap | LookDrift | **PhyFid** |
|---|---:|---:|---:|---:|---:|---:|
| worldcache | 0.955 | 0.449 | 0.427 | 0.001 | 1.91 | **76.5** |

ActorMotionDrift reuses the KineticDrift formula (`\|log(method/dense)\|`) on
ActorMotion(raw) instead of scene-wide KineticBudget — reported beside PhyFid,
not folded into its mean.

Per-episode MotionFidelity: 001471=0.58, 001533=2.19, 000255=0.61, 000397=0.44.
Per-episode ActorDynamics (dense): 001471=90.2, 001533=69.9, 000255=49.0, 000397=62.5.
Per-episode ActorMotionDrift (worldcache vs dense): 001471=0.455, 001533=0.351,
000255=0.808, 000397=0.096. Largest drift is 000255 — the near-static episode,
where a small absolute flow change reads as a large ratio; expected for a
log-ratio metric near zero, not a bug.

## Bonus: episode 000855 (picked up by default `--max-episodes 4`)

Dyn-only (qual/fid not run for it yet). Same pattern as 000255 — Continuity
saturated, ActorMotion near zero:

| method | ActorMotion (raw) | ActorDynamics | MotionContinuity | PhyDyn |
|---|---:|---:|---:|---:|
| dense | 0.032 | 49.9 | 99.6 | 74.8 |
| worldcache | 0.032 | 49.8 | 99.4 | 74.6 |

Two independent near-static episodes (000255, 000855) both land at
Continuity≈99.5 / PhyDyn≈74.7 / ActorMotion(raw)≈0.03 — consistent, not a
one-off artifact of a single clip.

See skill Insights for sanity verdicts.
