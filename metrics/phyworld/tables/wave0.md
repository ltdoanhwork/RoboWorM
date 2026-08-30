# PhyWorld Wave 0

Cell: `single_arm / multiview / ctrlworld / makovian`.
Methods: `dense`, `worldcache`.
Episodes: `001471` (good), `001533` (bad), `000255`, `000397`.
Date: 2026-08-30. Source: `sampling_dataset/phyworld_tables.json`.

## PhyQual-NR (higher better)

| method | SceneIntegrity | TaskLock | EntityPresence | **PhyQual** |
|---|---:|---:|---:|---:|
| dense | 39.4 | 21.2 | 20.6 | **27.1** |
| worldcache | 38.0 | 21.3 | 19.6 | 26.3 |

## PhyDyn-NR

| method | KineticBudget (raw) | ActorDynamics | MotionContinuity | **PhyDyn** |
|---|---:|---:|---:|---:|
| dense | 0.057 | 67.9 | 98.8 | 83.4 |
| worldcache | 0.094 | 71.0 | 98.8 | 84.9 |

## PhyFid-DR (worldcache vs dense; lower raw error, higher PhyFid)

| method | MotionFidelity | KineticDrift | ContinuityGap | LookDrift | **PhyFid** |
|---|---:|---:|---:|---:|---:|
| worldcache | 0.955 | 0.449 | 0.001 | 1.91 | **76.5** |

Per-episode MotionFidelity: 001471=0.58, 001533=2.19, 000255=0.61, 000397=0.44.
Per-episode ActorDynamics (dense): 001471=90.2, 001533=69.9, 000255=49.0, 000397=62.5.

See skill Insights for sanity verdicts.
