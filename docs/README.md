# RoboWorM documentation

Only what is needed to set up, train, run inference, evaluate and reproduce the paper.
Process notes, run hand-offs and analysis write-ups from development are archived in
`tmp/20261005_refactor/docs/` (index: `tmp/20261005_refactor/MANIFEST.md`).

| section | read this for |
|---|---|
| [setup/](setup/README.md) | the six Python environments, frozen package lists ([setup/env_freeze/](setup/env_freeze/)), secrets; full HF backup / restore incl. checkpoint repos `vgp_checkpoints_*` ([setup/hf_backup_restore.md](setup/hf_backup_restore.md)) |
| [datasets/](datasets/README.md) | the 5 source corpora, `datasets/` layout and download, Markovian / non-Markovian labelling, which prepared variant trains which checkpoint |
| [datasets/sampling_dataset.md](datasets/sampling_dataset.md) | benchmark rollouts: download (`--include` recipes), `sampling_dataset/{method}/{embodiment}/{input,output}/{view}/{model}/{split}` layout, input builders per cell |
| [datasets/cell_matrix.md](datasets/cell_matrix.md), [datasets/benchmark_stats.md](datasets/benchmark_stats.md) | audited episode / rollout counts per cell and method |
| [training/](training/README.md) | Table C1 training-input mapping (source → prepare → launcher → checkpoint) and the training commands per model family |
| [inference/](inference/README.md) | dense launchers for the 12 cells, the 9 acceleration methods and their knobs, speedup measurement |
| [inference/dense_generation_config.md](inference/dense_generation_config.md) | per-cell generation settings (steps, guidance, chunking, fps, resolution) |
| [acceleration/](acceleration/README.md) | the 9 acceleration methods: papers, code hooks per model family, flags and knobs, pitfalls, the full (cell × method) launcher / config / env / output matrix ([acceleration/configs.md](acceleration/configs.md)), speedup protocol |
| [metrics/](metrics/README.md) | what each paper metric measures, its source, code entrypoint and command; setup guides [metrics/kiva.md](metrics/kiva.md), [metrics/rigis.md](metrics/rigis.md) |
| [reproduce/](reproduce/README.md) | paper table / figure → producer script → output file, and a suggested regeneration order |
| [paper/](paper/) | the paper PDF (`RoboWM_ICLR2027_check.pdf`) |

Code-level references that live next to the code:
[scripts/README.md](../scripts/README.md) (script tree),
[metrics/README.md](../metrics/README.md) (metric registry / `python -m metrics.evaluate`),
[assets/scripts/README.md](../assets/scripts/README.md) (paper-asset producers),
[checkpoints/CHECKPOINTS.md](../checkpoints/CHECKPOINTS.md) (every checkpoint),
[results/README.md](../results/README.md) (metric outputs read by the paper).
