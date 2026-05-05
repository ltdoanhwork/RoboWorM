# Ctrl-World Robo Multiview Finetuning

This package keeps the new finetuning logic outside `models/Ctrl-World`.

The first target is Airbot/MMK2 data under:

```bash
datasets/humanoid/multiview/non_makovian
```

The default action interface is `eef_sim_pose_action` with 12 dimensions. This is intentional: Airbot raw action is 36D and Leju raw action is 54D, while the EEF pose fields provide a shared multi-embodiment interface.

## Prepare Airbot Latents

```bash
bash finetuning/scripts/prepare_airbot.sh
```

Useful smoke-test override:

```bash
MAX_EPISODES_PER_DATASET=2 LIMIT_RECORDS=4 bash finetuning/scripts/prepare_airbot.sh
```

Outputs are written to:

```bash
datasets/processed/ctrl_world_robo_multiview
```

Important files:

- `manifest.json`
- `meta/train_sample.json`
- `meta/val_sample.json`
- `meta/stat.json`
- `datasets/<dataset>/annotation/<split>/episode_*.json`
- `datasets/<dataset>/latent_videos/<split>/episode_*/<view>.pt`

## Dataset Smoke Test

```bash
python -m finetuning.data.dataset_robo_ctrl_world \
  --processed-root datasets/processed/ctrl_world_robo_multiview \
  --mode train \
  --batch-size 2
```

Expected shapes:

- `latent`: `[B, 11, 4, 96, 40]` for 4 views at `192x320`
- `action`: `[B, 11, 12]`

## Train

Single-node or debug:

```bash
GPUS_PER_NODE=1 bash finetuning/scripts/train_airbot_multinode.sh \
  --max-train-steps 100 \
  --checkpointing-steps 50 \
  --validation-steps 50 \
  --train-batch-size 1
```

Multi-node:

```bash
NNODES=2 GPUS_PER_NODE=8 NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR \
  bash finetuning/scripts/train_airbot_multinode.sh
```

The trainer saves:

```bash
datasets/processed/ctrl_world_robo_multiview/checkpoints/latest.pt
datasets/processed/ctrl_world_robo_multiview/checkpoints/checkpoint-<step>.pt
```

## Replay Eval

```bash
bash finetuning/scripts/eval_airbot.sh \
  --ckpt datasets/processed/ctrl_world_robo_multiview/checkpoints/latest.pt
```

Replay eval uses ground-truth actions. It does not use `pi0.5`.

## Notes

- `pi0.5` finetuning is intentionally separate. The world model can be trained and validated with replay actions first.
- Leju can be added by changing `--include-pattern` after Airbot is stable; missing fourth view is represented by a zero latent slot.
- Existing `models/Ctrl-World` files are not modified.

