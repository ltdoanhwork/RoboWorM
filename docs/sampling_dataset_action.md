# Action-Conditioned Video Generation Scripts

Run video generation conditioned on robot action sequences with the
`Cosmos-Predict2-2B-Sample-Action-Conditioned` model on three eval datasets:
`EgoDex_Eval`, `In-lab_Eval`, `DreamDojo-HV_Eval`.

## Action format notes

The model expects `action[:, :6]` = 6D EE wrist **delta** (pre-divided by 20 in
JSON; inference re-multiplies by 20) and `continuous_gripper_state` ∈ `[0, 1]`.
`scripts/extract_actions_for_eval.py` auto-detects the source format from each
dataset's `meta/modality.json`:

| Dataset            | Action layout | Source channel used        | Quality |
|--------------------|---------------|----------------------------|---------|
| In-lab_Eval        | 1027-D        | `delta_right_wrist[:, :6]` | Best (matches training) |
| EgoDex_Eval        | 52-D          | `np.diff(right_arm[:, :6])` | Best-effort (joint-space, OOD) |
| DreamDojo-HV_Eval  | 52-D          | `np.diff(right_arm[:, :6])` | Best-effort (joint-space, OOD) |

Common setup (run before any extract / inference command):

```bash
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
```

## 1. Prepare action data

### EgoDex_Eval

```bash
python scripts/extract_actions_for_eval.py \
  --dataset_path ./datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/EgoDex_Eval \
  --save_path ./sampling_dataset/PhysicalAI-Robotics-GR00T-Teleop-GR1/EgoDex_Eval/actions \
  --output_json ./sampling_dataset/PhysicalAI-Robotics-GR00T-Teleop-GR1/EgoDex_Eval/batch_input_action.json \
  --output_video_dir ./output/action_gen/EgoDex_Eval
```

### DreamDojo-HV_Eval

```bash
python scripts/extract_actions_for_eval.py \
  --dataset_path ./datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/DreamDojo-HV_Eval \
  --save_path ./sampling_dataset/PhysicalAI-Robotics-GR00T-Teleop-GR1/DreamDojo-HV_Eval/actions \
  --output_json ./sampling_dataset/PhysicalAI-Robotics-GR00T-Teleop-GR1/DreamDojo-HV_Eval/batch_input_action.json \
  --output_video_dir ./output/action_gen/DreamDojo-HV_Eval
```

### In-lab_Eval (10 sub-tasks)

`In-lab_Eval` contains 10 leaf datasets (`gr1_unified.<task>_robot/`). Loop over them:

```bash
IN_LAB_ROOT=./datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/In-lab_Eval
SAVE_ROOT=./sampling_dataset/PhysicalAI-Robotics-GR00T-Teleop-GR1/In-lab_Eval
OUT_ROOT=./output/action_gen/In-lab_Eval

for task_dir in "$IN_LAB_ROOT"/gr1_unified.*_robot; do
  task=$(basename "$task_dir")
  python scripts/extract_actions_for_eval.py \
    --dataset_path "$task_dir" \
    --save_path "$SAVE_ROOT/$task/actions" \
    --output_json "$SAVE_ROOT/$task/batch_input_action.json" \
    --output_video_dir "$OUT_ROOT/$task"
done
```

## 2. Run inference

### EgoDex_Eval

```bash
torchrun --nproc_per_node=${NUM_GPUS} -m models.dreamgen.examples.video2world_action \
  --model_size 2B \
  --batch_input_json ./sampling_dataset/PhysicalAI-Robotics-GR00T-Teleop-GR1/EgoDex_Eval/batch_input_action.json \
  --disable_guardrail \
  --autoregressive \
  --chunk_size 12 \
  --num_gpus ${NUM_GPUS}
```

### DreamDojo-HV_Eval

```bash
torchrun --nproc_per_node=${NUM_GPUS} -m models.dreamgen.examples.video2world_action \
  --model_size 2B \
  --batch_input_json ./sampling_dataset/PhysicalAI-Robotics-GR00T-Teleop-GR1/DreamDojo-HV_Eval/batch_input_action.json \
  --disable_guardrail \
  --autoregressive \
  --chunk_size 12 \
  --num_gpus ${NUM_GPUS}
```

### In-lab_Eval (loop over 10 sub-tasks)

```bash
SAVE_ROOT=./sampling_dataset/PhysicalAI-Robotics-GR00T-Teleop-GR1/In-lab_Eval

for task_dir in "$SAVE_ROOT"/gr1_unified.*_robot; do
  batch="$task_dir/batch_input_action.json"
  echo "=== Running $batch ==="
  torchrun --nproc_per_node=${NUM_GPUS} -m models.dreamgen.examples.video2world_action \
    --model_size 2B \
    --batch_input_json "$batch" \
    --disable_guardrail \
    --autoregressive \
    --chunk_size 12 \
    --num_gpus ${NUM_GPUS}
done
```

> [!NOTE]
> **Video length / FPS.** Model runs at 4 FPS, datasets are at 20 FPS, so the
> extract script subsamples by `--stride 5` (default) so each delta covers
> 1/4 s of motion (matches training). Without this, deltas are 5× too small
> and the arm appears nearly static.
>
> **Length control.** Each pipe call generates ~13 frames = ~3.25 s. Use
> `--autoregressive --chunk_size 12` to chain chunks; need ≥ `chunk_size * N`
> actions to get `~3.25 * N` seconds. For ~18 s of output: `N = 6` chunks =
> 72 actions (the strided JSON has more than enough).
>
> **Action space.** 7-D (6D EE-wrist delta + 1D gripper). Only `In-lab_Eval`
> matches training distribution. `EgoDex_Eval` / `DreamDojo-HV_Eval` fall back
> to joint-space differences (no `delta_right_wrist` in 52-D modality), so
> expect lower fidelity.
