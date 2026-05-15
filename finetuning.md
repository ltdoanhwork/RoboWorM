# DreamDojo Finetuning

## LAM (Latent Action Model)

```bash
export PYTHONPATH="./models/DreamDojo:${PYTHONPATH}"
source ./models/DreamDojo/.venv/bin/activate
torchrun --nnodes=1 --nproc_per_node=2 finetuning/dreamdojo/lam_project/main.py fit --config finetuning/dreamdojo/lam_project/config/lam_0.yaml
```

```bash
# Debug single GPU:
source ./models/DreamDojo/.venv/bin/activate
python finetuning/dreamdojo/lam_project/main.py fit --config finetuning/dreamdojo/lam_project/config/lam_0.yaml
```

LAM inference:
```bash
python finetuning/dreamdojo/lam_project/infer_video.py \
  --config finetuning/dreamdojo/lam_project/config/lam_0.yaml \
  --ckpt_path checkpoints/dreamdojo/LAM_400k.ckpt \
  --video_path ./datasets/droid_1.0.1_20chunks/videos/chunk-002/observation.images.exterior_1_left/episode_002000.mp4 \
  --output_dir lam_video_infer \
  --device cuda
```

## DreamDojo Posttraining (Action-Conditioned Video2World)

### Training

```bash
source ./models/DreamDojo/.venv/bin/activate

# ALOHA bimanual (2 GPUs, ~5s/iter steady-state, ~18h for 10k iters)
NPROC=2 bash finetuning/dreamdojo/launch.sh dreamdojo_2b_480_640_aloha

# DROID single-arm
NPROC=2 bash finetuning/dreamdojo/launch.sh dreamdojo_2b_480_640_droid_sv

# Single GPU debug
NPROC=1 bash finetuning/dreamdojo/launch.sh dreamdojo_2b_480_640_aloha

# Resume from iter_40k, train on all LeRobot roots under datasets/bimanual/singleview (54 tasks),
# save to a new job folder (not exp_aloha/aloha_bimanual_singleview)
NPROC=2 bash finetuning/dreamdojo/launch.sh dreamdojo_2b_480_640_aloha_resume40k_bimanual_sv
```

Configs: `models/DreamDojo/configs/2b_480_640_aloha.yaml`, `2b_480_640_droid_sv.yaml`, `2b_480_640_aloha_resume40k_bimanual_sv.yaml` (resume + full bimanual singleview list)

Checkpoints saved to: `exp_logs/dreamdojo_logs/dreamdojo/exp_<group>/<name>/checkpoints/`
Video samples logged to: `exp_logs/dreamdojo_logs/dreamdojo/exp_<group>/<name>/video_logs/`

Resume run above writes to `exp_logs/dreamdojo_logs/dreamdojo/exp_aloha_resume40k/aloha_bimanual_singleview_full/` and loads weights plus optimizer/scheduler/trainer state from `.../checkpoints/iter_000040000/`. If resume errors, edit that YAML and set `checkpoint.load_training_state: false` (weights only).

### Inference (test on dataset samples)

Given a dataset path and episode index, loads first frame + GT actions, generates predicted video, saves GT vs predicted comparison.

```bash
source ./models/DreamDojo/.venv/bin/activate

# Infer with pretrained 2B checkpoint on an ALOHA episode
bash finetuning/dreamdojo/infer.sh \
  --dataset_path /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/datasets/bimanual/singleview/makovian/close_cardboard_box \
  --episode_idx 0

# Infer with a finetuned checkpoint
bash finetuning/dreamdojo/infer.sh \
  --dataset_path /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/datasets/bimanual/singleview/makovian/close_cardboard_box \
  --episode_idx 0 \
  --ckpt_path /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/exp_logs/dreamdojo_logs/dreamdojo/exp_aloha/aloha_bimanual_singleview/checkpoints/iter_000010000/model

# 2) Action-conditioned batch infer (num-samples nhiều hơn)
export PYTHONPATH="./models/DreamDojo:${PYTHONPATH}"
source ./models/DreamDojo/.venv/bin/activate
python -m models.DreamDojo.examples.action_conditioned \
  -o ./output/dreamdojo_svg1_eval_finetuned \
  --checkpoints-dir ./exp_logs/dreamdojo_logs/dreamdojo/exp_aloha/aloha_bimanual_singleview/checkpoints \
  --experiment dreamdojo_2b_480_640_aloha \
  --save-dir ./output/dreamdojo_results/results_finetuned_40k \
  --num-frames 97 \
  --num-samples 5 \
  --dataset-path "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/datasets/bimanual/singleview/makovian/close_toolbox" \
  --data-split test \
  --deterministic-uniform-sampling 

# 3) Action-conditioned batch infer (num-samples nhiều hơn)
export PYTHONPATH="./models/DreamDojo:${PYTHONPATH}"
source ./models/DreamDojo/.venv/bin/activate
python -m models.DreamDojo.examples.action_conditioned \
  -o ./output/dreamdojo_svg1_eval_finetuned \
  --checkpoints-dir ./exp_logs/dreamdojo_logs/dreamdojo/exp_aloha_resume40k/aloha_bimanual_singleview_full/checkpoints \
  --experiment dreamdojo_2b_480_640_aloha \
  --save-dir ./output/dreamdojo_results/results_finetuned_resume40k \
  --num-frames 97 \
  --num-samples 5 \
  --dataset-path "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/datasets/bimanual/singleview/makovian/close_toolbox" \
  --data-split test \
  --deterministic-uniform-sampling 

# DROID single-arm batch infer
export PYTHONPATH="./models/DreamDojo:${PYTHONPATH}"
source ./models/DreamDojo/.venv/bin/activate
python -m models.DreamDojo.examples.action_conditioned \
  -o ./output/dreamdojo_droid_sv_eval_finetuned \
  --checkpoints-dir ./exp_logs/dreamdojo_logs/dreamdojo/exp_droid_sv/droid_singlearm_singleview/checkpoints \
  --experiment dreamdojo_2b_480_640_droid_sv \
  --save-dir ./output/dreamdojo_results/results_droid_finetuned_20k \
  --num-frames 97 \
  --num-samples 5 \
  --dataset-path "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/datasets/single_arm/singleview/makovian" \
  --data-split test \
  --deterministic-uniform-sampling

# DROID dataset
bash finetuning/dreamdojo/infer.sh \
  --dataset_path /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/datasets/single_arm/singleview/chunk-000 \
  --episode_idx 5 \
  --embodiment droid \
  --experiment_name dreamdojo_2b_480_640_droid_sv

# Options
#   --guidance 3.0        Classifier-free guidance scale
#   --seed 42             Random seed
#   --num_frames 13       Number of frames to generate (12 predicted + 1 conditional)
#   --output_dir ./infer_outputs
#   --fps 5               Output video FPS
```

Output: `infer_outputs/<dataset_name>_ep<N>_g<guidance>_s<seed>_{gt,pred,comparison}.mp4`

### Action dimension reference

| Embodiment | Range       | Dim |
|-----------|-------------|-----|
| GR-1      | [0, 29)     | 29  |
| GR-1 retarget | [29, 58) | 29  |
| G1        | [58, 101)   | 43  |
| YAM       | [101, 147)  | 46  |
| AgiBot    | [147, 169)  | 22  |
| ALOHA     | [169, 183)  | 14  |
| DROID     | [183, 190)  | 7   |
| Reserved  | [190, 220)  | 30  |
| MANO      | [220, 352)  | 132 |
| Latent    | [352, 384)  | 32  |




## DreamGen (Cosmos-Predict2 Video2World)

### Step 1: Prepare datasets

```bash
# Bimanual ALOHA
python finetuning/dreamgen/scripts/prepare_bimanual_dreamgen.py

# Single-arm DROID
python finetuning/dreamgen/scripts/prepare_single_arm_dreamgen.py

# Humanoid GR1 — generate LLM captions first, then write humanoid_dreamgen/
bash scripts/run_caption_humanoid_singleview.sh
```

### Step 2: Pre-compute T5-XXL embeddings (all 3 at once)

```bash
# All three datasets in one command
bash finetuning/dreamgen/scripts/compute_t5_embeddings.sh

# Or a single dataset
DATASETS=humanoid bash finetuning/dreamgen/scripts/compute_t5_embeddings.sh
```

### Step 3: Train

```bash
source models/dreamgen/.venv/bin/activate

# All 3 datasets combined (bimanual + single_arm + humanoid) — recommended
NPROC=8 bash finetuning/dreamgen/launch.sh predict2_video2world_training_2b_all_singleview

# Or train each embodiment separately
NPROC=8 bash finetuning/dreamgen/launch.sh predict2_video2world_training_2b_bimanual_singleview
NPROC=8 bash finetuning/dreamgen/launch.sh predict2_video2world_training_2b_single_arm_singleview
NPROC=8 bash finetuning/dreamgen/launch.sh predict2_video2world_training_2b_humanoid_singleview

# Single GPU debug
NPROC=1 bash finetuning/dreamgen/launch.sh predict2_video2world_training_2b_all_singleview
```

Checkpoints saved to: `exp_logs/dreamgen_logs/posttraining/singleview/all_singleview/`

### Step 4: Inference

```bash
bash finetuning/dreamgen/infer.sh \
    --ckpt_path exp_logs/dreamgen_logs/posttraining/singleview/all_singleview/checkpoints/model/iter_000005000.pt \
    --input_path /path/to/first_frame.png \
    --prompt "A GR-1 humanoid robot picks up a corn cob and places it on the plate." \
    --save_path output/dreamgen_all_infer.mp4
```