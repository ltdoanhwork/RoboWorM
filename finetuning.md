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
NPROC=2 bash finetuning/dreamdojo/launch.sh dreamdojo_2b_480_640_single_arm_sv

# Single GPU debug
NPROC=1 bash finetuning/dreamdojo/launch.sh dreamdojo_2b_480_640_aloha

# Resume from iter_40k, train on all LeRobot roots under datasets/bimanual/singleview (54 tasks),
# save to a new job folder (not exp_aloha/aloha_bimanual_singleview)
NPROC=2 bash finetuning/dreamdojo/launch.sh dreamdojo_2b_480_640_aloha_resume40k_bimanual_sv

# Resume from iter_70k → 100k (saves to exp_aloha_resume70k/aloha_bimanual_singleview_full/)
NPROC=2 bash finetuning/dreamdojo/launch.sh dreamdojo_2b_480_640_aloha_resume70k_bimanual_sv
```

Configs: `models/DreamDojo/configs/2b_480_640_aloha.yaml`, `2b_480_640_droid_sv.yaml`, `2b_480_640_aloha_resume40k_bimanual_sv.yaml` (resume + full bimanual singleview list), `2b_480_640_aloha_resume70k_bimanual_sv.yaml` (resume 70k→100k)

Checkpoints saved to: `exp_logs/dreamdojo_logs/dreamdojo/exp_<group>/<name>/checkpoints/`
Video samples logged to: `exp_logs/dreamdojo_logs/dreamdojo/exp_<group>/<name>/video_logs/`

Resume run above writes to `exp_logs/dreamdojo_logs/dreamdojo/exp_aloha_resume40k/aloha_bimanual_singleview_full/` and loads weights plus trainerizer/trainer/trainer state from `.../checkpoints/iter_000040000/`. If resume errors, edit that YAML and set `checkpoint.load_training_state: false` (weights only).

Resume 70k run writes to `exp_logs/dreamdojo_logs/dreamdojo/exp_aloha_resume70k/aloha_bimanual_singleview_full/` and loads full training state from `.../exp_aloha_resume40k/.../checkpoints/iter_000070000/`.

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
  --save-dir ./output/dreamdojo_results/results_droid_finetuned_40k \
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
NPROC=4 bash finetuning/dreamgen/launch.sh predict2_video2world_training_2b_all_singleview

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

NPROC=4 bash finetuning/dreamgen/launch.sh predict2_video2world_training_2b_all_multiview


NPROC=4 bash finetuning/dreamgen/launch.sh predict2_video2world_training_2b_all_singleview

# ctrl world


bash scripts/run_single_arm_multiview_ctrlworld.sh both

CUDA_VISIBLE_DEVICES=1 bash scripts/run_single_arm_singleview_dreamdojo_dense.sh

WORLDCACHE_REL_L1_THRESH=0.05 WORLDCACHE_PROBE_DEPTH=3 bash scripts/run_humanoid_singleview_dreamdojo_worldcache.sh


FASTERCACHE_MODEL_INTERVAL=3 FASTERCACHE_BLOCK_INTERVAL=2 bash scripts/run_humanoid_singleview_dreamdojo_fastercache.sh


CUDA_VISIBLE_DEVICES=1 bash /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/scripts/run_humanoid_singleview_dreamdojo_dicache.sh


CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_singleview_dreamdojo_worldcache.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_singleview_dreamdojo_fastercache.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_singleview_dreamdojo_dicache.sh

CUDA_VISIBLE_DEVICES=1 bash scripts/run_single_arm_singleview_dreamdojo_worldcache.sh
CUDA_VISIBLE_DEVICES=1 bash scripts/run_single_arm_singleview_dreamdojo_fastercache.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_singleview_dreamdojo_dicache.sh

CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_bimanual_multiview_dreamgen_dense.sh
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_humanoid_multiview_dreamgen_dense.sh

source models/DreamDojo/.venv/bin/activate
hf upload-large-folder doanh25032004/video_gen_physics_real_video /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/datasets  --repo-type dataset
rm -rf /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/exp_logs/dreamgen_logs/posttraining/multiview/all_multiview/checkpoints_40_per/trainer/iter_00004*
rm -rf /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/exp_logs/dreamgen_logs/posttraining/multiview/all_multiview/checkpoints_40_per/trainer/iter_00005*
rm -rf /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/exp_logs/dreamgen_logs/posttraining/multiview/all_multiview/checkpoints_40_per/trainer/iter_00006*
rm -rf /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/exp_logs/dreamgen_logs/posttraining/multiview/all_multiview/checkpoints_40_per/trainer/iter_00007*
rm -rf /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/exp_logs/dreamgen_logs/posttraining/multiview/all_multiview/checkpoints_40_per/trainer/iter_00008*

CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_all_singleview_dreamgen_dense.sh

CUDA_VISIBLE_DEVICES=0,1 EMBODIMENTS="bimanual" bash scripts/run_all_singleview_dreamgen_dense.sh
CUDA_VISIBLE_DEVICES=0,1 EMBODIMENTS="humanoid" bash scripts/run_all_singleview_dreamgen_dense.sh
CUDA_VISIBLE_DEVICES=0,1 EMBODIMENTS="single_arm" bash scripts/run_all_singleview_dreamgen_dense.sh

# Single arm (fps=15)
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash scripts/run_dreamgen_singleview_native_fps.sh
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 EMBODIMENT=single_arm bash scripts/run_dreamgen_singleview_native_fps.sh
# Bimanual (fps=30)
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 EMBODIMENT=bimanual bash scripts/run_dreamgen_singleview_native_fps.sh

# Humanoid (fps=20)
CUDA_VISIBLE_DEVICES=0 EMBODIMENT=humanoid bash scripts/run_dreamgen_singleview_native_fps.sh

SKIP_PREP=1 NPROC=4 bash finetuning/dreamgen/train_single_arm_static_16fps.sh
SKIP_PREP=1 NPROC=4 bash finetuning/dreamgen/train_bimanual_16fps.sh

NUM_EPISODES=10 CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_singleview_dreamdojo_long_autoregressive.sh

# 1. DreamGen humanoid singleview + SiTo
CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_singleview_dreamgen_sito.sh

# 2. DreamGen humanoid singleview + ITM
CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_singleview_dreamgen_itm.sh

# 3. Ctrl-World single_arm multiview + SiTo -> need to verified actually 
CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_multiview_ctrlworld_sito.sh both

# 4. Ctrl-World single_arm multiview + ITM -> need to verified actually
CUDA_VISIBLE_DEVICES=1 bash scripts/run_single_arm_multiview_ctrlworld_itm.sh both


CUDA_VISIBLE_DEVICES=0 USE_PROMPT_REFINEMENT=1 bash scripts/run_dreamgen_singleview_native_fps.sh

# 1. SiTo
CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_singleview_dreamgen_sito.sh

# 2. ITM
CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_singleview_dreamgen_itm.sh

# 3. PISA
CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_singleview_dreamgen_pisa.sh

# 4. Radial
CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_singleview_dreamgen_radial.sh

# 5. SVG1
CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_singleview_dreamgen_svg1.sh

# 6. SVG2
CUDA_VISIBLE_DEVICES=1 bash scripts/run_humanoid_singleview_dreamgen_svg2.sh

# 7. Prompt Refinement per Chunk
CUDA_VISIBLE_DEVICES=0,1 ITER_NAME=iter_000090000 bash scripts/run_humanoid_singleview_dreamgen_prompt_refine.sh



NPROC=4 bash finetuning/dreamgen/train_humanoid_16fps.sh
SKIP_PREP=1 NPROC=4 bash finetuning/dreamgen/train_single_arm_static_16fps.sh

CUDA_VISIBLE_DEVICES=0 ITER_NAME=iter_000080000 bash scripts/run_single_arm_singleview_dreamgen_static16fps.sh

CUDA_VISIBLE_DEVICES=0,1 ITER_NAME=iter_000090000 PROMPT_REFINE=1 bash scripts/run_single_arm_singleview_dreamgen_static16fps.sh

CUDA_VISIBLE_DEVICES=2,3 ITER_NAME=iter_000070000 PROMPT_REFINE=0 bash scripts/run_bimanual_singleview_dreamgen_16fps.sh -> khá tốt

CUDA_VISIBLE_DEVICES=2,3 ITER_NAME=iter_000070000 PROMPT_REFINE=0 bash scripts/run_bimanual_singleview_dreamgen_16fps.sh 

CUDA_VISIBLE_DEVICES=0,1 ITER_NAME=iter_000070000 PROMPT_REFINE=1 MASTER_PORT=29600 bash scripts/run_bimanual_singleview_dreamgen_16fps.sh 

CUDA_VISIBLE_DEVICES=2,3 ITER_NAME=iter_000120000 PROMPT_REFINE=0 bash scripts/run_bimanual_singleview_dreamgen_16fps.sh 

CUDA_VISIBLE_DEVICES=2,3 ITER_NAME=iter_000110000 PROMPT_REFINE=0 MASTER_PORT=29700 bash scripts/run_bimanual_singleview_dreamgen_16fps.sh

CUDA_VISIBLE_DEVICES=0,1,2,3 ITER_NAME=iter_000100000 PROMPT_REFINE=1 bash scripts/run_single_arm_singleview_dreamgen_static16fps.sh

CUDA_VISIBLE_DEVICES=0,1,2,3 PROMPT_REFINE=1 ITER_NAME=iter_000070000 bash scripts/run_single_arm_singleview_dreamgen_base.sh

CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash scripts/run_humanoid_singleview_dreamgen_base.sh

CUDA_VISIBLE_DEVICES=0,1, NUM_GPUS=2 ITER_NAME=iter_000100000 bash scripts/run_humanoid_singleview_dreamgen_static16fps.sh

CUDA_VISIBLE_DEVICES=2,3 NUM_GPUS=2 ITER_NAME=iter_000090000 PROMPT_REFINE=1 bash scripts/run_humanoid_singleview_dreamgen_static16fps.sh

CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_singleview_dreamdojo_worldcache.sh

CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_singleview_dreamdojo_dicache.sh



CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_singleview_dreamdojo_worldcache.sh

CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_singleview_dreamdojo_dicache.sh -> oke

CUDA_VISIBLE_DEVICES=1 bash scripts/run_single_arm_singleview_dreamdojo_fastercache.sh 

CUDA_VISIBLE_DEVICES=1 bash scripts/run_bimanual_singleview_dreamdojo_fastercache.sh

CUDA_VISIBLE_DEVICES=1 bash scripts/run_single_arm_multiview_dreamgen_worldcache.sh
CUDA_VISIBLE_DEVICES=1 bash scripts/run_single_arm_multiview_dreamgen_fastercache.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_multiview_dreamgen_dicache.sh

cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 SKIP_PREP=1 SKIP_T5=1 \
  bash finetuning/dreamgen/train_bimanual_multiview_grid.sh

CUDA_VISIBLE_DEVICES=1 NUM_GPUS=1 ITER_NAME=iter_000090000 MASTER_PORT=29650 bash scripts/run_single_arm_singleview_dreamgen_static16fps.sh
MAX_EPISODES=5 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_multiview_dreamgen_seg_test.sh



MAX_EPISODES=5 NUM_CONDITIONAL_FRAMES=5 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_multiview_dreamgen_seg_rollout.sh

MAX_EPISODES=5 NUM_CONDITIONAL_FRAMES=1 NUM_GPUS=2 CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_bimanual_multiview_dreamgen_seg_rollout.sh


CUDA_VISIBLE_DEVICES=0 CATEGORY=makovian MAX_EPISODES=4 \
  bash scripts/run_single_arm_multiview_dreamgen_itm.sh


Cấu hình	Total inference	Self-Attn / step	DiT MLP / step
Dense (baseline)
516.1s        4.66s                         1.88s


CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_multiview_dreamgen_itm.sh

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_single_arm_multiview_dreamgen_sito.sh 


cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics
CKPT=exp_logs/ctrlworld_logs/bimanual_multiview/checkpoint-120000.pt \
CUDA_VISIBLE_DEVICES=0 NUM_EPISODES=3 SPLIT=val GUIDANCE=2.5 \
  bash scripts/run_bimanual_multiview_ctrlworld.sh

CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 SPLITS="makovian" NUM_CHUNKS="2" bash scripts/run_single_arm_singleview_dreamgen_static16fps.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_singleview_dreamgen_worldcache.sh ; \
CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_singleview_dreamgen_fastercache.sh ; \
CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_singleview_dreamgen_dicache.sh

#run lai
CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_singleview_dreamdojo_all_methods.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_singleview_dreamdojo_all_methods.sh

CUDA_VISIBLE_DEVICES=1 bash scripts/run_single_arm_multiview_ctrlworld_sito.sh both


CKPT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/exp_logs/ctrlworld_logs/bimanual_multiview_5fps/checkpoint-280000.pt" \
DATASET_NAME=bimanual_multiview_5fps \
DOWN_SAMPLE=6 \
SPLIT=all \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_bimanual_multiview_ctrlworld.sh

CKPT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/exp_logs/ctrlworld_logs/humanoid_multiview/checkpoint-150000.pt" \
DATASET_NAME=humanoid_multiview \
DOWN_SAMPLE=6 \
SPLIT=all \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_humanoid_multiview_ctrlworld.sh


cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics

# WorldCache (tuned optimum: thresh 0.04 / ret 0.4)
CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_multiview_ctrlworld_worldcache.sh
CUDA_VISIBLE_DEVICES=1 bash scripts/run_bimanual_multiview_ctrlworld_dicache.sh

CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_multiview_dreamgen_dicache.sh && \
CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_multiview_dreamgen_worldcache.sh && \
CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_multiview_dreamgen_fastercache.sh

# ---- tmux submit: 3 DreamGen chia đều GPU0/1/2, 2 Ctrl-World tuần tự trên GPU3 ----
# (logs -> exp_logs/tmux_logs/, attach: tmux attach -t <name>)
tmux new-session -d -s dg_dicache_g0     "CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_multiview_dreamgen_dicache.sh     2>&1 | tee exp_logs/tmux_logs/dg_dicache.log; exec bash"
tmux new-session -d -s dg_worldcache_g1  "CUDA_VISIBLE_DEVICES=1 bash scripts/run_bimanual_multiview_dreamgen_worldcache.sh  2>&1 | tee exp_logs/tmux_logs/dg_worldcache.log; exec bash"
tmux new-session -d -s dg_fastercache_g2 "CUDA_VISIBLE_DEVICES=2 bash scripts/run_bimanual_multiview_dreamgen_fastercache.sh 2>&1 | tee exp_logs/tmux_logs/dg_fastercache.log; exec bash"
tmux new-session -d -s cw_cache_g3       "CUDA_VISIBLE_DEVICES=3 bash scripts/run_bimanual_multiview_ctrlworld_worldcache.sh 2>&1 | tee exp_logs/tmux_logs/cw_worldcache.log && CUDA_VISIBLE_DEVICES=3 bash scripts/run_bimanual_multiview_ctrlworld_dicache.sh 2>&1 | tee exp_logs/tmux_logs/cw_dicache.log; exec bash"

# ---- tmux submit: bimanual SINGLEVIEW dreamgen (iter 110k), 3 cache trên GPU0/1/3 ----
tmux new-session -d -s bsv_dicache_g0     "CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_singleview_dreamgen_dicache.sh     2>&1 | tee exp_logs/tmux_logs/bsv_dicache.log; exec bash"
tmux new-session -d -s bsv_worldcache_g1  "CUDA_VISIBLE_DEVICES=1 bash scripts/run_bimanual_singleview_dreamgen_worldcache.sh  2>&1 | tee exp_logs/tmux_logs/bsv_worldcache.log; exec bash"
tmux new-session -d -s bsv_fastercache_g3 "CUDA_VISIBLE_DEVICES=3 bash scripts/run_bimanual_singleview_dreamgen_fastercache.sh 2>&1 | tee exp_logs/tmux_logs/bsv_fastercache.log; exec bash"

# ---- tmux submit: bimanual SINGLEVIEW dreamgen sparse-attn (iter 110k) GPU0/1/2/3 ----
# (SVG2 uses .venv_svg w/ flashinfer0.3 + workspace .jitcache; set in the script.)
tmux new-session -d -s sp_svg1_g0   "CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_singleview_dreamgen_svg1.sh   2>&1 | tee exp_logs/tmux_logs/sp_svg1.log; exec bash"
tmux new-session -d -s sp_svg2_g1   "CUDA_VISIBLE_DEVICES=1 bash scripts/run_bimanual_singleview_dreamgen_svg2.sh   2>&1 | tee exp_logs/tmux_logs/sp_svg2.log; exec bash"
tmux new-session -d -s sp_radial_g2 "CUDA_VISIBLE_DEVICES=2 bash scripts/run_bimanual_singleview_dreamgen_radial.sh 2>&1 | tee exp_logs/tmux_logs/sp_radial.log; exec bash"
tmux new-session -d -s sp_pisa_g3   "CUDA_VISIBLE_DEVICES=3 bash scripts/run_bimanual_singleview_dreamgen_pisa.sh   2>&1 | tee exp_logs/tmux_logs/sp_pisa.log; exec bash"

CUDA_VISIBLE_DEVICES=0,3 MAX_EPISODES=5 bash scripts/run_humanoid_multiview_dreamgen_seg16.sh

# ---- tmux submit: single_arm MULTIVIEW dreamgen sparse-attn (14B, droid) GPU0/1/2/3 ----
# (SVG2 uses .venv_svg w/ flashinfer0.3; set in the script. 14B is heavy ~490s/ep dense.)
tmux new-session -d -s sam_pisa_g0   "CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_multiview_dreamgen_pisa.sh   2>&1 | tee exp_logs/tmux_logs/sam_pisa.log; exec bash"
tmux new-session -d -s sam_svg1_g1   "CUDA_VISIBLE_DEVICES=1 bash scripts/run_single_arm_multiview_dreamgen_svg1.sh   2>&1 | tee exp_logs/tmux_logs/sam_svg1.log; exec bash"
tmux new-session -d -s sam_radial_g2 "CUDA_VISIBLE_DEVICES=2 bash scripts/run_single_arm_multiview_dreamgen_radial.sh 2>&1 | tee exp_logs/tmux_logs/sam_radial.log; exec bash"
tmux new-session -d -s sam_svg2_g3   "CUDA_VISIBLE_DEVICES=3 bash scripts/run_single_arm_multiview_dreamgen_svg2.sh   2>&1 | tee exp_logs/tmux_logs/sam_svg2.log; exec bash"

# ---- tmux submit: single_arm SINGLEVIEW dreamgen sparse-attn (2B, static16fps/LVG) GPU0/1/2/3 ----
# (SVG2 uses .venv_svg; PISA/SVG1/Radial use .venv. Tuned: PISA d0.2, SVG1 sp0.15, SVG2 top_p0.7.)
tmux new-session -d -s sasv_pisa_g0   "CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_singleview_dreamgen_pisa.sh   2>&1 | tee exp_logs/tmux_logs/sasv_pisa.log; exec bash"
tmux new-session -d -s sasv_svg1_g1   "CUDA_VISIBLE_DEVICES=1 bash scripts/run_single_arm_singleview_dreamgen_svg1.sh   2>&1 | tee exp_logs/tmux_logs/sasv_svg1.log; exec bash"
tmux new-session -d -s sasv_radial_g2 "CUDA_VISIBLE_DEVICES=2 bash scripts/run_single_arm_singleview_dreamgen_radial.sh 2>&1 | tee exp_logs/tmux_logs/sasv_radial.log; exec bash" -> done
tmux new-session -d -s sasv_svg2_g3   "CUDA_VISIBLE_DEVICES=3 bash scripts/run_single_arm_singleview_dreamgen_svg2.sh   2>&1 | tee exp_logs/tmux_logs/sasv_svg2.log; exec bash" -> done 
# ---- tmux submit: bimanual MULTIVIEW dreamgen sparse-attn (iter 120k) GPU0/1/2/3 ----
# (Radial=decay0.5/start4 for multiview; SVG2 uses .venv_svg/flashinfer0.3.)
tmux new-session -d -s smv_pisa_g0   "CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_multiview_dreamgen_pisa.sh   2>&1 | tee exp_logs/tmux_logs/smv_pisa.log; exec bash"
tmux new-session -d -s smv_radial_g1 "CUDA_VISIBLE_DEVICES=1 bash scripts/run_bimanual_multiview_dreamgen_radial.sh 2>&1 | tee exp_logs/tmux_logs/smv_radial.log; exec bash"
tmux new-session -d -s smv_svg1_g2   "CUDA_VISIBLE_DEVICES=2 bash scripts/run_bimanual_multiview_dreamgen_svg1.sh   2>&1 | tee exp_logs/tmux_logs/smv_svg1.log; exec bash"
tmux new-session -d -s smv_svg2_g3   "CUDA_VISIBLE_DEVICES=3 bash scripts/run_bimanual_multiview_dreamgen_svg2.sh   2>&1 | tee exp_logs/tmux_logs/smv_svg2.log; exec bash"
