
# video_gen_physics


## Ctrl world

export CUDA_VISIBLE_DEVICES=0 
export PYTHONPATH="./models/Ctrl-World" 
python models/Ctrl-World/scripts/rollout_replay_traj.py \
  --dataset_root_path ./models/Ctrl-World/dataset_example \
  --dataset_meta_info_path ./models/Ctrl-World/dataset_meta_info \
  --dataset_names ./models/Ctrl-World/droid_subset \
  --svd_model_path ./checkpoints/stabilityai/stable-video-diffusion-img2vid \
  --clip_model_path ./checkpoints/openai/clip-vit-base-patch32 \
  --ckpt_path ./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt



CUDA_VISIBLE_DEVICES=0 python scripts/rollout_key_board.py  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info --dataset_names droid_subset --svd_model_path ${path to svd folder} --clip_model_path /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/checkpoints/openai/clip-vit-base-patch32 --ckpt_path ${path to ctrl-world ckpt} --task_type keyboard --keyboard lllrrr


### Default
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/Ctrl-World"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python models/Ctrl-World/scripts/rollout_interact_pi.py \
  --dataset_root_path ./models/Ctrl-World/dataset_example \
  --dataset_meta_info_path ./models/Ctrl-World/dataset_meta_info \
  --dataset_names ./models/Ctrl-World/droid_subset \
  --svd_model_path ./checkpoints/stabilityai/stable-video-diffusion-img2vid \
  --clip_model_path ./checkpoints/openai/clip-vit-base-patch32 \
  --ckpt_path ./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt \
  --task_type pickplace

### Use with Pisa
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/Ctrl-World"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python models/Ctrl-World/scripts/rollout_interact_pi.py \
  --dataset_root_path ./models/Ctrl-World/dataset_example \
  --dataset_meta_info_path ./models/Ctrl-World/dataset_meta_info \
  --dataset_names ./models/Ctrl-World/droid_subset \
  --svd_model_path ./checkpoints/stabilityai/stable-video-diffusion-img2vid \
  --clip_model_path ./checkpoints/openai/clip-vit-base-patch32 \
  --ckpt_path ./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt \
  --task_type pickplace \
  --use_pisa --pisa_density 0.5 --pisa_block_size 32

### Use-svg v1
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/Ctrl-World"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python models/Ctrl-World/scripts/rollout_interact_pi.py \
  --dataset_root_path ./models/Ctrl-World/dataset_example \
  --dataset_meta_info_path ./models/Ctrl-World/dataset_meta_info \
  --dataset_names ./models/Ctrl-World/droid_subset \
  --svd_model_path ./checkpoints/stabilityai/stable-video-diffusion-img2vid \
  --clip_model_path ./checkpoints/openai/clip-vit-base-patch32 \
  --ckpt_path ./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt \
  --task_type pickplace \
  --use_svg --svg_pattern SVG1 --svg_sparsity 0.25

### Use-svg v2
export VENV=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv
export LD_LIBRARY_PATH=$VENV/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$VENV/lib/python3.10/site-packages/nvidia/cusparse/lib:$VENV/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/Ctrl-World"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python models/Ctrl-World/scripts/rollout_interact_pi.py \
  --dataset_root_path ./models/Ctrl-World/dataset_example \
  --dataset_meta_info_path ./models/Ctrl-World/dataset_meta_info \
  --dataset_names ./models/Ctrl-World/droid_subset \
  --svd_model_path ./checkpoints/stabilityai/stable-video-diffusion-img2vid \
  --clip_model_path ./checkpoints/openai/clip-vit-base-patch32 \
  --ckpt_path ./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt \
  --task_type pickplace \
  --use_svg --svg_pattern SVG2 --svg_num_q_centroids 50 --svg_num_k_centroids 200


### use with radial attention (padding)
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/Ctrl-World"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python models/Ctrl-World/scripts/rollout_interact_pi.py \
  --dataset_root_path ./models/Ctrl-World/dataset_example \
  --dataset_meta_info_path ./models/Ctrl-World/dataset_meta_info \
  --dataset_names ./models/Ctrl-World/droid_subset \
  --svd_model_path ./checkpoints/stabilityai/stable-video-diffusion-img2vid \
  --clip_model_path ./checkpoints/openai/clip-vit-base-patch32 \
  --ckpt_path ./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt \
  --task_type pickplace \
  --use_radial \
  --radial_pad_small_layers 


### use with radial attention (no padding)

source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/Ctrl-World"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python models/Ctrl-World/scripts/rollout_interact_pi.py \
  --dataset_root_path ./models/Ctrl-World/dataset_example \
  --dataset_meta_info_path ./models/Ctrl-World/dataset_meta_info \
  --dataset_names ./models/Ctrl-World/droid_subset \
  --svd_model_path ./checkpoints/stabilityai/stable-video-diffusion-img2vid \
  --clip_model_path ./checkpoints/openai/clip-vit-base-patch32 \
  --ckpt_path ./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt \
  --task_type pickplace \
  --use_radial


### use with radial attention (no padding)

source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/Ctrl-World"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python models/Ctrl-World/scripts/rollout_interact_pi.py \
  --dataset_root_path ./models/Ctrl-World/dataset_example \
  --dataset_meta_info_path ./models/Ctrl-World/dataset_meta_info \
  --dataset_names ./models/Ctrl-World/droid_subset \
  --svd_model_path ./checkpoints/stabilityai/stable-video-diffusion-img2vid \
  --clip_model_path ./checkpoints/openai/clip-vit-base-patch32 \
  --ckpt_path ./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt \
  --task_type pickplace \
  --use_radial \
  --radial_decay_factor 0.5 \
  --radial_model_type hunyuan \
  --radial_first_layers_fp 0
## dreamgen base 
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0 
export NUM_GPUS=1
python -m scripts.prepare_batch_input_json \
  --dataset_path ./datasets/dream_gen_benchmark/gr1_env \
  --save_path ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_env \
  --output_path ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_env/batch_input.json

### dreamgen base - one gpu
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0 
export NUM_GPUS=1
torchrun -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS}

### dreamgen base - 4 gpu
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0,1,2,3 
export NUM_GPUS=4
torchrun -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS}

### dreamgen - pisa
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/methods/sparse_attention/piecewise-sparse-attention
uv pip install -e .

source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0 
export NUM_GPUS=1
torchrun -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_env/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS} \
  --use_pisa \
  --pisa_density 0.2 \
  --pisa_block_size 64 \
  --pisa_start_layer_idx 2

source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0,1
export NUM_GPUS=2
torchrun -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS} \
  --use_pisa \
  --pisa_density 0.1 \
  --pisa_block_size 64 \
  --pisa_use_bias

### dreamgen - radial
Radial is not supported in the default `DreamDojo/.venv`. It requires a separate env with `flashinfer`.

cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics
bash ./scripts/bootstrap_radial_env.sh

source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
torchrun -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS} \
  --use_radial \
  --radial_decay_factor 1.0 \
  --radial_block_size 64 \
  --radial_start_layer_idx 4 \
  --radial_model_type hunyuan

### dreamgen - svg
`SVG1` and `SVG2` have different environment requirements.

`SVG1` can run in the default `DreamDojo/.venv` if that Torch build exposes `torch.nn.attention.flex_attention`.

`SVG2` is not supported in the default `DreamDojo/.venv`. Use a separate env with `Sparse-VideoGen`, `flashinfer`, `cuvs`, and built SVG kernels.

Do not install `cuvs-cu12` into `DreamDojo/.venv`. That env currently uses a Torch CUDA 12.8 stack, and `cuvs-cu12` upgrades parts of the CUDA userspace to 12.9, which breaks `torch` import with `libcusparse.so.12` / `libnvJitLink.so.12` symbol errors.

cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics
bash ./scripts/bootstrap_svg_env.sh

source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1

### dreamgen - svg1
SVG1 uses DreamGen's `flex_attention + placement kernels` path. It does not require `flashinfer` or `cuvs`, but it does require a Torch build with `flex_attention`.

torchrun -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS} \
  --use_svg \
  --svg_variant svg1 \
  --svg_start_layer_idx 4 \
  --svg_dense_step_frac 0.1 \
  --svg_sparsity 0.25 \
  --svg_num_sampled_rows 64 \
  --svg_sample_mse_max_row 3000


Speed-focused (max throughput, some quality loss):

torchrun -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS} \
  --use_svg \
  --svg_variant svg1 \
  --svg_start_layer_idx 4 \
  --svg_dense_step_frac 0.05 \
  --svg_sparsity 0.15 \
  --svg_num_sampled_rows 16 \
  --svg_sample_mse_max_row 1500

Quality-focused (only SVG on spatially-local layers):


--svg_start_layer_idx 20 \
--svg_dense_step_frac 0.1 \
--svg_sparsity 0.25 \
--svg_num_sampled_rows 64 \
--svg_sample_mse_max_row 3000


### dreamgen - svg2
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1

torchrun --master_port=29601 -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS} \
  --use_svg \
  --svg_variant svg2 \
  --svg_start_layer_idx 1 \
  --svg_dense_step_frac 0.3 \
  --svg_num_q_centroids 150 \
  --svg_num_k_centroids 400 \
  --svg_top_p_kmeans 0.9 \
  --svg_min_kc_ratio 0.10 \
  --svg_kmeans_iter_init 20 \
  --svg_kmeans_iter_step 2

## dreamdojo
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/DreamDojo"
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
python -m cosmos_predict2._src.predict2.inference.inference_batch \
  --experiment "Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only_resume2" \
  --ckpt_path ./checkpoints/dreamdojo/2B_pretrain/iter_000140000/model \
  --batch_input ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.jsonl \
  --output_dir ./output/dream_gen_benchmark/output_base_2b \
  --resolution 480,640 \
  --guidance 5 \
  --num_output_frames 77 \
  --num_steps 35 \
  --use_neg_prompt





source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/DreamDojo"
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics

export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
hash -r
python -m models.DreamDojo.examples.action_conditioned \
  -o ./output/dreamdojo_hv_eval_2b_g1_gr1 \
  --checkpoints-dir ./checkpoints/dreamdojo/2B_GR1_post-train/iter_000050000 \
  --experiment dreamdojo_2b_480_640_gr1 \
  --save-dir ./output/dreamdojo_results/dreamdojo_GR1_robot_2b_g1_gr1 \
  --num-frames 49 \
  --num-samples 65 \
  --dataset-path "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/checkpoints/dreamdojo/datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot" \
  --data-split test \
  --deterministic-uniform-sampling \
  --checkpoint-interval 5000 \
  --infinite

### dreamdojo svg 1
python -m models.DreamDojo.examples.action_conditioned \
  -o ./output/dreamdojo_svg1_eval \
  --checkpoints-dir ./checkpoints/dreamdojo/2B_GR1_post-train/iter_000050000 \
  --experiment dreamdojo_2b_480_640_gr1 \
  --save-dir ./output/dreamdojo_results/svg1_results \
  --num-frames 49 \
  --num-samples 65 \
  --dataset-path "./checkpoints/dreamdojo/datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot" \
  --data-split test \
  --deterministic-uniform-sampling \
  --use_svg \
  --svg_variant svg1

python -m models.DreamDojo.examples.action_conditioned \
  -o ./output/dreamdojo_svg1_eval \
  --checkpoints-dir ./checkpoints/dreamdojo/2B_GR1_post-train/iter_000050000 \
  --experiment dreamdojo_2b_480_640_gr1 \
  --save-dir ./output/dreamdojo_results/svg1_results \
  --num-frames 49 \
  --num-samples 65 \
  --dataset-path "./checkpoints/dreamdojo/datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot" \
  --data-split test \
  --deterministic-uniform-sampling \
  --use_svg \
  --svg_variant svg1 \
  --svg_start_layer_idx 2 \
  --svg_sparsity 0.2 \
  --svg_dense_step_frac 0.1 \
  --svg_start_layer_idx 4

### dreadojo svg2
python -m models.DreamDojo.examples.action_conditioned \
  -o ./output/dreamdojo_svg2_eval \
  --checkpoints-dir ./checkpoints/dreamdojo/2B_GR1_post-train/iter_000050000 \
  --experiment dreamdojo_2b_480_640_gr1 \
  --save-dir ./output/dreamdojo_results/svg2_results \
  --num-frames 49 \
  --num-samples 65 \
  --dataset-path "./checkpoints/dreamdojo/datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot" \
  --data-split test \
  --deterministic-uniform-sampling \
  --use_svg \
  --svg_start_layer_idx 2 \
  --svg_variant svg2


cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics
python -m models.DreamDojo.examples.action_conditioned \
  -o ./output/dreamdojo_svg2_eval \
  --checkpoints-dir ./checkpoints/dreamdojo/2B_GR1_post-train/iter_000050000 \
  --experiment dreamdojo_2b_480_640_gr1 \
  --save-dir ./output/dreamdojo_results/svg2_results \
  --num-frames 49 \
  --num-samples 65 \
  --dataset-path "./checkpoints/dreamdojo/datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot" \
  --data-split test \
  --deterministic-uniform-sampling \
  --use_svg \
  --svg_variant svg2 \
  --svg_start_layer_idx 18 \
  --svg_num_q_centroids 8 \
  --svg_num_k_centroids 16 \
  --svg_top_p_kmeans 0.3 \
  --svg_kmeans_iter_init 2 \
  --svg_kmeans_iter_step 1

### dreamgen - sito
SiTo is inference-only, self-attention only, and single-GPU only in DreamGen v1.

```bash
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1

torchrun -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS} \
  --use_sito \
  --sito_start_layer_idx 6 \
  --sito_keep_last_n_dense 2
```

### dreamdojo - sito
SiTo uses post-patchify per-frame token grouping. For the common `(4, 60, 80)` latent grid, the DiT sequence is `4 * 30 * 40 = 4800` tokens.

```bash
python -m models.DreamDojo.examples.action_conditioned \
  -o ./output/dreamdojo_sito_eval \
  --checkpoints-dir ./checkpoints/dreamdojo/2B_GR1_post-train/iter_000050000 \
  --experiment dreamdojo_2b_480_640_gr1 \
  --save-dir ./output/dreamdojo_results/sito_results \
  --num-frames 49 \
  --num-samples 65 \
  --dataset-path "./checkpoints/dreamdojo/datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot" \
  --data-split test \
  --deterministic-uniform-sampling \
  --use_sito \
  --sito_start_layer_idx 4 \
  --sito_keep_last_n_dense 2
```

### ctrl-world - sito
SiTo is applied only on large spatial self-attention layers in Ctrl-World.

```bash
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/Ctrl-World"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python models/Ctrl-World/scripts/rollout_interact_pi.py \
  --dataset_root_path ./models/Ctrl-World/dataset_example \
  --dataset_meta_info_path ./models/Ctrl-World/dataset_meta_info \
  --dataset_names ./models/Ctrl-World/droid_subset \
  --svd_model_path ./checkpoints/stabilityai/stable-video-diffusion-img2vid \
  --clip_model_path ./checkpoints/openai/clip-vit-base-patch32 \
  --ckpt_path ./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt \
  --task_type pickplace \
  --use_sito \
  --sito_max_downsample_ratio 4
```
