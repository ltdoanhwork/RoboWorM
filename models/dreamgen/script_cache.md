## script to test env 
```
python scripts/test_environment.py
python scripts/test_environment.py --training
```

## You can install install flast-attn, transformer-engine, natten = wheel


```
conda create -y cosmospredict python==3.10 -y
conda activate cosmospredict 
pip install uv 

cd <dir/to/your/code>
uv pip install -e .
uv pip install -r requirements.txt

wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.6.3+cu124torch2.6-cp310-cp310-linux_x86_64.whl
wget https://pypi.nvidia.com/transformer-engine/transformer_engine-1.13.0-py3-none-any.whl#sha256=9a7482ed47512e888fded3361153a60846ad0dd6a98d86b77f69dd60194dab05
wget https://github.com/SHI-Labs/NATTEN/releases/download/v0.17.5/natten-0.17.5%2Btorch260cu124-cp310-cp310-linux_x86_64.whl

uv pip install (these downloaed .whl file above)


```

## my draft
```
uv pip install --no-build-isolation 'transformer_engine[pytorch]==1.13.0'
 CUDA_HOME=$CONDA_PREFIX uv pip install transformer-engine[pytorch]==1.13.0

CUDA_HOME=$CONDA_PREFIX uv pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext --cuda_ext" git+https://github.com/NVIDIA/apex.git

srun --pty --job-name=dreamgen \
--partition=main \
--nodes=1 \
--nodelist=worker-0,worker-1 \
--ntasks=1 \
--gpus=nvidia_h100_80gb_hbm3:2 \
--cpus-per-task=32 \
--mem=120G \
--time=48:00:00 \
bash -i

srun --pty --job-name=dreamgen \
--partition=main \
--nodes=1 \
--nodelist=worker-0,worker-1 \
--ntasks=1 \
--gpus=nvidia_h100_80gb_hbm3:4 \
--cpus-per-task=64 \
--mem=256G \
--time=240:00:00 \
bash -i

srun --pty --job-name=dreamgen \
--partition=main \
--ntasks=1 \
--gpus=nvidia_h100_80gb_hbm3:4 \
--cpus-per-task=64 \
--mem=256G \
--time=200:00:00 \
bash -i



```

## For training, we need apex but sometime, your server use CUDA toolkit 12.x but apex or torch use cuda 12.4
```
wget <you can search the internet to download cuda tookit.run file. E.x: cuda_12.4.0_550.54.14_linux.run>

# run the following commands
sh /mnt/data/sftp/data/vla_intern/workspace/binh/dreamgen/cuda_setup/cuda_12.4.0_550.54.14_linux.run --silent --toolkit --toolkitpath=/mnt/data/sftp/data/vla_intern/workspace/binh/dreamgen/cuda_setup/cuda_12.4

export CUDA_HOME=/mnt/data/sftp/data/vla_intern/workspace/binh/dreamgen/cuda_setup/cuda_12.4

pip install -v --no-cache-dir --no-build-isolation \
  --config-settings=--build-option=--cpp_ext \
  --config-settings=--build-option=--cuda_ext \
  git+https://github.com/NVIDIA/apex.git
```

## command to prepare `t5_xxl/....pickle` files:
```
python -m scripts.get_t5_embeddings \
  --dataset_path /mnt/data/sftp/data/vla_intern/workspace/binh/dreamgen/dreamgen/datasets/robocasa-100demos-5tasks-for-dg

python -m scripts.get_t5_embeddings \
  --dataset_path /mnt/data/sftp/data/vla_intern/workspace/binh/dreamgen/dreamgen/datasets/realworld-for-dg-training-v1
```


## create checkpoints folder to store checkpoints
```
export HF_TOKEN=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
nvidia/Cosmos-Predict2.5-2B
nvidia/Cosmos-Predict2-2B-Video2World

huggingface-cli download nvidia/Cosmos-Predict2-2B-Video2World --local-dir ./Cosmos-Predict2-2B-Video2World

huggingface-cli download nvidia/Cosmos-Predict2.5-2B --local-dir ./Cosmos-Predict2.5-2B

huggingface-cli download nvidia/Cosmos-Reason1-7B --local-dir ./Cosmos-Reason1-7B
```

## Sometime, server/environment fail to decode your videos, run following command to convert to correct mp4 file for decoder
```
for f in datasets/path/videos/*.mp4; do
    ffmpeg -y -i "$f" -c:v libx264 -pix_fmt yuv420p -c:a copy "${f}.tmp.mp4" && mv "${f}.tmp.mp4" "$f"
done
```

## command to debug inference
```
torchrun --nproc_per_node=4 examples/video2world_lvg.py \
  --num_gpus 4 \
  --model_size 2B \
  --dit_path /path/to/checkpoint.pt \
  --batch_input_json /tmp/manifest.json \
  --num_conditional_frames 1 \
  --num_chunks 3 \
  --disable_guardrail

torchrun --nproc_per_node=4 examples/video2world_lvg.py \
  --num_gpus 4 \
  --model_size 2B \
  --dit_path /mnt/data/sftp/data/vla_intern/workspace/binh/dreamgen/dreamgen/checkpoints/posttraining/video2world/2b_groot_gr1_480/checkpoints_40_per/model/iter_000000280.pt \
  --input_path data4infer/test_screenshots/frame_0.jpg \
  --prompt "robot pick all objects from cabinet to counter" \
  --num_conditional_frames 1 \
  --num_chunks 3 \
  --aspect_ratio "1:1" \
  --disable_guardrail 
```

## my draft
```
rsync -avzP vinrobot@slurm:/mnt/data/sftp/data/vla_intern/workspace/binh/dreamgen/dreamgen/chosen_checkpoints ./chosen 
 

bash bash-running-script/Inference-bash-dreamgen/inference-fpt.sh
bash bash-running-script/Finetuning-bash-dreamgen/debug.sh
bash bash-running-script/Finetuning-bash-dreamgen/robocasa_100_training.sh

bash bash-running-script/Finetuning-bash-dreamgen/realworld_training.sh
```

## check `extract_video_frame_for_inference.py` to prepare images for video inference