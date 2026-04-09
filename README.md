cd /work/HHRI-AI/POC/public/DreamGen/VLA-Humanoid
cat > README.md << 'EOF'
# VLA-Humanoid: DreamGen Fine-tuning with COSMOS Predict

This repository contains the implementation for fine-tuning NVIDIA's COSMOS Predict model on robotic datasets for video-to-world generation, based on the DreamGen benchmark.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Model Weights Preparation](#model-weights-preparation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Project Structure](#project-structure)
- [Configuration Details](#configuration-details)

## 🔧 Prerequisites

- NVIDIA GPU with sufficient VRAM (4-8 GPUs recommended)
- Docker or Singularity/Apptainer
- SLURM cluster (optional, for distributed training)
- Python 3.8+ with PyTorch
- Weights & Biases (W&B) account for experiment tracking
- Hugging Face account for downloading models and datasets

## 🐳 Environment Setup

### Docker Container

Pull the official NVIDIA COSMOS container:
```bash
docker pull nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2
```

### Required Python Packages
```bash
pip install --user wandb ffmpegcv opencv-python-headless
```

## 📦 Model Weights Preparation

### 1. Download COSMOS Predict Model Weights

Download the original COSMOS Predict model weights from NVIDIA:

- **COSMOS Predict 2.5 Collection**: [https://huggingface.co/collections/nvidia/cosmos-predict25](https://huggingface.co/collections/nvidia/cosmos-predict25)

Available models:
- 2B model (recommended for 4 GPUs)
- 14B model (requires 8 GPUs)

Place the model weights in:
```
/work/HHRI-AI/POC/public/pretraining_weights/NVIDIA/Cosmos-Predict/cosmos-predict2/checkpoints/
```

### 2. Download Text Encoder (T5-11B)

Download the T5-11B text encoder:
```bash
huggingface-cli download google-t5/t5-11b --local-dir /path/to/t5-11b
```

Or visit: [https://huggingface.co/google-t5/t5-11b](https://huggingface.co/google-t5/t5-11b)

## 📊 Dataset Preparation

### Dataset Structure Requirements

Your dataset must follow this structure:
```
your_dataset/
├── metas/
│   └── *.txt          # Text prompts for each video
├── videos/
│   └── *.mp4          # Video files (must have 100+ frames)
└── t5_xxl/
    └── *.pickle       # Pre-extracted T5 embeddings
```

### Supported Datasets

1. **GR1-100 Dataset** (DreamGen Benchmark)
```bash
   hf download nvidia/GR1-100 --repo-type dataset --local-dir datasets/benchmark_train/hf_gr1/
   
   mkdir -p datasets/benchmark_train/gr1/videos
   mv datasets/benchmark_train/hf_gr1/gr1/*.mp4 datasets/benchmark_train/gr1/videos/
   mv datasets/benchmark_train/hf_gr1/metadata.csv datasets/benchmark_train/gr1/
```

2. **RoboCasa Dataset** (Used in training script)
   - Path: `/work/HHRI-AI/POC/public/pretraining_data/Robotic-Data/Video-Simulation-Data/robocasa/`

3. **Libero Dataset** (Used in inference script)
   - Path: `/work/HHRI-AI/POC/public/pretraining_data/Robotic-Data/Video-Simulation-Data/Libero-data/`

### Dataset Preprocessing

#### 1. Extract T5 Embeddings (Required before training)
```bash
python -m scripts.extract_t5_embeddings \
  --dataset_path /path/to/your/dataset \
  --output_path /path/to/your/dataset/t5_xxl
```

#### 2. Create Text Prompts

For GR1 dataset:
```bash
python -m scripts.create_prompts_for_gr1_dataset --dataset_path datasets/benchmark_train/gr1
```

For custom datasets, ensure each video has a corresponding `.txt` file in the `metas/` directory.

## 🚀 Training

### Quick Start

The training script is located at:
```
bash-running-script.sh/Finetuning-bash-dreamgen/cosmos-predict-training.sh
```

### Configuration

Before running, configure the following in the training script:
```bash
# Dataset Path
DATASET_PATH="/work/HHRI-AI/POC/public/pretraining_data/Robotic-Data/Video-Simulation-Data/robocasa/robocasa_30_demos_lerobot_5_chosen_tasks_v3-training-format"

# GPU Configuration
NUM_GPUS=4  # Adjust based on your available GPUs

# Weights & Biases Configuration
export WANDB_API_KEY="your-wandb-api-key"
export WANDB_ENTITY="your-wandb-username"
export WANDB_PROJECT="your-project-name"
```

### Run Training
```bash
bash bash-running-script.sh/Finetuning-bash-dreamgen/cosmos-predict-training.sh
```

### Training Features

The training script includes:

1. **Automatic Video Filtering**: Filters videos with less than 100 frames
2. **Dataset Validation**: Checks for required files (videos, metas, T5 embeddings)
3. **Symlink Creation**: Creates expected dataset paths for config compatibility
4. **W&B Integration**: Automatic experiment tracking
5. **GPU Monitoring**: Validates GPU availability before training
6. **Incremental Processing**: Skips filtering if already processed

### Training Configuration Options

- **Experiment**: `predict2_video2world_training_2b_groot_gr1_480`
- **Batch Size**: Default 1 (adjustable via `BATCH_SIZE` env var)
- **Number of Workers**: Default 8 (adjustable via `NUM_WORKERS` env var)
- **Max Steps**: Optional (set via `MAX_STEPS` env var)

### Checkpoint Location

Checkpoints are saved to:
```
/work/HHRI-AI/POC/public/pretraining_weights/NVIDIA/Cosmos-Predict/cosmos-predict2/outputs/training_${EXPERIMENT}_${TIMESTAMP}/
```

### Training Outputs

- **Checkpoints**: Model weights saved periodically
- **Logs**: Training logs in `outputs/training_*/logs/`
- **Hydra Configs**: Configuration snapshots in `outputs/training_*/hydra_runs/`
- **W&B Dashboard**: Real-time metrics and visualizations

## 🎬 Inference

### Quick Start

The inference script is located at:
```
bash-running-script.sh/Inference-bash-dreamgen/inference-long-video-generation.sh
```

### Inference Features

1. **Batch Processing**: Processes multiple episodes in one run
2. **Automated Manifest Generation**: Creates JSON manifest from dataset
3. **Long Video Generation**: Generates videos in chunks (3 chunks by default)
4. **Skip Existing Videos**: Automatically skips already generated videos
5. **Multi-GPU Support**: Distributes generation across GPUs

### Configuration

Before running, configure the following in the inference script:
```bash
# GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_GPUS=4

# Dataset Paths
DATASET_DIR="/path/to/your/dataset/screenshots"
PROMPT_DIR="/path/to/your/prompts"
COSMOS_OUTPUT_DIR="/path/to/output/videos"

# Model Checkpoint
--dit_path /path/to/your/checkpoint.pt
```

### Input Data Structure

The inference script expects:

1. **Screenshot Directories**: Folders ending with `_screenshots`
```
   episode_000000_screenshots/
   ├── frame_0.jpg
   ├── frame_1.jpg
   └── ...
```

2. **Prompt Files**: Text files with naming pattern `{episode_name}_result.txt`
```
   episode_000000_result.txt
   episode_000001_result.txt
```

### Run Inference
```bash
bash bash-running-script.sh/Inference-bash-dreamgen/inference-long-video-generation.sh
```

### Inference Parameters

- **Model Size**: `2B` (configurable)
- **Conditional Frames**: `1` (first frame of video)
- **Number of Chunks**: `3` (for long video generation)
- **Guardrail**: Disabled for faster generation

### Output Format

Generated videos are saved as:
```
{output_dir}/{episode_name}_predicted_3chunks_new.mp4
```

### Example Inference Command
```bash
torchrun --nproc_per_node=4 examples/video2world_lvg.py \
  --num_gpus 4 \
  --model_size 2B \
  --dit_path /path/to/checkpoint.pt \
  --batch_input_json /tmp/manifest.json \
  --num_conditional_frames 1 \
  --num_chunks 3 \
  --disable_guardrail
```

## 📁 Project Structure
```
VLA-Humanoid/
├── bash-running-script.sh/
│   ├── Finetuning-bash-dreamgen/
│   │   └── cosmos-predict-training.sh              # Training script (4 GPUs)
│   └── Inference-bash-dreamgen/
│       └── inference-long-video-generation.sh      # Batch inference script
│   └── Processing-text-prompt/
│       └── processing-text-prompt.sh               # Batch processing text script
│  
│  
├── cosmos_predict2/                                # Core model code
│   ├── configs/                                    # Training configurations
│   ├── _src/                                       # Model implementations
│   └── experiments/                                # Experiment configs
├── scripts/
│   ├── train.py                                    # Training entry point
│   ├── extract_t5_embeddings.py                    # T5 preprocessing
│   └── create_prompts_for_gr1_dataset.py          # Prompt generation
├── examples/
│   ├── inference.py                                # Standard inference
│   └── video2world_lvg.py                         # Long video generation
├── datasets/                                       # Training datasets
│   └── benchmark_train/
│       └── gr1/                                    # Symlinked dataset
├── checkpoints/                                    # Model checkpoints
│   └── posttraining/
│       └── video2world/
└── outputs/                                        # Training outputs
    └── training_*/
        ├── logs/                                   # Training logs
        ├── checkpoints/                            # Saved models
        └── hydra_runs/                             # Hydra configs
```

## ⚙️ Configuration Details

### Environment Variables
```bash
# CUDA Configuration
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_SOCKET_IFNAME="ib0"
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO

# Python Path
export PYTHONPATH="/path/to/cosmos-predict2:$PYTHONPATH"

# Weights & Biases
export WANDB_API_KEY="your-api-key"
export WANDB_ENTITY="your-username"
export WANDB_PROJECT="your-project"
```

### Video Requirements

- **Minimum Frames**: 100 frames per video (automatically filtered)
- **Format**: MP4
- **Resolution**: 480p (training default)

### Naming Conventions

- **Videos**: `episode_XXXXXX.mp4`
- **Prompts**: `episode_XXXXXX.txt` or `episode_XXXXXX_result.txt`
- **Screenshots**: `episode_XXXXXX_screenshots/frame_X.jpg`
- **T5 Embeddings**: `episode_XXXXXX.pickle`

## 🔍 Monitoring and Debugging

### Check Training Progress
```bash
# View training logs
tail -f outputs/training_*/logs/training_output.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check W&B dashboard
# Visit: https://wandb.ai/{entity}/{project}
```

### Common Issues

1. **Missing T5 Embeddings**
```
   ERROR: No T5 embedding files found
```
   Solution: Run T5 embedding extraction before training

2. **Insufficient Frames**
```
   Videos filtered out due to <100 frames
```
   Solution: The script automatically filters these out

3. **GPU Memory Issues**
   - Reduce `NUM_GPUS`
   - Decrease batch size: `export BATCH_SIZE=1`

## 🔗 References

- [NVIDIA COSMOS Predict Collection](https://huggingface.co/collections/nvidia/cosmos-predict25)
- [GR1-100 Dataset](https://huggingface.co/datasets/nvidia/GR1-100)
- [T5-11B Text Encoder](https://huggingface.co/google-t5/t5-11b)
- [Weights & Biases](https://wandb.ai/)

## 📝 Notes

- Training uses 480p resolution by default to match the dataset
- Videos with less than 100 frames are automatically filtered
- The inference script skips already generated videos
- W&B API key should be set in environment for security
- Checkpoint conversion to PyTorch format may be needed for some inference scripts


## 📄 License

[Add your license information here]

---

**Last Updated**: January 2026  
**Maintained by**: HHRI-AI Team, Hon Hai (Foxconn)
EOF
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
