# Qwen2.5-VL Robotic Video Inference Pipeline

## 📖 Overview
This repository contains a high-throughput pipeline for generating text descriptions (instructions/captions) from robotic simulation videos using **Qwen2.5-VL-32B-Instruct**. 

The pipeline is designed to utilize **multi-GPU** setups with **vLLM** to process massive datasets efficiently. It handles:
1. **Parallel Batching:** Automatically splits video lists across available GPUs.
2. **Memory Management:** Uses Tensor Parallelism to fit large models.
3. **Resume Capability:** Skips videos that have already been processed.

---

## ⚙️ 1. Hardware & System Requirements

To run this pipeline successfully, the compute node must meet the following specifications:

### Hardware
* **GPU:** Minimum **2 NVIDIA GPUs** (e.g., A100/H100 80GB).
    * *Default Config:* The script is currently configured to use **4 GPUs** total.
    * *Tensor Parallelism:* The model requires **2 GPUs per instance** to load into memory (`TENSOR_PARALLEL_PER_JOB=2`).
* **VRAM:** High VRAM usage (Set to 90% utilization).
* **Storage:** Write permissions are **mandatory** for the output directory.

### Software Environment
* **OS:** Linux (Ubuntu/CentOS).
* **Python:** 3.9+.
* **Conda:** Recommended for environment isolation.

---

## 📦 2. Installation & Dependencies

Ensure your environment is active (`conda activate your_env`). Install the required Python packages:
```bash
pip install vllm torch transformers qwen-vl-utils accelerate
```

**Note:** The script forces multiprocessing to use 'spawn' mode. This is a required fix for vLLM compatibility and must not be removed from the Python script.

---

## 📂 3. Data & Directory Structure

The script expects a specific input hierarchy and will create a corresponding output hierarchy.

### Input Structure (Read-Only Source)
The script looks for `.mp4` files inside camera-specific subfolders:
```
/path/to/source/chunk-000/
├── left_image/      <-- Contains *.mp4
├── right_image/     <-- Contains *.mp4
└── wrist_image/     <-- Contains *.mp4
```

### Output Structure (Writable Destination)
The script generates `.txt` files containing the model's description for each video:
```
/path/to/your/output/chunk-000_captions/
├── left_image/
│   ├── video_001_result.txt
│   └── video_002_result.txt
└── ...
```

---

## 🚀 4. Usage Guide

### Step 1: Configure Paths
Open the Bash script (`run_inference.sh`) and update the **CONFIGURATION** section at the top:

| Variable | Description | Critical Note |
|----------|-------------|---------------|
| `MODEL_PATH` | Path to the local Qwen2.5-VL weights. | Must be the directory containing `config.json`. |
| `BASE_VIDEO_ROOT` | The root folder containing your video chunks. | Ensure you have Read permissions. |
| `OUTPUT_ROOT` | **CRITICAL:** The base folder for saving results. | Change this to a folder YOU own (e.g., your user directory). |
| `SCRIPT_DIR` | Location of the Python inference script. | Update if you move the python file. |

### Step 2: Configure Prompt
Update the `VIDEO_PROMPT` variable in the Bash script to change the instructions given to the model:
```bash
VIDEO_PROMPT="Can you describe what would be the best description instruction for this video related to robotic tasks?"
```

### Step 3: Run the Pipeline
Make the script executable and run it:
```bash
chmod +x run_inference.sh
./run_inference.sh
```

---

## 🔧 5. Advanced Configuration (Parallelism)

If you need to run on a different number of GPUs, modify the `process_specific_folder` function in the Bash script:
```bash
# Hardware Config
TOTAL_GPUS_TO_USE=4          # Total GPUs available on the node to use
TENSOR_PARALLEL_PER_JOB=2    # How many GPUs needed for ONE model instance
```

**Calculation:** `NUM_JOBS = TOTAL_GPUS / TENSOR_PARALLEL`  
**Default:** 4 GPUs / 2 per job = 2 Concurrent Jobs.

**Note:** Do not lower `TENSOR_PARALLEL_PER_JOB` below 2 for the 32B model, or you will likely encounter CUDA OOM (Out of Memory) errors.

---

## ❓ 6. Troubleshooting

### 1. "Permission denied creating output directory"
**Cause:** The `OUTPUT_ROOT` in the script is pointing to a shared/public folder where you do not have write access.  
**Fix:** Change `OUTPUT_ROOT` to your personal user folder (`/work/HHRI-AI/POC/users/YOUR_USERNAME/...`).

### 2. "CUDA Out of Memory"
**Cause:** The video is too long or the batch size is too high.  
**Fix:** The Python script limits input to 10 frames and 1 FPS automatically. If it still fails, increase `TENSOR_PARALLEL_PER_JOB` to 4.

### 3. "Zero description files were found"
**Cause:** The Python job crashed silently.  
**Fix:** Check the logs created inside the output folder: `.../results/chunk-000_captions/left_image/job_0/inference.log`.

### 4. Script hangs or Deadlocks
**Cause:** Multiprocessing conflict.  
**Fix:** Ensure `os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'` is present at the very top of the Python script.