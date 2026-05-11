# Setup

### Environment

The current code is tested with NVIDIA H100 80GB. We use [uv](https://docs.astral.sh/uv/) to manage the environment. We provide a bash script for quick installation.

```
bash install.sh
```

### Dataset

In this release, we provide the GR-1 post-training dataset along with evaluation sets as example datasets for training and inference. The datasets can be downloaded from [Hugging Face](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1) and can be placed in or linked to the `datasets` directory.

---

=> Next: [[Latent Action Model Training](https://github.com/NVIDIA/DreamDojo/blob/main/docs/LAM.md)]