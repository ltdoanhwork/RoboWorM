#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export NUM_GPUS=4
/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/cosmos-predict2.5/.venv/bin/torchrun --nproc_per_node=4 examples/video2world.py \
    --input_path assets/video2world/input0.jpg \
    --prompt "Verification test" \
    --save_path output/video2world_verification.mp4
