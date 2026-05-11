source /mnt/amlfs-01/shared/shenyuang/DreamDojo/.venv/bin/activate

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    main.py fit \
    --config config/lam_0.yaml \
    2>&1 | tee output_train.log