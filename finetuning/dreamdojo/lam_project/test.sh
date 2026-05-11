source /mnt/amlfs-01/shared/shenyuang/DreamDojo/.venv/bin/activate

python main.py test \
    --ckpt_path checkpoints/lam/epoch=1.ckpt \
    --config config/lam.yaml \
    2>&1 | tee output_test.log