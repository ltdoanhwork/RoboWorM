# DreamDojo Post-Training

The training configurations are managed by yaml under `configs`. To launch post-training:

1. Follow the same setup as the pretraining.
2. Change `load_path` in `cosmos_predict2/experiments/base/action.py` to the pretrained checkpoint and specify its training step. To use our pretrained model, set `load_path` to `checkpoints/DreamDojo/2B_pretrain/iter_000140000/` or `checkpoints/DreamDojo/14B_pretrain/iter_000140000/`.
3. (Optional) Disable the extraction of latent actions on the fly, as it will be reset to zero.
4. Send the desired experiment config (e.g., `dreamdojo_2b_480_640_gr1` to use `2b_480_640_gr1.yaml`).
    
    ```
    bash launch.sh dreamdojo_2b_480_640_gr1
    bash launch.sh dreamdojo_2b_480_640_g1
    bash launch.sh dreamdojo_2b_480_640_agibot
    bash launch.sh dreamdojo_2b_480_640_yam
    ```

For simplicity of implementation, we set the dimension of the first action projection layer to 384. This will slightly increase computational requirements but makes checkpoint resuming easier. You can also reduce it to the dimension actually used by the model after post-training. Our dimension assignment follows this rule:
- [0, 29): Fourier GR-1
- [29, 58): Retargeted GR-1 actions from Manus gloves
- [58, 101): Unitree G1
- [101, 147): Bimanual YAM
- [147, 169): AgiBot
- [169, 220): Reserved for other usages
- [220, 352): MANO actions
- [352, 384): Latent actions

> [!NOTE]
> The post-trained DreamDojo checkpoints for G1, GR-1, AgiBot, and YAM (2B and 14B) can be found at [Hugging Face](https://huggingface.co/nvidia/DreamDojo).

---

<= Previous: [[DreamDojo Pretraining](https://github.com/NVIDIA/DreamDojo/blob/main/docs/PRETRAIN.md)]

=> Next: [[DreamDojo Distillation](https://github.com/NVIDIA/DreamDojo/blob/main/docs/DISTILL.md)]