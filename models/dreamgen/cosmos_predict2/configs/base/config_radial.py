def build_radial_parameters(
    num_blocks: int,
    *,
    decay_factor: float,
    block_size: int,
    start_layer_idx: int,
    model_type: str = "hunyuan",
):
    if num_blocks <= 0:
        raise ValueError(f"`num_blocks` must be positive, got {num_blocks}.")
    if decay_factor <= 0:
        raise ValueError(f"`decay_factor` must be positive, got {decay_factor}.")
    if block_size not in (64, 128):
        raise ValueError(f"`block_size` must be 64 or 128, got {block_size}.")
    if start_layer_idx < 0:
        raise ValueError(f"`start_layer_idx` must be non-negative, got {start_layer_idx}.")
    if model_type not in {"hunyuan", "wan"}:
        raise ValueError(f"`model_type` must be 'hunyuan' or 'wan', got {model_type!r}.")

    return [
        None
        if layer_idx < start_layer_idx
        else {
            "decay_factor": decay_factor,
            "block_size": block_size,
            "model_type": model_type,
        }
        for layer_idx in range(num_blocks)
    ]
