def build_pisa_parameters(
    num_blocks: int,
    *,
    density: float,
    block_size: int,
    start_layer_idx: int,
    use_bias: bool = False,
):
    if num_blocks <= 0:
        raise ValueError(f"`num_blocks` must be positive, got {num_blocks}.")
    if not 0 < density <= 1:
        raise ValueError(f"`density` must be in (0, 1], got {density}.")
    if block_size <= 0:
        raise ValueError(f"`block_size` must be positive, got {block_size}.")
    if start_layer_idx < 0:
        raise ValueError(f"`start_layer_idx` must be non-negative, got {start_layer_idx}.")

    return [
        None
        if layer_idx < start_layer_idx
        else {
            "density": density,
            "block_size": block_size,
            "use_bias": use_bias,
        }
        for layer_idx in range(num_blocks)
    ]
