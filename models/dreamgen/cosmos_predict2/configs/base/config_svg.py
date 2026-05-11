def build_svg_parameters(
    num_blocks: int,
    *,
    variant: str,
    start_layer_idx: int,
    dense_step_frac: float,
    sparsity: float = 0.25,
    num_sampled_rows: int = 64,
    sample_mse_max_row: int = 3000,
    num_q_centroids: int = 50,
    num_k_centroids: int = 200,
    top_p_kmeans: float = 0.9,
    min_kc_ratio: float = 0.0,
    kmeans_iter_init: int = 5,
    kmeans_iter_step: int = 2,
):
    if num_blocks <= 0:
        raise ValueError(f"`num_blocks` must be positive, got {num_blocks}.")
    if variant not in {"svg1", "svg2"}:
        raise ValueError(f"`variant` must be 'svg1' or 'svg2', got {variant!r}.")
    if start_layer_idx < 0:
        raise ValueError(f"`start_layer_idx` must be non-negative, got {start_layer_idx}.")
    if not 0 <= dense_step_frac < 1:
        raise ValueError(f"`dense_step_frac` must be in [0, 1), got {dense_step_frac}.")

    if variant == "svg1":
        if not 0 < sparsity <= 1:
            raise ValueError(f"`sparsity` must be in (0, 1], got {sparsity}.")
        if num_sampled_rows <= 0:
            raise ValueError(f"`num_sampled_rows` must be positive, got {num_sampled_rows}.")
        if sample_mse_max_row <= 0:
            raise ValueError(f"`sample_mse_max_row` must be positive, got {sample_mse_max_row}.")
    else:
        if num_q_centroids <= 0:
            raise ValueError(f"`num_q_centroids` must be positive, got {num_q_centroids}.")
        if num_k_centroids <= 0:
            raise ValueError(f"`num_k_centroids` must be positive, got {num_k_centroids}.")
        if not 0 < top_p_kmeans <= 1:
            raise ValueError(f"`top_p_kmeans` must be in (0, 1], got {top_p_kmeans}.")
        if not 0 <= min_kc_ratio <= 1:
            raise ValueError(f"`min_kc_ratio` must be in [0, 1], got {min_kc_ratio}.")
        if kmeans_iter_init < 0:
            raise ValueError(f"`kmeans_iter_init` must be non-negative, got {kmeans_iter_init}.")
        if kmeans_iter_step < 0:
            raise ValueError(f"`kmeans_iter_step` must be non-negative, got {kmeans_iter_step}.")

    parameters = []
    for layer_idx in range(num_blocks):
        if layer_idx < start_layer_idx:
            parameters.append(None)
            continue

        params = {
            "variant": variant,
            "layer_idx": layer_idx,
            "dense_step_frac": dense_step_frac,
        }
        if variant == "svg1":
            params.update(
                {
                    "sparsity": sparsity,
                    "num_sampled_rows": num_sampled_rows,
                    "sample_mse_max_row": sample_mse_max_row,
                }
            )
        else:
            params.update(
                {
                    "num_q_centroids": num_q_centroids,
                    "num_k_centroids": num_k_centroids,
                    "top_p_kmeans": top_p_kmeans,
                    "min_kc_ratio": min_kc_ratio,
                    "kmeans_iter_init": kmeans_iter_init,
                    "kmeans_iter_step": kmeans_iter_step,
                }
            )
        parameters.append(params)
    return parameters
