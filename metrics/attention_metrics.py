import torch

def calculate_density(mask):
    """
    Calculate the density of a sparse mask.
    Density is the ratio of non-zero elements to total elements.
    Args:
        mask (torch.Tensor): Sparse attention mask (binary or weights).
    Returns:
        float: Density value in [0, 1].
    """
    if mask.numel() == 0:
        return 0.0
    return (torch.count_nonzero(mask).item() / mask.numel())

def calculate_recall(sparse_mask, dense_attn, top_k=None):
    """
    Calculate the recall of a sparse mask against a dense attention matrix.
    Recall measures how many of the 'important' elements from the dense 
    attention are captured by the sparse mask.
    
    Args:
        sparse_mask (torch.Tensor): Binary mask (1 where sparse attention is computed, 0 otherwise).
        dense_attn (torch.Tensor): The full dense attention matrix.
        top_k (int, optional): If provided, recall is calculated against the top-k elements of dense_attn.
                             If None, recall is calculated against ALL non-zero elements of dense_attn 
                             (or simply compares total mass if dense_attn is normalized).
    Returns:
        float: Recall value in [0, 1].
    """
    assert sparse_mask.shape == dense_attn.shape
    
    # Ensure binary mask
    mask = (sparse_mask > 0).float()
    
    if top_k is not None:
        # Calculate recall against top-k elements of dense attention
        # For each row, find the top-k indices
        _, indices = torch.topk(dense_attn, k=min(top_k, dense_attn.size(-1)), dim=-1)
        
        # Create a binary mask for top-k elements
        top_k_mask = torch.zeros_like(dense_attn)
        top_k_mask.scatter_(-1, indices, 1.0)
        
        # Intersection
        intersection = (mask * top_k_mask).sum()
        total_top_k = top_k_mask.sum()
        
        recall = (intersection / total_top_k).item() if total_top_k > 0 else 1.0
    else:
        # Calculate recall as the ratio of captured attention mass
        # Assuming dense_attn represents the ground truth attention weights
        total_mass = dense_attn.sum()
        captured_mass = (dense_attn * mask).sum()
        
        recall = (captured_mass / total_mass).item() if total_mass > 0 else 1.0
        
    return recall
