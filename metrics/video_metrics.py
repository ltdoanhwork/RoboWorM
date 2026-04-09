import torch
import torch.nn.functional as F
from torch.nn.functional import mse_loss

# Try to import lpips, but don't fail if not present
try:
    import lpips
    LPIPS_AVAILABLE = True
    lpips_model = lpips.LPIPS(net="vgg")
except ImportError:
    LPIPS_AVAILABLE = False
    lpips_model = None

def calculate_psnr(video1, video2, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two videos.
    Args:
        video1 (torch.Tensor): Shape (T, C, H, W) or (C, H, W)
        video2 (torch.Tensor): Shape (T, C, H, W) or (C, H, W)
        max_val (float): Maximum possible pixel value.
    Returns:
        float: Average PSNR across frames.
    """
    assert video1.shape == video2.shape, f"Shape mismatch: {video1.shape} vs {video2.shape}"
    
    if video1.dim() == 3:
        video1 = video1.unsqueeze(0)
        video2 = video2.unsqueeze(0)
        
    mse = mse_loss(video1, video2, reduction="none").mean(dim=(1, 2, 3))
    # Avoid log(0)
    mse = torch.clamp(mse, min=1e-10)
    
    psnr = 10 * torch.log10(max_val**2 / mse)
    return psnr.mean().item()

def calculate_ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    """
    Simplified Structural Similarity Index Measure (SSIM) implementation for PyTorch.
    Args:
        img1 (torch.Tensor): Shape (N, C, H, W)
        img2 (torch.Tensor): Shape (N, C, H, W)
    Returns:
        torch.Tensor: Mean SSIM for each image in batch.
    """
    # Compute means
    mu1 = F.avg_pool2d(img1, kernel_size=11, stride=1, padding=5)
    mu2 = F.avg_pool2d(img2, kernel_size=11, stride=1, padding=5)

    # Compute variances and covariances
    sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=11, stride=1, padding=5) - mu1**2
    sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=11, stride=1, padding=5) - mu2**2
    sigma12 = F.avg_pool2d(img1 * img2, kernel_size=11, stride=1, padding=5) - mu1 * mu2

    # SSIM calculation
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean(dim=(1, 2, 3))

def get_video_ssim(video1, video2):
    """
    Calculate average SSIM across video frames.
    Args:
        video1 (torch.Tensor): Shape (T, C, H, W)
        video2 (torch.Tensor): Shape (T, C, H, W)
    Returns:
        float: Average SSIM.
    """
    assert video1.shape == video2.shape
    if video1.dim() == 3:
        video1 = video1.unsqueeze(0)
        video2 = video2.unsqueeze(0)
        
    ssim_values = calculate_ssim(video1, video2)
    return ssim_values.mean().item()

def calculate_lpips(video1, video2, device="cuda"):
    """
    Calculate average LPIPS across video frames.
    Args:
        video1 (torch.Tensor): Shape (T, C, H, W)
        video2 (torch.Tensor): Shape (T, C, H, W)
        device (str): Device to run the model on.
    Returns:
        float: Average LPIPS.
    """
    if not LPIPS_AVAILABLE:
        print("Warning: LPIPS library not installed. Returning 0.0")
        return 0.0
    
    assert video1.shape == video2.shape
    if video1.dim() == 3:
        video1 = video1.unsqueeze(0)
        video2 = video2.unsqueeze(0)
        
    global lpips_model
    lpips_model = lpips_model.to(device)
    
    # LPIPS expects inputs in [-1, 1]. Assumes input is in [0, 1].
    v1 = video1.to(device) * 2 - 1
    v2 = video2.to(device) * 2 - 1
    
    with torch.no_grad():
        lpips_values = lpips_model(v1, v2)
    
    return lpips_values.mean().item()
