import torch
from metrics import (
    calculate_psnr, 
    get_video_ssim, 
    calculate_lpips, 
    calculate_density, 
    calculate_recall
)

def test_video_metrics():
    print("Testing Video Metrics...")
    # Create two dummy videos (T, C, H, W)
    v1 = torch.rand(4, 3, 64, 64)
    v2 = v1.clone()
    
    psnr_id = calculate_psnr(v1, v2)
    ssim_id = get_video_ssim(v1, v2)
    print(f"  Identical - PSNR: {psnr_id:.2f} dB, SSIM: {ssim_id:.4f}")
    
    v3 = torch.rand(4, 3, 64, 64)
    psnr_diff = calculate_psnr(v1, v3)
    ssim_diff = get_video_ssim(v1, v3)
    print(f"  Different - PSNR: {psnr_diff:.2f} dB, SSIM: {ssim_diff:.4f}")
    
    try:
        lpips_val = calculate_lpips(v1, v3, device="cpu")
        print(f"  LPIPS: {lpips_val:.4f}")
    except Exception as e:
        print(f"  LPIPS test failed or skipped: {e}")

def test_attention_metrics():
    print("\nTesting Attention Metrics...")
    # Create a dummy dense attention matrix (H, W)
    dense_attn = torch.abs(torch.randn(16, 16))
    dense_attn = dense_attn / dense_attn.sum(dim=-1, keepdim=True)
    
    # Create a sparse mask
    sparse_mask = torch.zeros_like(dense_attn)
    sparse_mask[:8, :8] = 1.0 # 1/4 density
    
    density = calculate_density(sparse_mask)
    recall = calculate_recall(sparse_mask, dense_attn)
    recall_topk = calculate_recall(sparse_mask, dense_attn, top_k=5)
    
    print(f"  Density: {density:.4f}")
    print(f"  Recall (Mass): {recall:.4f}")
    print(f"  Recall (Top-5): {recall_topk:.4f}")

if __name__ == "__main__":
    test_video_metrics()
    test_attention_metrics()
    print("\nAll basic tests completed!")
