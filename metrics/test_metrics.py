import torch
from metrics import calculate_psnr, get_video_ssim, calculate_lpips


def test_video_metrics():
    print("Testing Video Metrics...")
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


if __name__ == "__main__":
    test_video_metrics()
    print("\nAll basic tests completed!")
