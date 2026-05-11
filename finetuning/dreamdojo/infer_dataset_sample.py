"""Inference on a sample from an ALOHA/DROID dataset using a DreamDojo checkpoint.

Given a dataset path and episode index, loads the first frame + ground-truth actions,
runs the model to generate a predicted video, then saves both GT and predicted videos
side-by-side for visual comparison.

Usage:
    # Use pretrained checkpoint (default)
    python finetuning/dreamdojo/infer_dataset_sample.py \
        --dataset_path datasets/bimanual/singleview/makovian/close_cardboard_box \
        --episode_idx 0

    # Use a finetuned checkpoint
    python finetuning/dreamdojo/infer_dataset_sample.py \
        --dataset_path datasets/bimanual/singleview/makovian/close_cardboard_box \
        --episode_idx 0 \
        --ckpt_path exp_logs/dreamdojo_logs/dreamdojo/exp_aloha/aloha_bimanual_singleview/checkpoints/iter_000010000

    # DROID dataset
    python finetuning/dreamdojo/infer_dataset_sample.py \
        --dataset_path datasets/single_arm/singleview/chunk-000/... \
        --episode_idx 5 --embodiment droid
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DREAMDOJO_ROOT = PROJECT_ROOT / "models" / "DreamDojo"
sys.path.insert(0, str(DREAMDOJO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from groot_dreams.groot_configs import construct_modality_config_and_transforms
from groot_dreams.data.dataset import WrappedLeRobotSingleDataset
from groot_dreams.dataloader import VideoActionDataset


def load_dataset_sample(dataset_path: str, episode_idx: int, embodiment: str, num_frames: int = 13):
    """Load a sample from a LeRobot dataset, returning first frame + actions."""
    config, train_transform, _ = construct_modality_config_and_transforms(
        num_frames=num_frames, embodiment=embodiment
    )
    ds = VideoActionDataset(
        dataset_path=dataset_path,
        num_frames=num_frames,
        embodiment=embodiment,
    )
    sample = ds[episode_idx]
    return sample


def run_inference(
    ckpt_path: str,
    experiment_name: str,
    first_frame: np.ndarray,
    action: np.ndarray,
    guidance: float = 3.0,
    seed: int = 42,
):
    """Run model inference given first frame and actions."""
    from cosmos_predict2._src.predict2.action.inference.inference_pipeline import (
        ActionVideo2WorldInference,
    )

    pipeline = ActionVideo2WorldInference(
        experiment_name=experiment_name,
        ckpt_path=ckpt_path,
        s3_credential_path="",
        context_parallel_size=1,
    )

    next_frame, video = pipeline.step_inference(
        img_array=first_frame,
        action=action,
        guidance=int(guidance),
        seed=seed,
        num_latent_conditional_frames=1,
        prompt="",
        negative_prompt="",
        use_neg_prompt=False,
    )

    pipeline.cleanup()
    return video


def save_video(frames: np.ndarray, path: str, fps: int = 5):
    """Save numpy frames (T, H, W, C) as mp4."""
    frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, C, H, W)
    torchvision.io.write_video(path, frames.astype(np.uint8), fps=fps, video_codec="libx264")
    print(f"Saved video to {path}")


def save_comparison(gt_video: np.ndarray, pred_video: np.ndarray, path: str, fps: int = 5):
    """Save GT and predicted videos side-by-side."""
    T_gt = gt_video.shape[0]
    T_pred = pred_video.shape[0]
    T = min(T_gt, T_pred)

    gt_resized = gt_video[:T]
    pred_resized = pred_video[:T]

    if gt_resized.shape[1:3] != pred_resized.shape[1:3]:
        from torchvision.transforms.functional import resize
        h, w = pred_resized.shape[1], pred_resized.shape[2]
        gt_tensor = torch.from_numpy(gt_resized).permute(0, 3, 1, 2).float()
        gt_tensor = torch.nn.functional.interpolate(gt_tensor, size=(h, w), mode="bilinear")
        gt_resized = gt_tensor.permute(0, 2, 3, 1).numpy().astype(np.uint8)

    comparison = np.concatenate([gt_resized, pred_resized], axis=2)  # side-by-side
    torchvision.io.write_video(path, comparison.astype(np.uint8), fps=fps, video_codec="libx264")
    print(f"Saved comparison video to {path}")


def main():
    parser = argparse.ArgumentParser(description="Infer DreamDojo on a dataset sample")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to LeRobot dataset directory")
    parser.add_argument("--episode_idx", type=int, default=0,
                        help="Episode/sample index to use")
    parser.add_argument("--embodiment", type=str, default=None,
                        help="Embodiment tag (aloha/droid). Auto-detected if not specified.")
    parser.add_argument("--ckpt_path", type=str,
                        default=str(PROJECT_ROOT / "checkpoints/dreamdojo/2B_pretrain/iter_000140000"),
                        help="Path to model checkpoint directory")
    parser.add_argument("--experiment_name", type=str, default="dreamdojo_2b_480_640_aloha",
                        help="Experiment name for config resolution")
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_frames", type=int, default=13)
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "infer_outputs"))
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()

    # Auto-detect embodiment
    ds_lower = args.dataset_path.lower()
    if args.embodiment is None:
        if "aloha" in ds_lower or "bimanual" in ds_lower:
            args.embodiment = "aloha"
        elif "droid" in ds_lower or "single_arm" in ds_lower:
            args.embodiment = "droid"
        else:
            raise ValueError("Cannot auto-detect embodiment. Specify --embodiment.")
    print(f"Embodiment: {args.embodiment}")

    # Load dataset sample
    print(f"Loading sample {args.episode_idx} from {args.dataset_path}...")
    sample = load_dataset_sample(
        args.dataset_path, args.episode_idx, args.embodiment, args.num_frames
    )

    # Extract first frame and action from the sample
    video_frames = sample["video"]  # (C, T, H, W) uint8
    action = sample["action"]  # (T-1, 384)

    # video is (C, T, H, W) -> first frame (H, W, C)
    first_frame = video_frames[:, 0, :, :].permute(1, 2, 0).numpy()

    # GT video: (C, T, H, W) -> (T, H, W, C)
    gt_video = video_frames.permute(1, 2, 3, 0).numpy()

    print(f"First frame shape: {first_frame.shape}")
    print(f"Action shape: {action.shape}")
    print(f"Action non-zero dims: {(action.abs().sum(0) > 0).sum().item()}")

    # Run inference
    print(f"Running inference with checkpoint: {args.ckpt_path}")
    print(f"Guidance: {args.guidance}, Seed: {args.seed}")
    pred_video = run_inference(
        ckpt_path=args.ckpt_path,
        experiment_name=args.experiment_name,
        first_frame=first_frame,
        action=action.numpy(),
        guidance=args.guidance,
        seed=args.seed,
    )

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds_name = Path(args.dataset_path).name
    tag = f"{ds_name}_ep{args.episode_idx}_g{args.guidance:.0f}_s{args.seed}"

    save_video(gt_video, str(output_dir / f"{tag}_gt.mp4"), fps=args.fps)
    save_video(pred_video, str(output_dir / f"{tag}_pred.mp4"), fps=args.fps)
    save_comparison(gt_video, pred_video, str(output_dir / f"{tag}_comparison.mp4"), fps=args.fps)

    print(f"\nDone! Outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
