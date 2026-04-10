#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir

from cosmos_predict2.action_conditioned import inference
from cosmos_predict2.action_conditioned_config import (
    ActionConditionedInferenceArguments,
    ActionConditionedSetupArguments,
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    output_subdir = "dreamdojo_hv_eval_14b_g1"

    parser = argparse.ArgumentParser(description="Run DreamDojo action-conditioned eval from a local DCP checkpoint.")
    parser.add_argument(
        "--model",
        default="2B/robot/action-cond",
        help="Model key required by the setup schema. Experiment/checkpoint still control the actual local run.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "outputs" / "action_conditioned" / output_subdir),
        help="Auxiliary output directory used by the repo runtime.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        default=str(workspace_root / "checkpoints" / "dreamdojo" / "14B_G1_post-train"),
        help="Checkpoint root containing latest_checkpoint.txt and iter_* directories.",
    )
    parser.add_argument(
        "--experiment",
        default="dreamdojo_14b_480_640_gr1",
        help="DreamDojo experiment name.",
    )
    parser.add_argument(
        "--save-dir",
        default=str(workspace_root / "dream_gen_benchmark" / output_subdir),
        help="Directory where generated videos and metrics will be saved.",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(
            workspace_root
            / "checkpoints"
            / "dreamdojo"
            / "datasets"
            / "PhysicalAI-Robotics-GR00T-Teleop-GR1"
            / "DreamDojo-HV_Eval"
        ),
        help="Evaluation dataset path.",
    )
    parser.add_argument("--num-frames", type=int, default=49, help="Number of frames per sample including the first condition frame.")
    parser.add_argument("--num-samples", type=int, default=100, help="Maximum number of samples to evaluate.")
    parser.add_argument("--data-split", default="test", help="Dataset split to use.")
    parser.add_argument(
        "--deterministic-uniform-sampling",
        action="store_true",
        default=False,
        help="Uniformly sample across concatenated datasets.",
    )
    return parser.parse_args()


def resolve_checkpoint_path(checkpoints_dir: Path) -> Path:
    latest_checkpoint_file = checkpoints_dir / "latest_checkpoint.txt"
    if not latest_checkpoint_file.exists():
        raise FileNotFoundError(f"Missing latest checkpoint file: {latest_checkpoint_file}")

    latest_checkpoint = latest_checkpoint_file.read_text().strip()
    checkpoint_iter_dir = checkpoints_dir / latest_checkpoint
    checkpoint_model_dir = checkpoint_iter_dir / "model"

    if checkpoint_model_dir.exists():
        return checkpoint_model_dir
    raise FileNotFoundError(f"Missing DCP model directory: {checkpoint_model_dir}")


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    save_dir = Path(args.save_dir)
    dataset_path = Path(args.dataset_path)

    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoint root does not exist: {checkpoints_dir}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    checkpoint_path = resolve_checkpoint_path(checkpoints_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    setup_args = ActionConditionedSetupArguments(
        output_dir=output_dir,
        model=args.model,
        config_file="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
        checkpoints_dir=str(checkpoints_dir),
        experiment=args.experiment,
        save_dir=str(save_dir),
        num_frames=args.num_frames,
        num_samples=args.num_samples,
        dataset_path=str(dataset_path),
        data_split=args.data_split,
        deterministic_uniform_sampling=args.deterministic_uniform_sampling,
    )

    print("Running DreamDojo eval with:")
    print(f"  dataset_path:    {dataset_path}")
    print(f"  checkpoints_dir: {checkpoints_dir}")
    print(f"  checkpoint_path: {checkpoint_path}")
    print(f"  save_dir:        {save_dir}")
    print(f"  experiment:      {args.experiment}")
    print(f"  num_frames:      {args.num_frames}")
    print(f"  num_samples:     {args.num_samples}")
    print(f"  data_split:      {args.data_split}")

    init_environment()
    try:
        init_output_dir(setup_args.output_dir, profile=setup_args.profile)
        inference(setup_args, ActionConditionedInferenceArguments, checkpoint_path)
    finally:
        cleanup_environment()


if __name__ == "__main__":
    main()
