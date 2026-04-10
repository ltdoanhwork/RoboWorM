import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
import re


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", help="One or more directories to process")
    args = parser.parse_args()

    plt.figure(figsize=(8, 6))

    for dir_path in args.dirs:
        dir_path = Path(dir_path)
        name = dir_path.name

        step_numbers = []
        avg_psnrs = []

        step_dirs = [d for d in dir_path.iterdir() if re.fullmatch(r"iter_\d+", d.name)]
        for step_dir in sorted(step_dirs, key=lambda x: int(x.name.split("_")[-1])):
            if len(list(step_dir.iterdir())) == 0:
                continue
            if len(list(step_dir.iterdir())) != 501:
                continue
            psnrs = []
            for f in Path(step_dir).glob("*_metrics.json"):
                with open(f, "r") as file:
                    data = json.load(file)
                psnrs.append(data["psnr"])
            avg_psnr = sum(psnrs) / len(psnrs) if psnrs else 0.0

            step_num = int(step_dir.name.split("_")[-1])
            step_numbers.append(step_num)
            avg_psnrs.append(avg_psnr)

            print(f"Directory: {step_dir}")
            print(f"Average PSNR: {avg_psnr:.2f} dB ({len(psnrs)} samples)")

        plt.plot(step_numbers, avg_psnrs, marker="o", label=name)

    plt.xlabel("Step", fontsize=14)
    plt.ylabel("Average PSNR (dB)", fontsize=14)
    plt.title("Average PSNR per Step", fontsize=16)
    plt.grid(True)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        frameon=False,
        fontsize=14
    )

    plt.savefig("psnr.png", bbox_inches="tight")
