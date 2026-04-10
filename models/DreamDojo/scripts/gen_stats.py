import json
from pathlib import Path
from groot_dreams.data.dataset import calculate_dataset_statistics

LE_ROBOT_DATA_FILENAME_SINGLE = "data/*/*.parquet"
LE_ROBOT_DATA_FILENAME_SUBSET = "*/data/*/*.parquet"


if __name__ == "__main__":
    parquet_paths = []
    dataset_paths = "/mnt/amlfs-02/shared/datasets/oss_testing/G1/g1_shelf,/mnt/amlfs-03/shared/datasets/g1-shelf,/mnt/amlfs-03/shared/datasets/Gear2025CoRLDemo"
    for dataset_path in dataset_paths.split(","):
        dataset_path = dataset_path.strip()
        parquet_paths.extend(list((Path(dataset_path).glob(LE_ROBOT_DATA_FILENAME_SINGLE))))
        parquet_paths.extend(list((Path(dataset_path).glob(LE_ROBOT_DATA_FILENAME_SUBSET))))
    dataset_stats = calculate_dataset_statistics(parquet_paths)
    with open("stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=4)
