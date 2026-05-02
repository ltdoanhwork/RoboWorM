from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Using the official lerobot/droid_1.0.1 which is in v3.0 format
repo_id = "lerobot/droid_1.0.1"
# DROID 1.0.1 has ~92,230 episodes. 20% is 18,446.
episodes_20p = list(range(18446))

print(f"Starting download of 20% data ({len(episodes_20p)} episodes) from {repo_id}...")

dataset = LeRobotDataset(
    repo_id,
    episodes=episodes_20p,
    root="./datasets/droid_20p",
)

print(f"Download complete! Data saved to: ./datasets/droid_20p")