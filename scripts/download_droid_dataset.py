import time
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

repo_id = "cadene/droid_1.0.1"
local_dir = "datasets/droid_1.0.1_20chunks"

for i in range(20):
    chunk = f"chunk-{i:03d}"

    allow_patterns = [
        "meta/*",
        f"data/{chunk}/*",
        f"videos/{chunk}/**",
    ]

    print(f"\nDownloading {chunk}...")

    for attempt in range(1, 21):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                max_workers=1,
            )
            print(f"Finished {chunk}")
            break

        except HfHubHTTPError as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                wait_time = min(300, 60 * attempt)
                print(f"Rate limited. Sleep {wait_time}s then retry...")
                time.sleep(wait_time)
            else:
                raise

    # nghỉ giữa các chunk để không spam API
    time.sleep(5*60)