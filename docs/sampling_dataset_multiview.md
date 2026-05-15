# Humanoid 
## Run Multi-view Video Generation in DreamGen

This document provides instructions and examples on how to run multi-view video generation using the Cosmos-Predict2-Multiview model from the **root directory** (without `cd`).

## Environment Setup

To run DreamGen scripts from the root directory, you must add `models/dreamgen` to your `PYTHONPATH` so that internal imports like `from examples...` can be resolved:

```bash
# From the root: /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics
export PYTHONPATH=$PYTHONPATH:$(pwd)/models/dreamgen
```

## Running the Multi-view Script

The multi-view pipeline generates **7 views** with **29 frames** at **720p**.

Here is the command to run multi-view inference from the root directory:

```bash
torchrun --nproc_per_node=1 --master_port=12341 \
  -m examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant droid \
  --prompt "A multi-view video shows that a robot pick the lid and put it on the pot The video is split into four views: The top-left view shows the robotic arm from the left side, the top-right view shows it from the right side, the bottom-left view shows a first-person perspective from the robot's end-effector (gripper), and the bottom-right view is a black screen (inactive view). The robot pick the lid and put it on the pot" \
  --input_path models/dreamgen/assets/sample_gr00t_dreams_droid/episode_000408.png \
  --prompt_prefix "" \
  --num_gpus 1 \
  --disable_guardrail \
  --save_path output/generated_video_droid.mp4

```

### Parameter Explanations
- `--model_size 2B`: Uses the 2-Billion parameter multi-view checkpoint.
- `--input_path`: Path to your initial image (relative to the root).
- `--num_conditional_frames 1`: Uses only the first frame (the image) as the condition.
- `--n_views 7`: Generates 7 different camera views.
- `--save_path`: The output location for the generated `mp4` video.

### Multi-GPU Support
To run with Context Parallelism using multiple GPUs:

```bash
torchrun --nproc_per_node=2 --master_port=12341 \
  -m examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant droid \
  --prompt "A multi-view video shows that a robot pick the lid and put it on the pot The video is split into four views: The top-left view shows the robotic arm from the left side, the top-right view shows it from the right side, the bottom-left view shows a first-person perspective from the robot's end-effector (gripper), and the bottom-right view is a black screen (inactive view). The robot pick the lid and put it on the pot" \
  --input_path models/dreamgen/assets/sample_gr00t_dreams_droid/episode_000408.png \
  --prompt_prefix "" \
  --num_gpus 2 \
  --disable_guardrail \
  --save_path output/generated_video_droid.mp4
  
```
# Đảm bảo đang ở thư mục root: /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics

source models/DreamDojo/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/models/dreamgen

torchrun --nproc_per_node=2 --master_port=12341 \
  -m examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant droid \
  --prompt "A multi-view video shows that a robot picks up a pomegranate with its left hand and puts it in the storage box, then picks up a mango with its right hand and puts it in the storage box. The video is split into four views: top-left (front view), top-right (head view), bottom-left (wrist view), and bottom-right (black screen)." \
  --input_path models/dreamgen/airbot_multiview_input.png \
  --prompt_prefix "" \
  --num_gpus 2 \
  --disable_guardrail \
  --save_path output/generated_video_airbot_droid.mp4


# Chạy tất cả 6 Airbot (5 episodes mỗi bộ, FPS=10)
NUM_GPUS=1 MAX_EPISODES=0 bash scripts/run_multiview_airbot_all.sh

# Hoặc override
MAX_EPISODES=3 FPS=8 bash scripts/run_multiview_airbot_all.sh


# Single-arm (datasets/single_arm/multiview)

Full-batch prep + DreamGen (`2×2` grid: `exterior_1_left`, `exterior_2_left`, `wrist_left`, black quadrant):

```bash
source models/dreamgen/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/models/dreamgen

NUM_GPUS=2 bash scripts/run_single_arm_multiview_dreamgen_all.sh
```

One branch only (`makovian` or `non_makovian`):

```bash
DATASET_PATH=datasets/single_arm/multiview/makovian NUM_GPUS=2 \
  bash scripts/run_single_arm_multiview_dreamgen_batch.sh
```

**Độ dài video (số frame thật):** `examples.video2world_gr00t` **không** có cờ `--num_frames`; độ dài mỗi lần sinh do checkpoint / `state_t` cố định (tài liệu multiview: khoảng **29** frame @ 16fps training). Để **video dài hơn**, dùng **LVG** (nối nhiều chunk):

```bash
USE_LVG=1 NUM_CHUNKS=4 NUM_GPUS=1 \
  DATASET_PATH=datasets/single_arm/multiview/makovian \
  bash scripts/run_single_arm_multiview_dreamgen_batch.sh
```

`NUM_CHUNKS` càng lớn → clip càng dài (tốn GPU/time hơn). `--fps` trên bản **không**-LVG chỉ đổi tốc độ phát khi encode MP4; bản LVG hiện encode nội bộ **16 fps**.

**Tại sao trước đây có hàng loạt `[WARN] … missing video for exterior_1_left`?** Script cũ đi theo **toàn bộ** `episode_index` trong `meta/episodes.jsonl` (có thể hàng chục nghìn dòng), trong khi trên đĩa bạn chỉ có **một subset** file `episode_XXXXXX.mp4` (dataset chưa tải đủ / chỉ giữ shard). Index kiểu `005698` nằm trong meta nhưng **không có file** ⇒ cảnh báo. **Mặc định mới:** prepare dùng **`--episode_source disk`**: chỉ lấy các episode mà **đủ cả 3 camera** đều có mp4 dưới `videos/chunk-*/…` (khoảng 128 episode trong ví dụ makovian của bạn). Prompt vẫn lấy từ meta khi trùng index; thiếu dòng meta thì fallback câu generic. Nếu cố tình muốn đi theo đúng meta và chấp nhận skip + warn: `EPISODE_SOURCE=meta`.

**Gợi ý cũ (`MIN_EPISODE_INDEX`)** vẫn dùng được để giới hạn dải index sau khi đã lọc theo disk:

```bash
MAX_EPISODES=50 MIN_EPISODE_INDEX=1000 NUM_GPUS=2 \
  DATASET_PATH=datasets/single_arm/multiview/makovian \
  bash scripts/run_single_arm_multiview_dreamgen_batch.sh
```

Output mặc định: `sampling_dataset/dense/single_arm/input/multiview/dreamgen/<category>/` (grid PNG + `batch_input.json`) và `sampling_dataset/dense/single_arm/output/multiview/dreamgen/<category>/` (mp4). Thư mục `dense/` là tuyến video chuẩn (full steps); sau này có thể thêm các tuyến khác (vd. video tăng tốc / sparse) cạnh `dense/` — xem skill `sampling-dataset-structure`.


# Bimanual
## Aloha_robot

### DreamGen batch (2×2 grid, droid variant)

LeRobot tasks live under `datasets/bimanual/multiview/{makovian,non_makovian}/<task>/`.
Grid inputs and `batch_input.json` are written to `sampling_dataset/bimanual/input/multiview/dreamgen/<category>/<task>/`.
Generated videos default to `sampling_dataset/bimanual/output/multiview/dreamgen/<category>/<task>/`.

Prepare only (3 static+dynamic slots: high, low, left wrist + black quadrant):

```bash
python scripts/prepare_bimanual_multiview_dreamgen_batch.py \
  --dataset_path datasets/bimanual/multiview/makovian/close_toolbox
```

Four live views (high, low, left wrist, right wrist):

```bash
python scripts/prepare_bimanual_multiview_dreamgen_batch.py \
  --dataset_path /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/datasets/bimanual/multiview/non_makovian/fold_bath_towel \
  --four_views
```

Single-task prepare + inference:

```bash
source models/DreamDojo/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/models/dreamgen

DATASET_PATH=datasets/bimanual/multiview/makovian/close_toolbox \
  bash scripts/run_bimanual_multiview_dreamgen_batch.sh
```
```bash
source models/dreamgen/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/models/dreamgen

DATASET_PATH=datasets/bimanual/multiview/non_makovian/fold_bath_towel \
  bash scripts/run_bimanual_multiview_dreamgen_batch.sh
```
### Tự động toàn bộ task

Script [scripts/run_bimanual_multiview_dreamgen_all.sh](scripts/run_bimanual_multiview_dreamgen_all.sh) lần lượt xử lý **mọi thư mục task** có `meta/episodes.jsonl` dưới `datasets/bimanual/multiview/makovian/*` và `.../non_makovian/*` (thứ tự tên thư mục). Mỗi task: (1) tạo lưới + `batch_input.json` dưới `sampling_dataset/bimanual/input/multiview/dreamgen/<category>/<task>/`, (2) chạy DreamGen lên batch đó.

Từ root repo (đã có venv trong script):

```bash
# Cả makovian + non_makovian, infer toàn episode, 2 GPU
NUM_GPUS=2 bash scripts/run_bimanual_multiview_dreamgen_all.sh

# Chỉ một nhánh
CATEGORY=makovian NUM_GPUS=2 bash scripts/run_bimanual_multiview_dreamgen_all.sh

# Bốn ô camera (không ô đen)
FOUR_VIEWS=1 NUM_GPUS=2 bash scripts/run_bimanual_multiview_dreamgen_all.sh

# Chỉ chuẩn bị batch + PNG (ffmpeg), không cần GPU — chạy infer sau
PREPARE_ONLY=1 bash scripts/run_bimanual_multiview_dreamgen_all.sh

# Batch đã có sẵn: chỉ infer
SKIP_PREPARE=1 NUM_GPUS=2 bash scripts/run_bimanual_multiview_dreamgen_all.sh

# Giới hạn episode mỗi task (test nhanh)
MAX_EPISODES=3 NUM_GPUS=1 bash scripts/run_bimanual_multiview_dreamgen_all.sh
```

Biến môi trường khác: `MULTIVIEW_ROOT`, `INPUT_DIR`, `OUTPUT_DIR_ROOT` (ghi đè thư mục video ra), `DYNAMIC_WRIST`, `MODEL_SIZE`, `MASTER_PORT`, `FPS`.

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ABQCFW5HLSyivpYQpzRsAgg/data/gr00t/brush_screws_into_dustpan_human_brush_left_hold?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AHgy4x95OLbl04yHDw7IqqI/data/gr00t/brush_screws_into_dustpan_left_brush_human_hold?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ACnI8RPDbWA9msMM0rX1_Xo/data/gr00t/close_cardboard_box?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AONAPYNvhZNtiKMuMOHvHJ4/data/gr00t/close_toolbox?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AONAPYNvhZNtiKMuMOHvHJ4/data/gr00t/close_toolbox?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ABhzbzBluSd85DYlgHT5B58/data/gr00t/find_hole_and_insert_into_gear?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AINcWNtzLGhS_FzI3Lf5MWI/data/gr00t/find_insert_large_gear_shaft?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AKkjdJU9oXcPF3LEFiDhhDw/data/gr00t/find_insert_small_gear_shaft?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AJRxziYsrEd7EHOC1r9BdBk/data/gr00t/fit_large_gear_shaft?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ADyb3_zrmEjmJ3W30IGuwTc/data/gr00t/fit_small_gear_shaft?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/APX3uyoq6RM9uVPVCcqrH08/data/gr00t/fold_bath_towel?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AIlA8EILMD-qArXYKTXU_IY/data/gr00t/fold_big_towel?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AHGRwiMtscgUoFF5bwucFMM/data/gr00t/fold_blue_towel?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AFfYrj8EiiPJrjRYKxPDK78/data/gr00t/fold_green_towel?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ANvXLKPkOPYUv1kiBQveORk/data/gr00t/fold_light_blue_towel?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AP_wBpVW5skU6g0u946MBAM/data/gr00t/fold_orange_towel?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ALJFsCCQEi9su0nxMIioJwE/data/gr00t/fold_towel_assist?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AA7cxdu5SwhzDeJ-FCBgxl4/data/gr00t/fold_towel_in_random_places?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AEZRF3x26iROaMlM0bkHl6c/data/gr00t/fold_yellow_towel?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ALMTPQGnShstWMSAAKzIvio/data/gr00t/handover_and_wipe_sponge?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ACGtaeyN40UK0-3LSxOkROo/data/gr00t/handover_clear_zip_bag_upright?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ACZOmKyuBDwR5T0JUTmRYZU/data/gr00t/handover_metallic_zip_bag_upright?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AM9LQZq43EHxl0W7k_HTihc/data/gr00t/hit_mark_with_hammer?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AG0WpnaOmp5j4z7X3tXWsm0/data/gr00t/hook_cable_8pin?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ADbEe5GDgrw6LnQ1mvic8r0/data/gr00t/hook_cable_narrow_8pin?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AC9UiUcRRXtEdY7sH_U9FQA/data/gr00t/hook_rubber_shaft?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AD5JRM8YqEMVZWOlvMwWz4M/data/gr00t/insert_gear_onto_shaft_sliding?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ADrhGfTGCDRY3Uzvm-A5jXQ/data/gr00t/insert_lan_cable_into_the_hub?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AE46iX-ZsUjAQO5uXcmaekE/data/gr00t/insert_large_gear_shaft?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/APbrrl5RFt3VnVD2MXW3_Jw/data/gr00t/insert_random_shaft_random_medium_gear?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ACEhPyU1hNfV4EY55W3uLLw/data/gr00t/insert_random_shaft_random_small_gear?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AIlPo-ooDKg4ZuWMvpvkGOE/data/gr00t/insert_rod_board?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AM94k3jyBMZ5xp9yRDfkuHQ/data/gr00t/insert_rod_moved_board?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AMLTcOVuZci-xNjnwp2AoeE/data/gr00t/insert_shaft_into_gear?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AP6Ueermvw4bDt6KyBQusmU/data/gr00t/insert_small_gear_shaft?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ABiQzdMEaR-7ZvkBBk1SWlM/data/gr00t/insert_usb_cable?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ADZih17xms9qLmda3U6sA1c/data/gr00t/insert_usb_cable_board?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AIdL9EvMO2gApRkgYTd9RDk/data/gr00t/insert_usb_cable_fix_parts?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AKwIgy3sl_JoZtNK0NhMHYE/data/gr00t/insert_usb_cable_fixed_by_robot?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AHyVCUFAc84GUtk7OAIyJ2U/data/gr00t/insert_usb_pcbox?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AH4mcBhwUDDoltT848Osl9s/data/gr00t/insert_usb_port_hub?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AFL7RqZu2OMTJX50cagxxG8/data/gr00t/insert_washer_shaft?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AKLXZKEpCH6kJuc8m3xXabs/data/gr00t/insert_washer_shaft_both_hands?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AMibO4nLGXVc9WnYvfPmgRA/data/gr00t/wipe_the_desk_with_a_blue_cloth?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AOJ6Ym5zCIDehwGr1NonQ8I/data/gr00t/wipe_the_desk_with_a_sponge?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/APIqxuoWkm6cQNwU_9M8pvY/data/gr00t/take_out_reel_clear_zip_bag?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ADnOB_uBb2pweSO-Uqg0P1o/data/gr00t/tape_closed_cardboard?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AAhtW5rPDugeOZ6eH27ek5A/data/gr00t/turn_towel_over?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ALzSwJTUYhEFlCTlniaq6Dg/data/gr00t/spoon_yellow_block_bowl?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AOL52QGW3zMZ3XawKWV9SWU/data/gr00t/spoon_red_block_bowl?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AE1aieZSLWx-75Ca4qmjU64/data/gr00t/remove_box_from_case_right_arm?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AI1Ln8ryvKGEyNiKE1Q4qaA/data/gr00t/remove_box_from_case?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AJQO0XN02iDKRzz-Qf_Sg4Y/data/gr00t/put_tape_white_case_alternating_arms?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AK4Fz8EnnTiplbXtuVassz0/data/gr00t/put_cup_moved_red_handle_holder?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0

https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AFGVgKQpj3R8NP98F8qFoJ0/data/gr00t/place_the_chinese_spoons_on_the_cloth?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=0


