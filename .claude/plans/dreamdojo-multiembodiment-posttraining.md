# DreamDojo Multi-Embodiment Posttraining (ALOHA bimanual + DROID single-arm)

## Context

DreamDojo posttraining (under `models/DreamDojo/`) hiện chỉ register sẵn 4 embodiment: `gr1`, `g1`, `agibot`, `yam` — đều dùng `action_dim=384` thống nhất với các sub-range action được phân bổ cố định trong `docs/POSTTRAIN.md`. User muốn mở rộng để posttrain trên 2 dataset mới ở format LeRobot Parquet+MP4:

- `datasets/bimanual/singleview/{makovian,non_makovian}/<task>/` — ALOHA, fps=30, 480×640 (cam_high), action 14-dim qpos.
- `datasets/single_arm/singleview/` — chưa tồn tại; sẽ extract single-view (`exterior_1_left`) từ `datasets/droid_1.0.1_20chunks/`. Franka, fps=15, 180×320, action_dim=7 (`action.original` = ee_xyz3 + ee_euler3 + gripper1).

Mục tiêu: train **một** action-conditioned video2world model dùng chung 384-dim action embedding cho cả 2 embodiment, đúng tinh thần của POSTTRAIN.md (allocate sub-range trong `[169, 220)` "Reserved").

## Confirmed facts (đã verify trên repo)

- **YAML auto-loader** ở `models/DreamDojo/cosmos_predict2/experiments/base/action.py:297-316` tự scan `models/DreamDojo/configs/*.yaml` → đăng ký experiment `dreamdojo_<filename_stem>`. Vì vậy chỉ cần drop YAML, **không cần file `exp_*.py`** mới.
- **Multi-embodiment dispatcher** ở `models/DreamDojo/groot_dreams/dataloader.py::MultiVideoActionDataset` (line 148+) auto-detect embodiment bằng substring match (`gr1`/`agibot`/`g1`/`yam`) trong `dataset_path`.
- **Embodiment tag enum** ở `models/DreamDojo/groot_dreams/data/embodiment_tags.py` cần thêm 2 thành viên.
- **Modality/stats JSON fallback** ở `models/DreamDojo/shared_meta/` (đã có `GR1_*`, `G1_*`, `YAM_*`, `AgiBot_*`).
- **Action ranges** đã dùng: `[0,29) [29,58) [58,101) [101,147) [147,169)` GR1/G1/YAM/AgiBot. Range `[169, 220)` là Reserved — sẵn sàng cấp cho ALOHA và DROID.
- **Action format đã verify trực tiếp từ Parquet:**
  - ALOHA `action`: shape (14,) float32 (qpos motor_0..13). 800 frames/episode.
  - DROID `action.original`: shape (7,) float32 (cartesian6 + gripper1) — đúng format Bridge `Dataset_3D` (`action_dim=7`).

## Action-dim allocation (chuẩn)

Update `models/DreamDojo/docs/POSTTRAIN.md`, append vào danh sách hiện có:

| Range | Embodiment | Raw dim | Source |
|---|---|---|---|
| `[169, 183)` | **ALOHA bimanual** | 14 | parquet `action[0:14]` (qpos delta) |
| `[183, 190)` | **DROID single-arm** | 7 | parquet `action.original[0:7]` (cartesian + gripper) |
| `[190, 220)` | Reserved (chưa dùng) | – | – |

Phần còn lại (`[0,169)`, `[220,384)`) zero-pad.

## Files to modify / create

### Modify

1. **`models/DreamDojo/groot_dreams/data/embodiment_tags.py`** — thêm `ALOHA = "aloha"` và `DROID = "droid"`.
2. **`models/DreamDojo/groot_dreams/dataloader.py`** (line 148+ và `is_lerobot_dataset` line 13):
   - Mở rộng `is_lerobot_dataset` với keys mới: `"aloha"`, `"/bimanual/"`, `"droid"`, `"/single_arm/"`.
   - Trong `MultiVideoActionDataset.__init__`, thêm 2 nhánh resolve embodiment **TRƯỚC** nhánh `gr1` (vì path "single_arm" không chứa "g1" nhưng cẩn thận với collision):
     ```python
     elif "aloha" in path.lower() or "/bimanual/" in path.lower():
         embodiment = "aloha"
     elif "droid" in path.lower() or "/single_arm/" in path.lower():
         embodiment = "droid"
     ```
3. **`models/DreamDojo/groot_dreams/groot_configs.py::construct_modality_config_and_transforms`** — thêm 2 nhánh `elif embodiment == "aloha"` và `elif embodiment == "droid"`:
   - **aloha**: `video_key="video.cam_high"`, `state.qpos[0:14]`, `action.qpos[0:14]`, `timestep_interval=2` (30 fps native → 13 frames @ 15 Hz hiệu dụng ≈ 0.83 s).
   - **droid**: `video_key="video.exterior_1_left"`, `state.cartesian_position + state.gripper_position`, `action.cartesian_position + action.gripper_position` (concat → 7), `timestep_interval=1` (15 fps native).
   - Train transform giữ `VideoResize(480, 640, "linear")` — DROID 180×320 sẽ upscale 2.67× (xem Risk #1).
4. **`models/DreamDojo/groot_dreams/data/dataset.py::WrappedLeRobotSingleDataset.__getitem__`** (~line 1059-1071) — thêm 2 nhánh **trước** nhánh `gr1`, đặt action delta vào đúng slice 384-dim:
   ```python
   elif "aloha" in str(self.dataset_path).lower() or "/bimanual/" in str(self.dataset_path).lower():
       action_seq[:, 169:183] = delta_actions      # 14 dims
   elif "droid" in str(self.dataset_path).lower() or "/single_arm/" in str(self.dataset_path).lower():
       action_seq[:, 183:190] = delta_actions      # 7 dims
   ```
   Cùng file, cập nhật `_get_metadata` (~line 251-268) và `_get_lerobot_modality_meta` (~line 468-485) cho 2 tag mới: trỏ tới `shared_meta/ALOHA_*.json` và `shared_meta/DROID_*.json`.
5. **`models/DreamDojo/docs/POSTTRAIN.md`** — append 2 dòng cho ALOHA và DROID range.

### Create

6. **`models/DreamDojo/shared_meta/ALOHA_modality.json`** — `state.qpos[0:14]`, `action.qpos[0:14]`, `video.cam_high.original_key="observation.images.cam_high"`. (Có thể copy từ `meta/modality.json` của 1 task ALOHA bất kỳ.)
7. **`models/DreamDojo/shared_meta/ALOHA_stats.json`** — concat min/max/mean/std từ stats per-task có sẵn (hoặc compute lại trên 1 task, các task khác có `meta/stats.json` riêng đã được loader load trước fallback).
8. **`models/DreamDojo/shared_meta/DROID_modality.json`** — định nghĩa `state.cartesian_position[0:6]` + `state.gripper_position[0:1]`, `action.cartesian_position[0:6]` + `action.gripper_position[0:1]`, `video.exterior_1_left.original_key="observation.images.exterior_1_left"`.
9. **`models/DreamDojo/shared_meta/DROID_stats.json`** — compute bằng `groot_dreams.data.dataset.calculate_dataset_statistics` trên ~100 episodes của `datasets/single_arm/singleview/`.
10. **`models/DreamDojo/configs/2b_480_640_aloha.yaml`** — copy từ `2b_480_640_yam.yaml`, đổi `dataset_path` sang list 54 task subdirs dưới `datasets/bimanual/singleview/{makovian,non_makovian}/`. Giữ `state_t=4`, `action_dim=384`, `num_action_per_chunk=12`, `batch_size=4`. Group: `exp_aloha`, name: `aloha_bimanual_singleview`.
11. **`models/DreamDojo/configs/2b_480_640_droid_sv.yaml`** — tương tự, `dataset_path: [datasets/single_arm/singleview/]`. Group: `exp_droid_sv`, name: `droid_singlearm_singleview`. `batch_size=4` (giảm xuống 2 nếu OOM khi load 95k episodes).
12. **`finetuning/dreamdojo/scripts/extract_droid_singleview.py`** — script extract DROID singleview:
    - Source: `datasets/droid_1.0.1_20chunks/`
    - Target: `datasets/single_arm/singleview/`
    - Hardlink (`os.link`) hoặc symlink các parquet `data/chunk-*/*.parquet` và **chỉ** MP4 dưới `videos/chunk-*/observation.images.exterior_1_left/`.
    - Copy `meta/{episodes.jsonl, episodes_stats.jsonl, tasks.jsonl}`; patch `meta/info.json` để `features` chỉ còn `observation.images.exterior_1_left` trong các video keys.
    - Viết `meta/modality.json` mirror `shared_meta/DROID_modality.json`.
    - Khuyến nghị: prototype với 5 chunk đầu (`chunk-000..chunk-004`, ~5000 episodes / 1.5M frames) trước khi extract toàn bộ 95 chunks.

## Reusable existing code (KHÔNG viết lại)

- `models/DreamDojo/groot_dreams/dataloader.py::MultiVideoActionDataset` — đã handle multi-source weighting, sample dispatching, output shape `(C, T-1, H, W)` + 384-dim action.
- `models/DreamDojo/groot_dreams/data/dataset.py::WrappedLeRobotSingleDataset` — đã có pipeline parquet+MP4, delta-action computation (`action[t:t+4] - action[t-1]`), normalization theo `meta/stats.json`.
- `models/DreamDojo/groot_dreams/data/dataset.py::LeRobotSingleDataset` — đã handle parquet decoding, video frame fetching, modality config.
- `models/DreamDojo/cosmos_predict2/_src/predict2/action/networks/action_conditioned_minimal_v1_lvg_dit.py` — action embedder đã chấp nhận 384-dim, không cần đổi.
- `models/DreamDojo/cosmos_predict2/experiments/base/action.py` (line 297-316) — YAML auto-register, không cần đổi.
- `models/DreamDojo/configs/2b_480_640_yam.yaml` — template tốt nhất để clone (cũng là robot bimanual).
- `models/DreamDojo/launch.sh` — invoke không đổi: `bash launch.sh dreamdojo_2b_480_640_aloha`.

## Implementation sequence

1. **DROID extraction** (script ở step 12) — chạy parallel với code edits, chậm nhất.
2. Patch `embodiment_tags.py` (1).
3. Patch `groot_configs.py` thêm 2 branch (3).
4. Patch `dataset.py`: action slice + metadata fallback (4).
5. Patch `dataloader.py`: dispatcher + `is_lerobot_dataset` (2).
6. Tạo `shared_meta/ALOHA_*` và `DROID_*` JSON (6-9).
7. Drop `2b_480_640_aloha.yaml` (10).
8. Drop `2b_480_640_droid_sv.yaml` (11) — sau khi DROID extraction xong.
9. Update `docs/POSTTRAIN.md` (5).
10. Smoke test → 1-step dry run (xem Verification).

ALOHA path low-risk (480×640 native, không cần extract). DROID path high-risk (extraction + upscale + 95k episodes scan time).

## Verification

### Standalone dataset smoke test (ALOHA)

```bash
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/DreamDojo
PYTHONPATH=. python -c "
from groot_dreams.dataloader import MultiVideoActionDataset
ds = MultiVideoActionDataset(
    num_frames=13,
    dataset_path=['datasets/bimanual/singleview/makovian/close_cardboard_box'],
    data_split='train',
)
b = ds[0]
print('video', b['video'].shape, b['video'].dtype)         # (3, 12, 480, 640) uint8
print('action', b['action'].shape, b['action'].dtype)      # (12, 384) float32
print('non-zero range [169:183]:', b['action'][:, 169:183].abs().mean().item())  # > 0
print('zero range [0:169]   :', b['action'][:, :169].abs().sum().item())          # == 0
print('zero range [183:]    :', b['action'][:, 183:].abs().sum().item())          # == 0
"
```
Lặp lại cho `datasets/single_arm/singleview/` (sau extract), verify slice `[183:190)` non-zero.

### 1-step training dry run

```bash
cd models/DreamDojo
NPROC=1 bash launch.sh dreamdojo_2b_480_640_aloha \
  trainer.max_iter=2 \
  trainer.logging_iter=1 \
  checkpoint.save_iter=10000000 \
  job.wandb_mode=disabled
```
Pass criteria: dataset load thành công, 2 step train không lỗi, log cho thấy `action.shape=(B,12,384)` đi đúng vào model. Lặp lại với `dreamdojo_2b_480_640_droid_sv`.

### End-to-end multi-embodiment (optional)

Drop 1 YAML thứ ba liệt kê CẢ 2 root vào `dataset_path:` để test 1 batch mix ALOHA+DROID, sample-by-sample, slice action đúng cho từng embodiment.

## Risks & open questions

1. **DROID upscale 180×320 → 480×640** dùng `interpolation="linear"` có thể blur. Mitigation: switch sang `bicubic` hoặc thêm config `2b_180_320_droid_sv.yaml` với resolution native nếu chất lượng giảm. Quyết định sau khi xem sample đầu.
2. **ALOHA action delta wraparound**: `action.qpos` là absolute joint pos, delta `t - (t-1)` có thể spike khi gripper toggle. Mitigation: clip `±0.5 rad` trước khi assign vào slice `[169,183)`. Theo dõi loss sau dry-run.
3. **DROID `language_instruction`** là per-row string column (không phải `task_index → tasks.jsonl` style). v1 để text rỗng (loader đã `setdefault("ai_caption","")`), v2 thêm `get_language` đọc trực tiếp từ parquet. Không block dry-run.
4. **timestep_interval ALOHA**: chọn `=2` cho 30 fps native → 15 Hz hiệu dụng. Nếu motion 13 frames quá nhanh, đổi sang `=4` (matching YAM/AgiBot).
5. **Quy mô ALOHA YAML**: 54 task subdirs trong `dataset_path:` list — cân nhắc thêm glob support `Path(p).parent.glob(Path(p).name)` trong `MultiVideoActionDataset.__init__` (low priority).
6. **Pretrained checkpoint** cần set ở `cosmos_predict2/experiments/base/action.py` qua `load_path` (theo POSTTRAIN.md step 2). Default trỏ tới `checkpoints/DreamDojo/2B_pretrain/iter_000140000/`. Confirm file tồn tại trên storage trước khi launch.

## Critical files (paths đầy đủ)

- `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/DreamDojo/groot_dreams/data/embodiment_tags.py`
- `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/DreamDojo/groot_dreams/dataloader.py`
- `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/DreamDojo/groot_dreams/groot_configs.py`
- `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/DreamDojo/groot_dreams/data/dataset.py`
- `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/DreamDojo/shared_meta/ALOHA_modality.json` (new)
- `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/DreamDojo/shared_meta/ALOHA_stats.json` (new)
- `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/DreamDojo/shared_meta/DROID_modality.json` (new)
- `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/DreamDojo/shared_meta/DROID_stats.json` (new)
- `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/DreamDojo/configs/2b_480_640_aloha.yaml` (new)
- `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/DreamDojo/configs/2b_480_640_droid_sv.yaml` (new)
- `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/DreamDojo/docs/POSTTRAIN.md` (append)
- `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/finetuning/dreamdojo/scripts/extract_droid_singleview.py` (new)
