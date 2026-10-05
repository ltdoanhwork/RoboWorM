# Chạy nhiều GPU = chạy trên nhiều shell ID

Trên cluster này **mỗi shell ID = 1 máy = 1 GPU** (`GPU 0`, A100 80GB).
Muốn dùng 7 GPU thì phải phóng job vào 7 shell ID khác nhau.

Tất cả shell **dùng chung filesystem** `/pfss/mlde/workspaces/...`, nên code và
output là một — chỉ cần chia việc sao cho 2 shell không làm trùng nhau.

---

## 1. Xem mình có bao nhiêu shell

```bash
det shell list | grep VLA_doanh | grep RUNNING
```

Kiểm tra GPU của từng ID có rảnh không:

```bash
det shell open <ID> -- 'nvidia-smi --query-gpu=memory.used --format=csv,noheader'
# 0 MiB = rảnh. Có sẵn job khác thì bỏ qua ID đó.
```

## 2. Phóng job (LUÔN dùng tmux)

`nohup`/`&` **bị giết** ngay khi `det shell open` đóng. Chỉ `tmux` sống sót.

```bash
PROJ=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics

det shell open <ID> -- "
tmux new-session -d -s myjob '
  cd $PROJ &&
  bash scripts/run_humanoid_singleview_dreamgen_pisa.sh > $PROJ/logs/exp_logs/tmux_logs/pisa_<ID8>.log 2>&1
'"
```

**Mỗi `det shell open` chỉ làm 1 việc.** Gộp `kill-session; pkill; new-session`
vào 1 lệnh thì phần cuối bị cắt âm thầm (job không chạy mà vẫn tưởng chạy).

## 3. Chia việc cho N shell

### Cách A — mỗi shell 1 method (dễ nhất, dùng cho DreamGen/DreamDojo)

```bash
# shell 1 -> pisa, shell 2 -> svg1, shell 3 -> radial, ...
det shell open <ID1> -- "tmux new-session -d -s m_pisa 'cd $PROJ && bash scripts/run_humanoid_singleview_dreamgen_pisa.sh > $PROJ/logs/exp_logs/tmux_logs/pisa.log 2>&1'"
det shell open <ID2> -- "tmux new-session -d -s m_svg1 'cd $PROJ && bash scripts/run_humanoid_singleview_dreamgen_svg1.sh > $PROJ/logs/exp_logs/tmux_logs/svg1.log 2>&1'"
```

Không cần merge gì — mỗi method ghi vào cây `sampling_dataset/<method>/` riêng.

### Cách B — cùng 1 method, chia episode (Ctrl-World)

`NUM_SHARDS` = tổng số shell, `SHARD_ID` = thứ tự shell (0,1,2,…):

```bash
# shell thứ i trong 4 shell
det shell open <ID_i> -- "
tmux new-session -d -s cw_s<i> '
  cd $PROJ &&
  NUM_SHARDS=4 SHARD_ID=<i> bash scripts/run_humanoid_multiview_ctrlworld_worldcache.sh \
    > $PROJ/logs/exp_logs/tmux_logs/cw_shard<i>.log 2>&1
'"
```

Xong **tất cả** shard mới merge (chạy ở đâu cũng được, 1 lần):

```bash
python scripts/merge_ctrlworld_shard_summaries.py \
  --output-root sampling_dataset/worldcache/humanoid/output/multiview/ctrlworld
```

### Cách C — để script tự phát việc (timing pass)

```bash
WATCH=1 bash scripts/_dispatch_timing_cells.sh
```

Nó tự tìm shell `VLA_doanh` RUNNING có GPU rảnh, giao 1 cell/shell, tự tránh
trùng, và nhận thêm shell mới khi shell cũ chết.

### Cách D — chấm điểm NR nhiều máy

Mỗi shell nhận 1 dải slot, **`MACHINES` và `WORKERS` phải giống nhau ở mọi shell**:

```bash
MACHINES=28 WORKERS=2 bash scripts/run_dense_nr_slots.sh 0 4    # shell 1
MACHINES=28 WORKERS=2 bash scripts/run_dense_nr_slots.sh 4 4    # shell 2
```

Chi tiết: [`docs/dense_nr_10gpu_slots.md`](dense_nr_10gpu_slots.md).

## 4. Theo dõi

Log nằm trên FS dùng chung → **đọc trực tiếp, không cần connect lại**:

```bash
tail -n 30 logs/exp_logs/tmux_logs/pisa.log
```

Còn sống không:

```bash
det shell open <ID> -- 'tmux has-session -t myjob && echo RUNNING || echo DONE'
```

## 5. Dừng

```bash
det shell open <ID> -- 'tmux kill-session -t myjob'
```

---

## Bẫy (đều đã gặp thật)

| Hiện tượng | Nguyên nhân |
|---|---|
| Job biến mất sau vài phút, log 0 byte | dùng `nohup`/`&` thay vì tmux |
| `pkill` chạy nhưng job mới không lên | gộp nhiều lệnh vào 1 `det shell open` → bị cắt phần cuối |
| Shell "im lặng" không chạy gì | `tmux new-session -d` với tên đã tồn tại = no-op → `tmux ls` trước |
| 2 shell làm trùng việc, phần còn lại không ai làm | quên `SHARD_ID` khác nhau, hoặc `NUM_SHARDS` lệch giữa các shell |
| Job bỏ gần hết episode | env rác leak qua tmux → `unset NUM_SHARDS SHARD_ID NUM_EPISODES OUTPUT_ROOT` trước khi chạy |
| Shell bị cluster thu hồi giữa đêm | bình thường — mọi run script đều resumable, phóng lại đúng dải việc của shell đó |

Shell `TERMINATED` thì không exec vào được, phải `det shell start` shell mới.
