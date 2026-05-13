# Markovian Task Classification — Run Commands

> Chạy từ thư mục gốc:
> `cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics`

---

## Bước 1 — Khởi động vLLM server (Terminal riêng, giữ chạy nền)

```bash
bash scripts/launch_qwen3vl_server.sh
```

Chờ đến khi thấy: `Application startup complete` (~2–5 phút lần đầu tải model).

---

## Bước 2 — Phân loại DROID (20 chunks)

```bash
python scripts/classify_markovian_qwen.py
```

Output:
- `scripts/task_labels_cache.jsonl` — cache resumable
- `scripts/task_labels.csv` — ~49k task labels

---

## Bước 3 — Join labels về episodes + sample 130/label (DROID)

```bash
python scripts/join_labels_to_episodes.py
```

Output:
- `scripts/episode_markovian_split_full.csv` — toàn bộ episodes trong 20 chunks
- `scripts/episode_markovian_split.csv` — **130 markovian + 130 non_markovian**

---

## Bước 4 — Phân loại GR1 humanoid (DreamDojo-HV + EgoDex)

```bash
python scripts/classify_markovian_gr1.py
```

Output:
- `scripts/gr1_task_labels_cache.jsonl` — cache resumable
- `scripts/gr1_task_labels.csv` — toàn bộ 133 episodes
- `scripts/gr1_episode_sampled.csv` — **≤33/label/source** (cân bằng 2 nguồn)

---

## Tóm tắt output files

| File | Mô tả |
|---|---|
| `scripts/task_labels.csv` | DROID: 49k task strings + label |
| `scripts/episode_markovian_split_full.csv` | DROID: tất cả episodes trong 20 chunks |
| `scripts/episode_markovian_split.csv` | DROID: 260 episodes sampled (130×2) |
| `scripts/gr1_task_labels.csv` | GR1: 133 episodes + label |
| `scripts/gr1_episode_sampled.csv` | GR1: ~132 episodes sampled (33×2×2) |
