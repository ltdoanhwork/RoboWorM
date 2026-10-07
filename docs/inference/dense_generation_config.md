# Dense generation configuration (measured)

Every video property below is measured with `ffprobe` from the prediction
clips in `sampling_dataset/dense/`; sampler settings come from the run
config / canonical run script (`cfg_src` in the JSON). Regenerate with:

```bash
python3 scripts/tables/make_dense_generation_table.py \
  --md docs/inference/dense_generation_config.md \
  --json results/reports/dense_generation_config.json \
  --tex results/reports/dense_generation_config.tex
```

Cells: 12 | clips probed: 3165 / 3166 on disk | **1 unreadable**: `single_arm/output/singleview/dreamgen/makovian/iter_000090000_static16fps/global/episode_000056.mp4`

Frames/duration per clip are **median [min–max]**, not a constant: every clip
is generated to match its ground-truth episode length, so clip length varies
within a cell.

## Per generator family (paper layout)

### Ctrl-World

| | DROID-MV<br>(single_arm/multiview) | ALOHA-MV-grid<br>(bimanual/multiview) | RoboCOIN-MV-grid<br>(humanoid/multiview) |
|---|---|---|---|
| **Source data** | DROID (cadene/droid_1.0.1) | AIST Bimanual (ALOHA) | RoboCOIN (AIRBOT MMK2) |
| **Generator** | Ctrl-World (SVD img2vid U-Net, action-cond.) | Ctrl-World (4-view 2x2 latent canvas) | Ctrl-World (4-view 2x2 latent canvas) |
| **Checkpoint** | Ctrl-World/checkpoint-10000.pt (released DROID pretrain) | bimanual_multiview_4view_grid/checkpoint-110000.pt | humanoid_multiview_4view_grid/checkpoint-140000.pt |
| **Views / clip** | 3 (3x1 latent strip) | 4 (2x2 latent grid) | 4 (2x2 latent grid) |
| **Resolution (per view)** | 320x192 | 320x192 | 320x192 |
| **Resolution (canvas)** | 960x192 | 1280x192 | 1280x192 |
| **FPS** | 5 | 5 | 5 |
| **Frames per clip** | 81 [22-434] | 165 [41-497] | 45 [13-157] |
| **Duration per clip** | 16.20 s [4.40-86.80] | 33.00 s [8.20-99.40] | 9.00 s [2.60-31.40] |
| **Guidance scale** | 1.0 | 1.0 | 1.0 |
| **Sampling steps** | 50 | 50 | 50 |
| **Seed** | unseeded | unseeded | unseeded |
| **Clips on disk** | 257 | 300 | 248 |

### DreamGen

| | DROID-MV<br>(single_arm/multiview) | ALOHA-MV<br>(bimanual/multiview) | RoboCOIN-MV<br>(humanoid/multiview) | RoboChallenge-SV<br>(single_arm/singleview) | ALOHA-SV<br>(bimanual/singleview) | GR1-SV<br>(humanoid/singleview) |
|---|---|---|---|---|---|---|
| **Source data** | DROID (cadene/droid_1.0.1) | AIST Bimanual (ALOHA) | RoboCOIN (AIRBOT MMK2) | RoboChallenge (Traly/RoboChallenge-lerobot) | AIST Bimanual (ALOHA), cam_high | GR00T Teleop GR1 (nvidia/PhysicalAI-...-GR1) |
| **Generator** | Cosmos-Predict2 14B (GR00T-Dreams-DROID) | Cosmos-Predict2 2B (post-trained) | Cosmos-Predict2 2B (post-trained) | Cosmos-Predict2 2B (post-trained, static 16fps) | Cosmos-Predict2 2B (post-trained, 16fps) | Cosmos-Predict2 2B (post-trained, static 16fps) |
| **Checkpoint** | GR00T-dreams-DROID 14B (video2world_gr00t, --gr00t_variant droid) | bimanual_multiview_seg16/iter_000120000 | humanoid_multiview_seg16/iter_000100000 | single_arm_static_singleview_16fps_v2/iter_000090000 | bimanual_singleview_16fps/iter_000110000 | humanoid_singleview_16fps/iter_000090000 |
| **Views / clip** | 3 (2x2 grid (3 active + blank)) | 4 (2x2 grid) | 4 (2x2 grid) | 1 (single view) | 1 (single view) | 1 (single view) |
| **Resolution (per view)** | 384x216 | 384x216 | 384x216 | 640x480 | 768x432 | 768x432 |
| **Resolution (canvas)** | 768x432 | 768x432 | 768x432 | 640x480 | 768x432 | 768x432 |
| **FPS** | 10 | 16 | 16 | 16 | 16 | 16 |
| **Frames per clip** | 93 [93-93] | 535 [135-1602] | 154 [50-505] | 369 [185-645] | 1013 [185-1657] | 277 [185-277] |
| **Duration per clip** | 9.30 s [9.30-9.30] | 33.44 s [8.44-100.12] | 9.62 s [3.12-31.56] | 23.06 s [11.56-40.31] | 63.31 s [11.56-103.56] | 17.31 s [11.56-17.31] |
| **Guidance scale** | 7 | 7 | 7 | 7 | 7 | 7 |
| **Sampling steps** | 35 | 35 | 35 | 35 | 35 | 35 |
| **Seed** | 0 | 0 + segment index | 0 + segment index | 0 | 0 | 0 |
| **Clips on disk** | 257 | 300 | 248 | 240 | 300 | 260 |

### DreamDojo

| | RoboChallenge-SV<br>(single_arm/singleview) | ALOHA-SV<br>(bimanual/singleview) | GR1-SV<br>(humanoid/singleview) |
|---|---|---|---|
| **Source data** | RoboChallenge (Traly/RoboChallenge-lerobot) | AIST Bimanual (ALOHA), cam_high | GR00T Teleop GR1 (nvidia/PhysicalAI-...-GR1) |
| **Generator** | DreamDojo 2B action-cond. DiT (autoregressive) | DreamDojo 2B action-cond. DiT (autoregressive) | DreamDojo 2B action-cond. DiT (autoregressive) |
| **Checkpoint** | dreamdojo_2b_480_640_single_arm_sv/iter_000030000 | dreamdojo_2b_480_640_aloha_resume70k_bimanual_sv/iter_000100000 | dreamdojo/2B_GR1_post-train/iter_000050000 |
| **Views / clip** | 1 (single view) | 1 (single view) | 1 (single view) |
| **Resolution (per view)** | 640x480 | 640x480 | 640x480 |
| **Resolution (canvas)** | 640x480 | 640x480 | 640x480 |
| **FPS** | 15 | 10 | 10 |
| **Frames per clip** | 187 [133-577] | 493 [121-1489] | 98 [49-1420] |
| **Duration per clip** | 12.47 s [8.87-38.47] | 49.30 s [12.10-148.90] | 9.80 s [4.90-142.00] |
| **Guidance scale** | 1.5 | 1.5 | 1.5 |
| **Sampling steps** | 35 | 35 | 35 |
| **Seed** | per-chunk index | per-chunk index | per-chunk index |
| **Clips on disk** | 240 | 300 | 216 |

## All cells (flat)

| Cell | Source data | Generator | Views | Res (per view) | Res (canvas) | FPS | Frames/clip | Duration/clip | Guid. | Steps | Seed | Clips |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `single_arm/multiview/ctrlworld` | DROID (cadene/droid_1.0.1) | Ctrl-World (SVD img2vid U-Net, action-cond.) | 3 (3x1 latent strip) | 320x192 | 960x192 | 5 | 81 [22–434] | 16.20 s | 1.0 | 50 | unseeded | 257 |
| `bimanual/multiview/ctrlworld_4view_grid` | AIST Bimanual (ALOHA) | Ctrl-World (4-view 2x2 latent canvas) | 4 (2x2 latent grid) | 320x192 | 1280x192 | 5 | 165 [41–497] | 33.00 s | 1.0 | 50 | unseeded | 300 |
| `humanoid/multiview/ctrlworld_4view_grid` | RoboCOIN (AIRBOT MMK2) | Ctrl-World (4-view 2x2 latent canvas) | 4 (2x2 latent grid) | 320x192 | 1280x192 | 5 | 45 [13–157] | 9.00 s | 1.0 | 50 | unseeded | 248 |
| `single_arm/multiview/dreamgen` | DROID (cadene/droid_1.0.1) | Cosmos-Predict2 14B (GR00T-Dreams-DROID) | 3 (2x2 grid (3 active + blank)) | 384x216 | 768x432 | 10 | 93 [93–93] | 9.30 s | 7 | 35 | 0 | 257 |
| `bimanual/multiview/dreamgen` | AIST Bimanual (ALOHA) | Cosmos-Predict2 2B (post-trained) | 4 (2x2 grid) | 384x216 | 768x432 | 16 | 535 [135–1602] | 33.44 s | 7 | 35 | 0 + segment index | 300 |
| `humanoid/multiview/dreamgen` | RoboCOIN (AIRBOT MMK2) | Cosmos-Predict2 2B (post-trained) | 4 (2x2 grid) | 384x216 | 768x432 | 16 | 154 [50–505] | 9.62 s | 7 | 35 | 0 + segment index | 248 |
| `single_arm/singleview/dreamgen` | RoboChallenge (Traly/RoboChallenge-lerobot) | Cosmos-Predict2 2B (post-trained, static 16fps) | 1 (single view) | 640x480 | 640x480 | 16 | 369 [185–645] | 23.06 s | 7 | 35 | 0 | 240 |
| `bimanual/singleview/dreamgen` | AIST Bimanual (ALOHA), cam_high | Cosmos-Predict2 2B (post-trained, 16fps) | 1 (single view) | 768x432 | 768x432 | 16 | 1013 [185–1657] | 63.31 s | 7 | 35 | 0 | 300 |
| `humanoid/singleview/dreamgen` | GR00T Teleop GR1 (nvidia/PhysicalAI-...-GR1) | Cosmos-Predict2 2B (post-trained, static 16fps) | 1 (single view) | 768x432 | 768x432 | 16 | 277 [185–277] | 17.31 s | 7 | 35 | 0 | 260 |
| `single_arm/singleview/dreamdojo` | RoboChallenge (Traly/RoboChallenge-lerobot) | DreamDojo 2B action-cond. DiT (autoregressive) | 1 (single view) | 640x480 | 640x480 | 15 | 187 [133–577] | 12.47 s | 1.5 | 35 | per-chunk index | 240 |
| `bimanual/singleview/dreamdojo` | AIST Bimanual (ALOHA), cam_high | DreamDojo 2B action-cond. DiT (autoregressive) | 1 (single view) | 640x480 | 640x480 | 10 | 493 [121–1489] | 49.30 s | 1.5 | 35 | per-chunk index | 300 |
| `humanoid/singleview/dreamdojo` | GR00T Teleop GR1 (nvidia/PhysicalAI-...-GR1) | DreamDojo 2B action-cond. DiT (autoregressive) | 1 (single view) | 640x480 | 640x480 | 10 | 98 [49–1420] | 9.80 s | 1.5 | 35 | per-chunk index | 216 |
