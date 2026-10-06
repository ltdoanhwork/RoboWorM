# Fill-missing progress

Last checked: 2026-09-15 07:13 UTC  
Active tmux: `fill_missing_20260915_g0`  
Log: `logs/fill_missing/fill_missing_20260915_g0.log`

## Canonical progress

| Method | Target | Valid | Remaining |
|---|---:|---:|---:|
| dense, single-arm SV DreamGen non_makovian | 120 | 120 | 0 |
| sito, single-arm SV DreamGen non_makovian | 120 | 120 | 0 |
| itm, single-arm SV DreamGen non_makovian | 120 | 120 | 0 |
| pisa, single-arm SV DreamGen non_makovian | 120 | 120 | 0 |
| radial, single-arm SV DreamGen non_makovian | 120 | 120 | 0 |
| svg1, single-arm SV DreamGen non_makovian | 120 | 120 | 0 |
| svg2, single-arm SV DreamGen non_makovian | 120 | 104 | 16 |
| fastercache, bimanual SV DreamGen non_makovian | 150 | 21 | 129 |

Current remaining: **145 valid episodes**.

## Execution

`archived:jobs_fill_missing.sh` runs `WAVE=all` sequentially:

1. WAVE 1: single-arm SV, `NUM_CHUNKS="4 6 7"`; dense → SiTo → ITM → PISA → Radial → SVG1 → SVG2.
2. WAVE 2: bimanual FasterCache, `NUM_CHUNKS="3 5 6 18"`.

Resume command:

```bash
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics
CUDA_VISIBLE_DEVICES=0 WAVE=all DRY_RUN=0 bash archived:jobs_fill_missing.sh
```

## Incident notes

- Previous runs stopped with `OSError: [Errno 122] Disk quota exceeded` while saving SVG2.
- Removed only old fill log and zero-byte MP4s. Checkpoint/model logs remain untouched.
- Zero-byte outputs must be removed before resume; runner skips any existing `is_file()`.
- Verify with valid-size MP4 count, not directory entry count.
