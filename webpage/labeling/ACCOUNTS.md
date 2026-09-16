# Rater accounts

| Person | Institution | Account |
|---|---|---|
| Nhi Phạm | MPI-INF | `nhi_pham` |
| Hoàng Trần | VinUni | `hoang_tran` |
| Minh Lợi | VinUni | `minh_loi` |
| Lưu Minh Tùng | KAIST | `luu_minh_tung` |
| Christopher Wewer | MPI-INF | `christopher_wewer` |
| Minh Đức | VinUni | `minh_duc` |
| Bảo Ngọc | VinUni | `bao_ngoc` |
| Gia Bình | VinUni | `gia_binh` |
| Đức Duy | VinUni | `duc_duy` |
| Kiên Bùi | VinUni | `kien_bui` |
| Doanh 1 | — | `doanh_1` |
| Doanh 2 | — | `doanh_2` |

Each account has one fixed assignment: 50 Geometry videos and 48 or 49
Hallucinate videos, for 98 or 99 total. Across all twelve accounts, every one of
the 300 Geometry items and 291 Hallucinate items has exactly two distinct
assigned raters.

Geometry catalog order is the 300-row order from
`sampling_dataset/data_for_web/data_for_web.csv`. Hallucinate assignment is
atomic by video: each assigned rater scores every rateable segment, while
conflict detection compares the two resulting overall video scores.

## Conflict resolution

| Check | Two raters agree | Two raters disagree |
|---|---|---|
| Geometry | that score wins (`consensus: raters`) | the arbiter score wins (`consensus: arbiter`) |
| Hallucinate | that score wins (`consensus: raters`) | a checker decides (`consensus: checker`/`checkers`) |

The **geometry arbiter** is `thanh_check`, who scored all 300 geometry clips
offline — that pass is the `score` column of `data_for_web.csv`. Because it
covers the whole catalog rather than only the conflicts a checker happened to
reach, it settles every geometry conflict on its own, so geometry conflicts are
never queued as checker work. Load it with:

```bash
python3 scripts/import_geo_arbiter_labels.py --dry-run   # verify mapping
python3 scripts/import_geo_arbiter_labels.py             # upsert 300 rows
```

Row *i* of the CSV is catalog id `{i:05d}`. The importer re-verifies that against
`metrics.multiview.json` (whose builder checks each rendered clip's `meta.json`
against the CSV `video_path`) and aborts on any mismatch, since a wrong mapping
would attach a score to a different video.

Hallucinate has no arbiter, so `quang_check` / `thanh_check` still adjudicate
those conflicts in the UI.
