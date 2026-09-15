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
