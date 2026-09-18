# Cách một con số ra đời — flow tính, giải thích đơn giản

Tài liệu này trả lời đúng 3 câu: **không ghép cặp thì số ra thế nào**, **ghép cặp thì thế nào**,
và **chỗ nào là trung bình, chỗ nào không**.

Câu trả lời ngắn cho câu cuối: **chỉ có 1 chỗ dùng trung bình** (gộp 3 cột thành một điểm ở
Option 1/2, và đó là bước *tuỳ chọn*). Mọi bước gộp khác đều là **trung vị** hoặc **phân vị**,
cố ý chọn vậy vì một clip lỗi nặng sẽ kéo lệch trung bình.

---

## Tầng 0 — từ video ra 1 số cho 1 clip

Mỗi metric có cách gộp frame → clip riêng. Không cái nào là "trung bình đơn giản":

| Metric | Gộp các frame lại bằng | Vì sao không dùng trung bình |
|---|---|---|
| **Visual Integrity** | **phân vị 10** của CLIP-IQA+ các frame, ×100 | Trung bình che mất một đoạn ngắn bị đen/vỡ. p10 hỏi "gần như mọi frame còn dùng được không". |
| **Text Alignment** | **trung bình** CLIPScore các frame | Đây là ngoại lệ — CLIPScore đã bounded 0–100, không có đuôi dài. |
| **Actor Activity** | camera tĩnh/đầu: **phân vị 95** của flow vùng robot; camera cổ tay: **trung vị** cả frame | p95 bắt đỉnh chuyển động; trung vị cho wrist vì cả frame đang động. |
| **Active Rate** | **tỉ lệ** cặp frame có chuyển động ≥ 1 px | Là tỉ lệ, không phải mức. |
| **AMC** | **trung bình có trọng số** theo mức hoạt động của từng bộ-3-frame | Frame robot đứng yên thì nội suy nào cũng dễ → phải hạ trọng số, không thì điểm bị thổi. |

Mỗi clip ra 1 file JSON. Ví dụ thật:

```
visual_integrity          37.15     (p10 của CLIP-IQA+, ×100)
text_condition_alignment  27.79     (CLIPScore trung bình)
actor_motion_raw           5.28     (px/frame, p95 vùng robot)
actor_motion_active_rate   0.853    (85% cặp frame có chuyển động)
active_motion_continuity  74.99     (VFI gain có trọng số, ×100)
```

### Cửa chặn: một số có thể bị bỏ thành `None`

Trước khi số được dùng, nó phải qua 4 cửa. Đây là lý do một ô trong bảng có thể trống:

```
theo dõi được robot < 50% khung          -> bỏ Actor Activity + AMC
tổng trọng số hoạt động < 3.0            -> bỏ AMC
số bộ-3 "đang hoạt động" < 3             -> bỏ AMC
số bộ-3 hoạt động < 5% tổng số cặp       -> bỏ AMC
clip không có câu lệnh text              -> bỏ Text Alignment
```

Riêng AMC còn một cửa ở mức *từng bộ-3 frame*: nếu hai frame kề gần như trùng nhau
(`copy_err < COPY_ERR_FLOOR`) thì trọng số của bộ-3 đó về 0. Nếu không có cửa này, một
đoạn video đứng yên sẽ cho "nội suy hoàn hảo" và thổi điểm AMC lên.

---

## Tầng 1 — hai cách đi từ nhiều clip lên một con số

Đây là chỗ bạn đang hỏi. Cùng một tập clip, hai đường khác nhau:

### Đường A — KHÔNG ghép cặp ("cứ xài chính video đó thôi")

```
clip 1 của dicache:  VI = 20.1
clip 2 của dicache:  VI = 24.8
clip 3 của dicache:  VI = 18.2
...
28 clip
        |
        v  TRUNG VỊ (không phải trung bình)
   dicache VI = 20.80,  khoảng tin cậy [18.84, 22.00]

Làm y vậy cho dense:
   dense   VI = 21.50,  khoảng tin cậy [20.85, 21.95]

So sánh = xem hai khoảng có chồng nhau không.
```

**Số output**: một điểm tuyệt đối trên thang gốc của metric (VI 13–60, AMC 19–87,
TextAlign 16–36, ActorActivity 0–27 px/frame, ActiveRate 0–1).

**Ưu**: dùng được **mọi clip** (2,683 thay vì 1,610 — nhiều hơn 67%), không cần đối tác,
dễ giải thích, có thể để `dense` và `GT` làm 2 dòng mốc ngay trong cùng bảng.

**Nhược, đo thật**: khoảng tin cậy của dicache `[18.84, 22.00]` **chồng lên** dense
`[20.85, 21.95]` → không kết luận được gì. Mà bảng ghép cặp thì thấy rõ dicache
kém hơn `−1.28`.

### Đường B — CÓ ghép cặp (đang dùng)

```
Ghép theo pair_key = cùng episode, cùng góc camera, cùng độ dài, cùng fps:

  episode "gấp khăn" cam_high:   dense 30.2  |  dicache 28.9   ->  hiệu = -1.3
  episode "cầm cốc"  cam_high:   dense 41.7  |  dicache 40.8   ->  hiệu = -0.9
  episode "mở hộp"   cam_low :   dense 22.4  |  dicache 21.2   ->  hiệu = -1.2
  ...
  28 cặp
        |
        v  TRUNG VỊ của các HIỆU
   dicache VI delta = -1.28,  khoảng tin cậy [-1.57, -0.79],  p = 0.004
```

**Số output**: một **hiệu số**, cùng đơn vị với metric gốc. Âm = tệ hơn dense.

**Ưu**: chú ý hai episode trên — `30.2` và `41.7` chênh nhau **11 điểm** vì nội dung
khác nhau, nhưng *hiệu* của cả hai đều quanh `−1`. Phép trừ **loại bỏ phần "episode nào"**
và chỉ để lại phần "method nào". Đó là lý do khoảng tin cậy hẹp hơn **4.1 lần**.

**Nhược**: cần đối tác dense hợp lệ → mất 40% clip. Và số đọc khó hơn ("−1.28" phải giải
thích là so với cái gì).

### Đo thẳng cái giá của việc bỏ ghép cặp

| | Đường A (không ghép) | Đường B (ghép cặp) |
|---|---|---|
| clip dùng được | **2,683** | 1,610 |
| độ rộng khoảng tin cậy | 3.99 | **0.98** (hẹp hơn 4.1×) |
| số hiệu ứng phát hiện được | 47 | **128** |

**81 trong 128 hiệu ứng biến mất** nếu bỏ ghép cặp (mất 63%). Và đường A còn báo thêm
**6 ô mà đường B nói là không có gì** — tức nó "phát hiện" khác biệt vốn do tập episode
tình cờ khác nhau, không do method. Đó là dương tính giả.

Ví dụ đầy đủ trên `ctrlworld_4view/humanoid`, Visual Integrity:

```
                 KHÔNG ghép cặp              CÓ ghép cặp
dense       21.50 [20.85, 21.95]  (mốc)
dicache     20.80 [18.84, 22.00]  chồng     ->  -0.38  không kết luận
worldcache  21.41 [19.03, 22.10]  chồng     ->  -0.78  RESOLVED  <-- chỉ ghép cặp thấy
pisa        20.53 [19.05, 21.62]  chồng     ->  -0.68  RESOLVED  <-- chỉ ghép cặp thấy
sito        18.88 [17.80, 19.46]  TÁCH RÕ   ->  -8.80  RESOLVED
itm         17.37 [15.93, 18.62]  TÁCH RÕ   ->  -4.30  RESOLVED
```

**Kết luận thực dụng**: 3 method hỏng nặng (ITM, SiTo, SVG2) thì **cả hai đường đều thấy**.
6 method còn lại thì chỉ ghép cặp mới thấy. Nếu paper chỉ cần claim "pruning phá chất lượng,
caching thì không" → đường A đủ và dễ đọc hơn. Nếu cần nói "dicache mất 1.2%" → phải ghép cặp.

---

## Tầng 2 — gộp lên bảng (chỗ duy nhất có trung bình)

Hai kiểu gộp, dùng cho hai câu hỏi khác nhau:

### Xếp hạng method → gộp theo CẶP, không gộp theo ô

```
tất cả 227 cặp của dicache (mọi model, mọi robot, mọi camera)
        |
        v  TRUNG VỊ
   dicache VI = -1.22  [-1.57, -0.79]   <- đây là bảng T3b, số lên headline
```

**Cảnh báo đã đo**: KHÔNG được lấy trung bình các ô để xếp hạng. Làm vậy cho Spearman
**+0.883 chứ không phải +1.0** so với T3b — nó đẩy `radial` từ nhất (−0.07) xuống thứ 6,
vì các method không chạy trên cùng tập cấu hình và ô ít mẫu bị cân bằng như ô nhiều mẫu.

### Chỉ ra method vỡ ở đâu → chia ô, mỗi ô tính riêng

```
ô = (model, robot, MV/SV) x method
mỗi ô: trung vị các hiệu trong ô đó + khoảng tin cậy + p
```

### Nếu muốn 1 số/ô thay vì 5 bảng (Option 1/2 — **chỗ duy nhất dùng trung bình**)

```
trong ô đó, với mỗi cột:
   Retention = TRUNG VỊ theo cặp của (accel / dense) x 100
                ^^^^^^^ vẫn trung vị

   VI:        98.8
   AMC:      101.3     -> TRUNG BÌNH 3 số này = 100.0
   TextAlign:  99.9       ^^^^^^^^^ CHỖ DUY NHẤT DÙNG TRUNG BÌNH
```

Trung bình được phép ở đây vì mỗi số đã là **phần trăm của chính dense cột đó** — cùng đơn vị,
cùng mốc 100. Nếu lấy trung bình các hiệu số thô thì sai, vì `−1.22 VI` và `−0.005 ActiveRate`
khác thang nhau 200 lần.

**Hai quy tắc cứng ở bước này:**

1. **Actor Activity và Active Rate KHÔNG được vào trung bình.** Chúng mô tả — method làm
   tay robot động *nhiều hơn* không phải tốt hơn. Chúng đi thành bảng riêng, và chính đó
   mới lộ ra kiểu hỏng riêng của ITM/SiTo: Actor Activity retention xuống **40%**, tức
   chuyển động bị **xoá**, không chỉ mờ. Trộn vào trung bình là mất bằng chứng đó.
2. **`n < 8` phải có ký hiệu riêng, không được in `ns`.** 11/37 ô không kết luận được là
   *vì thiếu dữ liệu*, không phải vì không có hiệu ứng. Trước khi sửa, cả dòng
   `DreamGen/Bimanual/SV` in `ns` cho 9 method — đọc thành "ở đây method nào cũng miễn phí",
   thực tế là n=1–4, chưa kiểm định gì.

---

## Tầng 3 — khi nào được phát biểu "method này làm hỏng"

Cần **cả 3** điều kiện, và **3 cửa phủ quyết**:

```
CẦN:  khoảng tin cậy 95% không chứa 0   (chặt: 0 nằm đúng biên là KHÔNG đạt)
      p < 0.05  (Wilcoxon)
      n >= 8 cặp

PHỦ QUYẾT (dù 3 điều trên đạt):
      cột fail bài test "video thật phải là trần"     -> Visual Integrity, chỉ khi xếp hạng model
      hiệu DƯƠNG trên cột AMC khi so với dense        -> là artifact làm mượt quá mức, không phải cải thiện
      hai clip không cùng phép đo (fps/độ dài lệch)   -> loại khỏi cặp ngay từ đầu
```

Cửa phủ quyết thứ 2 là mới, tìm ra ngày 2026-09-17: 3 ô có AMC dương với p tới 2e-05, đo
pixel thì thấy jerk (độ khó nội suy) giảm ×0.86–0.91 — tức method làm video *mượt hơn*,
và AMC vốn thưởng đúng cái đó. Không phải chất lượng tốt hơn.

---

## Tóm một dòng

```
video -> [p10 / trung bình / p95 / tỉ lệ / trung bình-có-trọng-số] -> 1 số/clip
      -> [trừ theo cặp cùng episode] -> 1 hiệu/cặp        (bỏ bước này = mất 63% phát hiện)
      -> [TRUNG VỊ các hiệu] -> 1 số/ô hoặc /method
      -> [chia dense, rồi TRUNG BÌNH 3 cột] -> 1 số/ô     (tuỳ chọn, chỗ duy nhất có trung bình)
      -> [CI + p + n>=8 + 3 cửa phủ quyết] -> được phép claim
```
