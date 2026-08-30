# Hướng dẫn đánh giá chất lượng robot trong video

## Nhiệm vụ

Bạn sẽ xem các video robot do mô hình sinh ra và đánh giá độ ảo giác (hallucination) — tức những lỗi mà một video quay robot thật không bao giờ có. Chất lượng của robot bao gồm: robot có bị biến dạng, mờ, loang màu, chuyển động kì lạ hay chuyển động không đúng vật lý thông thường hay không.

Mỗi video được chia thành các đoạn ngắn (segment). Bạn chấm điểm từng đoạn theo thang:

**Thang điểm:** Good – Medium – Bad

Trong đó Good nghĩa là biến dạng rất ít hoặc không có — nhìn như một video sạch.

Lưu ý: chấm mỗi đoạn độc lập. Một đoạn có thể bẩn nhưng các đoạn sau đó robot sạch trở lại — khi đó đoạn sau vẫn chấm Good, đừng để đoạn trước ảnh hưởng đoạn sau.

## Trọng tâm đánh giá

Chỉ tập trung vào bản thân robot:

- Hình dáng: robot có bị biến dạng, kéo giãn, gãy khúc, đứt rời tay/gripper, mờ nhòe hay tan biến không?
- Màu sắc / bề mặt: có loang màu, nhấp nháy, đổi màu bất thường không?
- Chuyển động: có mượt và đúng vật lý như robot thật không — hay giật cục, nhảy vị trí tức thời (teleport), khớp gập theo cách không thể xảy ra?
- Tính nhất quán theo thời gian: robot ở khung hình sau có còn đúng là robot ở khung hình trước không (không mất chi tiết, không mọc thêm bộ phận)?
- Độ ổn định camera: lý tưởng là camera hoàn toàn tĩnh; rung lắc hay xoay/dịch góc máy bất thường được tính là lỗi.

## Quy tắc về sự kiện bất thường

Các sự kiện bất thường xảy ra với con robot đều được tính là lỗi, bao gồm nhưng không giới hạn: robot mọc thêm tay hoặc bộ phận; mất tay/gripper; đột ngột di chuyển hoặc nhảy sang vị trí khác; hành vi lạ hoặc phi vật lý; biến dạng, biến mất / xuất hiện đột ngột. Chỉ xét trên con robot, không xét các vật thể khác trong khung hình.

## Yếu tố không xét khi đánh giá

- Tương tác của robot với môi trường (gắp trúng hay hụt, nhiệm vụ thành công hay thất bại).
- Chất lượng nén video hay độ phân giải.

## Tiêu chí chấm điểm (cho từng đoạn)

| Mức | Tiêu chuẩn |
|---|---|
| **Good** | Robot nhìn như video thật: hình dáng nguyên vẹn, màu sắc ổn định, chuyển động mượt và đúng vật lý. Nếu có biến dạng thì rất nhỏ và thoáng qua. Camera gần như tĩnh. |
| **Medium** | Có lỗi nhìn thấy được (biến dạng nhẹ, mờ cục bộ, giật nhẹ, loang màu thoáng qua) nhưng robot vẫn nhận diện rõ và hành động vẫn theo dõi được. Camera có thể giật gián đoạn hoặc xoay/dịch nhẹ. |
| **Bad** | Lỗi rõ rệt, nghiêm trọng và kéo dài trong đoạn: robot biến dạng nặng hoặc tan rã, nhảy vị trí tức thời, chuyển động phi vật lý; khó hoặc không thể theo dõi hành động. Camera có thể xoay/dịch với biên độ lớn. |
