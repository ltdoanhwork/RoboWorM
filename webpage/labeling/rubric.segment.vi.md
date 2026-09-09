# Hướng Dẫn Đánh Giá Chất Lượng Segment Video Robot

## Nhiệm vụ
Bạn sẽ xem các segment video robot do mô hình tạo ra và đánh giá mức độ "hallucination" — các lỗi tạo hình mà video robot thật không bao giờ có. Chất lượng robot bao gồm việc robot có bị biến dạng, mờ nhòe, chảy màu, chuyển động bất thường, hay chuyển động vi phạm vật lý thông thường hay không.

Mỗi video được chia thành các segment ngắn. Đánh giá từng segment theo thang điểm:

**Thang điểm: Good – Medium – Bad**

Trong đó **Good** là mức tốt nhất (ít/không biến dạng — segment trông sạch, giống video thật), **Bad** là mức lỗi nghiêm trọng nhất.

Lưu ý: chấm mỗi đoạn độc lập. Một đoạn có thể bẩn nhưng các đoạn sau đó robot sạch trở lại — khi đó đoạn sau vẫn chấm Good, đừng để đoạn trước ảnh hưởng đoạn sau.

## Những gì cần tập trung
Tập trung vào bản thân robot:

- **Hình dạng:** robot có bị biến dạng, kéo giãn, gãy rời, mất tay/gripper, mờ nhòe, hay tan biến không?
- **Màu sắc / bề mặt:** có hiện tượng chảy màu, nhấp nháy, hay đổi màu bất thường không?
- **Chuyển động:** chuyển động có mượt và hợp lý về mặt vật lý như robot thật, hay giật cục, dịch chuyển tức thời (teleport), hoặc khớp bẻ cong phi vật lý?
- **Tính nhất quán theo thời gian:** robot ở frame sau có còn là robot ở frame trước không (không mất chi tiết, không mọc thêm bộ phận lạ)?
- **Độ ổn định camera:** camera lý tưởng nên đứng yên; rung lắc hoặc xoay/dịch chuyển bất thường tính là lỗi.

## Quy tắc cho các hiện tượng bất thường
Các hiện tượng bất thường xảy ra với robot đều tính là lỗi, bao gồm nhưng không giới hạn: robot mọc thêm tay/bộ phận lạ; mất tay/gripper; đột ngột dịch chuyển hoặc "nhảy" sang vị trí khác; hành vi kỳ lạ hoặc phi vật lý; biến dạng, biến mất/xuất hiện đột ngột. Chỉ xét robot — không xét các vật thể khác trong khung hình.

## Những gì không cần xét
- Tương tác của robot với môi trường (gắp có thành công hay không, task có hoàn thành hay không).
- Chất lượng nén video hoặc độ phân giải.

## Tiêu chí chấm điểm (theo từng segment)

| Mức độ | Tiêu chuẩn |
|---|---|
| **Good** | Segment nét, cánh tay robot chuyển động giật bình thường, quan sát rõ robot đang làm gì. Cánh tay không biến dạng. Robot không bị nhòe, không chảy xệ. |
| **Medium** | Segment tương đối mờ nhưng cánh tay robot không quá biến dạng; hoặc cánh tay robot biến dạng ít. Robot vẫn nhận diện rõ ràng và có thể theo dõi được hành động. |
| **Bad** | Segment mờ, hình ảnh chảy xệ (nhòe/méo nặng); hoặc segment nét nhưng gripper bị biến dạng; hoặc cánh tay robot biến dạng mạnh; hoặc mọc ra thành phần lạ / robot biến thành vật thể khác. Hành động khó hoặc không thể theo dõi. |
