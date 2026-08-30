Bạn sẽ đánh giá sự đồng bộ và nhất quán giữa 2 góc nhìn của video tạo sinh quay cảnh robot thực hiện hành động. Ngoài sự nhất quán về mặt không gian, độ ổn định của góc quay camera cũng là yếu tố đánh giá (trường hợp lý tưởng là cả 2 góc quay đều tĩnh, không bị rung lắc hay xoay dịch góc nhìn).

**Thang điểm:** Good - Medium - Bad

## Nguyên tắc chung

### Trọng tâm đánh giá

Tính nhất quán về chuyển động, hình dáng, màu sắc, và vị trí tương quan của vật thể/robot trong vùng không gian giao nhau mà cả 2 góc nhìn đều quan sát được; cùng với độ ổn định của 2 góc nhìn (lý tưởng là các camera hoàn toàn tĩnh).

### Quy tắc bất thường đồng bộ

Đánh giá độ đồng bộ **KHÔNG** xét đến ngữ nghĩa hay tính thực tế của các sự kiện bất thường (bao gồm nhưng không giới hạn: hành vi lạ, phi vật lí; vật thể  biến dạng, biến mất / xuất hiện, tự di chuyển...). Miễn là các hiện tượng bất thường đó xuất hiện giống hệt nhau và đồng thời ở cả 2 góc nhìn thì vẫn được coi là nhất quán.

### Yếu tố không xét trong quá trình đánh giá

Mức độ hoàn thành/tuân thủ tác vụ, tính thẩm mỹ, độ chuẩn xác của hành động robot.

## Tiêu chí chấm điểm

| Mức điểm | Tiêu chuẩn Đồng bộ & Độ ổn định Camera | Điều kiện Ngưỡng (Diện tích & Thời lượng)* |
|----------|----------------------------------------|---------------------------------------------|
| **Good** | • Chuyển động, hình dạng, màu sắc và vị trí tương quan giữa các view gần như tương đồng.<br>• Camera gần như tĩnh. | Vùng bị lỗi chỉ chiếm < 10% diện tích khung hình và xuất hiện < 10% thời lượng video. |
| **Medium** | • Xuất hiện các sự bất đồng giữa 2 view (ví dụ: sai khác nhỏ về chuyển động, hình dáng, vị trí tương quan, ...) nhưng vẫn có thể  kết luận robot ở hai góc nhìn đang thực hiện cùng một hành động.<br>• Camera có thể bị giật gián đoạn, xoay/dịch góc máy nhỏ. | Vùng bị lỗi chiếm ≤ 30% diện tích khung hình và xuất hiện ≤ 30% thời lượng video. |
| **Bad** | • Các sự bất đồng bộ xuất hiện rõ rệt, nghiêm trọng và kéo dài, dễ thấy robot ở hai góc nhìn đang thực hiện hành động khác nhau.<br>• Camera có thể bị xoay/dịch với biên độ lớn. | Vùng bị lỗi chiếm > 30% diện tích khung hình hoặc kéo dài > 30% thời lượng video. |

\* *Lưu ý: Các logic kết hợp về mặt định lượng % diện tích khung hình và % thời lượng video mang tính chất tham khảo.*
