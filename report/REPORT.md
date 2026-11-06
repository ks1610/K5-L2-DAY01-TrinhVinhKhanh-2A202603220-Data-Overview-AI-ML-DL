# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** T4 GPU

**Python / PyTorch / Ultralytics:** 
- Python: 3.13.16
- PyTorch: 2.11.0+cu130
- Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Chương trình cũ vẫn được giữ nguyên, để tiện cho quan sát kết quả, học viên đã duplicate các chương trình `phát hiện vật thể `và `phân đoạn đối tượng`. 
Cụ thể:
 - `phát hiện vật thể `: nhân đôi để thay đổi ngưỡng thành 0.15 và 0.55 để quan sát kết quả. Đồng thời đọc output được in trên collab với sample_id = kitchen.
 - `phân đoạn đối tượng`: nhân đôi để điều chỉnh, đọc output được in trên collab với sample_id = kitchen. 
 <br>Chi tiết: `https://colab.research.google.com/drive/1ZXkYJ5u3Hoq3ZqIJ0__sB_-XCPV0MtrG?usp=sharing`
> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468; `class_name`=cab ;`rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
```
Record nói về cả bức ảnh, cụ thể là lớp có điểm số cao nhất trong ảnh. 
 Nguyên nhân:
 1, Đầu vào của mô hình là một bức ảnh hoàn chỉnh thông qua source=image.
 2, Mô hình duyệt toàn bộ bức ảnh:
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
 3, Trong chương trình không có code liên quan đến tọa độ hay id của một vật thể riêng biệt.
 4, Kết quả trả về 5 lớp có điềm số cao nhất top_scores = probs.top5conf.tolist().
```
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
```
class list có được từ thông tin từ cấu hình nhãn gắn với taxonomy và cấu hình mô của mô hình huấn luyện.
Model lấy thông tin từ các nhãn được huấn luyện sẵn, ImageNet-1K gồm 1.000 lớp, mỗi lớp tương ứng 1 WordNet Synset và các lớp được định danh bằng Synset ID [1]. Nên mô hình không thể tự tạo nhãn.
[1]: https://docs.ultralytics.com/datasets/classify/imagenet
```
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
```
Việc lưu trữ đồng thời class_id, class_name và taxonomy_name giúp đảm bảo tính nhất quán và khả năng truy xuất của dữ liệu phân loại. Trong đó: 
1, class_id đóng vai trò định danh lớp, 
2, class_name cung cấp thông tin mô tả dễ hiểu cho người sử dụng, 
3, taxonomy_name xác định hệ thống phân loại mà nhãn đó thuộc về. 

Theo tài liệu Ultralytics, kết quả phân loại cung cấp ánh xạ giữa class index và class name thông qua thuộc tính result.names [2]. 
Bên cạnh đó, ImageNet-1K tổ chức 1.000 lớp theo hệ thống phân cấp WordNet, trong đó mỗi lớp tương ứng với một Synset [1]. 

Vì vậy, việc lưu đầy đủ ba thuộc tính giúp dữ liệu dễ dàng được kiểm tra, đối chiếu và sử dụng trong các quy trình đánh giá mô hình sau này.
[2]: https://docs.ultralytics.com/modes/predict
[1]: https://docs.ultralytics.com/datasets/classify/imagenet
```
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
```
Guidline cần quy định rõ tiêu chí lựa chọn một nhãn đại diện duy nhất cho toàn bộ bức ảnh, cần gán nhãn ảnh theo nhãn cho score cao nhất hay một vật thể nhất định.
```
6. Vì sao không được lấy `score` của model làm ground truth?
```
Không được sử dụng score của mô hình làm Ground Truth vì hai thành phần này có bản chất và chức năng hoàn toàn khác nhau.

Ground Truth là nhãn thực tế đã được xác định độc lập thông qua dữ liệu có sẵn hoặc quá trình gán nhãn và kiểm duyệt. Trong khi đó, score là giá trị do mô hình tạo ra trong quá trình inference, thể hiện mức độ mà mô hình gán cho một lớp dự đoán.

Theo tài liệu Evaluating ML Models của Amazon Web Services, quá trình đánh giá mô hình cần sử dụng dữ liệu đã có nhãn Ground Truth và so sánh kết quả dự đoán với nhãn thực tế tương ứng [3]. 

Bên cạnh đó, nghiên cứu Responsible Model Deployment via Model-Agnostic Uncertainty Learning chỉ ra rằng mô hình có thể đưa ra dự đoán sai ngay cả khi confidence score ở mức cao. Điều này cho thấy score không phải bằng chứng xác nhận tính chính xác của nhãn dự đoán [4].

[3]: https://docs.aws.amazon.com/machine-learning/latest/dg/evaluating_models.html
[4]: https://link.springer.com/article/10.1007/s10994-022-06248-y
```
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person; `score`=0.912625 ; `bbox_xyxy`=[385.33,69.24,498.92,348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
```
Góc trên bên trái ở vị trí tọa độ (385.33, 69.24), Góc dưới bên phảiở vị trí có tọa độ (498.92, 348.92). Chiều rộng là 113.58 pixel. Chiều cao là 279.68 pixel.
```
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
```
Sample: kitchen;
Đặt ngưỡng yolo11n.pt: 0.15; số vật thể phát hiện: 20
Đặt ngưỡng yolo11n.pt: 0.55; số vật thể phát hiện: 6
```
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
```
Reviewer phải dành nhiều thời gian hơn để kiểm tra, phân loại và loại bỏ các prediction sai.
```
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
```
Quy tắc: 
1, Mỗi vật thể cần được bao quanh bởi một bounding box hình chữ nhật nhỏ nhất có thể, đảm bảo chứa toàn bộ phần vật thể nhìn thấy trong ảnh. 
2, Bốn cạnh của bounding box phải sát với các mép ngoài của vật thể, không bao gồm quá nhiều vùng nền xung quanh. 
3, Nếu vật thể bị che khuất một phần, chỉ gán nhãn theo phần thực sự nhìn thấy, không tự suy đoán hoặc mở rộng hộp để bao phủ phần bị che khuất.
```
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
```
Trong ảnh bị che khuất là person 0.61 ở phía góc bên trái của ảnh chỉ nhìn thấy một phần cánh tay và bàn tay.
Quyết định: Escalation – Hỏi Lab Coach. Lý do:
1, Không thể xác định đầy đủ hình dạng và phạm vi của người vì phần lớn cơ thể nằm ngoài khung hình.
2, Chưa rõ quy định gán nhãn có cho phép tạo bounding box cho một đối tượng chỉ còn một phần nhỏ xuất hiện trong ảnh hay không.
3, Nếu tự đưa ra quy tắc mà chưa được xác nhận, những người gán nhãn khác có thể xử lý các trường hợp tương tự không thống nhất. 
```
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[446.0,70.0],[445.0,71.0],[444.0,71.0],[443.0,72.0],[442.0,72.0],
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
```
Polygon cung cấp thông tin chi tiết hơn về đường biên và hình dạng thực tế của vật thể so với box.
```
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
```
instance_id là mã định danh riêng cho từng đối tượng được phát hiện trong một ảnh. Nó giúp phân biệt các đối tượng khác nhau, kể cả khi chúng thuộc cùng một class.

- class_id: Xác định đối tượng thuộc lớp nào.
- instance_id: Phân biệt từng đối tượng riêng biệt trong kết quả phát hiện. 
- tracking_id: Theo dõi cùng một đối tượng qua nhiều khung hình (frame) trong video.
```
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
```
1, Người gán nhãn phải vẽ đường biên Polygon bám sát đường viền thực tế của vật thể, chỉ bao phủ những pixel thuộc về vật thể đó. 
2, Không đưa vùng nền hoặc các vật thể khác vào mask. Đối với những vật thể có hình dạng không đều, cần đặt đủ các điểm tọa độ tại những vị trí đường biên thay đổi để mô tả chính xác hình dạng. 
3, Khi vật thể bị che khuất, chỉ gán nhãn phần có thể quan sát được, không tự suy đoán hoặc vẽ thêm phần bị che khuất.
```
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
```
Ở vị trí ở góc bên trái ảnh, các chiếc bát bowl 0.53 và bowl 0.74 xếp chồng lên nhau. 

Lựa chọn: Escalation (chuyển cấp để xác nhận). Lý do:
- Không thể xác định chắc chắn đường biên thực tế tại vùng hai chiếc bát tiếp xúc.
- Nếu tự suy đoán đường biên, người gán nhãn có thể tạo ra mask sai lệch.
- Cần người phụ trách hoặc chuyên gia xác nhận cách xử lý thống nhất để tránh mỗi người gán nhãn theo một cách khác nhau.
```
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Ảnh có các nhãn, lớp đại diện cho nội dung chính của ảnh. | Khung cảnh chứa nhiều đối tượng: Ảnh hiển thị một tuyến đường đông đúc với nhiều xe bus, xe van, ô tô con. Mô hình bị mơ hồ và dự đoán cab (score: 0.5109), minibus (score: 0.1643), police_van, streetcar cho toàn bộ bức ảnh. | Gán nhãn cho toàn bộ ảnh dựa trên guideline (ví dụ: chọn lớp chủ đạo duy nhất như traffic_jam/street_scene hoặc gán danh sách multi-label gồm cả bus, car, van). | Kiểm tra nhãn được gán có đúng với quy định trong guideline đối với ảnh đa đối tượng hay không, đảm bảo không áp đặt nhãn đơn (single-label) không phù hợp cho cảnh phức tạp. |
| Phát hiện vật thể | Bounding box kèm nhãn lớp tương ứng cho từng đối tượng riêng biệt. | 1. Nhầm lẫn/Bị che khuất: Bàn tay chỉ ở mép trái bị đoán nhầm là person (score: 0.61, bbox: [0.12, 263.18, 61.55, 310.78]). <br><br>2. Bỏ sót (False Negative): Nhiều xoong nồi treo, dụng cụ làm bếp, đồ ăn trên bàn bị bỏ sót (JSON chỉ trả về 11 vật thể). <br><br> 3. Bounding box thiếu chính xác: Nhãn oven 0.69 trùm lên cả khu vực bếp/bàn.    | Vẽ khung bounding box ôm sát từng đối tượng; gán nhãn chính xác (phân biệt rõ giữa person đầy đủ với các bộ phận cơ thể bị che khuất như hand nếu guideline quy định); gán đầy đủ nhãn cho các vật thể nhỏ/bị che một phần. | Kiểm tra độ chính xác của bounding box (không vẽ thừa/thiếu pixel), xác định các trường hợp bị nhận diện sai nhãn (bàn tay -> person, bếp -> oven), và đối chiếu với guideline về quy tắc vẽ vật thể bị che khuất . |
| Instance segmentation | Chuỗi tọa độ đa giác polygon (polygon_xy: [[x1, y1], [x2, y2], ...]) hoặc pixel mask chi tiết kèm theo tên lớp và ID định danh riêng cho từng đối tượng. | 1. Ranh giới mask tràn/thiếu chính xác: Mask của dining table 0.58 phủ đè lên cả các dụng cụ trên bàn. Mask của person 0.90 (JSON kitchen-001, polygon_point_count: 348) có viền còn lem nhem quanh mái tóc. <br><br> 2. Bị đè lấn ranh giới: Vùng bàn bị đè lấn bởi nhiều mask khác nhau (dining table 0.58, person 0.43, bowl 0.74, bowl 0.70, bowl 0.55) <br><br> 3. Nhiều xoong chảo treo bị bỏ sót.| Vẽ polygon hoặc mask tô kín chuẩn xác ranh giới từng điểm ảnh cho từng thể hiện đối tượng; tách biệt các vùng bị che khuất/đè lên nhau; gán đúng class_name và phân biệt rõ vật thể với nền. | Kiểm tra độ ôm sát ranh giới pixel, phát hiện ranh giới mask bị lem/tràn qua vật thể khác, xử lý các vùng mask bị cắt/đè lấn không hợp lý và kiểm tra nhãn nhầm/bỏ sót. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
```
Chỉ sử dụng dữ liệu được cấp phép cho mục đích bài lab, không tự ý sao chép, chia sẻ, phát tán hoặc tải dữ liệu lên các nền tảng bên ngoài khi chưa được cho phép.
```
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
```
Tôi sẽ dừng ngay việc xử lý dữ liệu đó, không tự ý sử dụng hoặc chia sẻ, đồng thời báo cáo cho  Lab Coach hoặc người phụ trách quản lý dữ liệu của bài lab để được hướng dẫn xử lý.
```

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [v] `REPORT.md` (file này đã điền)
- [v] `day1_lab_outputs/classification_predictions.json`
- [v] `day1_lab_outputs/detection_predictions.json`
- [v] `day1_lab_outputs/segmentation_predictions.json`
- [v] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [v] `day1_lab_outputs/visuals/classification_top5.png`
- [v] `day1_lab_outputs/visuals/detection_predictions.png`
- [v] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [v] Ô validation cuối notebook báo `PASS`
- [v] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [v] Đã nộp **link repository** trên LMS
