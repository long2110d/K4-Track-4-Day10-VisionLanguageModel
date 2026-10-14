Họ tên: Đỗ Đình Long
`dataset_version`: d10-21e4070301

> **Bản nháp (15/10/2026).** Đã điền phần đã chạy xong: dữ liệu, E0, mask loss, meta huấn luyện E1/E2 và số tham số E3.
> Các ô **⏳** chờ `report/evidence/e1.json`–`e3.json`, `e4_*.json` và `report/extra/*.json` từ Colab.

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | – | – | 0.063 | [0.043, 0.089] | 1.28 | 0.173 | 0.010 | 0.116 |
| E1 | LoRA LLM | 33,030,144 | 5.87 | 46.3 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| E2 | + projector | 87,609,344 | 9.20 | 45.8 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| E3 | + LoRA ViT | 93,900,800 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| E4 | rank r = 4 / r = 64 (alpha = 2r) trên setup tốt nhất | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

Cấu hình chung E1–E3: 300 bước tối ưu, batch 2 × tích luỹ 4 (8 ảnh/bước), lr 2e-4 (warmup 5%, cosine), seed 42, 1422 ảnh train, GPU Colab T4.
E0 chi tiết: n = 311, P = 0.075, R = 0.055, mean IoU (cặp khớp) = 0.583.

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?

⏳ Số tham số từng khối đo bằng `count_blocks` nằm trong `report/extra/model_blocks.json` (chưa lấy về). Đối chiếu tạm theo GUIDE: LLM + embedding ≈ 4,0B, ViT ≈ 415M, projector ≈ 109M (`visual.merger` ≈ 27M + `visual.deepstack_merger_list` ≈ 82M).

Khối được cập nhật và số tham số huấn luyện **đo được**, kiểm lại bằng tay:

| Setup | Khối cập nhật | Module LoRA | Tham số huấn luyện |
|---|---|---|---|
| E1 | LoRA trên 7 phép chiếu (q, k, v, o, gate, up, down) của 36 lớp LLM | 252 | 33,030,144 |
| E2 | E1 + projector (merger + 3 deepstack merger) huấn luyện đầy đủ | 252 | 87,609,344 (projector 54,579,200) |
| E3 | E2 + LoRA trên qkv, proj, linear_fc1, linear_fc2 của 24 khối ViT | 348 | 93,900,800 |

- **E1:** LoRA r = 16 thêm r·(d_in + d_out) tham số cho mỗi module. Một lớp LLM (hidden 2560, q 4096, k/v 1024, MLP 9728) có tổng (d_in + d_out) = 6656 + 3584 + 3584 + 6656 + 3 × 12288 = 57,344. Vậy 16 × 57,344 × 36 = **33,030,144**, khớp chính xác.
- **E3 − E2 = 6,291,456:** một khối ViT (hidden 1024, MLP 4096) có tổng (d_in + d_out) = 4096 + 2048 + 5120 + 5120 = 16,384. Vậy 16 × 16,384 × 24 = **6,291,456**, khớp chính xác.
- **E2 − E1 = 54,579,200**, chỉ bằng một nửa con số ≈ 109M của GUIDE. Phân tích ở Câu 5.

ViT và LLM gốc luôn đóng băng; E1 chỉ học adapter của LLM, E2 học thêm toàn bộ projector, E3 học thêm adapter của ViT.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?

`parse_valid_rate` = **0.733**: 228/311 ảnh cho JSON đọc được, 83 ảnh không đọc được. Kể cả khi JSON hợp lệ, model gốc vẫn kém: P = 0.075, R = 0.055, F1 = 0.063.

1. **Bịa box và gán sai lớp, gần như luôn là `manhole`.** FP/ảnh = 1.28, cao hơn nhiều so với số box đúng trung bình (583/311 ≈ 1.87 box/ảnh mà recall chỉ 5.5%). Cả 5 output thô in ra đều chỉ có nhãn `manhole` hoặc `[]`, dù tập test có 299 box crack. Crack gần như không được nhận ra: F1 crack = 0.010.
2. **Lặp box trùng hệt nhau:** cùng một box được sinh hai lần, nên chắc chắn có một FP. Ví dụ output thô (ảnh test thứ 3):
   ```
   [{"label": "manhole", "bbox_2d": [322, 684, 393, 714]}, {"label": "manhole", "bbox_2d": [322, 684, 393, 714]}]
   ```
   Ở ví dụ thứ 5, model sinh hai box cùng kích thước nằm sát nhau ([262, 632, 317, 660] và [317, 632, 372, 660]), có vẻ "trượt" toạ độ theo mẫu thay vì nhìn ảnh.

⏳ Cần mở `work/e0.jsonl` để phân loại 83 output hỏng (bọc ```` ```json ````, chữ thừa hay bị cắt ở 256 token) trước khi chốt câu này.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?

Ảnh có kích thước 640 × 360. Nếu đem output thang 0–1000 chấm như pixel, toạ độ x bị phóng 1000/640 ≈ 1.56 lần và y bị phóng 1000/360 ≈ 2.78 lần. Box bị kéo xuống góc dưới phải và phình to, gần như không cặp nào còn IoU ≥ 0.5, nên F1 về sát 0 dù JSON vẫn hợp lệ. Trường hợp ngược lại (pixel chấm như 0–1000), box bị co lại về góc trên trái và cũng hỏng như vậy. Thang 0–1000 không phụ thuộc độ phân giải, đúng với cách Qwen3-VL được tiền huấn luyện cho grounding, nên cùng một nhãn dùng được cho mọi kích thước ảnh.

Kiểm chứng:
- **Nhãn:** đổi YOLO → 0–1000 → pixel rồi so với box gốc trên một ảnh train, sai số lớn nhất **0.28 px**, tức chỉ do làm tròn.
- **Chấm:** ⏳ `coord_check` trên setup tốt nhất ghi 4 F1 (đúng thang, 0–1000 chấm như pixel, pixel chấm như 0–1000, pixel chấm như pixel) vào `report/extra/coord_check.json`.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?

Trên 40 ví dụ train, một chuỗi trung bình có **448.6 token**: **220 token ảnh** (ảnh 640 × 360), khoảng 145 token prompt và chat template, và chỉ **83.6 token đáp án**. Vậy phần cần học chỉ chiếm khoảng 19% chuỗi.

- **Token ảnh** (`<|image_pad|>`) là chỗ giữ chỗ cho embedding từ ViT. Bắt LLM "dự đoán" chúng là vô nghĩa: không có nhãn văn bản thật nào ở đó.
- **Prompt** giống hệt nhau ở mọi mẫu. Tính loss trên prompt sẽ dạy model thuộc lòng câu lệnh: loss giảm rất nhanh, trông như học tốt nhưng không liên quan đến việc định vị.
- Nếu tính trên cả chuỗi, khoảng 81% gradient đổ vào các token đó và tín hiệu của JSON đáp án bị pha loãng khoảng 5 lần. Với chỉ 300 bước, model sẽ học chậm phần toạ độ.

Cách kiểm tra: giải mã riêng các token có `labels != -100` của ví dụ đầu tiên. Kết quả đúng bằng JSON đáp án nối với `<|im_end|>`. Như vậy model học cả việc **dừng** sau JSON, đó là lý do `<|im_end|>` được giữ trong phần tính loss.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?

**Vì sao không thể huấn luyện nếu bị lượng tử hoá:** trọng số NF4 được lưu dưới dạng chỉ số 4-bit nén hai số vào một byte (`uint8`), cộng hằng số absmax theo khối. Đây là tra bảng rời rạc, không có gradient liên tục theo từng trọng số, và optimizer không thể cộng một bước cập nhật nhỏ vào. Với khối 4-bit, ta chỉ có thể gắn LoRA lên trên (QLoRA), không thể "mở" toàn bộ trọng số. Vì vậy cấu hình nạp model dùng `llm_int8_skip_modules=["visual", "lm_head"]` để giữ ViT và projector ở fp16.

**Vì sao phải dùng float32:**
1. Với lr 2e-4, mỗi bước cập nhật nhỏ hơn khoảng cách giữa hai số fp16 kề nhau (≈ 9.8e-4 quanh giá trị 1). Cộng vào trọng số fp16 thì bị làm tròn mất, nên projector thực tế không học.
2. `GradScaler` chỉ unscale được gradient fp32. Tôi gặp đúng lỗi này ở E2: `Attempting to unscale FP16 gradients`. Nguyên nhân là projector được nạp ở fp16, và việc ép kiểu về fp32 đặt sai chỗ so với lúc bật gradient checkpointing. Cách sửa: ép mọi tham số đang học về fp32 **sau** khi bật gradient checkpointing và **trước** khi tạo optimizer (commit `9555b9e`).

**Điểm đáng ngờ (cần kiểm tra lại):** số tham số projector đo được là **54,579,200**, trong khi theo kiến trúc phải là ≈ 109M. Phép tính khớp chính xác với giả thuyết các **trọng số linear của merger đang ở dạng `Params4bit`**, tức `numel()` chỉ đếm một nửa vì nén 2 số/byte:
- 4 merger × (4096×4096 + 4096×2560) = 109,051,904 trọng số linear, một nửa là 54,525,952.
- Cộng bias (26,624) và LayerNorm (26,624), không bị lượng tử hoá: **54,579,200**.

Nếu đúng vậy, `llm_int8_skip_modules=["visual"]` đã **không** giữ được merger ở fp16, và việc "huấn luyện" projector ở E2/E3 thực ra đang tác động lên dữ liệu 4-bit đã nén. Điều này khớp với việc loss của E2 luôn cao hơn E1: trung bình 5 lần log cuối là 0.698 so với 0.585. ⏳ Cần xác nhận trên Colab bằng `type(model.visual.merger.linear_fc1.weight)` và `model_blocks.json`.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?

Ảnh được trích từ video dashcam. Các khung hình liên tiếp trong cùng một phiên quay gần như trùng nhau: cùng đoạn đường, cùng ổ gà, chỉ dịch vài pixel. Nếu chia ngẫu nhiên, "anh em sinh đôi" của ảnh test nằm sẵn trong train. Khi đó model chỉ cần nhớ lại chứ không cần tổng quát hoá, F1 test bị thổi phồng và không phản ánh hiệu năng trên đoạn đường mới. Chia theo `session` giữ toàn bộ một phiên quay trong cùng một split. Notebook kiểm tra cả ba cặp split và xác nhận **không có phiên quay nào trùng**.

| Split | Ảnh | Box | pothole (ảnh / box) | crack (ảnh / box) | manhole (ảnh / box) |
|---|---|---|---|---|---|
| train | 1422 | 3515 | 664 / 1074 | 969 / 1750 | 543 / 691 |
| val | 276 | 639 | 55 / 66 | 235 / 470 | 79 / 103 |
| test | 311 | 583 | 76 / 121 | 171 / 299 | 137 / 163 |

Hệ quả của việc chia theo phiên: tỉ lệ lớp lệch giữa các split. Pothole chiếm 31% box train nhưng chỉ 10% box val, còn manhole chiếm 28% box test so với 20% ở train.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?

⏳ Chờ F1 và CI95 của E1–E3. Số đã có:
- **VRAM:** E2 cần 9.20 GB, so với 5.87 GB của E1 (**+3.33 GB**, +57%). Phần tăng đến từ 54.6M tham số projector fp32 cùng gradient và hai trạng thái Adam (mỗi tham số khoảng 16 byte), và từ activation của ViT phải giữ lại để lan truyền ngược vào projector.
- **Thời gian:** gần như không đổi (46.3 so với 45.8 phút cho 300 bước). Phần lớn thời gian nằm ở forward/backward qua LLM 4-bit.
- **Loss:** E2 không giảm thấp bằng E1 (0.698 so với 0.585, trung bình 5 lần log cuối). Xem nghi vấn ở Câu 5.
- E3 thêm 6.3M tham số LoRA trên ViT; ⏳ VRAM và thời gian của E3.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?

CI95 bootstrap theo ảnh của E0 là [0.043, 0.089]: rộng 0.046 điểm F1, tức khoảng ±37% quanh giá trị 0.063. Với 311 ảnh (583 box), việc vài ảnh khó lọt vào hay rơi khỏi mẫu bootstrap cũng làm F1 dao động đáng kể. Riêng từng lớp còn kém tin cậy hơn: test chỉ có 121 box pothole trong 76 ảnh.

Hệ quả: hai setup chỉ được coi là khác nhau khi CI95 **không chồng nhau**. Chênh lệch nhỏ hơn độ rộng CI (cỡ vài phần trăm F1) không đủ để kết luận. ⏳ Áp dụng cho các cặp E1/E2/E3 khi có số.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.

Ở E0, **crack kém nhất** (F1 0.010), sau đó là manhole (0.116) và pothole (0.173). Crack kém dù có nhiều mẫu nhất (1750 box train, 299 box test), vì các lý do sau:
- **Hình dạng:** vết nứt mảnh và dài, box của nó rất dẹt hoặc chéo. Chỉ cần lệch vài đơn vị theo chiều hẹp là IoU tụt dưới 0.5.
- **Ranh giới mơ hồ:** một vết nứt dài có thể được gán thành một box lớn hay nhiều box nhỏ (ảnh mẫu train có 4 box crack nối nhau). Vì vậy kể cả khi "thấy" vết nứt, model vẫn khó khớp đúng cách chia box của người gán nhãn.
- **Ngữ nghĩa:** vết nứt dễ lẫn với vạch sơn, mép vá đường và bóng.

⏳ F1 từng lớp của E1–E3 và ảnh `report/figures/errors_<lớp>.jpg`, `<best>_good_bad.jpg`, `e0_examples.jpg`.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?

Biến được đổi là **rank LoRA**: r = 4 và r = 64 (alpha = 2r), trên setup tốt nhất của E1–E3. Mọi thứ khác giữ nguyên: lr, số bước, seed, dữ liệu. Với E1, r = 4 cho khoảng 8.3M tham số và r = 64 cho khoảng 132M, so với 33M ở r = 16. ⏳ Chờ `e4_rank_4.json`, `e4_rank_64.json` để so F1 và CI95.

## Nhìn lại

- **Số tham số projector bằng một nửa kỳ vọng** (Câu 5). Công cụ chấm không bắt được lỗi này vì E2 vẫn chạy, vẫn ra loss hữu hạn và vẫn lưu được `projector.pt` (24 tensor). Nếu merger thực sự bị lượng tử hoá, so sánh E1/E2/E3 sẽ không phản ánh tác dụng thật của việc mở projector.
- **Lỗi fp16 với GradScaler ở E2** chỉ lộ ra khi mở projector, vì E1 toàn là tham số LoRA fp32 nên chạy bình thường.
- **Colab bị ngắt giữa chừng** làm mất kết quả trong `/content`. Tôi đã chuyển `work/` và `report/` lên Google Drive và thêm checkpoint trạng thái optimizer mỗi 50 bước để chạy tiếp được.
- E0 lặp lại cùng một box hai lần. Bộ chấm tính đúng là FP, nhưng `parse_valid_rate` vẫn coi output đó là hợp lệ, nên chỉ số này không phản ánh lỗi lặp.
