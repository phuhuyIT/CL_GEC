# Kế hoạch huấn luyện BARTpho & ViT5 với Contrastive Learning

## 1. Mục tiêu dự án

- Nâng cao hiệu quả sửa lỗi ngữ pháp & phong cách viết học thuật tiếng Việt trên bộ viGEC.
- Giảm hiện tượng *over‑correction* trong miền mật độ lỗi thấp bằng Contrastive Learning (CL) như mô tả trong bài báo **“Grammatical Error Correction with Contrastive Learning in Low‑Error‑Density Domains”**.

## 2. Mô tả tổng quan pipeline

1. **Chuẩn bị dữ liệu**
   - Tải tập `phuhuy-se1/viGEC`; làm sạch, chuẩn hoá UTF‑8 NFC.
2. **Tiền xử lý & Tokeniser**
   - Giới hạn `max_length` 384 token.
3. **Fine‑tune cơ sở (CE) và tuning hyperparameter để tìm model best**
   - Huấn luyện 5‑10 epoch với Cross‑Entropy + label‑smoothing 0.1.
   - Tuning hyperparameter với optuna để tìm model best
   - Quy trình thực tế gợi ý
     1. **Coarse search** baseline → chọn LR, smoothing, batch “tốt” nhất sau \~30 trial.
     2. **Fix** các siêu tham số này, *tách* branch → áp CL 3-5 epoch.
     3. **Fine search** `λ, γ, k` trên CL checkpoint (mỗi trial chỉ 1-2 epoch) để chọn cấu hình hợp lý.
     4. **Early stop** theo F0.5-val & IE/OE. Đừng dùng loss để dừng vì CL có hai thành phần loss không đồng nhất.
   - Kiểm tra F0.5 trên tập validation → mốc nền.
4. **Sinh negative samples**
   - Beam size = 5, lấy top‑3 kết quả ≠ gold.
   - Thêm self‑pair ⟨s, s⟩ cho câu chứa lỗi.
5. **Huấn luyện Contrastive Learning**
   - Hàm mất mát: `L = L_CE + λ·L_CL` (λ = 1.0, γ = 0.25).
   - 3‑5 epoch, R‑Drop + fp16.
6. **Suy diễn**
   - Dùng Contrastive Search (beam = 1, p = 0.7).
7. **Đánh giá & Phân tích**
   - BLEU, F0.5, Precision, Recall.
8. **Lặp lại** → tinh chỉnh k, γ, λ.

