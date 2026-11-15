# Sử dụng AdapterForge

AdapterForge là dự án fine-tuning LoRA thực sự trên mô hình SmolLM2-135M-Instruct. Đầu vào là yêu cầu hỗ trợ bằng tiếng Anh; đầu ra gồm `intent`, `urgency`, `order_id`. Đây là bản thử nghiệm trên dữ liệu tổng hợp, chưa phải hệ thống hỗ trợ khách hàng thực tế.

## Mở dự án

```powershell
cd "C:\Users\LOQ\Documents\Codex\2026-11-06\cj\outputs\AdapterForge"
```

Môi trường `.venv`, mô hình gốc và adapter được tạo cục bộ. Không cần nhập API key. Không commit thư mục `.venv`, `private` hoặc `reports`.

## Kiểm tra mã

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

Các bài kiểm tra giao diện/API dùng mô hình giả để kiểm tra hành vi chương trình; chúng không đo độ chính xác của mô hình. Kết quả mô hình thật nằm trong báo cáo đánh giá riêng.

Sau khi có adapter đã huấn luyện, có thể kiểm tra tích hợp bằng mô hình thật:

```powershell
.\.venv\Scripts\python.exe -X utf8 verify_local.py
```

Lệnh này tải adapter vào API và giao diện, thực hiện một yêu cầu rồi lưu `reports/local-smoke.json`. Đây là kiểm tra khả năng chạy và tính nhất quán UI/API, không phải đánh giá độ chính xác trên tập test.

## Mở giao diện

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8517
```

Mở `http://127.0.0.1:8517`. Chọn adapter đã huấn luyện, nhập yêu cầu rồi bấm **Run local model**. Ví dụ tự thử:

```text
Please urgently track my shipment for ORD-3197.
```

Theo quy tắc tác vụ, kết quả mong đợi là `delivery`, `urgent`, `ORD-3197`. Mô hình vẫn có thể dự đoán sai. So sánh với lựa chọn `base` để quan sát sự khác biệt. JSON hợp lệ chỉ chứng minh đúng định dạng, không chứng minh nội dung chính xác.

## Đánh giá lại adapter đã lưu

```powershell
.\.venv\Scripts\python.exe -X utf8 evaluate.py --adapter private/run-v1/best --device auto --output reports/my-comparison.json
Get-Content reports/my-comparison.json -Encoding utf8
```

Lệnh này chạy mô hình thật trên 64 mẫu test và 8 tình huống thử thách. Báo cáo giữ nguyên đầu ra thô và các lỗi, so sánh adapter với base và bộ luật. Không chỉnh mô hình theo kết quả test rồi gọi đó là đánh giá độc lập.

## Chạy API cục bộ

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
```

Mở `http://127.0.0.1:8000/docs`, chọn `POST /extract`, bấm **Try it out** và nhập:

```json
{"text":"Please urgently track my shipment for ORD-3197."}
```

Bấm **Execute**. API này không yêu cầu đăng nhập khi chạy cục bộ. Mã 422 nghĩa là đầu vào không hợp lệ/quá dài theo giới hạn token; mã 503 nghĩa là mô hình chưa sẵn sàng hoặc đang bận. Không mở cổng ra Internet với cấu hình demo này.

## Huấn luyện một phiên bản mới (không bắt buộc)

```powershell
.\.venv\Scripts\python.exe -X utf8 train.py --device cuda --output private/run-v2
.\.venv\Scripts\python.exe -X utf8 evaluate.py --adapter private/run-v2/best --output reports/run-v2.json
```

Thư mục đầu ra phải chưa tồn tại. Trong UI có thể chọn phiên bản mới; API mặc định vẫn trỏ tới `run-v1`. Nếu không có CUDA, dùng `--device cpu`, nhưng thời gian sẽ lâu hơn. Đừng chạy đồng thời huấn luyện và các cửa sổ suy luận khi GPU thiếu bộ nhớ.

## Đọc kết quả đúng phạm vi

- `training.json`: cấu hình, loss từng epoch, checkpoint được chọn bằng validation loss và thông tin GPU.
- `adapter_model.safetensors`: trọng số LoRA đã học, cần mô hình gốc để suy luận.
- `comparison.json`: điểm trên bộ dữ liệu tổng hợp cố định, không phải cam kết độ chính xác trên dữ liệu khách hàng.
- `schema_valid_rate`: đúng cấu trúc JSON; `exact_match`: cả ba trường khớp nhãn.

Dữ liệu hiện tại bằng tiếng Anh. Không sử dụng thông tin khách hàng thật trong demo và không tuyên bố đây là mô hình đa ngôn ngữ đã kiểm chứng.
