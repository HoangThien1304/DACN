# MedAI Diagnostic Portal

Dự án demo chẩn đoán hình ảnh ngực bằng Python + FastAPI + SQLite, tập trung vào quy trình làm việc thực tế của một hệ thống y tế hỗ trợ bác sĩ: quản lý bệnh nhân, upload ảnh X-quang, phân tích hình ảnh bằng mô hình AI, hiển thị nhãn bệnh, và xuất báo cáo Word.

> Lưu ý quan trọng: Đây là phiên bản demo, không phải thiết bị y tế, không phải mô hình đã được kiểm định lâm sàng, và không nên dùng để chẩn đoán thay cho bác sĩ.

## 1. Mục tiêu dự án

- Quản lý hồ sơ bệnh nhân trong cơ sở dữ liệu SQLite
- Upload 2 ảnh của cùng một bệnh nhân: frontal và lateral
- Gọi mô hình AI để dự đoán các nhãn bất thường trên X-quang ngực
- Chuyển kết quả sang tiếng Việt để người bệnh dễ hiểu
- Cho phép xem lịch sử phiên chụp và xuất báo cáo Word
- Mô phỏng đúng quy trình một hệ thống báo cáo bệnh viện, nhưng ở mức demo nội bộ

## 2. Kiến trúc tổng thể

```text
DACN/
├─ backend/              # API, database, mô hình AI, training
│  ├─ main.py            # Endpoint FastAPI và xử lý nghiệp vụ chính
│  ├─ database.py        # CRUD SQLite cho bệnh nhân và các phiên chụp
│  ├─ ai_model.py        # Mô hình AI demo / checkpoint loader / dịch nhãn
│  ├─ train.py           # Script train baseline mô hình Pure Transformer
│  ├─ requirements.txt   # Dependencies chạy app demo
│  └─ requirements-ai.txt # Dependencies cho phần AI / PyTorch
├─ frontend/             # Giao diện đơn trang (HTML/CSS/JS)
│  └─ index.html         # UI chính cho bệnh nhân và đánh giá ảnh
├─ data/
│  ├─ hospital_ehr.db    # SQLite database demo
│  ├─ uploads/           # Ảnh được upload từ frontend
│  ├─ checkpoints/       # Chứa checkpoint model nếu có
│  └─ datasets/          # Dữ liệu training (nếu triển khai mô hình thật)
├─ README.md             # Tài liệu hướng dẫn dự án
└─ .venv/                # Môi trường ảo Python (nếu đã tạo)
```

## 3. Tính năng chính

### 3.1 Quản lý bệnh nhân
- Thêm bệnh nhân mới
- Hiển thị danh sách bệnh nhân đã có
- Lưu mã bệnh nhân, tên, ngày sinh, giới tính, số điện thoại

### 3.2 Quản lý phiên chụp hình ảnh
- Upload ảnh frontal và lateral
- Kiểm tra định dạng ảnh (PNG/JPG/WebP)
- Lưu ảnh vào thư mục data/uploads
- Tạo bản ghi phiên chụp trong database

### 3.3 Phân tích bằng AI
- Nếu có checkpoint hợp lệ, mô hình sẽ dự đoán các nhãn bệnh
- Nếu không có checkpoint hoặc thiếu dependency AI, API trả về lỗi 503 rõ ràng và không gán dự đoán sai
- Kết quả được thể hiện theo 14 nhãn chuẩn, rồi chuyển sang tiếng Việt

### 3.4 Báo cáo và xuất Word
- Tạo report text bằng findings và impression
- Xuất file .docx cho bác sĩ hoặc người dùng

## 4. Triển khai và chạy dự án

### Bước 1: Tạo môi trường ảo

```powershell
cd DACN
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Bước 2: Cài đặt dependency nền tảng

```powershell
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

### Bước 3: Khởi động server

```powershell
uvicorn backend.main:app --reload
```

### Bước 4: Mở ứng dụng

Truy cập địa chỉ:

```text
http://127.0.0.1:8000
```

- Giao diện HTML nằm ở frontend/index.html
- FastAPI sẽ serve page gốc ở URL `/`
- Các API backend nằm ở `/api/...`

## 5. Chạy với mô hình AI thực

Nếu bạn muốn bật phần AI phân tích hình ảnh thật, cài thêm dependency dành cho PyTorch:

```powershell
pip install -r backend/requirements-ai.txt
```

Sau đó cần có checkpoint huấn luyện có tên mặc định:

```text
data/checkpoints/medai.pt
```

Nếu file checkpoint không tồn tại, ứng dụng vẫn chạy nhưng phần phân tích ảnh sẽ trả về lỗi thay vì bịa ra kết quả.

## 6. Dữ liệu và checkpoint

### Cấu trúc dữ liệu
- SQLite: `data/hospital_ehr.db`
- Ảnh upload: `data/uploads/`
- Dataset train: `data/datasets/`
- Checkpoint: `data/checkpoints/medai.pt`

### Định dạng dataset cho train

Bản baseline này có thể huấn luyện trên CSV chứa các cột:

```text
frontal,lateral,labels,findings,impression
```

Trong đó:
- `frontal`: đường dẫn ảnh tư thế thẳng
- `lateral`: đường dẫn ảnh tư thế nghiêng
- `labels`: JSON chứa 14 nhãn bệnh với giá trị 0/1
- `findings`: mô tả phát hiện bệnh lý
- `impression`: đánh giá tổng kết

## 7. Giải thích kiến trúc kỹ thuật

### backend/main.py
- Quản lý route của FastAPI
- Kiểm tra xác thực request và file upload
- Gọi `database.create_exam()` để lưu kết quả
- Gọi `CheckpointInference.predict()` để lấy nhãn bệnh
- Xuất báo cáo Word bằng `python-docx`

### backend/database.py
- Tạo bảng `patients` và `exams`
- Thêm và đọc bệnh nhân
- Lưu `labels_json` dưới dạng JSON trong SQLite
- Trả về dữ liệu sẵn sàng cho frontend

### backend/ai_model.py
- Định nghĩa danh sách 14 nhãn bệnh chuẩn
- Map từ tên tiếng Anh sang tiếng Việt
- Tính `score` và mức độ cảnh báo: Cao, Trung bình, Thấp
- Nạp checkpoint nếu có, hoặc báo lỗi nếu thiếu mô hình

### frontend/index.html
- Giao diện người dùng theo kiểu dashboard bệnh viện
- Hiển thị bệnh nhân, lịch sử phiên chụp, nhãn dự đoán
- Gọi API backend để thêm bệnh nhân và phân tích hình ảnh
- Render các chỉ số, báo cáo và mức độ nguy cơ

## 8. Giới hạn và cảnh báo

- Repo không đi kèm dataset nghiên cứu thực tế và không có checkpoint đã được huấn luyện sẵn
- Đây là demo để minh họa luồng tổng thể, không phải mô hình lâm sàng đã chứng minh hiệu quả
- Kết quả dự đoán không nên dùng để chẩn đoán hoặc điều trị
- Nếu triển khai mô hình thật, cần đánh giá độc lập trên dữ liệu riêng và kiểm tra leakage bệnh nhân

## 9. Tài liệu tham khảo

Dự án dựa trên ý tưởng của bài báo nghiên cứu:

- Wang et al., “Automated Radiographic Report Generation Purely on Transformer: A Multicriteria Supervised Approach”, IEEE TMI 2022
- DOI: https://doi.org/10.1109/TMI.2022.3171661

Nội dung này chỉ mang tính nền tảng về kiến trúc và mô hình, không phải bản triển khai nghiên cứu đầy đủ hoặc mô hình đã được chứng minh lâm sàng.
