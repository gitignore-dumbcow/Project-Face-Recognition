# Hệ Thống Nhận Diện Khuôn Mặt

## Giới Thiệu
Đây là một ứng dụng nhận diện khuôn mặt sử dụng Python và TensorFlow, với giao diện đồ họa được xây dựng bằng Tkinter. Ứng dụng cho phép người dùng chụp ảnh khuôn mặt, huấn luyện mô hình và thực hiện nhận diện khuôn mặt trong thời gian thực.

## Cấu Trúc Thư Mục
```
.
├── main.py              # File chính chứa giao diện người dùng
├── utils/              # Thư mục chứa các module phụ trợ
│   ├── data/          # Thư mục chứa dữ liệu huấn luyện
│   │   ├── train/     # Ảnh huấn luyện
│   │   └── validation/# Ảnh kiểm định
│   ├── models/        # Thư mục lưu mô hình đã huấn luyện
│   ├── plots/         # Thư mục lưu biểu đồ huấn luyện
│   ├── prepare_data.py# Module xử lý dữ liệu
│   ├── train.py       # Module huấn luyện mô hình
│   ├── detect_faces.py# Module nhận diện khuôn mặt
│   └── model.py       # Định nghĩa kiến trúc mô hình
└── requirements.txt    # Danh sách thư viện cần thiết
```

## Các Tính Năng Chính

### 1. Chụp Ảnh Khuôn Mặt
- Cho phép người dùng chụp ảnh khuôn mặt thông qua webcam
- Tự động phát hiện khuôn mặt và lưu ảnh
- Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm định (20%)
- Lưu ảnh với chất lượng cao (95%)

### 2. Huấn Luyện Mô Hình
- Sử dụng kiến trúc CNN cho nhận diện khuôn mặt
- Hiển thị quá trình huấn luyện theo thời gian thực
- Hiển thị các metrics: accuracy, loss, validation accuracy, validation loss
- Tự động lưu mô hình tốt nhất
- Dừng sớm (early stopping) khi không cải thiện

### 3. Nhận Diện Khuôn Mặt
- Nhận diện khuôn mặt trong thời gian thực
- Hiển thị xác suất cho mỗi lớp
- Vẽ khung xung quanh khuôn mặt được phát hiện

## Cách Sử Dụng

### Cài Đặt
1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

### Chạy Ứng Dụng
```bash
python main.py
```

### Các Bước Sử Dụng
1. **Chụp Ảnh**:
   - Nhập tên người dùng
   - Nhấn nút "Chụp ảnh"
   - Đứng trước camera và chờ chụp đủ số ảnh

2. **Huấn Luyện**:
   - Nhấn nút "Huấn luyện"
   - Theo dõi quá trình huấn luyện qua biểu đồ và metrics
   - Mô hình sẽ tự động lưu khi đạt kết quả tốt nhất

3. **Nhận Diện**:
   - Nhấn nút "Nhận diện"
   - Đứng trước camera để nhận diện
   - Xem kết quả nhận diện và xác suất

## Các Thông Số Kỹ Thuật

### Mô Hình
- Kiến trúc CNN với các lớp:
  - Convolutional layers
  - MaxPooling layers
  - Dense layers
  - Dropout để tránh overfitting

### Xử Lý Ảnh
- Kích thước ảnh đầu vào: 64x64 pixels
- Chuyển đổi sang ảnh xám
- Chuẩn hóa pixel values (0-1)

### Huấn Luyện
- Batch size: 32
- Số epoch tối đa: 50
- Early stopping với patience: 10
- Data augmentation cho tập huấn luyện

## Xử Lý Lỗi
- Kiểm tra và xử lý lỗi khi mở camera
- Xử lý lỗi khi lưu ảnh
- Kiểm tra và tạo thư mục nếu chưa tồn tại
- Xử lý lỗi khi tải mô hình

## Giao Diện Người Dùng
- Cửa sổ chính: 1920x1080 pixels
- Chia làm 2 phần:
  - Bên trái: Camera và điều khiển
  - Bên phải: Biểu đồ huấn luyện
- Hiển thị metrics theo thời gian thực
- Thanh tiến trình cho các hoạt động

## Lưu ý
- Đảm bảo đủ ánh sáng khi chụp ảnh
- Giữ khuôn mặt trong khung camera
- Đợi quá trình huấn luyện hoàn tất trước khi nhận diện
- Kiểm tra thư mục data có đủ ảnh trước khi huấn luyện

