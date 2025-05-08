import tensorflow as tf
from tensorflow.keras import layers, models

def cnn_model(input_shape=(64, 64, 3), num_classes=2):
    """
    Mô hình nhận dạng khuôn mặt đơn giản sử dụng CNN
    
    Args:
        input_shape: Kích thước của ảnh đầu vào (chiều cao, chiều rộng, số kênh màu)
        num_classes: Số lượng người cần nhận dạng
    
    Returns:
        Mô hình Keras đã biên dịch
    """
    model = models.Sequential([
        # Khối Convolutional đầu tiên
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Khối Convolutional thứ hai
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Khối Convolutional thứ ba
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten và Dense Layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),  # Ngăn chặn overfitting
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Biên dịch mô hình với:
    # - optimizer='adam': Thuật toán tối ưu Adam để cập nhật trọng số
    # - loss='sparse_categorical_crossentropy': Hàm mất mát cho bài toán phân loại nhiều lớp
    # - metrics=['accuracy']: Đo lường độ chính xác của mô hình trong quá trình huấn luyện
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Test mô hình
    model = cnn_model()
    model.summary() 