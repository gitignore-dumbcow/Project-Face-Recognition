import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
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
        layers.Input(shape=input_shape),
        # Khối Convolutional đầu tiên
        layers.Conv2D(32, (3, 3), activation='relu'),
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
        layers.Dense(1, activation='sigmoid')  # Changed to 1 output for binary classification
    ])
    
    # Biên dịch mô hình với:
    # - optimizer='adam': Thuật toán tối ưu Adam để cập nhật trọng số
    # - loss='binary_crossentropy': Hàm mất mát cho bài toán phân loại nhị phân
    # - metrics=['accuracy']: Đo lường độ chính xác của mô hình trong quá trình huấn luyện
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Test mô hình
    model = cnn_model()
    model.summary() 