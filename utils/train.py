import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from utils.model import cnn_model
from utils.prepare_data import get_data_generators

# Define paths
MODELS_PATH = os.path.join('utils', 'models')
PLOTS_PATH = os.path.join('utils', 'plots')

def train_model(epochs=50, callback=None):
    """
    Huấn luyện mô hình nhận dạng khuôn mặt
    
    Args:
        epochs: Số epoch huấn luyện
        callback: Callback tùy chỉnh để cập nhật GUI
    """
    # Tạo thư mục models nếu chưa tồn tại
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    # Lấy data generators
    train_generator, validation_generator = get_data_generators()
    
    # Tạo mô hình
    model = cnn_model()
    
    # Callbacks
    callbacks = []
    
    # Thêm callback tùy chỉnh nếu có
    if callback:
        callbacks.append(callback)
    
    # Thêm các callbacks mặc định
    callbacks.extend([
        ModelCheckpoint(
            os.path.join(MODELS_PATH, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ])
    
    # Huấn luyện mô hình
    print("Bắt đầu huấn luyện...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Lưu mô hình cuối cùng
    model.save(os.path.join(MODELS_PATH, 'final_model.keras'))
    print("Đã lưu mô hình cuối cùng")
    
    return model, history

def plot_training_history(history):
    """
    Vẽ đồ thị quá trình huấn luyện
    
    Args:
        history: Lịch sử huấn luyện từ model.fit()
    """
    # Tạo thư mục plots nếu chưa tồn tại
    os.makedirs(PLOTS_PATH, exist_ok=True)
    
    # Vẽ đồ thị accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Vẽ đồ thị loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'training_history.png'))
    plt.close()

if __name__ == "__main__":
    model, history = train_model()
    plot_training_history(history) 