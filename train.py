import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from model import cnn_model
from prepare_data import get_data_generators

def train_model(epochs=50):
    """
    Huấn luyện model với các tham số và callback
    
    Args:
        epochs: Số epoch huấn luyện
    """
    # Lấy data generators
    train_generator, validation_generator = get_data_generators()
    
    # Tạo model
    num_classes = len(train_generator.class_indices)
    model = cnn_model(num_classes=num_classes)
    
    # Tạo callbacks
    callbacks = [
        # Lưu model tốt nhất
        ModelCheckpoint(
            'models/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Dừng sớm nếu không cải thiện
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Huấn luyện model
    print("Bắt đầu huấn luyện...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    # Vẽ đồ thị
    plot_training_history(history)
    
    return model, history

def plot_training_history(history):
    """
    Vẽ đồ thị quá trình huấn luyện
    
    Args:
        history: Lịch sử huấn luyện từ model.fit()
    """
    # Tạo figure với 2 subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Vẽ accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Vẽ loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    # Lưu đồ thị
    plt.savefig('training_history.png')
    plt.close()

if __name__ == "__main__":
    # Huấn luyện model
    model, history = train_model()
    
    # Lưu model cuối cùng
    model.save('models/final_model.keras')
    print("Huấn luyện hoàn tất!") 