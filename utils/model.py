import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def cnn_model(input_shape=(64, 64, 3), num_classes=4):
    """
    Mô hình nhận dạng khuôn mặt cải tiến sử dụng CNN
    
    Args:
        input_shape: Kích thước của ảnh đầu vào (chiều cao, chiều rộng, số kênh màu)
        num_classes: Số lượng người cần nhận dạng
    
    Returns:
        Mô hình Keras đã biên dịch
    """
    # Sử dụng functional API để tạo mô hình phức tạp hơn
    inputs = layers.Input(shape=input_shape)
    
    # Khối Convolutional đầu tiên với Batch Normalization
    x = layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Khối Convolutional thứ hai
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Khối Convolutional thứ ba
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Khối Convolutional thứ tư
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Flatten và Dense Layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Biên dịch mô hình với:
    # - optimizer: Adam với learning rate thấp hơn để học chậm và ổn định hơn
    # - loss: sparse_categorical_crossentropy cho phân loại đa lớp
    # - metrics: accuracy và top-k accuracy
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
    )
    
    return model

if __name__ == "__main__":
    # Test mô hình
    model = cnn_model(input_shape=(64, 64, 3), num_classes=4)
    model.summary() 