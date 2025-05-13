import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Define paths
MODEL_PATH = os.path.join('utils', 'models', 'best_model.keras')
DATA_PATH = os.path.join('utils', 'data', 'train')

def preprocess_face(face_img):
    """
    Preprocess face image for model input.
    
    Args:
        face_img: Face image in BGR format
    
    Returns:
        Preprocessed face image
    """
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    face_resized = cv2.resize(face_rgb, (64, 64))
    
    # Normalize pixel values
    face_normalized = face_resized / 255.0
    
    # Add batch dimension
    face_batch = np.expand_dims(face_normalized, axis=0)
    
    return face_batch

def detect_faces():
    """
    Nhận diện khuôn mặt realtime từ webcam
    """
    # Tải model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")
    
    model = load_model(MODEL_PATH)
    
    # Tải class names từ thư mục data/train
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")
    
    class_names = sorted(os.listdir(DATA_PATH))
    if not class_names:
        raise ValueError("No classes found in training data directory")
    
    # Khởi tạo webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")
        
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("Bắt đầu nhận diện khuôn mặt. Nhấn 'q' để thoát.")
    print(f"Classes: {', '.join(class_names)}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể lấy khung hình")
            break
        
        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện khuôn mặt
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Xử lý từng khuôn mặt
        for (x, y, w, h) in faces:
            # Cắt khuôn mặt
            face_roi = frame[y:y+h, x:x+w]
            
            # Tiền xử lý
            processed_face = preprocess_face(face_roi)
            
            # Dự đoán
            predictions = model.predict(processed_face, verbose=0)[0]
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class]
            
            # Vẽ kết quả
            label = f"{class_names[predicted_class]}: {confidence:.2f}"
            color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Hiển thị khung hình
        cv2.imshow('Face Recognition', frame)
        
        # Thoát nếu nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_faces() 