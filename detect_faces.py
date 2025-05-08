import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def load_face_recognition_model(model_path='models/best_model.keras'):
    """
    Tải model đã huấn luyện
    
    Args:
        model_path: Đường dẫn đến file model
    """
    try:
        model = load_model(model_path)
        print("Đã tải model thành công!")
        return model
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        return None

def preprocess_face(face_img):
    """
    Tiền xử lý ảnh khuôn mặt
    
    Args:
        face_img: Ảnh khuôn mặt từ webcam
    """
    # Resize về kích thước 64x64
    face_img = cv2.resize(face_img, (64, 64))
    # Chuyển đổi về float32 và normalize
    face_img = face_img.astype('float32') / 255.0
    # Thêm batch dimension
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

def detect_faces():
    """
    Nhận diện khuôn mặt realtime từ webcam
    """
    # Tải model
    model = load_face_recognition_model()
    if model is None:
        return
    
    # Tải class names từ thư mục Dataset/train
    import os
    class_names = sorted(os.listdir('Dataset/train'))
    
    # Khởi tạo webcam
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("Bắt đầu nhận diện khuôn mặt. Nhấn 'q' để thoát.")
    
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
            predictions = model.predict(processed_face, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
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