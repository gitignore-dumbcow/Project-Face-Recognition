import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_dataset_structure():
    """Tạo cấu trúc thư mục cần thiết cho tập dữ liệu"""
    directories = ['Dataset/train', 'Dataset/validation']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
    print("Đã tạo thành công các thư mục dữ liệu")

def capture_face_images(person_name, num_images=50):
    """
    Chụp ảnh khuôn mặt bằng webcam
    
    Args:
        person_name: Tên của người dùng
        num_images: Số lượng ảnh cần chụp
    """
    # Tạo thư mục cho người dùng
    train_dir = os.path.join('Dataset/train', person_name)
    val_dir = os.path.join('Dataset/validation', person_name)
    
    for directory in [train_dir, val_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Khởi tạo webcam
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    count = 0
    print(f"Đang chụp {num_images} ảnh cho {person_name}. Nhấn 'c' để chụp, 'q' để thoát.")
    
    while count < num_images:
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
        
        # Vẽ hình chữ nhật xung quanh khuôn mặt
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'Đã chụp: {count}/{num_images}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Hiển thị khung hình
        cv2.imshow('Chụp Khuôn Mặt', frame)
        
        # Đợi nhấn phím
        key = cv2.waitKey(1) & 0xFF
        
        # Nếu nhấn 'c' và phát hiện được khuôn mặt, lưu ảnh
        if key == ord('c') and len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (64, 64))
            
            # Lưu 80% vào tập huấn luyện, 20% vào tập kiểm định
            if count < int(num_images * 0.8):
                save_path = os.path.join(train_dir, f'face_{count}.jpg')
            else:
                save_path = os.path.join(val_dir, f'face_{count}.jpg')
                
            cv2.imwrite(save_path, face_roi)
            count += 1
            print(f"Đã chụp ảnh {count}/{num_images}")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Đã chụp {count} ảnh cho {person_name}")

def get_data_generators():
    """
    Tạo data generators cho training và validation
    """
    # Data augmentation cho training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Chỉ rescale cho validation
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Tạo generators
    train_generator = train_datagen.flow_from_directory(
        'Dataset/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='sparse',
        shuffle=True
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        'Dataset/validation',
        target_size=(64, 64),
        batch_size=32,
        class_mode='sparse',
        shuffle=False
    )
    
    return train_generator, validation_generator

if __name__ == "__main__":
    # Kiểm tra các hàm
    create_dataset_structure()
    person_name = input("Nhập tên người dùng: ")
    capture_face_images(person_name)