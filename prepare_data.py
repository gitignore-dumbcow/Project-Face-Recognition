import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_dataset_structure():
    """Create the necessary directory structure for the dataset"""
    directories = ['Dataset/train', 'Dataset/validation']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
    print("Dataset directories created successfully")

def capture_face_images(person_name, num_images=50):
    """
    Capture face images using webcam
    
    Args:
        person_name: Name of the person
        num_images: Number of images to capture
    """
    # Create person's directories
    train_dir = os.path.join('Dataset/train', person_name)
    val_dir = os.path.join('Dataset/validation', person_name)
    
    for directory in [train_dir, val_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    count = 0
    print(f"Capturing {num_images} images for {person_name}. Press 'c' to capture, 'q' to quit.")
    
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Draw rectangle around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'Captured: {count}/{num_images}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Capture Faces', frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # If 'c' is pressed and face is detected, save the image
        if key == ord('c') and len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (64, 64))
            
            # Save 80% to training, 20% to validation
            if count < int(num_images * 0.8):
                save_path = os.path.join(train_dir, f'face_{count}.jpg')
            else:
                save_path = os.path.join(val_dir, f'face_{count}.jpg')
                
            cv2.imwrite(save_path, face_roi)
            count += 1
            print(f"Captured image {count}/{num_images}")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {count} images for {person_name}")

def get_data_generators():
    """
    Create data generators for training and validation
    
    Returns:
        train_generator, validation_generator
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        'Dataset/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='sparse'
    )
    
    validation_generator = val_datagen.flow_from_directory(
        'Dataset/validation',
        target_size=(64, 64),
        batch_size=32,
        class_mode='sparse'
    )
    
    return train_generator, validation_generator

if __name__ == "__main__":
    # Test the functions
    create_dataset_structure()
    person_name = input("Enter person's name: ")
    capture_face_images(person_name) 