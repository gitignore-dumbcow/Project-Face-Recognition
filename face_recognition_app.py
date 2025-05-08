import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from PIL import Image, ImageTk
import threading
from prepare_data import capture_face_images, create_dataset_structure
from train import train_model
from detect_faces import load_face_recognition_model, preprocess_face

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("800x600")
        
        # Biến để kiểm soát luồng
        self.is_capturing = False
        self.is_detecting = False
        self.cap = None
        
        # Tạo giao diện
        self.create_widgets()
        
        # Tạo thư mục dataset nếu chưa tồn tại
        create_dataset_structure()
    
    def create_widgets(self):
        # Frame chính
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame cho video
        self.video_frame = ttk.LabelFrame(main_frame, text="Camera View", padding="5")
        self.video_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Label cho video
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.grid(row=0, column=0, padx=5, pady=5)
        
        # Frame cho controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Entry cho tên người dùng
        ttk.Label(control_frame, text="Tên người dùng:").grid(row=0, column=0, padx=5, pady=5)
        self.name_entry = ttk.Entry(control_frame)
        self.name_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Buttons
        self.capture_btn = ttk.Button(control_frame, text="Chụp ảnh", command=self.start_capture)
        self.capture_btn.grid(row=0, column=2, padx=5, pady=5)
        
        self.train_btn = ttk.Button(control_frame, text="Huấn luyện", command=self.start_training)
        self.train_btn.grid(row=0, column=3, padx=5, pady=5)
        
        self.detect_btn = ttk.Button(control_frame, text="Nhận diện", command=self.start_detection)
        self.detect_btn.grid(row=0, column=4, padx=5, pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='determinate')
        self.progress.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Sẵn sàng")
        self.status_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
    
    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update()
    
    def start_capture(self):
        if not self.name_entry.get():
            messagebox.showerror("Lỗi", "Vui lòng nhập tên người dùng")
            return
        
        if self.is_capturing:
            self.stop_capture()
            return
        
        self.is_capturing = True
        self.capture_btn.config(text="Dừng chụp")
        self.update_status("Đang chụp ảnh...")
        
        # Chạy capture trong thread riêng
        threading.Thread(target=self.capture_faces, daemon=True).start()
    
    def stop_capture(self):
        self.is_capturing = False
        self.capture_btn.config(text="Chụp ảnh")
        if self.cap is not None:
            self.cap.release()
        self.update_status("Đã dừng chụp ảnh")
    
    def capture_faces(self):
        try:
            capture_face_images(self.name_entry.get())
            self.root.after(0, lambda: self.update_status("Đã chụp xong ảnh"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
        finally:
            self.root.after(0, self.stop_capture)
    
    def start_training(self):
        if self.is_detecting:
            self.stop_detection()
        
        self.train_btn.config(state='disabled')
        self.update_status("Đang huấn luyện...")
        
        # Chạy training trong thread riêng
        threading.Thread(target=self.train_model, daemon=True).start()
    
    def train_model(self):
        try:
            model, history = train_model()
            self.root.after(0, lambda: self.update_status("Đã huấn luyện xong"))
            self.root.after(0, lambda: messagebox.showinfo("Thành công", "Huấn luyện hoàn tất!"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
        finally:
            self.root.after(0, lambda: self.train_btn.config(state='normal'))
    
    def start_detection(self):
        if self.is_detecting:
            self.stop_detection()
            return
        
        self.is_detecting = True
        self.detect_btn.config(text="Dừng nhận diện")
        self.update_status("Đang nhận diện...")
        
        # Chạy detection trong thread riêng
        threading.Thread(target=self.detect_faces, daemon=True).start()
    
    def stop_detection(self):
        self.is_detecting = False
        self.detect_btn.config(text="Nhận diện")
        if self.cap is not None:
            self.cap.release()
        self.update_status("Đã dừng nhận diện")
    
    def detect_faces(self):
        try:
            # Tải model
            model = load_face_recognition_model()
            if model is None:
                raise Exception("Không thể tải model")
            
            # Tải class names
            class_names = sorted(os.listdir('Dataset/train'))
            
            # Khởi tạo webcam
            self.cap = cv2.VideoCapture(0)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            while self.is_detecting:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Phát hiện khuôn mặt
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                
                # Xử lý từng khuôn mặt
                for (x, y, w, h) in faces:
                    face_roi = frame[y:y+h, x:x+w]
                    processed_face = preprocess_face(face_roi)
                    
                    predictions = model.predict(processed_face, verbose=0)
                    predicted_class = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_class]
                    
                    label = f"{class_names[predicted_class]}: {confidence:.2f}"
                    color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255)
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Chuyển đổi frame để hiển thị
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Cập nhật video frame
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
        finally:
            self.root.after(0, self.stop_detection)
    
    def on_closing(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop() 