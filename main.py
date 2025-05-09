import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from PIL import Image, ImageTk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.prepare_data import capture_face_images, create_dataset_structure
from utils.train import train_model
from utils.detect_faces import load_face_recognition_model, preprocess_face

# Define paths
DATASET_PATH = os.path.join('utils', 'data')
MODELS_PATH = os.path.join('utils', 'models')
PLOTS_PATH = os.path.join('utils', 'plots')

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1920x1080")
        
        # State variables
        self.is_capturing = False
        self.is_detecting = False
        self.is_training = False
        self.cap = None
        self.current_mode = "default"  # default, capture, train, detect
        self.capture_count = 0
        self.capture_target = 50
        self.training_metrics = {
            'epoch': 0,
            'accuracy': 0,
            'loss': 0,
            'val_accuracy': 0,
            'val_loss': 0
        }
        self.training_history = {
            'accuracy': [],
            'loss': [],
            'val_accuracy': [],
            'val_loss': []
        }
        
        # Create GUI
        self.create_widgets()
        
        # Start camera in default mode
        self.start_camera()
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left frame for video and controls
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, padx=5, pady=5)
        
        # Video frame
        self.video_frame = ttk.LabelFrame(left_frame, text="Camera View", padding="5")
        self.video_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video label
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.grid(row=0, column=0, padx=5, pady=5)
        
        # Control frame
        control_frame = ttk.LabelFrame(left_frame, text="Controls", padding="5")
        control_frame.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Name entry
        ttk.Label(control_frame, text="Tên người dùng:").grid(row=0, column=0, padx=5, pady=5)
        self.name_entry = ttk.Entry(control_frame)
        self.name_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Buttons
        self.capture_btn = ttk.Button(control_frame, text="Chụp ảnh", command=self.toggle_capture_mode)
        self.capture_btn.grid(row=0, column=2, padx=5, pady=5)
        
        self.train_btn = ttk.Button(control_frame, text="Huấn luyện", command=self.toggle_training_mode)
        self.train_btn.grid(row=0, column=3, padx=5, pady=5)
        
        self.detect_btn = ttk.Button(control_frame, text="Nhận diện", command=self.toggle_detection_mode)
        self.detect_btn.grid(row=0, column=4, padx=5, pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(left_frame, mode='determinate')
        self.progress.grid(row=2, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Status label
        self.status_label = ttk.Label(left_frame, text="Sẵn sàng")
        self.status_label.grid(row=3, column=0, padx=5, pady=5)
        
        # Metrics frame
        self.metrics_frame = ttk.LabelFrame(left_frame, text="Metrics", padding="5")
        self.metrics_frame.grid(row=4, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Metrics labels
        self.metrics_labels = {}
        metrics = ['Epoch', 'Accuracy', 'Loss', 'Val Accuracy', 'Val Loss']
        for i, metric in enumerate(metrics):
            ttk.Label(self.metrics_frame, text=f"{metric}:").grid(row=0, column=i*2, padx=5, pady=5)
            self.metrics_labels[metric] = ttk.Label(self.metrics_frame, text="0")
            self.metrics_labels[metric].grid(row=0, column=i*2+1, padx=5, pady=5)
        
        # Right frame for plots
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, padx=5, pady=5)
        
        # Create figure for plots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plots
        self.ax1.set_title('Model Accuracy')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Accuracy')
        self.ax1.grid(True)
        
        self.ax2.set_title('Model Loss')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Loss')
        self.ax2.grid(True)
        
        plt.tight_layout()
    
    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update()
    
    def start_camera(self):
        """Start the camera in default mode"""
        self.cap = cv2.VideoCapture(0)
        self.update_camera()
    
    def update_camera(self):
        """Update camera feed based on current mode"""
        if self.cap is None:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            return
            
        # Process frame based on current mode
        if self.current_mode == "capture":
            frame = self.process_capture_mode(frame)
        elif self.current_mode == "train":
            frame = self.process_training_mode(frame)
        elif self.current_mode == "detect":
            frame = self.process_detection_mode(frame)
        
        # Convert frame to PhotoImage
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update video label
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
        # Schedule next update
        self.root.after(10, self.update_camera)
    
    def process_capture_mode(self, frame):
        """Process frame in capture mode"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{self.name_entry.get()}: {self.capture_count}/{self.capture_target}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame
    
    def process_training_mode(self, frame):
        """Process frame in training mode"""
        # Add training metrics overlay
        metrics_text = f"Epoch: {self.training_metrics['epoch']}\n"
        metrics_text += f"Accuracy: {self.training_metrics['accuracy']:.2f}\n"
        metrics_text += f"Loss: {self.training_metrics['loss']:.2f}\n"
        metrics_text += f"Val Accuracy: {self.training_metrics['val_accuracy']:.2f}\n"
        metrics_text += f"Val Loss: {self.training_metrics['val_loss']:.2f}"
        
        y0, dy = 30, 30
        for i, line in enumerate(metrics_text.split('\n')):
            y = y0 + i*dy
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def process_detection_mode(self, frame):
        """Process frame in detection mode"""
        if not hasattr(self, 'model') or self.model is None:
            try:
                self.model = load_face_recognition_model(MODELS_PATH)
                if self.model is None:
                    return frame
            except Exception as e:
                print(f"Error loading model: {e}")
                return frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            processed_face = preprocess_face(face_roi)
            
            predictions = self.model.predict(processed_face, verbose=0)
            class_names = sorted(os.listdir(os.path.join(DATASET_PATH, 'train')))
            
            # Display all class probabilities
            y_offset = y
            for i, (class_name, prob) in enumerate(zip(class_names, predictions[0])):
                text = f"{class_name}: {prob:.2%}"
                cv2.putText(frame, text, (x+w+10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return frame
    
    def toggle_capture_mode(self):
        """Toggle capture mode"""
        if self.current_mode == "capture":
            self.current_mode = "default"
            self.capture_btn.config(text="Chụp ảnh")
            self.is_capturing = False
        else:
            if not self.name_entry.get():
                messagebox.showerror("Lỗi", "Vui lòng nhập tên người dùng")
                return
            self.current_mode = "capture"
            self.capture_btn.config(text="Dừng chụp")
            self.is_capturing = True
            self.capture_count = 0
            threading.Thread(target=self.capture_faces, daemon=True).start()
    
    def toggle_training_mode(self):
        """Toggle training mode"""
        if self.current_mode == "train":
            self.current_mode = "default"
            self.train_btn.config(text="Huấn luyện")
            self.is_training = False
        else:
            self.current_mode = "train"
            self.train_btn.config(text="Dừng huấn luyện")
            self.is_training = True
            threading.Thread(target=self.train_model, daemon=True).start()
    
    def toggle_detection_mode(self):
        """Toggle detection mode"""
        if self.current_mode == "detect":
            self.current_mode = "default"
            self.detect_btn.config(text="Nhận diện")
            self.is_detecting = False
        else:
            self.current_mode = "detect"
            self.detect_btn.config(text="Dừng nhận diện")
            self.is_detecting = True
    
    def capture_faces(self):
        """Capture face images"""
        try:
            capture_face_images(self.name_entry.get(), self.capture_target)
            self.root.after(0, lambda: self.update_status("Đã chụp xong ảnh"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
        finally:
            self.root.after(0, self.toggle_capture_mode)
    
    def train_model(self):
        """Train the model"""
        try:
            # Create a custom callback to update GUI
            class MetricsCallback(tf.keras.callbacks.Callback):
                def __init__(self, app):
                    super().__init__()
                    self.app = app
                
                def on_epoch_end(self, epoch, logs=None):
                    if logs:
                        self.app.training_metrics.update({
                            'epoch': epoch + 1,
                            'accuracy': logs.get('accuracy', 0),
                            'loss': logs.get('loss', 0),
                            'val_accuracy': logs.get('val_accuracy', 0),
                            'val_loss': logs.get('val_loss', 0)
                        })
                        self.app.root.after(0, self.app.update_metrics_display)
            
            # Get the model and train with callback
            model, history = train_model(callback=MetricsCallback(self))
            
            self.root.after(0, lambda: self.update_status("Đã huấn luyện xong"))
            self.root.after(0, lambda: messagebox.showinfo("Thành công", "Huấn luyện hoàn tất!"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
        finally:
            self.root.after(0, self.toggle_training_mode)
    
    def update_plots(self):
        """Update training plots"""
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot accuracy
        self.ax1.set_title('Model Accuracy')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Accuracy')
        self.ax1.grid(True)
        if self.training_history['accuracy']:
            epochs = range(1, len(self.training_history['accuracy']) + 1)
            self.ax1.plot(epochs, self.training_history['accuracy'], label='Training')
            self.ax1.plot(epochs, self.training_history['val_accuracy'], label='Validation')
            self.ax1.legend()
        
        # Plot loss
        self.ax2.set_title('Model Loss')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Loss')
        self.ax2.grid(True)
        if self.training_history['loss']:
            epochs = range(1, len(self.training_history['loss']) + 1)
            self.ax2.plot(epochs, self.training_history['loss'], label='Training')
            self.ax2.plot(epochs, self.training_history['val_loss'], label='Validation')
            self.ax2.legend()
        
        plt.tight_layout()
        self.canvas.draw()
    
    def update_metrics_display(self):
        """Update metrics display"""
        for metric, value in self.training_metrics.items():
            if metric in self.metrics_labels:
                self.metrics_labels[metric].config(text=f"{value:.4f}")
        
        # Update training history
        for metric in ['accuracy', 'loss', 'val_accuracy', 'val_loss']:
            if metric in self.training_metrics:
                self.training_history[metric].append(self.training_metrics[metric])
        
        # Update plots
        self.update_plots()
    
    def on_closing(self):
        """Handle window closing"""
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop() 