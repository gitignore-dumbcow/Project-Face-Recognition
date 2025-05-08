# Face Recognition System

A real-time face recognition system built with Python, OpenCV, and face_recognition library.

## Features

- Real-time face detection and recognition using webcam
- Support for multiple known faces
- Simple and intuitive interface
- High accuracy face recognition

## Requirements

- Python 3.7 or higher
- Webcam
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Add face images to the `Dataset` directory:
   - Use clear, front-facing photos
   - Name the files with the person's name (e.g., `john.jpg`, `sarah.png`)
   - Supported formats: JPG, JPEG, PNG

2. Run the face recognition system:
   ```
   python face_recognition_system.py
   ```

3. The system will:
   - Load known faces from the Dataset directory
   - Start your webcam
   - Display real-time face recognition results
   - Press 'q' to quit the application

## Notes

- Make sure you have good lighting for better recognition
- Keep your face clearly visible to the camera
- The system works best with front-facing faces
- Recognition accuracy depends on the quality of the reference images

## Troubleshooting

If you encounter any issues:
1. Ensure all dependencies are correctly installed
2. Check if your webcam is working properly
3. Verify that the Dataset directory contains valid face images
4. Make sure you have sufficient lighting

