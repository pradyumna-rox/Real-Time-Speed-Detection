# Real-Time Speed Detection Using YOLOv11

This project implements a real-time vehicle speed detection system using YOLOv11. It detects vehicles in video streams, tracks their movement across frames, and calculates their speed. This system can be used for traffic monitoring, law enforcement, and road safety applications.

## 🚀 Features

- **YOLOv11-Based Detection**: High-accuracy object detection.
- **Vehicle Speed Calculation**: Speed estimation based on frame difference.
- **YOLOv5 Comparison**: Analyze and compare performance between YOLOv11 and YOLOv5.
- **Modular Codebase**: Separate scripts for detection, tracking, and analysis.

## 📁 Project Structure

Real-Time-Speed-Detection/
├── Analysis_with_yolo11.py # Main script for YOLOv11 analysis
├── Analysis_with_yolo5.py # YOLOv5 comparison script
├── Frames_creation.py # Extracts frames from input video
├── tracker3.py # Vehicle tracking and speed calculation
├── classeswithdata.py # Helper classes and data structures
├── yolov11_frames/ # Output frames from YOLOv11
├── yolov5_frames/ # Output frames from YOLOv5
├── yolo11s.pt # YOLOv11 model weights
├── yolov5s.pt # YOLOv5 model weights
└── performance-comparison-by-ultralytics.png




## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/pradyumna-rox/Real-Time-Speed-Detection.git
cd Real-Time-Speed-Detection



(Optional) Create and activate a virtual environment:


python -m venv venv
venv\Scripts\activate  # For Windows
Install dependencies:


pip install -r requirements.txt
If requirements.txt is not available, install: ultralytics, opencv-python, numpy, etc.

## 🎥 Usage
Extract frames from a video

python Frames_creation.py --video path_to_video.mp4 --output yolov11_frames
Run YOLOv11 analysis

python Analysis_with_yolo11.py --frames yolov11_frames
Run YOLOv5 analysis (for comparison)

python Analysis_with_yolo5.py --frames yolov5_frames
Track and calculate speed

python tracker3.py --input yolov11_frames
Adjust paths and parameters inside the scripts as needed.

## 📊 Performance Comparison
The file performance-comparison-by-ultralytics.png shows benchmark results comparing YOLOv11 and YOLOv5 in terms of speed, accuracy, and performance on this dataset.

## ⚠️ Notes
Large files (e.g. .mp4, .avi) are not pushed to GitHub due to size limits.

Model weights (yolo11s.pt, yolov5s.pt) must be placed in the project root.

Use external hosting (Google Drive, Dropbox) for large inputs if needed.
