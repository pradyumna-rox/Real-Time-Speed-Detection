from ultralytics import YOLO

# Load a pre-trained YOLO model
model = YOLO("yolov5s.pt")

# Start tracking objects in a video
# You can also use live video streams or webcam input
model.predict("SBP_Traffic.mp4",save=True, imgsz=640, conf=0.3)


