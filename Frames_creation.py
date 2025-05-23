import cv2
import os

video_path = "runs\detect\predict2\SBP_Traffic(YOLOV5).avi"
output_dir = "yolov5_frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_id = 0
saved_id = 0
frame_rate = 10  # Extract every 10th frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_id % frame_rate == 0:
        filename = os.path.join(output_dir, f"frame_{saved_id:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved_id += 1
    frame_id += 1

cap.release()
print(f"✅ Extracted {saved_id} frames to '{output_dir}' folder.")
