from ultralytics import YOLO
import os

models = [
    "cv_pipeline/models/yolov12m.pt",
    "cv_pipeline/models/yolov12m-pose.pt",
    "cv_pipeline/models/yolov8m.pt",
    "cv_pipeline/models/yolov8m-pose.pt",
    "cv_pipeline/models/yolo26s.pt",
    "cv_pipeline/models/yolo26s-pose.pt"
]

for m in models:
    print(f"Testing {m}...")
    if not os.path.exists(m):
        print(f"  Result: File not found")
        continue
    try:
        model = YOLO(m)
        print(f"  Result: Success")
    except Exception as e:
        print(f"  Result: Failed - {e}")
