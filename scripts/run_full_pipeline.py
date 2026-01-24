import cv2
import time
import argparse
import sys
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cv_pipeline.detection.yolo_detector import YOLODetector
from cv_pipeline.tracking.boxmot_tracker import PersonTracker
from cv_pipeline.pose_estimation.pose_estimator import PoseEstimator
from cv_pipeline.emotion_analysis.emotion_analyzer import EmotionAnalyzer
from cv_pipeline.utils.scene_describer import SceneDescriber

def run_pipeline(video_path, output_path="output.avi"):
    # 1. Initialization
    print(f"I: Initializing pipeline for {video_path}...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"E: Could not open video: {video_path}")
        return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video writer
    # MJPG (avi) is widely supported on Windows without extra dlls
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize Modules (Optimized for GTX 1650 / 16GB RAM)
    # Switched from Large/XLarge to Medium (yolov8m) for better balance of speed/accuracy
    detector = YOLODetector(
        human_model_path="yolov8m.pt",
        pose_model_path="yolov8m-pose.pt",
        face_model_path="yolov8n-face.pt" # Explicitly use downloaded face model
    )
    # Initialize tracker with specific reid_weights path if needed, or default
    # Note: BoxMOT might need some downloads on first run
    try:
        tracker = PersonTracker(tracker_type='deepocsort', device=detector.device)
    except Exception as e:
        print(f"E: Tracker failed to init: {e}")
        return
        
    # pose_estimator = PoseEstimator() # DISABLED: Unstable on crops. Using YOLO native.
    emotion_analyzer = EmotionAnalyzer()
    
    # ... (skipping lines) ...
    
    scene_describer = SceneDescriber(log_file="scene_log.txt")

    print("I: Pipeline initialized. Starting loop...")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # 2. Detection (YOLO)
        detections = detector.detect(frame)
        
        # 3. Tracking (BoxMOT)
        detections = tracker.update(frame, detections)
        
        # 4 & 5. Pose & Emotion for each tracked person
        for det in detections:
            bbox = det['bbox']
            
            # --- Pose Estimation ---
            # Using YOLOv8-pose native keypoints (already in det['pose_keypoints'])
            
            # --- Emotion Analysis ---
            if frame_count % 5 == 0:
                 # Prefer face detection box if available
                 if det['faces']:
                     face_bbox = det['faces'][0]['bbox']
                     dom_emotion, scores = emotion_analyzer.analyze(frame, face_bbox)
                     if dom_emotion:
                         det['emotion'] = dom_emotion
                 else:
                     # Fallback to analyzing upper body region if face not explicitly detected?
                     # Or just skip. For now, skip to avoid noise.
                     pass

        # 6. Scene Description
        desc_text = scene_describer.describe(detections, frame_count)
        scene_describer.save_log(desc_text)

        # 7. Visualization
        # Use detector.draw() for basic boxes/yolo-pose, then overlay extra stuff
        # Enable draw_skeleton=True to use YOLO pose
        drawn_frame = detector.draw(frame, detections, draw_skeleton=True, draw_faces=True)
        
        # (MediaPipe manual drawing removed)
        for det in detections:
            # Draw Emotion
            if 'emotion' in det:
                x1, y1, x2, y2 = map(int, det['bbox'])
                cv2.putText(drawn_frame, det['emotion'], (x1, y1 - 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 8. Output
        # Resize for display to fit 1080p screens better (1280x720)
        display_frame = cv2.resize(drawn_frame, (1280, 720))
        cv2.imshow('IA Camera Challenge', display_frame)
        out_writer.write(drawn_frame) # Write full resolution to file
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("I: Stopped by user.")
            break
            
        # Logging progress
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_curr = frame_count / (elapsed + 1e-6)
            print(f"Processing frame {frame_count}/{total_frames} (FPS: {fps_curr:.2f})", end='\r')

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    
    print(f"\nProcessing complete. Video saved to {output_path}")
    print(f"Scene logs saved to {scene_describer.log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs='?', default="vd2.mp4", help="Path to input video")
    args = parser.parse_args()
    
    run_pipeline(args.video)
