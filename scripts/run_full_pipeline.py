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

def run_pipeline(video_path, output_path="output.mp4"):
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
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # original
    fourcc = cv2.VideoWriter_fourcc(*'avc1') # Better compatibility
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize Modules
    detector = YOLODetector()
    # Initialize tracker with specific reid_weights path if needed, or default
    # Note: BoxMOT might need some downloads on first run
    try:
        tracker = PersonTracker(tracker_type='deepocsort', device=detector.device)
    except Exception as e:
        print(f"E: Tracker failed to init: {e}")
        return
        
    pose_estimator = PoseEstimator()
    emotion_analyzer = EmotionAnalyzer()
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
        # Returns list of dicts: {'bbox': (x1,y1,x2,y2), 'conf': float, 'pose_keypoints': ..., 'faces': ...}
        detections = detector.detect(frame)
        
        # 3. Tracking (BoxMOT)
        # Updates 'detections' with 'track_id'
        detections = tracker.update(frame, detections)
        
        # 4 & 5. Pose & Emotion for each tracked person
        for det in detections:
            bbox = det['bbox']
            
            # --- Pose Estimation ---
            # detector.detect() already does pose if using YOLO-pose model, 
            # but let's use our MediaPipe wrapper if we want "MediaPipe" specifically as requested.
            # However, the YOLODetector code ALREADY has a pose model loaded (YOLOv8-pose).
            # The user explicitly asked for "MediaPipe BlazePose".
            # So, we will overwrite or augment the YOLO keypoints with MediaPipe keypoints.
            
            mp_kpts, _ = pose_estimator.estimate(frame, bbox)
            if mp_kpts:
                # We store it in a specific key to avoid confusion with YOLO pose
                det['mediapipe_keypoints'] = mp_kpts
            
            # --- Emotion Analysis ---
            # Run emotion analysis every 5 frames to save speed, or if track is new
            # For simplicity in this demo, run every frame (can be slow) or modulo
            if frame_count % 5 == 0:
                dom_emotion, scores = emotion_analyzer.analyze(frame, bbox)
                if dom_emotion:
                    det['emotion'] = dom_emotion
            
            # Persist emotion from previous frames if skipping (simple logic omitted for now)
            # A real system would cache emotions by ID.

        # 6. Scene Description
        desc_text = scene_describer.describe(detections, frame_count)
        scene_describer.save_log(desc_text)

        # 7. Visualization
        # Use detector.draw() for basic boxes/yolo-pose, then overlay extra stuff
        drawn_frame = detector.draw(frame, detections, draw_skeleton=False, draw_faces=False)
        
        # Draw MediaPipe skeletons manually
        for det in detections:
            if 'mediapipe_keypoints' in det:
                pose_estimator.draw_skeleton(drawn_frame, det['mediapipe_keypoints'])
            
            # Draw Emotion
            if 'emotion' in det:
                x1, y1, x2, y2 = map(int, det['bbox'])
                cv2.putText(drawn_frame, det['emotion'], (x1, y1 - 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 8. Output
        cv2.imshow('IA Camera Challenge', drawn_frame)
        out_writer.write(drawn_frame)
        
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
