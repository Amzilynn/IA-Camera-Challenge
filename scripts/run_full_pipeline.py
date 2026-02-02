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
# from cv_pipeline.pose_estimation.pose_estimator import PoseEstimator
from cv_pipeline.emotion_analysis.emotion_analyzer import EmotionAnalyzer
from cv_pipeline.emotion_analysis.mivolo_analyzer import MivoloAnalyzer
from cv_pipeline.social_interaction.social_analyzer import SocialAnalyzer
from cv_pipeline.utils.scene_describer import SceneDescriber

def run_pipeline(video_path, output_path="output.avi"):
    # 1. Initialization
    print(f"I: Initializing pipeline for {video_path}...")
    
    # Handle webcam input (integer)
    if str(video_path).isdigit():
        video_path = int(video_path)
        
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
    mivolo_analyzer = MivoloAnalyzer()
    social_analyzer = SocialAnalyzer(fps=fps)
    scene_describer = SceneDescriber(log_file="scene_log.txt")

    print("I: Pipeline initialized. Starting loop...")

    frame_count = 0
    start_time = time.time()
    
    from collections import deque, Counter
    
    # Store persistent attributes for tracked persons to prevent flickering
    # Structure: {track_id: {'emotion_history': deque, 'age_history': deque, 'gender_history': deque, 'stable': {}}}
    person_history = {} 
    MAX_HISTORY = 15

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
            track_id = det.get('track_id', -1)
            
            # Initialize history if new track
            if track_id != -1 and track_id not in person_history:
                person_history[track_id] = {
                    'emotion_history': deque(maxlen=MAX_HISTORY),
                    'age_history': deque(maxlen=MAX_HISTORY),
                    'gender_history': deque(maxlen=MAX_HISTORY),
                    'stable': {}
                }
            
            # --- Emotion/Age/Gender Analysis ---
            if frame_count % 5 == 0:
                 # Prefer face detection box if available
                 analysis_bbox = None
                 if det['faces']:
                     analysis_bbox = det['faces'][0]['bbox']
                 else:
                     analysis_bbox = det['bbox']
                 
                 
                 # 1. Emotion (DeepFace)
                 emotion_results = emotion_analyzer.analyze(frame, analysis_bbox)
                 
                 # 2. Age & Gender (MiVOLO)
                 # Note: MiVOLO needs person bbox (det['bbox']) AND face bbox (analysis_bbox if it is a face)
                 person_bbox = det['bbox']
                 # Ensure proper bbox format for MiVOLO
                 mivolo_results = mivolo_analyzer.analyze(frame, person_bbox, det['faces'][0]['bbox'] if det['faces'] else None)

                 if track_id != -1:
                     hist = person_history[track_id]
                     
                     if emotion_results:
                        hist['emotion_history'].append(emotion_results['emotion'])
                     
                     if mivolo_results:
                        hist['age_history'].append(mivolo_results['age'])
                        hist['gender_history'].append(mivolo_results['gender'])

                     
                     # Calculate stable results
                     if len(hist['gender_history']) > 0:
                         # Majority vote for gender
                         gender_counts = Counter(hist['gender_history'])
                         hist['stable']['gender'] = gender_counts.most_common(1)[0][0]
                         
                         # Moving average for age
                         hist['stable']['age'] = int(np.mean(hist['age_history']))
                         
                         # Majority vote for emotion (or just most recent)
                         emotion_counts = Counter(hist['emotion_history'])
                         hist['stable']['emotion'] = emotion_counts.most_common(1)[0][0]
                         
                 # Apply results to current detection immediately as well
                 if emotion_results:
                     det.update(emotion_results)
                 if mivolo_results:
                     det.update(mivolo_results)

            # Assign from stable history if available
            if track_id != -1 and track_id in person_history:
                det.update(person_history[track_id]['stable'])

        # 6. Social Interaction Analysis (STAS)
        interactions, _ = social_analyzer.analyze(detections)

        # 7. Scene Description
        desc_text = scene_describer.describe(detections, frame_count, interactions)
        scene_describer.save_log(desc_text)

        # 8. Visualization
        drawn_frame = detector.draw(frame, detections, draw_skeleton=True, draw_faces=True)
        
        # Draw Interactions
        for inter in interactions:
            id1, id2 = inter['ids']
            label = inter['type']
            # Find center between the two people to place the label
            p1 = [d for d in detections if d.get('track_id') == id1][0]
            p2 = [d for d in detections if d.get('track_id') == id2][0]
            c1 = [(p1['bbox'][0]+p1['bbox'][2])/2, (p1['bbox'][1]+p1['bbox'][3])/2]
            c2 = [(p2['bbox'][0]+p2['bbox'][2])/2, (p2['bbox'][1]+p2['bbox'][3])/2]
            cx, cy = int((c1[0]+c2[0])/2), int((c1[1]+c2[1])/2)
            
            cv2.putText(drawn_frame, f"<< {label} >>", (cx - 50, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.line(drawn_frame, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), (0, 255, 0), 1)

        for det in detections:
            # Draw Emotion, Age, Gender
            if 'emotion' in det:
                x1, y1, x2, y2 = map(int, det['bbox'])
                text = f"{det['emotion']} | {det['age']} | {det['gender']}"
                
                # Position text above person ID label if it exists
                text_y = y1 - 35
                cv2.putText(drawn_frame, text, (x1, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

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
