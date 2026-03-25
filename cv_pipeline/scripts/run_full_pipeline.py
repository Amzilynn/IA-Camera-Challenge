import cv2
import time
import argparse
import sys
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cv_pipeline.detection.yolo_detector import YOLODetector
from cv_pipeline.tracking.boxmot_tracker import PersonTracker
# from cv_pipeline.pose_estimation.pose_estimator import PoseEstimator
from cv_pipeline.emotion_analysis.emotion_analyzer import EmotionAnalyzer
from cv_pipeline.social_interaction.social_analyzer import SocialAnalyzer
from cv_pipeline.utils.scene_describer import SceneDescriber
from cv_pipeline.utils.ui_processor import UIProcessor
from cv_pipeline.utils.metrics_tracker import HUDMetricsTracker

def run_pipeline(video_path, output_path="final_output.mp4", headless=False, log_file="scene_log.json"):
    # 1. Initialization
    print(f"I: Initializing pipeline for {video_path}...")
    
    # Resolve paths relative to project root
    def get_root_path(rel_path):
        return str(project_root / rel_path)
    
    # Handle webcam input (integer)
    if str(video_path).isdigit():
        video_path = int(video_path)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"E: Could not open video: {video_path}")
        sys.exit(1)

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video writer
    # avc1 (H.264) is universally supported by web browsers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    
    # Adjust output dimensions if we plan to downscale
    out_width, out_height = width, height
    if width > 1920:
        out_width, out_height = 1920, 1080
    
    # Ensure extension is .mp4
    if not output_path.endswith('.mp4'):
        output_path = str(Path(output_path).with_suffix('.mp4'))
        
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    # Initialize Modules (Absolute State-of-Art for 2025/2026)
    detector = YOLODetector(
        human_model_path=get_root_path("cv_pipeline/models/yolo26s.pt"),
        pose_model_path=get_root_path("cv_pipeline/models/yolo26s-pose.pt"),
        face_model_path=get_root_path("cv_pipeline/models/yolov8n.pt")
    )
    # Initialize tracker with specific reid_weights path if needed, or default
    # Note: BoxMOT might need some downloads on first run
    try:
        tracker = PersonTracker(tracker_type='deepocsort', device=detector.device)
    except Exception as e:
        print(f"E: Tracker failed to init: {e}")
        sys.exit(1)
        
    # pose_estimator = PoseEstimator() # DISABLED: Unstable on crops. Using YOLO native.
    emotion_analyzer = EmotionAnalyzer() # SOTA hsemotion/deepface
    social_analyzer = SocialAnalyzer(fps=fps)
    
    # Ensure log file is absolute if not already
    if not os.path.isabs(log_file):
        log_file = get_root_path(log_file)
        
    scene_describer = SceneDescriber(log_file=log_file)
    print(f"I: Video Properties: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")
    
    # 3. HUD and Metrics Stabilization
    ui_processor = UIProcessor()
    # 30 frames (1s) gives a snappy but stable dashboard given the new BoxMOT coasting
    metrics_tracker = HUDMetricsTracker(window_size=30, ema_alpha=0.1)

    print("I: Pipeline initialized. Starting loop...")

    frame_count = 0
    start_time = time.time()
    
    from collections import deque, Counter
    
    # Store persistent attributes for tracked persons to prevent flickering
    # Structure: {track_id: {'emotion_history': deque, 'age_history': deque, 'gender_history': deque, 'stable': {}}}
    person_history = {} 
    MAX_HISTORY = 30 # Increased for better temporal smoothing
    
    # --- Cross-Scene ID Tracking ---
    # gallery: {original_id: {'embedding': np.array, 'gender': str, 'age': int}}
    # id_mapping: {current_track_id: original_id}
    gallery = {}
    id_mapping = {}
    next_original_id = 0
    
    def get_cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)

    # --- Intelligence Summary Buffer ---
    intelligence_buffer = [] # list of (frame_id, detections, interactions)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Optimization: Downscale 4K to 1080p for significantly faster processing
        if width > 1920:
            frame = cv2.resize(frame, (1920, 1080))
            
        frame_count += 1
        
        # 2. Detection (YOLO)
        detections = detector.detect(frame)
        
        # 3. Tracking (BoxMOT)
        detections = tracker.update(frame, detections)
        
        # 4 & 5. Pose & Emotion for each tracked person
        for det in detections:
            track_id = det.get('track_id', -1)
            
            # 1. Handle ID mapping for cross-scene consistency
            if track_id != -1:
                if track_id not in id_mapping:
                    best_match_id = -1
                    best_score = 0
                    if det.get('faces'):
                        face_bbox = det['faces'][0]['bbox']
                        curr_emb = emotion_analyzer.get_embedding(frame, face_bbox)
                        if curr_emb is not None:
                            for gid, data in gallery.items():
                                score = get_cosine_similarity(curr_emb, data['embedding'])
                                match = (score > 0.75) 
                                if match and score > best_score:
                                    best_score = score
                                    best_match_id = gid
                    
                    if best_match_id != -1:
                        id_mapping[track_id] = best_match_id
                    else:
                        id_mapping[track_id] = track_id
                
                det['track_id_original'] = id_mapping[track_id]
                orig_id = det['track_id_original']
            else:
                orig_id = -1

            # Initialize history if new track
            if orig_id != -1 and orig_id not in person_history:
                person_history[orig_id] = {
                    'emotion_history': deque(maxlen=MAX_HISTORY),
                    'age_history': deque(maxlen=MAX_HISTORY),
                    'gender_history': deque(maxlen=MAX_HISTORY),
                    'stable': {},
                    'embeddings': [] 
                }
            
            # --- Emotion/Age/Gender Analysis ---
            if frame_count % 15 == 0:
                analysis_bbox = det['faces'][0]['bbox'] if det.get('faces') else det['bbox']
                emotion_results = emotion_analyzer.analyze(frame, analysis_bbox)

                if orig_id != -1:
                    hist = person_history[orig_id]
                    if emotion_results:
                        hist['emotion_history'].append(emotion_results['emotion'])
                        hist['age_history'].append(emotion_results['age'])
                        hist['gender_history'].append(emotion_results['gender'])

                    # Temporal Smoothing
                    if len(hist['gender_history']) > 0:
                        most_common_gender, count = Counter(hist['gender_history']).most_common(1)[0]
                        if count >= 3: hist['stable']['gender'] = most_common_gender
                    if len(hist['age_history']) > 0:
                        hist['stable']['age'] = int(np.mean(hist['age_history']))
                    if len(hist['emotion_history']) > 0:
                        most_common_emo, count = Counter(hist['emotion_history']).most_common(1)[0]
                        current_stable_emo = hist['stable'].get('emotion')
                        if current_stable_emo is None or (count > (len(hist['emotion_history']) * 0.4) and most_common_emo != current_stable_emo):
                            hist['stable']['emotion'] = most_common_emo

            # Assign from stable history if available
            if orig_id != -1 and orig_id in person_history:
                det.update(person_history[orig_id]['stable'])
                det['track_id_display'] = orig_id

        # 6. Social Interaction Analysis (STAS/STIE)
        social_detections = []
        for d in detections:
            sd = d.copy()
            if 'track_id_original' in d:
                sd['track_id'] = d['track_id_original']
            social_detections.append(sd)
            
        interactions, person_statuses = social_analyzer.analyze(social_detections)
        
        # Buffer for intelligence summary
        intelligence_buffer.append({
            'frame': frame_count,
            'detections': social_detections,
            'interactions': interactions,
            'status': person_statuses
        })

        if len(intelligence_buffer) >= 100:
            generate_scene_summary(intelligence_buffer, scene_describer)
            intelligence_buffer = []

        # Update original detections with status info
        for det in detections:
            tid = det.get('track_id_original', det.get('track_id'))
            if tid in person_statuses:
                det.update(person_statuses[tid])

        # 7. Scene Description
        desc_text = scene_describer.describe(social_detections, frame_count, width=frame.shape[1], height=frame.shape[0], interactions=interactions)
        scene_describer.save_log(desc_text)

        # 8. Visualization (Premium HUD)
        # Current Frame Raw Stats
        current_persons = [d for d in detections if d['type'] == 'person']
        total_current = len(current_persons)
        
        # Staff vs Visitor (Checking role assignment)
        staff_count = sum(1 for d in current_persons if 'staff' in str(d.get('role', '')).lower())
        
        engaged_people = set()
        for interaction in interactions:
            if interaction.get('type') not in ['Group_Bond', 'Proximity']:
                engaged_people.update(interaction.get('ids', []))
        active_engagements = len(engaged_people)
        
        # Raw Satisfaction Index from emotions (Handling HSEmotion labels)
        idx_raw = 75.0
        if total_current > 0:
            # Common emotion labels from HSEmotion/DeepFace
            pos_labels = ['happy', 'happiness', 'surprise', 'surprised']
            neu_labels = ['neutral']
            
            # Use a more nuanced sentiment mapping:
            # Positive: 100%, Neutral: 80%, Negative: 50%
            emotions = [str(d.get('emotion', 'neutral')).lower() for d in current_persons]
            weights = []
            for e in emotions:
                if any(label in e for label in pos_labels):
                    weights.append(1.0)
                elif any(label in e for label in neu_labels):
                    weights.append(0.8)
                else: 
                    # Negative or others (Sad, Angry, Fear, Disgust)
                    weights.append(0.5)
            
            idx_raw = float(sum(weights) / total_current * 100.0)
        
        # 9. APPLY HUD METRICS TRACKER (SMOOTHING & CONSISTENCY)
        raw_metrics = {
            'total_people': total_current,
            'staff_count': staff_count,
            'active_engagements': active_engagements,
            'satisfaction_index': idx_raw
        }
        
        global_metrics = metrics_tracker.update(raw_metrics)
        
        # 10. Render HUD
        drawn_frame = ui_processor.render_hud(frame, detections, interactions, global_metrics)

        if not headless:
            display_frame = cv2.resize(drawn_frame, (1280, 720))
            cv2.imshow('IA Camera Challenge STIE', display_frame)
        
        out_writer.write(drawn_frame) 
        if not headless and cv2.waitKey(1) & 0xFF == ord('q'): break
            
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_curr = frame_count / (elapsed + 1e-6)
            print(f"STIE Pipeline: {frame_count}/{total_frames} (FPS: {fps_curr:.2f})", flush=True)

    cap.release()
    out_writer.release()
    if not headless: cv2.destroyAllWindows()
    print(f"\nProcessing complete. Video saved to {output_path}")

def draw_badge(img, text, pos, bg_color, text_color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5; thickness = 1
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x, y - h - 10), (x + w + 10, y + 5), bg_color, -1)
    cv2.putText(img, text, (x + 5, y - 5), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return h + 15

def generate_scene_summary(buffer, scene_describer):
    """Generates a Scene_Intelligence_Summary for the user-requested 100-frame window."""
    last_frame = buffer[-1]
    interactions = last_frame['interactions']
    statuses = last_frame['status']
    
    print("\n" + "="*50)
    print(f"--- SCENE INTELLIGENCE SUMMARY (Frames {buffer[0]['frame']}-{buffer[-1]['frame']}) ---")
    
    # 1. Active Groups
    groups = [i for i in interactions if i['type'] == 'Group_Bond']
    if groups:
        print("Active Groups:", [f"{g['ids']} -> GroupBonded" for g in groups])
    
    # 2. Actionable Insights
    alerts = []
    for tid, s in statuses.items():
        if s.get('intent') == 'Pre-emptive Service':
            alerts.append(f"Visitor {tid} is showing confusion cues; nearest Staff may need to assist.")
    
    if alerts:
        print("Actionable Insights:")
        for a in alerts: print(f" - {a}")
    else:
        print("Actionable Insights: None (All behaviors normal)")
    
    print("="*50 + "\n")
    print(f"Scene logs saved to {scene_describer.log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs='?', default="vd2.mp4", help="Path to input video")
    parser.add_argument("--output", default="final_output.mp4", help="Path to output video")
    parser.add_argument("--log", default="scene_log.json", help="Path to scene log")
    parser.add_argument("--headless", action="store_true", help="Run without GUI display")
    args = parser.parse_args()
    
    run_pipeline(args.video, output_path=args.output, headless=args.headless, log_file=args.log)
