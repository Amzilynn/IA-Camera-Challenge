import cv2
import time
import argparse
import sys
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cv_pipeline.detection.yolo_detector import YOLODetector
from cv_pipeline.tracking.boxmot_tracker import PersonTracker
# from cv_pipeline.pose_estimation.pose_estimator import PoseEstimator
from cv_pipeline.emotion_analysis.emotion_analyzer import EmotionAnalyzer
from cv_pipeline.emotion_analysis.mivolo_analyzer import MivoloAnalyzer
from cv_pipeline.social_interaction.social_analyzer import SocialAnalyzer
from cv_pipeline.utils.scene_describer import SceneDescriber

def run_pipeline(video_path, output_path="final_output.mp4"):
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
    # mp4v is better for MP4 output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    
    # Adjust output dimensions if we plan to downscale
    out_width, out_height = width, height
    if width > 1920:
        out_width, out_height = 1920, 1080
        
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    # Initialize Modules (Absolute State-of-Art for 2025/2026)
    detector = YOLODetector(
        human_model_path="cv_pipeline/models/yolov12m.pt",
        pose_model_path="cv_pipeline/models/yolov12m-pose.pt",
        face_model_path="cv_pipeline/models/yolov8n-face.pt"
    )
    # Initialize tracker with specific reid_weights path if needed, or default
    # Note: BoxMOT might need some downloads on first run
    try:
        tracker = PersonTracker(tracker_type='deepocsort', device=detector.device)
    except Exception as e:
        print(f"E: Tracker failed to init: {e}")
        return
        
    # pose_estimator = PoseEstimator() # DISABLED: Unstable on crops. Using YOLO native.
    emotion_analyzer = EmotionAnalyzer(backend='opencv') # Fast initialization
    mivolo_analyzer = MivoloAnalyzer()
    social_analyzer = SocialAnalyzer(fps=fps)
    scene_describer = SceneDescriber(log_file="scene_log.json")

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
                    # New track detected - try to match with gallery
                    best_match_id = -1
                    best_score = 0
                    
                    # Only try ReID if we can get an embedding (face available)
                    if det['faces']:
                        face_bbox = det['faces'][0]['bbox']
                        curr_emb = emotion_analyzer.get_embedding(frame, face_bbox)
                        
                        if curr_emb is not None:
                            for gid, data in gallery.items():
                                score = get_cosine_similarity(curr_emb, data['embedding'])
                                
                                # Use attributes for filtering if available
                                # (Wait until we have enough history for gender/age to be stable)
                                match = (score > 0.75) # threshold for Facenet
                                
                                if match and score > best_score:
                                    best_score = score
                                    best_match_id = gid
                    
                    if best_match_id != -1:
                        # Re-identified person from gallery
                        id_mapping[track_id] = best_match_id
                        print(f"I: Re-identified track {track_id} as original ID {best_match_id} (score: {best_score:.2f})")
                    else:
                        # Unknown person - assign new global ID
                        new_gid = track_id # Use the tracker's ID as the base for original ID
                        id_mapping[track_id] = new_gid
                        # We'll populate gallery data once attributes are stable
                
                # Apply the mapping
                det['track_id_original'] = id_mapping[track_id]
                # Update track_id for visual consistency if desired, or just use track_id_original for logging
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
                    'embeddings': [] # Collect a few embeddings to average
                }
            
            # --- Emotion/Age/Gender Analysis ---
            # Increased stride for 4K/heavy videos
            if frame_count % 15 == 0:
                # Prefer face detection box if available
                analysis_bbox = None
                if det['faces']:
                    analysis_bbox = det['faces'][0]['bbox']
                else:
                    analysis_bbox = det['bbox']
                
                # 1. Emotion (DeepFace)
                emotion_results = emotion_analyzer.analyze(frame, analysis_bbox)
                
                # 2. Age & Gender (MiVOLO)
                person_bbox = det['bbox']
                mivolo_results = mivolo_analyzer.analyze(frame, person_bbox, det['faces'][0]['bbox'] if det['faces'] else None)

                if orig_id != -1:
                    hist = person_history[orig_id]
                    
                    if emotion_results:
                        hist['emotion_history'].append(emotion_results['emotion'])
                        # Fallback for Age/Gender if MiVOLO fails
                        if not mivolo_results:
                            hist['age_history'].append(emotion_results['age'])
                            hist['gender_history'].append(emotion_results['gender'])
                    
                    if mivolo_results:
                        hist['age_history'].append(mivolo_results['age'])
                        hist['gender_history'].append(mivolo_results['gender'])

                    # --- Temporal Smoothing (Independent) ---
                    
                    # 1. Gender: High-consensus majority vote
                    if len(hist['gender_history']) > 0:
                        gender_counts = Counter(hist['gender_history'])
                        most_common_gender, count = gender_counts.most_common(1)[0]
                        if count >= 3: # Require at least 3 detections for stability
                             hist['stable']['gender'] = most_common_gender
                    
                    # 2. Age: Moving average (EMA-like)
                    if len(hist['age_history']) > 0:
                        hist['stable']['age'] = int(np.mean(hist['age_history']))
                        
                    # 3. Emotion: Hysteresis-based switching
                    if len(hist['emotion_history']) > 0:
                        emotion_counts = Counter(hist['emotion_history'])
                        most_common_emo, count = emotion_counts.most_common(1)[0]
                        
                        current_stable_emo = hist['stable'].get('emotion')
                        if current_stable_emo is None:
                            hist['stable']['emotion'] = most_common_emo
                        else:
                            # Requirement: New emotion must have at least 40% dominance to override
                            if count > (len(hist['emotion_history']) * 0.4) and most_common_emo != current_stable_emo:
                                hist['stable']['emotion'] = most_common_emo
                    
                    # Update Gallery with average embedding and attributes
                    if det['faces'] and len(hist['embeddings']) < 5:
                        face_bbox = det['faces'][0]['bbox']
                        emb = emotion_analyzer.get_embedding(frame, face_bbox)
                        if emb is not None:
                            hist['embeddings'].append(emb)
                            
                            # Update average embedding in gallery
                            avg_emb = np.mean(hist['embeddings'], axis=0)
                            gallery[orig_id] = {
                                'embedding': avg_emb,
                                'gender': hist['stable'].get('gender'),
                                'age': hist['stable'].get('age')
                            }

                    # Calculate Mood Trajectory (Sentiment Trend)
                    if len(hist['emotion_history']) >= 5:
                        emotion_map = {
                            'happy': 1.0, 'surprise': 0.5, 'neutral': 0.1,
                            'sad': -0.5, 'angry': -1.0, 'fear': -0.8, 'disgust': -1.0
                        }
                        scores = [emotion_map.get(e, 0) for e in hist['emotion_history']]
                        
                        # Compare first half vs second half trend
                        mid = len(scores) // 2
                        first_half = np.mean(scores[:mid])
                        second_half = np.mean(scores[mid:])
                        trend = second_half - first_half
                        
                        if trend > 0.15:
                            hist['stable']['mood_trend'] = "Improving ↗"
                        elif trend < -0.15:
                            hist['stable']['mood_trend'] = "Declining ↘"
                        else:
                            hist['stable']['mood_trend'] = "Stable →"
                
                # NOTE: We do NOT apply results to det immediately to prevent flickering.
                # We wait for the 'stable' update below.
                pass

            # Assign from stable history if available
            if orig_id != -1 and orig_id in person_history:
                det.update(person_history[orig_id]['stable'])
                # Override the displayed ID with the original one
                det['track_id_display'] = orig_id

        # 6. Social Interaction Analysis (STAS)
        # Use original IDs for social interaction to keep them consistent across cuts
        social_detections = []
        for d in detections:
            sd = d.copy()
            if 'track_id_original' in d:
                sd['track_id'] = d['track_id_original']
            social_detections.append(sd)
            
        interactions, person_statuses = social_analyzer.analyze(social_detections)
        
        # Update original detections with status info (posture, etc.)
        for det in detections:
            tid = det.get('track_id_original', det.get('track_id'))
            if tid in person_statuses:
                det.update(person_statuses[tid])

        # 7. Scene Description
        desc_text = scene_describer.describe(social_detections, frame_count, interactions)
        scene_describer.save_log(desc_text)

        # 8. Visualization
        drawn_frame = detector.draw(frame, detections, draw_skeleton=True, draw_faces=True)
        
        # Draw Interactions
        for inter in interactions:
            id1, id2 = inter['ids']
            label = inter['type']
            # Find center between the two people to place the label
            p1 = [d for d in detections if d.get('track_id_original', d.get('track_id')) == id1][0]
            p2 = [d for d in detections if d.get('track_id_original', d.get('track_id')) == id2][0]
            c1 = [(p1['bbox'][0]+p1['bbox'][2])/2, (p1['bbox'][1]+p1['bbox'][3])/2]
            c2 = [(p2['bbox'][0]+p2['bbox'][2])/2, (p2['bbox'][1]+p2['bbox'][3])/2]
            cx, cy = int((c1[0]+c2[0])/2), int((c1[1]+c2[1])/2)
            
            cv2.putText(drawn_frame, f"<< {label} >>", (cx - 50, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.line(drawn_frame, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), (0, 255, 0), 1)

        # Helper for drawing badge labels
        def draw_badge(img, text, pos, bg_color, text_color=(255, 255, 255)):
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            x, y = pos
            # Background rectangle
            cv2.rectangle(img, (x, y - h - 10), (x + w + 10, y + 5), bg_color, -1)
            # Text
            cv2.putText(img, text, (x + 5, y - 5), font, font_scale, text_color, thickness, cv2.LINE_AA)
            return h + 15 # Return height used to stack next badge

        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            
            # --- Draw Emotion & Role Badges ---
            curr_y = y1 - 10
            
            # 1. Role Badge
            role = det.get('role', 'Analyzing...')
            role_color = (255, 100, 0) if "Staff" in role else (100, 100, 100)
            curr_y -= draw_badge(drawn_frame, f"ROLE: {role}", (x1, curr_y), role_color)
            
            # 2. Emotion Badge
            emotion = det.get('emotion', 'N/A')
            emo_colors = {
                'happy': (0, 200, 0),        # Green
                'sad': (200, 0, 0),          # Blue
                'angry': (0, 0, 200),        # Red
                'surprise': (0, 200, 200),   # Yellow
                'fear': (150, 0, 150),       # Purple
                'neutral': (150, 150, 150),  # Gray
            }
            emo_color = emo_colors.get(emotion.lower(), (100, 100, 100))
            mood_trend = det.get('mood_trend', "")
            display_emo = f"EMO: {emotion} {mood_trend}"
            curr_y -= draw_badge(drawn_frame, display_emo, (x1, curr_y), emo_color)

            # 3. Secondary Info (Age/Gender) - Smaller/Subtle
            age = det.get('age', 'N/A')
            gender = det.get('gender', 'N/A')
            info_text = f"{gender} | Age: {age}"
            draw_badge(drawn_frame, info_text, (x1, curr_y), (50, 50, 50))

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
