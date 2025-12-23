import cv2
import numpy as np
import mediapipe as mp
import time

class MediaPipePoseEstimator:
    """
    MediaPipe BlazePose implementation with enhanced skeleton visualization.
    """
    
    def __init__(self, 
                 model_complexity=1,
                 static_image_mode=False,
                 smooth_landmarks=True,
                 min_detection_conf=0.5,
                 min_tracking_conf=0.5):
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            static_image_mode=static_image_mode,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf
        )
        
        self.model_complexity = model_complexity
        print(f"✓ MediaPipe Pose initialized (complexity: {model_complexity})")
        
        self.pose_connections = self.mp_pose.POSE_CONNECTIONS
        
    def estimate_pose(self, frame, detections):
        """Estimate pose for each detected person."""
        height, width = frame.shape[:2]
        
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            
            margin = 10
            x1_crop = max(0, x1 - margin)
            y1_crop = max(0, y1 - margin)
            x2_crop = min(width, x2 + margin)
            y2_crop = min(height, y2 + margin)
            
            person_img = frame[y1_crop:y2_crop, x1_crop:x2_crop]
            
            if person_img.size == 0:
                continue
                
            # Convert to RGB
            rgb_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_img)
            
            if results.pose_landmarks:
                keypoints = []
                
                for landmark in results.pose_landmarks.landmark:
                    px = int(landmark.x * (x2_crop - x1_crop) + x1_crop)
                    py = int(landmark.y * (y2_crop - y1_crop) + y1_crop)
                    pz = landmark.z
                    visibility = landmark.visibility
                    
                    keypoints.append({
                        'x': px,
                        'y': py,
                        'z': pz,
                        'visibility': visibility
                    })
                
                det['keypoints'] = keypoints
                
                # Extract face bbox from facial landmarks
                if len(keypoints) >= 11:
                    face_points = [keypoints[i] for i in range(11)]
                    face_xs = [p['x'] for p in face_points if p['visibility'] > 0.5]
                    face_ys = [p['y'] for p in face_points if p['visibility'] > 0.5]
                    
                    if face_xs and face_ys:
                        face_x1 = max(0, min(face_xs) - 20)
                        face_y1 = max(0, min(face_ys) - 30)
                        face_x2 = min(width, max(face_xs) + 20)
                        face_y2 = min(height, max(face_ys) + 10)
                        
                        det['face_bbox'] = [face_x1, face_y1, face_x2, face_y2]
        
        return detections
    
    def draw_pose(self, image, detection):
        """Draw enhanced pose skeleton with better visualization."""
        if 'keypoints' not in detection or detection['keypoints'] is None:
            return image
            
        keypoints = detection['keypoints']
        track_id = detection.get('track_id', -1)
        
        # Get color based on track ID
        if track_id != -1:
            base_color = detection.get('track_color', (0, 255, 0))
        else:
            base_color = (0, 255, 0)
        
        # Define body part connections with colors
        connections = {
            'face': [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10)],
            'torso': [(11, 12), (11, 23), (12, 24), (23, 24)],
            'arms': [(11, 13), (13, 15), (12, 14), (14, 16), (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22)],
            'legs': [(23, 25), (25, 27), (27, 29), (27, 31), (24, 26), (26, 28), (28, 30), (28, 32)]
        }
        
        # Draw connections with varying thickness
        for part, conns in connections.items():
            if part == 'face':
                thickness = 1
                alpha = 0.5
            elif part == 'torso':
                thickness = 3
                alpha = 1.0
            else:
                thickness = 2
                alpha = 0.8
            
            for start_idx, end_idx in conns:
                if start_idx < len(keypoints) and end_idx < len(keypoints):
                    start_kp = keypoints[start_idx]
                    end_kp = keypoints[end_idx]
                    
                    if start_kp['visibility'] > 0.5 and end_kp['visibility'] > 0.5:
                        start_point = (start_kp['x'], start_kp['y'])
                        end_point = (end_kp['x'], end_kp['y'])
                        
                        # Apply alpha blending for semi-transparent lines
                        overlay = image.copy()
                        cv2.line(overlay, start_point, end_point, base_color, thickness)
                        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # Draw keypoints with different sizes
        key_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]  # Important joints
        
        for i, point in enumerate(keypoints):
            if point['visibility'] > 0.5:
                radius = 5 if i in key_joints else 3
                cv2.circle(image, (point['x'], point['y']), radius, base_color, -1)
                cv2.circle(image, (point['x'], point['y']), radius + 1, (255, 255, 255), 1)
                
        return image

