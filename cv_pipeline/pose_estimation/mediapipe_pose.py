import cv2
import numpy as np
import mediapipe as mp
import time

class MediaPipePoseEstimator:
    """
    MediaPipe BlazePose implementation for extracting body pose keypoints.
    """
    
    def __init__(self, 
                 model_complexity=1,     # 0, 1, or 2 (higher is more accurate but slower)
                 static_image_mode=False, # Set to False for video (tracking-based optimization)
                 smooth_landmarks=True,   # Temporal filter for smoother keypoints
                 min_detection_conf=0.5,  # Minimum confidence for detection
                 min_tracking_conf=0.5,   # Minimum confidence for tracking
                ):
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create pose estimator
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            static_image_mode=static_image_mode,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf
        )
        
        # Store parameters
        self.model_complexity = model_complexity
        print(f"✓ MediaPipe Pose initialized (complexity: {model_complexity})")
        
        # Define pose connections for visualization
        self.pose_connections = self.mp_pose.POSE_CONNECTIONS
        
    def process_frame(self, frame):
        """
        Process a frame to extract pose landmarks.
        
        Args:
            frame: RGB input frame
            
        Returns:
            Pose landmarks if detected, None otherwise
        """
        # Convert to RGB (MediaPipe requires RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        return results
    
    def estimate_pose(self, frame, detections):
        """
        Estimate pose for each detected person in the frame.
        
        Args:
            frame: Input BGR frame
            detections: List of detection dictionaries from YOLODetector
            
        Returns:
            Updated detections with pose keypoints added
        """
        height, width = frame.shape[:2]
        
        for det in detections:
            # Extract person bounding box
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            
            # Extract person region with margin
            margin = 10
            x1_crop = max(0, x1 - margin)
            y1_crop = max(0, y1 - margin)
            x2_crop = min(width, x2 + margin)
            y2_crop = min(height, y2 + margin)
            
            person_img = frame[y1_crop:y2_crop, x1_crop:x2_crop]
            
            if person_img.size == 0:
                continue
                
            # Process person image to get pose
            results = self.process_frame(person_img)
            
            if results.pose_landmarks:
                # Convert normalized landmarks to pixel coordinates in original frame
                keypoints = []
                
                for landmark in results.pose_landmarks.landmark:
                    # Convert from relative coordinates in the crop to absolute in the full frame
                    px = int(landmark.x * (x2_crop - x1_crop) + x1_crop)
                    py = int(landmark.y * (y2_crop - y1_crop) + y1_crop)
                    pz = landmark.z  # Keep depth as is
                    visibility = landmark.visibility
                    
                    keypoints.append({
                        'x': px,
                        'y': py,
                        'z': pz,
                        'visibility': visibility
                    })
                
                # Add keypoints to detection
                det['keypoints'] = keypoints
                
                # Extract face bounding box (approximate from facial landmarks)
                if len(keypoints) >= 11:  # Need at least facial landmarks
                    face_points = [keypoints[i] for i in range(11)]
                    face_xs = [p['x'] for p in face_points]
                    face_ys = [p['y'] for p in face_points]
                    
                    if face_xs and face_ys:  # If face points exist
                        face_x1 = max(0, min(face_xs) - 10)
                        face_y1 = max(0, min(face_ys) - 10)
                        face_x2 = min(width, max(face_xs) + 10)
                        face_y2 = min(height, max(face_ys) + 10)
                        
                        det['face_bbox'] = [face_x1, face_y1, face_x2, face_y2]
        
        return detections
    
    def draw_pose(self, image, detection):
        """
        Draw pose skeleton on an image.
        
        Args:
            image: Input BGR image
            detection: Single detection dictionary with keypoints
            
        Returns:
            Image with pose drawn
        """
        if 'keypoints' not in detection or detection['keypoints'] is None:
            return image
            
        keypoints = detection['keypoints']
        track_id = detection.get('track_id', -1)
        
        # Get color based on track ID or use default
        if track_id != -1:
            color = detection.get('track_color', (0, 255, 0))
        else:
            color = (0, 255, 0)
            
        # Define connections for skeleton drawing
        connections = [
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            # Torso
            (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            # Legs
            (11, 23), (12, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32)
        ]
        
        # Draw connections
        for conn in connections:
            start_idx, end_idx = conn
            
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_point = (keypoints[start_idx]['x'], keypoints[start_idx]['y'])
                end_point = (keypoints[end_idx]['x'], keypoints[end_idx]['y'])
                
                # Only draw if both points are visible
                if keypoints[start_idx]['visibility'] > 0.5 and keypoints[end_idx]['visibility'] > 0.5:
                    cv2.line(image, start_point, end_point, color, 2)
        
        # Draw keypoints
        for point in keypoints:
            if point['visibility'] > 0.5:
                cv2.circle(image, (point['x'], point['y']), 4, color, -1)
                
        return image