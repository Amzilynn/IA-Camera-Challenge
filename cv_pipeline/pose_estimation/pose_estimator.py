import cv2
import mediapipe as mp
import numpy as np

class PoseEstimator:
    def __init__(self, mode=False, complexity=1, smooth_landmarks=True,
                 enable_segmentation=False, smooth_segmentation=True,
                 detection_confidence=0.5, tracking_confidence=0.5):
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=mode,
            model_complexity=complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

    def estimate(self, frame, bbox):
        """
        Estimate pose for a cropped person image.
        Args:
            frame: The full video frame
            bbox: Bounding box (x1, y1, x2, y2) of the person
        Returns:
            keypoints: List of (x, y, z, visibility) normalized to crop
            world_landmarks: 3D world landmarks
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w, c = frame.shape
        
        # Ensure bbox is within frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None, None

        person_crop = frame[y1:y2, x1:x2]
        crop_h, crop_w, _ = person_crop.shape
        
        # Convert to RGB for MediaPipe
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_crop)
        
        keypoints = []
        if results.pose_landmarks:
            # MediaPipe returns normalized coordinates [0, 1] relative to the image (crop)
            for lm in results.pose_landmarks.landmark:
                # Store normalized coordinates relative to the crop
                # To draw on full frame later: 
                # global_x = x1 + lm.x * crop_w
                # global_y = y1 + lm.y * crop_h
                keypoints.append({
                    'x': lm.x, 
                    'y': lm.y, 
                    'z': lm.z, 
                    'visibility': lm.visibility,
                    'global_x': x1 + lm.x * crop_w,
                    'global_y': y1 + lm.y * crop_h
                })
                
        return keypoints, results.pose_landmarks

    def draw_skeleton(self, frame, keypoints, color=(0, 255, 255)):
        """
        Draw skeleton on the full frame using calculated global coordinates.
        """
        if not keypoints:
            return frame
            
        # Draw connections (using mp_pose.POSE_CONNECTIONS)
        # We need to map the MediaPipe connection indices to our keypoints list
        
        # Define connections manually or map from mp_pose.POSE_CONNECTIONS
        connections = self.mp_pose.POSE_CONNECTIONS
        
        for start_idx, end_idx in connections:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                pt1 = keypoints[start_idx]
                pt2 = keypoints[end_idx]
                
                if pt1['visibility'] > 0.5 and pt2['visibility'] > 0.5:
                    x1, y1 = int(pt1['global_x']), int(pt1['global_y'])
                    x2, y2 = int(pt2['global_x']), int(pt2['global_y'])
                    cv2.line(frame, (x1, y1), (x2, y2), color, 2)
                    
        return frame
