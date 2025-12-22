import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO

class YOLODetector:
    """
    YOLOv8 detector for person detection in video frames.
    """
    
    def __init__(self, 
                 model_path='yolov8x.pt',  # Use 'x' size for better accuracy
                 device='cpu',            # Use 'cuda' or 'cpu'
                 conf_threshold=0.5,       # Confidence threshold
                 classes=[0],              # Default to detect only persons (class 0)
                ):
        
        self.conf_threshold = conf_threshold
        self.device = device
        self.classes = classes
        
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            print(f"✓ Loaded YOLOv8 model from {model_path} on {device}")
        except Exception as e:
            print(f"⚠️ Error loading YOLO model: {e}")
            print(f"Attempting to download default model...")
            self.model = YOLO("yolov8n.pt")  # Fallback to smaller model
            
        # For visualization
        self.class_names = self.model.names
    
    def detect(self, frame, verbose=False):
        """
        Detect persons in a frame and return their bounding boxes.
        
        Args:
            frame: Input frame
            verbose: Whether to print detection info
            
        Returns:
            List of dictionaries with detection information
        """
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Run YOLO inference
        start_time = time.time()
        results = self.model(frame, 
                             conf=self.conf_threshold, 
                             classes=self.classes,
                             verbose=False)
        
        # Process results
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i, box in enumerate(boxes):
                # Extract detection data
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # Only add person detections (or whatever classes we're looking for)
                if class_id in self.classes:
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'conf': conf,
                        'class_id': class_id,
                        'class_name': self.class_names[class_id],
                        'track_id': -1,  # Will be assigned by tracker
                        'keypoints': None  # Will be filled by pose estimator
                    }
                    detections.append(detection)
        
        if verbose:
            inference_time = (time.time() - start_time) * 1000
            print(f"Detection: {len(detections)} persons, {inference_time:.1f}ms")
            
        return detections
    
    def draw(self, frame, detections, draw_skeleton=True, draw_faces=False):
        """
        Draw detection boxes and optionally skeleton and face on the frame.
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            draw_skeleton: Whether to draw pose skeleton
            draw_faces: Whether to draw face bounding boxes
            
        Returns:
            Frame with drawings
        """
        output = frame.copy()
        
        for det in detections:
            # Extract data
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            conf = det['conf']
            track_id = det.get('track_id', -1)
            
            # Assign color by track ID if available, otherwise use default
            if track_id != -1:
                color = det.get('track_color', (0, 255, 0))
                
                # Draw tracking ID
                cv2.putText(output, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                color = (0, 255, 0)
                
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence
            cv2.putText(output, f"{conf:.2f}", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw skeleton if available and requested
            if draw_skeleton and 'keypoints' in det and det['keypoints'] is not None:
                # This will be implemented when we have a pose_estimator
                pass
                
            # Draw face box if available and requested
            if draw_faces and 'face_bbox' in det and det['face_bbox'] is not None:
                fx1, fy1, fx2, fy2 = [int(v) for v in det['face_bbox']]
                cv2.rectangle(output, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
        
        return output

    def draw_skeleton(self, frame, detection, pose_estimator):
        """
        Draw skeleton for a detection on the frame.
        
        Args:
            frame: Input frame
            detection: Detection dictionary with keypoints
            pose_estimator: MediaPipePoseEstimator instance
            
        Returns:
            Frame with skeleton drawn
        """
        if 'keypoints' not in detection or detection['keypoints'] is None:
            return frame
        
        return pose_estimator.draw_pose(frame, detection)