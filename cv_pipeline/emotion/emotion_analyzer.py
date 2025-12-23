import cv2
import numpy as np
from deepface import DeepFace
import time
from collections import deque, defaultdict

class EmotionAnalyzer:
    """
    Emotion analysis using DeepFace with temporal smoothing and tracking.
    """
    
    def __init__(self, 
                 detector_backend='opencv',
                 enforce_detection=False,
                 history_length=30):
        """
        Initialize emotion analyzer.
        
        Args:
            detector_backend: Backend for face detection ('opencv', 'retinaface', 'mtcnn', 'ssd')
            enforce_detection: If True, raises error when no face is detected
            history_length: Number of frames to keep for emotion tracking history
        """
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection
        self.history_length = history_length
        
        # Track emotion history per person ID
        self.emotion_history = defaultdict(lambda: deque(maxlen=history_length))
        
        # Emotion colors for visualization (BGR format)
        self.emotion_colors = {
            'angry': (0, 0, 255),      # Red
            'disgust': (0, 128, 0),    # Dark Green
            'fear': (128, 0, 128),     # Purple
            'happy': (0, 255, 255),    # Yellow
            'sad': (255, 0, 0),        # Blue
            'surprise': (255, 165, 0), # Orange
            'neutral': (192, 192, 192) # Gray
        }
        
        print(f"✓ EmotionAnalyzer initialized with {detector_backend} backend")
    
    def analyze_face(self, face_img):
        """
        Analyze emotion from a single face image.
        
        Args:
            face_img: Cropped face image (BGR format)
            
        Returns:
            Dictionary with emotion analysis results
        """
        if face_img is None or face_img.size == 0:
            return None
        
        try:
            # Ensure minimum face size
            if face_img.shape[0] < 48 or face_img.shape[1] < 48:
                # Resize to minimum required size
                face_img = cv2.resize(face_img, (48, 48))
            
            # Analyze emotion using DeepFace
            result = DeepFace.analyze(
                face_img,
                actions=['emotion'],
                detector_backend=self.detector_backend,
                enforce_detection=self.enforce_detection,
                silent=True
            )
            
            # Handle both list and dict returns
            if isinstance(result, list):
                result = result[0]
            
            return {
                'dominant_emotion': result['dominant_emotion'],
                'emotion_scores': result['emotion'],
                'region': result.get('region', None)
            }
            
        except Exception as e:
            # Silently handle detection failures
            return None
    
    def update_emotions(self, frame, detections):
        """
        Update emotion analysis for all detected persons.
        
        Args:
            frame: Input BGR frame
            detections: List of detection dictionaries
            
        Returns:
            Updated detections with emotion data
        """
        height, width = frame.shape[:2]
        
        for det in detections:
            # Skip if no face bbox
            if 'face_bbox' not in det or det['face_bbox'] is None:
                continue
            
            # Extract face region
            fx1, fy1, fx2, fy2 = [int(v) for v in det['face_bbox']]
            
            # Ensure valid coordinates
            fx1 = max(0, fx1)
            fy1 = max(0, fy1)
            fx2 = min(width, fx2)
            fy2 = min(height, fy2)
            
            if fx2 <= fx1 or fy2 <= fy1:
                continue
            
            # Crop face with some padding
            padding = 10
            fx1_pad = max(0, fx1 - padding)
            fy1_pad = max(0, fy1 - padding)
            fx2_pad = min(width, fx2 + padding)
            fy2_pad = min(height, fy2 + padding)
            
            face_img = frame[fy1_pad:fy2_pad, fx1_pad:fx2_pad]
            
            if face_img.size == 0:
                continue
            
            # Analyze emotion
            emotion_result = self.analyze_face(face_img)
            
            if emotion_result is not None:
                track_id = det.get('track_id', -1)
                
                # Add to detection
                det['emotion'] = emotion_result['dominant_emotion']
                det['emotion_scores'] = emotion_result['emotion_scores']
                
                # Update history if tracked
                if track_id != -1:
                    self.emotion_history[track_id].append({
                        'emotion': emotion_result['dominant_emotion'],
                        'scores': emotion_result['emotion_scores'],
                        'timestamp': time.time()
                    })
                    
                    # Add smoothed emotion (most common in recent history)
                    det['emotion_smoothed'] = self._get_smoothed_emotion(track_id)
                else:
                    det['emotion_smoothed'] = emotion_result['dominant_emotion']
        
        return detections
    
    def _get_smoothed_emotion(self, track_id):
        """
        Get smoothed emotion based on recent history.
        
        Args:
            track_id: ID of the tracked person
            
        Returns:
            Most common emotion in recent history
        """
        if track_id not in self.emotion_history or len(self.emotion_history[track_id]) == 0:
            return 'neutral'
        
        # Count emotions in history
        emotion_counts = defaultdict(int)
        for entry in self.emotion_history[track_id]:
            emotion_counts[entry['emotion']] += 1
        
        # Return most common emotion
        return max(emotion_counts.items(), key=lambda x: x[1])[0]
    
    def get_emotion_timeline(self, track_id):
        """
        Get emotion timeline for a specific track.
        
        Args:
            track_id: ID of the tracked person
            
        Returns:
            List of emotion entries with timestamps
        """
        if track_id not in self.emotion_history:
            return []
        
        return list(self.emotion_history[track_id])
    
    def draw_emotion(self, frame, detection):
        """
        Draw emotion information on the frame.
        
        Args:
            frame: Input frame
            detection: Detection dictionary with emotion data
            
        Returns:
            Frame with emotion visualization
        """
        if 'emotion' not in detection or detection['emotion'] is None:
            return frame
        
        output = frame.copy()
        
        # Get data
        emotion = detection.get('emotion_smoothed', detection['emotion'])
        emotion_scores = detection.get('emotion_scores', {})
        track_id = detection.get('track_id', -1)
        face_bbox = detection.get('face_bbox')
        
        if face_bbox is None:
            return output
        
        fx1, fy1, fx2, fy2 = [int(v) for v in face_bbox]
        
        # Get emotion color
        color = self.emotion_colors.get(emotion, (255, 255, 255))
        
        # Draw face box with emotion color
        cv2.rectangle(output, (fx1, fy1), (fx2, fy2), color, 2)
        
        # Draw emotion label
        label = f"{emotion.upper()}"
        if emotion_scores:
            confidence = emotion_scores.get(emotion, 0)
            label += f" {confidence:.1f}%"
        
        # Background for text
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(output, (fx1, fy1 - text_h - 10), (fx1 + text_w + 10, fy1), color, -1)
        cv2.putText(output, label, (fx1 + 5, fy1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw emotion bar chart if we have scores
        if emotion_scores and track_id != -1:
            self._draw_emotion_chart(output, emotion_scores, fx2 + 10, fy1)
        
        return output
    
    def _draw_emotion_chart(self, frame, emotion_scores, x, y):
        """
        Draw a small emotion bar chart.
        
        Args:
            frame: Input frame
            emotion_scores: Dictionary of emotion scores
            x, y: Top-left position for the chart
        """
        bar_width = 100
        bar_height = 15
        margin = 2
        
        # Sort emotions by score
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for i, (emotion, score) in enumerate(sorted_emotions):
            y_pos = y + i * (bar_height + margin)
            
            # Check if position is within frame
            if y_pos + bar_height >= frame.shape[0]:
                break
            
            # Draw background
            cv2.rectangle(frame, (x, y_pos), (x + bar_width, y_pos + bar_height), (50, 50, 50), -1)
            
            # Draw filled bar
            filled_width = int((score / 100) * bar_width)
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            cv2.rectangle(frame, (x, y_pos), (x + filled_width, y_pos + bar_height), color, -1)
            
            # Draw emotion label
            cv2.putText(frame, f"{emotion[:3]}", (x + 2, y_pos + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

