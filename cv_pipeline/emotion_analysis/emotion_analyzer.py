from deepface import DeepFace
import cv2
import numpy as np

class EmotionAnalyzer:
    def __init__(self, backend='opencv'):
        self.backend = backend
        # No heavy initialization needed for DeepFace as it loads models lazyly or globally
        
    def analyze(self, frame, bbox):
        """
        Analyze emotions for a cropped face.
        Args:
            frame: The full video frame
            bbox: Face bounding box (x1, y1, x2, y2)
        Returns:
            dominant_emotion: String (e.g. "happy", "score")
            scores: Dictionary with emotion scores
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w, c = frame.shape
        
        # Validations
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None, None
            
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None, None
            
        try:
            # DeepFace expects path or numpy array
            results = DeepFace.analyze(
                img_path=face_crop,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend=self.backend,
                silent=True
            )
            
            if not results:
                return None, None
                
            # DeepFace can return a list if multiple faces are found in the crop
            # Since we cropped one face, we take the first result
            res = results[0]
            return res['dominant_emotion'], res['emotion']
            
        except Exception as e:
            # print(f"Emotion analysis failed: {e}")
            return None, None
