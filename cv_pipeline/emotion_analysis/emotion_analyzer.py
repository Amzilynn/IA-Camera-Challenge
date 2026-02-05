from deepface import DeepFace
import cv2
import numpy as np

class EmotionAnalyzer:
    def __init__(self, backend='opencv'):
        self.backend = backend
        
    def analyze(self, frame, bbox=None):
        """
        Analyze emotions, age, and gender for a face or person crop.
        Args:
            frame: The full video frame
            bbox: Face or person bounding box (x1, y1, x2, y2). 
                  If None, DeepFace will try to find a face in the image (not recommended).
        Returns:
            dict: {
                'emotion': str,
                'age': int,
                'gender': str
            } or None
        """
        face_crop = frame
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            h, w, c = frame.shape
            
            # Validations
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
                
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                return None
            
        try:
            # DeepFace focuses ONLY on emotion now, MiVOLO handles age/gender
            results = DeepFace.analyze(
                img_path=face_crop,
                actions=['emotion'],
                enforce_detection=False, 
                detector_backend=self.backend,
                silent=True
            )
            
            if not results:
                return None
                
            res = results[0]
            
            return {
                'emotion': res['dominant_emotion']
            }
            
        except Exception as e:
            # print(f"DeepFace analysis failed: {e}")
            return None

    def get_embedding(self, frame, bbox):
        """
        Extract face embedding for ReID.
        """
        if bbox is None:
            return None
            
        x1, y1, x2, y2 = map(int, bbox)
        h, w, c = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        try:
            # Using Facenet for fast and robust embeddings
            embeddings = DeepFace.represent(
                img_path=face_crop,
                model_name="Facenet",
                enforce_detection=False,
                detector_backend=self.backend
            )
            if embeddings:
                return np.array(embeddings[0]["embedding"])
        except:
            pass
        return None
