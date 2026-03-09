from deepface import DeepFace
import cv2
import numpy as np
try:
    from hsemotion.facial_emotions import HSEmotionRecognizer
    HSEMOTION_AVAILABLE = True
except ImportError:
    HSEMOTION_AVAILABLE = False

class EmotionAnalyzer:
    def __init__(self, backend='hsemotion'):
        self.backend = backend
        self.fer_model = None
        
        if HSEMOTION_AVAILABLE:
            try:
                # Using a balanced model: enet_b2_8 for 8 emotions
                self.fer_model = HSEmotionRecognizer(model_name='enet_b2_8', device='cuda')
                print("I: HSEmotion (SOTA) initialized successfully on GPU.")
            except Exception as e:
                try:
                    self.fer_model = HSEmotionRecognizer(model_name='enet_b2_8', device='cpu')
                    print("I: HSEmotion initialized successfully on CPU.")
                except Exception as e2:
                    print(f"W: HSEmotion initialization failed. Falling back to DeepFace. Error1: {e}, Error2: {e2}")
                    self.fer_model = None
        else:
            print("W: HSEmotion not installed. Falling back to DeepFace.")
        
    def analyze(self, frame, bbox=None):
        """
        Analyze emotions, age, and gender using SOTA HSEmotion.
        """
        if bbox is None:
            return None
            
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None
            
        try:
            # 1. SOTA Emotion Prediction
            if self.fer_model:
                emotion, scores = self.fer_model.predict_emotions(face_crop, logits=False)
            else:
                # Fallback to DeepFace for emotion
                res = DeepFace.analyze(img_path=face_crop, actions=['emotion'], 
                                       enforce_detection=False, detector_backend='skip', silent=True)[0]
                emotion = res['dominant_emotion']
            
            # 2. Age/Gender via DeepFace
            results = DeepFace.analyze(
                img_path=face_crop,
                actions=['age', 'gender'],
                enforce_detection=False, 
                detector_backend='skip',
                silent=True
            )
            
            age = 25
            gender = "unknown"
            if results:
                res = results[0]
                age = int(res['age'])
                gender = "male" if res['dominant_gender'].lower() in ["man", "male"] else "female"
            
            return {
                'emotion': emotion,
                'age': age,
                'gender': gender
            }
            
        except Exception as e:
            return None

    def get_embedding(self, frame, bbox):
        """
        Extract face embedding for ReID.
        """
        if bbox is None:
            return None
            
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
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
                detector_backend='skip'
            )
            if embeddings:
                return np.array(embeddings[0]["embedding"])
        except:
            pass
        return None
