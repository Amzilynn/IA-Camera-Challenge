from deepface import DeepFace
import cv2
import numpy as np
import os

def test_emotion():
    print("I: Initializing DeepFace test...")
    # Create a dummy face-like pattern or just a gray square
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.putText(dummy_img, "FACE", (50, 112), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    try:
        results = DeepFace.analyze(
            img_path=dummy_img,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='skip',
            silent=False
        )
        if results:
            print(f"I: Success! Dominant emotion: {results[0]['dominant_emotion']}")
        else:
            print("W: DeepFace returned no results.")
    except Exception as e:
        print(f"E: DeepFace analysis failed: {e}")

if __name__ == "__main__":
    test_emotion()
