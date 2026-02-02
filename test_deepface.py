import cv2
from deepface import DeepFace
import numpy as np
import sys

# Set encoding for safe printing
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_deepface():
    print("Testing DeepFace...")
    # Create a small image
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    try:
        print("Starting DeepFace.analyze for emotion, age, gender...")
        results = DeepFace.analyze(
            img_path=dummy_img,
            actions=['emotion', 'age', 'gender'],
            enforce_detection=False,
            silent=False
        )
        print("Results received successfully.")
        print(results)
    except Exception as e:
        print("DeepFace analysis failed with an exception.")
        # Print only ASCII to avoid encoding issues
        print("Error message (safe):", str(e).encode('ascii', 'ignore').decode('ascii'))

if __name__ == "__main__":
    test_deepface()
