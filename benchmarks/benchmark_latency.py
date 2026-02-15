import time
import torch
import numpy as np
import cv2
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
os.chdir(project_root) # Ensure relative paths to 'models/' work
sys.path.insert(0, str(project_root))

from cv_pipeline.detection.yolo_detector import YOLODetector
from cv_pipeline.emotion_analysis.emotion_analyzer import EmotionAnalyzer
from cv_pipeline.emotion_analysis.mivolo_analyzer import MivoloAnalyzer

def benchmark_latency():
    print("=== Pipeline Component Benchmark ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. Initialize models
    print("\n[1/4] Initializing Models...")
    start = time.time()
    detector = YOLODetector()
    emotion = EmotionAnalyzer()
    mivolo = MivoloAnalyzer()
    print(f"Initialization took: {time.time() - start:.2f}s")

    # Create dummy data
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    person_bbox = [100, 100, 400, 800]
    face_bbox = [200, 150, 300, 250]
    
    iters = 10
    
    # 2. Benchmark YOLO Detection
    print("\n[2/4] Benchmarking YOLO Detection (Human + Pose)...")
    latencies = []
    for _ in range(iters):
        t0 = time.time()
        _ = detector.detect(frame)
        latencies.append(time.time() - t0)
    print(f"Avg Latency: {np.mean(latencies)*1000:.2f}ms | FPS: {1/np.mean(latencies):.1f}")

    # 3. Benchmark Emotion Analysis (DeepFace)
    print("\n[3/4] Benchmarking Emotion Analysis (DeepFace)...")
    latencies = []
    crop = frame[100:400, 100:400]
    for _ in range(iters):
        t0 = time.time()
        _ = emotion.analyze(frame, person_bbox)
        latencies.append(time.time() - t0)
    print(f"Avg Latency: {np.mean(latencies)*1000:.2f}ms")

    # 4. Benchmark MiVOLO (Age/Gender)
    print("\n[4/4] Benchmarking MiVOLO (Age/Gender)...")
    latencies = []
    for _ in range(iters):
        t0 = time.time()
        _ = mivolo.analyze(frame, person_bbox, face_bbox)
        latencies.append(time.time() - t0)
    print(f"Avg Latency: {np.mean(latencies)*1000:.2f}ms")

    print("\n" + "="*35)
    print("Recommendation: If total latency > 100ms, increase frame skipping (stride) in your main loop.")

if __name__ == "__main__":
    benchmark_latency()
