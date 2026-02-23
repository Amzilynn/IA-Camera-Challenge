import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cv_pipeline.social_interaction.social_analyzer import SocialAnalyzer

def test_metrics():
    # Mock parameters
    fps = 30
    analyzer = SocialAnalyzer(fps=fps, history_seconds=5)
    
    print("Testing Approach...")
    # Step 1: People approaching (30 frames = 1s)
    # ID 1 moves from 100 to 130. ID 2 moves from 1000 to 970.
    # Total distance decreases from 900 to 840.
    # Approach rate = (840 - 900) / (1/30) = -60 / 0.033 = -1800.
    for i in range(30):
        dets = [
            {
                'track_id': 1, 
                'bbox': [100 + i, 100, 150 + i, 200], 
                'pose_keypoints': np.array([[0,0,0]]*5 + [[125+i, 160, 0.9], [125+i, 140, 0.9]]) # Facing Right (+x)
            },
            {
                'track_id': 2, 
                'bbox': [1000 - i, 100, 1050 - i, 200], 
                'pose_keypoints': np.array([[0,0,0]]*5 + [[1025-i, 140, 0.9], [1025-i, 160, 0.9]]) # Facing Left (-x)
            }
        ]
        analyzer.analyze(dets)
        time.sleep(0.01)
    
    # Step 2: People talking (Stationary and Close)
    print("Testing Talking...")
    for i in range(60): # 2 seconds
        dets = [
            {
                'track_id': 1, 
                'bbox': [470, 100, 520, 200], 
                'pose_keypoints': np.array([[0,0,0]]*5 + [[495, 160, 0.9], [495, 140, 0.9]]) # Facing Right
            },
            {
                'track_id': 2, 
                'bbox': [530, 100, 580, 200], 
                'pose_keypoints': np.array([[0,0,0]]*5 + [[555, 140, 0.9], [555, 160, 0.9]]) # Facing Left
            }
        ]
        # Centers: (495, 150) and (555, 150). Dist = 60.
        analyzer.analyze(dets)
        time.sleep(0.01)

    # Step 3: End interactions
    print("Ending interactions...")
    for i in range(10):
        dets = [
            {'track_id': 1, 'bbox': [0, 0, 50, 50], 'pose_keypoints': None},
            {'track_id': 2, 'bbox': [2000, 2000, 2050, 2050], 'pose_keypoints': None}
        ]
        analyzer.analyze(dets)
    
    metrics = analyzer.get_metrics()
    print("\nMetrics results:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Assertions
    talking_key = (1, 2, "Talking")
    approaching_key = (1, 2, "Approaching")
    
    assert talking_key in metrics['interaction_durations'], "Talking interaction not recorded"
    assert metrics['interaction_durations'][talking_key] > 0, "Talking duration should be > 0"
    
    print("\nTesting Waiting Duration...")
    # Step 4: Person 3 waiting (stationary for 1.5s)
    # Need enough history for _is_stationary (requires 1s)
    for i in range(60): 
        dets = [
            {'track_id': 3, 'bbox': [800, 800, 850, 900], 'pose_keypoints': None}
        ]
        analyzer.analyze(dets)
        time.sleep(0.01)
    
    # Move person 3 to end waiting
    for i in range(10):
        dets = [
            {'track_id': 3, 'bbox': [800 + i*20, 800, 850 + i*20, 900], 'pose_keypoints': None}
        ]
        analyzer.analyze(dets)

    metrics = analyzer.get_metrics()
    print(f"Waiting duration for ID 3: {metrics['waiting_durations'].get(3, 0)}")
    assert metrics['waiting_durations'].get(3, 0) > 0, "Waiting duration for ID 3 should be > 0"

    print("\nALL TESTS PASSED!")

if __name__ == "__main__":
    test_metrics()
