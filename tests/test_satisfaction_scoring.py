import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cv_pipeline.social_interaction.satisfaction_analyzer import SatisfactionAnalyzer

def test_satisfaction():
    analyzer = SatisfactionAnalyzer(history_seconds=2, fps=30)
    
    print("Testing Satisfaction Scoring...")
    
    # Case 1: Neutral/Waiting (30 frames)
    print("Simulating Person 1 waiting with neutral emotions...")
    for _ in range(30):
        analyzer.update(
            track_id=1, 
            emotion_scores={'neutral': 80, 'happy': 10, 'sad': 10}, 
            interaction_type=None, 
            is_stationary=True
        )
    
    score_waiting = analyzer.compute_score(1)
    print(f"Score after waiting: {score_waiting:.2f}")
    
    # Case 2: Positive Interaction (30 frames)
    print("Simulating Person 1 talking and happy...")
    for _ in range(30):
        analyzer.update(
            track_id=1, 
            emotion_scores={'happy': 95, 'neutral': 5}, 
            interaction_type='Talking', 
            is_stationary=True
        )
    
    score_talking = analyzer.compute_score(1)
    print(f"Score after interaction: {score_talking:.2f}")
    
    assert score_talking > score_waiting, "Score should increase after positive interaction"
    
    # Case 3: Negative/Frustrated (30 frames)
    print("Simulating Person 2 angry and stationary...")
    for _ in range(30):
        analyzer.update(
            track_id=2, 
            emotion_scores={'angry': 90, 'sad': 10}, 
            interaction_type=None, 
            is_stationary=True
        )
    
    score_angry = analyzer.compute_score(2)
    print(f"Score for angry person: {score_angry:.2f}")
    
    assert score_angry < score_waiting, "Score for angry person should be lower than neutral waiting"
    
    print("\nALL SATISFACTION TESTS PASSED!")

if __name__ == "__main__":
    test_satisfaction()
