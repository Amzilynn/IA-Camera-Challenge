import time
from collections import deque
import numpy as np

class SatisfactionAnalyzer:
    """
    Combines emotion, behavioral, and interaction metrics into a Satisfaction Score.
    Score range: 0 to 100.
    """
    def __init__(self, history_seconds=10, fps=30):
        self.fps = fps
        self.history_len = int(history_seconds * fps)
        
        # history[track_id] = deque of {timestamp, emotion_scores, interaction_type, is_stationary}
        self.history = {}
        self.current_scores = {}

    def update(self, track_id, emotion_scores, interaction_type, is_stationary):
        """
        Update the history for a specific person.
        emotion_scores: dict from DeepFace (e.g. {'happy': 90, 'sad': 2, ...}) or None
        interaction_type: string (e.g. 'Talking', 'Approaching') or None
        is_stationary: bool
        """
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.history_len)
            
        self.history[track_id].append({
            'time': time.time(),
            'emotions': emotion_scores,
            'interaction': interaction_type,
            'stationary': is_stationary
        })

    def compute_score(self, track_id):
        """
        Calculate satisfaction score for a track ID based on historical window.
        """
        if track_id not in self.history or not self.history[track_id]:
            return 50.0 # Neutral starting point

        hist = list(self.history[track_id])
        
        # 1. EMOTION COMPONENT (40%)
        emotion_score = self._get_emotion_score(hist)
        
        # 2. ENGAGEMENT COMPONENT (30%)
        # Positive: Talking, Service/Helping, Walking Together
        # Negative: Disengaged (Stationary for long without social contact)
        engagement_score = self._get_engagement_score(hist)
        
        # 3. WAITING/EFFICIENCY COMPONENT (30%)
        # Penalize if stationary but not socially engaged
        wait_score = self._get_efficiency_score(hist)
        
        # Final weighted sum
        final_score = (
            (emotion_score * 0.4) +
            (engagement_score * 0.3) +
            (wait_score * 0.3)
        )
        
        self.current_scores[track_id] = np.clip(final_score, 0, 100)
        return self.current_scores[track_id]

    def _get_emotion_score(self, hist):
        valid_emotions = [h['emotions'] for h in hist if h['emotions'] is not None]
        if not valid_emotions:
            return 50.0  # Neutral
        
        # Weighted average of emotions across the window
        # Positive: happy (1.0), surprise (0.5), neutral (0.2)
        # Negative: angry (-1.0), sad (-0.8), fear (-0.6), disgust (-1.0)
        total_e_score = 0
        for e in valid_emotions:
            p = (e.get('happy', 0) * 1.0 + e.get('surprise', 0) * 0.5 + e.get('neutral', 0) * 0.2)
            n = (e.get('angry', 0) * 1.0 + e.get('sad', 0) * 0.8 + e.get('fear', 0) * 0.6 + e.get('disgust', 0) * 1.0)
            # Map diff to 0-100
            diff = p - n
            total_e_score += (diff + 100) / 2 # Shift -100..100 to 0..100
            
        return total_e_score / len(valid_emotions)

    def _get_engagement_score(self, hist):
        social_types = ["Talking", "Service/Helping", "Walking Together"]
        engagement_count = sum(1 for h in hist if h['interaction'] in social_types)
        # Ratio of time spent engaged
        ratio = engagement_count / len(hist)
        return ratio * 100

    def _get_efficiency_score(self, hist):
        # penalize being stationary without interaction
        bad_wait_count = sum(1 for h in hist if h['stationary'] and h['interaction'] is None)
        ratio = bad_wait_count / len(hist)
        return (1.0 - ratio) * 100 # 100 is good (no waiting), 0 is bad (all waiting)

    def get_all_scores(self):
        return self.current_scores
