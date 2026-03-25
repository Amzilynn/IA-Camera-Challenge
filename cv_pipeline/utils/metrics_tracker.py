import numpy as np
from collections import deque

class HUDMetricsTracker:
    """
    Stabilizes and ensures consistency of HUD dashboard metrics.
    Uses windowed median filters and Exponential Moving Averages (EMA).
    Includes logic to bridge short gaps in detection.
    """
    def __init__(self, window_size=45, ema_alpha=0.05):
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        
        # History buffers for smoothing
        self.history = {
            'total_people': deque(maxlen=window_size),
            'staff_count': deque(maxlen=window_size),
            'active_engagements': deque(maxlen=window_size)
        }
        
        # Current smoothed state
        self.smoothed_metrics = {
            'total_people': 0,
            'staff_count': 0,
            'visitor_count': 0,
            'active_engagements': 0,
            'satisfaction_index': 75.0
        }
        
        # Persistence counter (to bridge gaps of up to 10 frames)
        self.gap_count = 0
        self.max_gap = 15

    def update(self, raw_metrics):
        """
        Update the tracker with new frame-level metrics.
        """
        total_in_frame = raw_metrics.get('total_people', 0)
        
        # 1. BRIDGE GAPS
        # If no one is detected, we might be in a temporary flicker
        if total_in_frame == 0 and self.smoothed_metrics['total_people'] > 0 and self.gap_count < self.max_gap:
            self.gap_count += 1
            # Return current state without updating history with "zeros"
            return self.smoothed_metrics.copy()
        
        # If we reached max gap or found someone, reset gap count
        if total_in_frame > 0:
            self.gap_count = 0
        
        # 2. UPDATE HISTORY
        for key in self.history:
            if key in raw_metrics:
                self.history[key].append(raw_metrics[key])
        
        # 3. SMOOTHING via Median Filter (for counts)
        self.smoothed_metrics['total_people'] = int(np.median(self.history['total_people'])) if self.history['total_people'] else 0
        self.smoothed_metrics['staff_count'] = int(np.median(self.history['staff_count'])) if self.history['staff_count'] else 0
        
        # Engagements: Use a slightly more "optimistic" window to avoid flicker
        # (Median of last available frames)
        if self.history['active_engagements']:
            self.smoothed_metrics['active_engagements'] = int(np.median(list(self.history['active_engagements'])))
        
        # 4. EMA Smoothing for Satisfaction Index
        # Only update satisfaction if people are actually present to avoid baseline drift
        if total_in_frame > 0:
            current_satisfaction = raw_metrics.get('satisfaction_index', 75.0)
            self.smoothed_metrics['satisfaction_index'] = (self.ema_alpha * current_satisfaction + 
                                                          (1 - self.ema_alpha) * self.smoothed_metrics['satisfaction_index'])
        
        # 5. ENFORCE INTERNAL CONSISTENCY
        if self.smoothed_metrics['staff_count'] > self.smoothed_metrics['total_people']:
            self.smoothed_metrics['staff_count'] = self.smoothed_metrics['total_people']
            
        self.smoothed_metrics['visitor_count'] = max(0, self.smoothed_metrics['total_people'] - self.smoothed_metrics['staff_count'])
        
        if self.smoothed_metrics['active_engagements'] > self.smoothed_metrics['total_people']:
            self.smoothed_metrics['active_engagements'] = self.smoothed_metrics['total_people']
            
        return self.smoothed_metrics.copy()
