import numpy as np
import time
from collections import deque

class SocialAnalyzer:
    """
    STAS: Spatial-Temporal Interaction Mode
    Analyzes relationships between tracked persons over time.
    """
    def __init__(self, fps=30, history_seconds=5):
        self.fps = int(fps) if fps > 0 else 30
        self.dt = 1.0 / self.fps
        self.history_len = int(self.fps * history_seconds)
        
        # history[track_id] = deque of {timestamp, position, bbox, facing_vector}
        self.history = {}
        
        # Current active interactions: {(id1, id2): "Interaction Type"}
        # We use sorted tuple as key to ensure (A, B) and (B, A) are treated same
        self.active_interactions = {} # {(id1, id2, type): start_time}
        self.active_waiting = {}      # {id: start_time}
        
        # Persistent metrics
        self.metrics = {
            'interaction_durations': {}, # {(id1, id2, type): total_seconds}
            'approach_times': {},        # {(id1, id2): [durations]}
            'waiting_durations': {},     # {id: total_seconds}
            'service_counts': {}         # {(id1, id2): count}
        }

    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def _compute_iou(self, b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        if x2 < x1 or y2 < y1: return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def _get_facing_vector(self, kpts):
        """
        Estimate facing direction using shoulder points (indices 5, 6).
        """
        if kpts is None or len(kpts) <= 6:
            return None
            
        left_sh = kpts[5][:2]  # [x, y]
        right_sh = kpts[6][:2]
        conf_l = kpts[5][2]
        conf_r = kpts[6][2]
        
        if conf_l < 0.5 or conf_r < 0.5:
            return None
            
        # Shoulder vector (L -> R)
        sh_vec = right_sh - left_sh
        # Perpendicular vector (Facing out from chest)
        # Assuming camera is front-ish, facing is approximately normal to shoulders
        facing = np.array([-sh_vec[1], sh_vec[0]]) 
        norm = np.linalg.norm(facing)
        return facing / norm if norm > 0 else None

    def _calculate_angle(self, vec1, vec2):
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0: return 180
        cos_theta = np.dot(vec1, vec2) / (norm1 * norm2)
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    def update_history(self, detections, current_time):
        for det in detections:
            track_id = det.get('track_id', -1)
            if track_id == -1: continue
            
            if track_id not in self.history:
                self.history[track_id] = deque(maxlen=self.history_len)
            
            pos = self._get_center(det['bbox'])
            facing = self._get_facing_vector(det.get('pose_keypoints'))
            
            self.history[track_id].append({
                'time': current_time,
                'pos': pos,
                'bbox': det['bbox'],
                'facing': facing
            })

        # Cleanup old tracks
        active_ids = {det.get('track_id') for det in detections}
        self.history = {tid: h for tid, h in self.history.items() if tid in active_ids or len(h) > 0}

    def analyze(self, detections):
        current_time = time.time()
        self.update_history(detections, current_time)
        found_interactions = []
        
        tracked_dets = [d for d in detections if d.get('track_id', -1) != -1]
        active_ids = {d['track_id'] for d in tracked_dets}
        
        # 1. Detect current pair interactions
        current_pair_states = set()
        for i in range(len(tracked_dets)):
            for j in range(i + 1, len(tracked_dets)):
                det_a, det_b = tracked_dets[i], tracked_dets[j]
                id1, id2 = det_a['track_id'], det_b['track_id']
                pair_ids = tuple(sorted((id1, id2)))
                
                interaction_type = self._detect_pair_interaction(det_a, det_b)
                if interaction_type:
                    state_key = (pair_ids[0], pair_ids[1], interaction_type)
                    current_pair_states.add(state_key)
                    found_interactions.append({'ids': pair_ids, 'type': interaction_type})

        # 2. Update interaction durations and counts
        # Handle ended interactions
        ended_interactions = [k for k in self.active_interactions if k not in current_pair_states]
        for key in ended_interactions:
            duration = current_time - self.active_interactions[key]
            self.metrics['interaction_durations'][key] = self.metrics['interaction_durations'].get(key, 0) + duration
            
            # Specific logic for Approach Time
            if key[2] == "Approaching":
                pair = (key[0], key[1])
                if pair not in self.metrics['approach_times']: self.metrics['approach_times'][pair] = []
                self.metrics['approach_times'][pair].append(duration)
                
            del self.active_interactions[key]

        # Handle new interactions
        for key in current_pair_states:
            if key not in self.active_interactions:
                self.active_interactions[key] = current_time
                # Increment service frequency
                if key[2] == "Service/Helping":
                    pair = (key[0], key[1])
                    self.metrics['service_counts'][pair] = self.metrics['service_counts'].get(pair, 0) + 1

        # 3. Waiting Duration Logic
        # A person is "waiting" if stationary AND not in a social interaction (Talking/Service)
        socially_engaged = {tid for (id1, id2, itype) in self.active_interactions if itype in ["Talking", "Service/Helping"] for tid in (id1, id2)}
        
        for det in tracked_dets:
            tid = det['track_id']
            is_stationary = self._is_stationary(tid)
            
            if is_stationary and tid not in socially_engaged:
                if tid not in self.active_waiting:
                    self.active_waiting[tid] = current_time
            else:
                if tid in self.active_waiting:
                    duration = current_time - self.active_waiting[tid]
                    self.metrics['waiting_durations'][tid] = self.metrics['waiting_durations'].get(tid, 0) + duration
                    del self.active_waiting[tid]

        return found_interactions

    def _is_stationary(self, track_id, threshold=10):
        if track_id not in self.history or len(self.history[track_id]) < self.fps:
            return False
        recent = list(self.history[track_id])[-self.fps:]
        # Calculate average speed over last 1s
        speeds = []
        for k in range(1, len(recent)):
            d = np.linalg.norm(recent[k]['pos'] - recent[k-1]['pos'])
            speeds.append(d / self.dt)
        return np.mean(speeds) < threshold

    def get_metrics(self):
        """Returns a snapshot of accumulated metrics."""
        return self.metrics

    def _detect_pair_interaction(self, det_a, det_b):
        hist_a = self.history[det_a['track_id']]
        hist_b = self.history[det_b['track_id']]
        
        if len(hist_a) < 2 or len(hist_b) < 2: return None
        
        # Current data
        curr_a, curr_b = hist_a[-1], hist_b[-1]
        prev_a, prev_b = hist_a[-2], hist_b[-2]
        
        # 1. SPATIAL FEATURES
        pos_a, pos_b = curr_a['pos'], curr_b['pos']
        dist = np.linalg.norm(pos_a - pos_b)
        
        # Facing logic
        vector_a_to_b = pos_b - pos_a
        facing_a = curr_a['facing']
        facing_b = curr_b['facing']
        
        angle_a = self._calculate_angle(facing_a, vector_a_to_b) if facing_a is not None else 180
        angle_b = self._calculate_angle(facing_b, -vector_a_to_b) if facing_b is not None else 180
        
        facing_each_other = angle_a < 45 and angle_b < 45
        
        # 2. TEMPORAL FEATURES
        vel_a = (pos_a - prev_a['pos']) / self.dt
        vel_b = (pos_b - prev_b['pos']) / self.dt
        speed_a = np.linalg.norm(vel_a)
        speed_b = np.linalg.norm(vel_b)
        
        dist_prev = np.linalg.norm(prev_a['pos'] - prev_b['pos'])
        approach_rate = (dist - dist_prev) / self.dt
        
        # Stationary check (avg speed over last 1s)
        recent_a = list(hist_a)[-self.fps:]
        avg_speed_a = np.mean([np.linalg.norm(recent_a[k]['pos'] - recent_a[k-1]['pos'])/self.dt for k in range(1, len(recent_a))]) if len(recent_a)>1 else speed_a
        
        # 3. HEURISTICS
        
        # Rule 1: Talking
        if dist < 150 and facing_each_other and avg_speed_a < 10:
            return "Talking"
            
        # Rule 2: Approaching
        if approach_rate < -50 and (angle_a < 60 or angle_b < 60):
            return "Approaching"
            
        # Rule 3: Walking Together
        if dist < 200 and speed_a > 20 and speed_b > 20:
            vel_sim = np.dot(vel_a, vel_b) / (speed_a * speed_b + 1e-6)
            if vel_sim > 0.8:
                return "Walking Together"
                
        # Rule 4: Physical Contact
        iou = self._compute_iou(det_a['bbox'], det_b['bbox'])
        if iou > 0.1:
            return "Physical Contact"
            
        # Rule 5: Service (Heuristic: one stationary, one close/facing)
        if dist < 180 and (avg_speed_a < 5) and facing_each_other:
            return "Service/Helping"

        return None
