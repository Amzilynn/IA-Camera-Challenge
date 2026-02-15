import numpy as np
import time
from collections import deque, defaultdict

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
            'service_counts': {},        # {(id1, id2): count}
            'personal_space_violations': {}, # {id: total_seconds}
            'emotion_trajectories': {}   # {id: [emotions]}
        }
        
        # New: Interaction Smoothing Buffer
        # {(id1, id2): deque([itype, itype, ...])}
        self.interaction_buffer = defaultdict(lambda: deque(maxlen=10)) 
        
        # New: Group management
        self.groups = [] # List of sets {id1, id2, ...}
        
        # New: Role discovery stats
        self.role_stats = {} # {id: {'total_distance': 0, 'unique_interactions': set(), 'time_in_scene': 0}}

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
        Refined Facing Direction: Prioritize Head Orientation (Nose/Eyes)
        Indices: 0: Nose, 1: L-Eye, 2: R-Eye, 5: L-Shoulder, 6: R-Shoulder
        """
        if kpts is None or len(kpts) < 7:
            return None
            
        # Try Head Vector first (Nose relative to mid-eye)
        nose = kpts[0]
        l_eye = kpts[1]
        r_eye = kpts[2]
        
        if nose[2] > 0.5 and l_eye[2] > 0.5 and r_eye[2] > 0.5:
            mid_eye = (l_eye[:2] + r_eye[:2]) / 2
            facing = nose[:2] - mid_eye
        else:
            # Fallback to Shoulder Vector
            left_sh = kpts[5][:2]
            right_sh = kpts[6][:2]
            sh_vec = right_sh - left_sh
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
                
                # Fetch pre-discovered roles for context
                role_a = self._discover_role(id1, current_time)
                role_b = self._discover_role(id2, current_time)
                
                interaction_type = self._detect_pair_interaction(det_a, det_b, role_a, role_b)
                
                # 1b. Smooth the interaction (Temporal Consensus)
                self.interaction_buffer[pair_ids].append(interaction_type)
                
                # Only "confirm" the interaction if it appears in > 70% of recent frames
                recent = [i for i in self.interaction_buffer[pair_ids] if i is not None]
                if recent:
                    from collections import Counter
                    most_common, count = Counter(recent).most_common(1)[0]
                    if count >= 7: # 7 out of 10 frames
                        state_key = (pair_ids[0], pair_ids[1], most_common)
                        current_pair_states.add(state_key)
                        found_interactions.append({'ids': pair_ids, 'type': most_common})

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

        # 3. Group Clustering (Connected Components of interactions)
        self.groups = self._cluster_groups(current_pair_states)
        
        # 4. Waiting Duration, Space Violations & Role Stats
        socially_engaged = {tid for (id1, id2, itype) in self.active_interactions if itype in ["Talking", "Service/Helping"] for tid in (id1, id2)}
        
        person_statuses = {}
        for det in tracked_dets:
            tid = det['track_id']
            is_static = self.is_stationary(tid)
            
            # Update role stats
            if tid not in self.role_stats:
                self.role_stats[tid] = {'total_distance': 0, 'unique_interactions': set(), 'start_time': current_time}
            
            # Track distance for role discovery
            hist = self.history[tid]
            if len(hist) > 1:
                dist = np.linalg.norm(hist[-1]['pos'] - hist[-2]['pos'])
                self.role_stats[tid]['total_distance'] += dist
            
            # Track space violations (Social Force lite)
            space_violation = self._check_space_violation(tid, tracked_dets)
            if space_violation:
                self.metrics['personal_space_violations'][tid] = self.metrics['personal_space_violations'].get(tid, 0) + self.dt
            
            # Update waiting duration metrics
            if is_static and tid not in socially_engaged:
                if tid not in self.active_waiting:
                    self.active_waiting[tid] = current_time
            else:
                if tid in self.active_waiting:
                    duration = current_time - self.active_waiting[tid]
                    self.metrics['waiting_durations'][tid] = self.metrics['waiting_durations'].get(tid, 0) + duration
                    del self.active_waiting[tid]
            
            # Determine Role (Heuristic)
            role = self._discover_role(tid, current_time)
            
            # Compile status for satisfaction analysis
            person_statuses[tid] = {
                'stationary': is_static,
                'engaged': tid in socially_engaged,
                'interaction_type': next((itype for (id1, id2, itype) in self.active_interactions if tid in (id1, id2)), None),
                'posture': self._detect_posture(det),
                'activity': self._detect_movement(tid),
                'space_violated': space_violation,
                'role': role,
                'group_id': self._get_group_id(tid)
            }

        return found_interactions, person_statuses

    def _detect_posture(self, det):
        """
        Detect posture (Standing, Sitting, Bending) based on keypoints and bbox aspect ratio.
        """
        bbox = det['bbox']
        kpts = det.get('pose_keypoints')
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        aspect_ratio = h / (w + 1e-6)

        if kpts is not None:
            # Indices: 11, 12: Hips, 15, 16: Ankles
            if len(kpts) > 16:
                hip_y = (kpts[11][1] + kpts[12][1]) / 2
                ankle_y = (kpts[15][1] + kpts[16][1]) / 2
                shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
                
                torso_h = hip_y - shoulder_y
                leg_h = ankle_y - hip_y
                
                if leg_h < 0.5 * torso_h and aspect_ratio < 1.4:
                    return "Sitting"
                elif hip_y > y1 + 0.7 * h: # Hips very low in box
                    return "Crouching"
                
        if aspect_ratio > 1.8:
            return "Standing"
        elif aspect_ratio < 0.8:
            return "Lying Down"
        elif aspect_ratio < 1.2:
            return "Sitting/Bending"
        
        return "Unknown"

    def _detect_movement(self, track_id):
        """
        Classify movement activity (Stationary, Walking, Running).
        """
        if track_id not in self.history or len(self.history[track_id]) < self.fps:
            return "Unknown"
            
        recent = list(self.history[track_id])[-self.fps:]
        speeds = []
        for k in range(1, len(recent)):
            d = np.linalg.norm(recent[k]['pos'] - recent[k-1]['pos'])
            speeds.append(d / self.dt)
            
        avg_speed = np.mean(speeds)
        
        if avg_speed < 15:
            return "Stationary"
        elif avg_speed < 60:
            return "Walking"
        else:
            return "Running"

    def is_stationary(self, track_id, threshold=10):
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

    def _detect_pair_interaction(self, det_a, det_b, role_a="Unknown", role_b="Unknown"):
        hist_a = self.history[det_a['track_id']]
        hist_b = self.history[det_b['track_id']]
        
        if len(hist_a) < 2 or len(hist_b) < 2: return None
        
        # Current data
        curr_a, curr_b = hist_a[-1], hist_b[-1]
        prev_a, prev_b = hist_a[-2], hist_b[-2]
        
        # 1. SPATIAL FEATURES (Perspective Aware)
        pos_a, pos_b = curr_a['pos'], curr_b['pos']
        dist = np.linalg.norm(pos_a - pos_b)
        
        # Get average height of the two people to scale thresholds
        h_a = curr_a['bbox'][3] - curr_a['bbox'][1]
        h_b = curr_b['bbox'][3] - curr_b['bbox'][1]
        avg_h = (h_a + h_b) / 2
        
        # Thresholds scaled by person height (Social context: 0.8h is close, 1.5h is social)
        PROXIMITY_THRES = avg_h * 0.8
        WALKING_THRES = avg_h * 1.2
        
        # Facing logic
        vector_a_to_b = pos_b - pos_a
        facing_a = curr_a['facing']
        facing_b = curr_b['facing']
        
        angle_a = self._calculate_angle(facing_a, vector_a_to_b) if facing_a is not None else 180
        angle_b = self._calculate_angle(facing_b, -vector_a_to_b) if facing_b is not None else 180
        
        facing_each_other = angle_a < 50 and angle_b < 50
        
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
        
        # Rule 1: Service/Helping (High Priority)
        # If one is Staff and they are facing each other while close
        if (role_a == "Staff" or role_b == "Staff") and dist < PROXIMITY_THRES * 1.5:
            if facing_each_other or (role_a == "Staff" and angle_b < 45) or (role_b == "Staff" and angle_a < 45):
                return "Service/Helping"

        # Rule 2: Talking (Facing + Close + Stationary)
        if dist < PROXIMITY_THRES and facing_each_other and avg_speed_a < 15:
            return "Talking"
            
        # Rule 3: Walking Together
        if dist < WALKING_THRES and speed_a > 20 and speed_b > 20:
            vel_sim = np.dot(vel_a, vel_b) / (speed_a * speed_b + 1e-6)
            if vel_sim > 0.85:
                return "Walking Together"

        # Rule 4: Approaching
        if approach_rate < -60 and dist > PROXIMITY_THRES and (angle_a < 45 or angle_b < 45):
            return "Approaching"
                
        # Rule 5: Physical Contact (Keypoint-based precision)
        kpts_a = det_a.get('pose_keypoints')
        kpts_b = det_b.get('pose_keypoints')
        if kpts_a is not None and kpts_b is not None:
            # Check if hands of A are near body of B
            hands_a = [kpts_a[9], kpts_a[10]] # Wrists
            body_b = kpts_b[5:13] # Torso keypoints
            for h in hands_a:
                if h[2] > 0.5:
                    for b in body_b:
                        if b[2] > 0.5 and np.linalg.norm(h[:2] - b[:2]) < 30:
                            return "Physical Contact"

        # Fallback to IoU Physical Contact
        if self._compute_iou(det_a['bbox'], det_b['bbox']) > 0.2:
            return "Physical Contact"
            
        return None

    def _cluster_groups(self, pair_states):
        """Find connected components of interacting people."""
        adj = defaultdict(set)
        for id1, id2, itype in pair_states:
            if itype in ["Talking", "Walking Together", "Physical Contact"]:
                adj[id1].add(id2)
                adj[id2].add(id1)
        
        groups = []
        visited = set()
        for node in adj:
            if node not in visited:
                group = set()
                stack = [node]
                while stack:
                    curr = stack.pop()
                    if curr not in visited:
                        visited.add(curr)
                        group.add(curr)
                        stack.extend(adj[curr] - visited)
                if len(group) > 1:
                    groups.append(group)
        return groups

    def _get_group_id(self, track_id):
        for i, group in enumerate(self.groups):
            if track_id in group:
                return i
        return -1

    def _check_space_violation(self, track_id, all_detections):
        """Social Force: Detect if someone is too deep in personal bubble."""
        if track_id not in self.history: return False
        pos_self = self.history[track_id][-1]['pos']
        
        for det in all_detections:
            tid = det['track_id']
            if tid == track_id: continue
            
            pos_other = self._get_center(det['bbox'])
            dist = np.linalg.norm(pos_self - pos_other)
            
            # Intimate space (< 50px) is a violation if not in same group
            if dist < 60:
                if self._get_group_id(track_id) == -1 or self._get_group_id(track_id) != self._get_group_id(tid):
                    return True
        return False

    def _discover_role(self, track_id, current_time):
        """Unsupervised Role Discovery based on mobility and Social Network centrality."""
        stats = self.role_stats.get(track_id)
        if not stats: return "Unknown"
        
        # Add current partner to interaction set
        for (id1, id2, itype) in self.active_interactions:
            if track_id == id1: stats['unique_interactions'].add(id2)
            if track_id == id2: stats['unique_interactions'].add(id1)
            
        time_elapsed = current_time - stats['start_time']
        
        # Heuristic: Staff move a lot and talk to many different people
        if time_elapsed > 30: # Need at least 30s of data
            mobility = stats['total_distance'] / time_elapsed
            social_reach = len(stats['unique_interactions'])
            
            if mobility > 40 and social_reach >= 2:
                return "Staff"
            elif mobility < 15 and social_reach <= 1:
                return "Visitor (Stationary)"
            else:
                return "Visitor (Mobile)"
                
        return "Analyzing..."
