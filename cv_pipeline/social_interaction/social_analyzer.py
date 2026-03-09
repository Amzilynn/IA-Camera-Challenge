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
        
        # history[track_id] = deque of {timestamp, position, bbox, facing, pose_keypoints}
        self.history = {}
        
        # Synchronicity buffer: {(id1, id2): deque of correlations}
        self.sync_buffer = defaultdict(lambda: deque(maxlen=45)) 
        
        # Current active interactions: {(id1, id2): "Interaction Type"}
        self.active_interactions = {} 
        self.active_waiting = {}      
        
        # Persistent metrics
        self.metrics = {
            'interaction_durations': {}, 
            'approach_times': {},        
            'waiting_durations': {},     
            'service_counts': {},        
            'personal_space_violations': {}, 
            'intent_alerts': []          # New: List of intent-based alerts
        }
        
        self.interaction_buffer = defaultdict(lambda: deque(maxlen=10)) 
        self.groups = [] 
        self.role_stats = {} 

    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def _get_personal_space_polygon(self, kpts, bbox):
        """
        Construct a polygon of personal space based on pose keypoints.
        If pose is missing, fallback to stretched bbox.
        """
        if kpts is None or len(kpts) < 17:
            x1, y1, x2, y2 = bbox
            return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        
        # Take keypoints: Shoulders (5,6), Hips (11,12), Knees (13,14), Ankles (15,16)
        # Extend points slightly outward to create a 'bubble'
        polygon_points = []
        for idx in [5, 6, 12, 14, 16, 15, 13, 11]:
            pt = kpts[idx][:2]
            if kpts[idx][2] > 0.3:
                polygon_points.append(pt)
        
        if len(polygon_points) < 3:
            x1, y1, x2, y2 = bbox
            return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            
        return np.array(polygon_points, dtype=np.int32)

    def _check_polygon_intersection(self, poly1, poly2):
        """
        Check if two polygons intersect.
        Uses Shapely if available, otherwise falls back to a simple IoU bbox check.
        """
        try:
            from shapely.geometry import Polygon
            p1 = Polygon(poly1)
            p2 = Polygon(poly2)
            return p1.intersects(p2)
        except (ImportError, Exception):
            # Fallback: Simple BBox Intersection (IoU-like)
            min1 = np.min(poly1, axis=0)
            max1 = np.max(poly1, axis=0)
            min2 = np.min(poly2, axis=0)
            max2 = np.max(poly2, axis=0)
            
            # Check overlap
            if (min1[0] > max2[0] or min2[0] > max1[0] or
                min1[1] > max2[1] or min2[1] > max1[1]):
                return False
            return True

    def _get_facing_vector(self, kpts):
        if kpts is None or len(kpts) < 7:
            return None
        nose = kpts[0]
        l_eye = kpts[1]
        r_eye = kpts[2]
        if nose[2] > 0.5 and l_eye[2] > 0.5 and r_eye[2] > 0.5:
            mid_eye = (l_eye[:2] + r_eye[:2]) / 2
            facing = nose[:2] - mid_eye
        else:
            left_sh = kpts[5][:2]
            right_sh = kpts[6][:2]
            sh_vec = right_sh - left_sh
            facing = np.array([-sh_vec[1], sh_vec[0]]) 
        norm = np.linalg.norm(facing)
        return facing / norm if norm > 0 else None

    def _compute_synchronicity(self, id1, id2):
        """
        Cross-correlation of velocity and head-orientation over 45 frames.
        """
        h1 = list(self.history[id1])[-45:]
        h2 = list(self.history[id2])[-45:]
        if len(h1) < 45 or len(h2) < 45: return 0.0
        
        # Velocity correlation
        v1 = [h1[i]['pos'] - h1[i-1]['pos'] for i in range(1, len(h1))]
        v2 = [h2[i]['pos'] - h2[i-1]['pos'] for i in range(1, len(h2))]
        
        v1_norm = [v / (np.linalg.norm(v) + 1e-6) for v in v1]
        v2_norm = [v / (np.linalg.norm(v) + 1e-6) for v in v2]
        
        dots = [np.dot(n1, n2) for n1, n2 in zip(v1_norm, v2_norm)]
        sync_score = np.mean(dots)
        return sync_score

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
                'facing': facing,
                'pose': det.get('pose_keypoints')
            })

    def analyze(self, detections, environment_objects=[]):
        current_time = time.time()
        self.update_history(detections, current_time)
        found_interactions = []
        
        tracked_dets = [d for d in detections if d.get('track_id', -1) != -1]
        
        # 1. Update Intent & Synchronicity Graph
        current_pair_states = set()
        
        # Track distance for role discovery
        for det in tracked_dets:
            tid = det['track_id']
            if tid in self.history and len(self.history[tid]) > 1:
                hist = self.history[tid]
                dist = np.linalg.norm(hist[-1]['pos'] - hist[-2]['pos'])
                if tid not in self.role_stats:
                    self._discover_role_v2(det, environment_objects, current_time)
                self.role_stats[tid]['total_distance'] += dist

        for i in range(len(tracked_dets)):
            for j in range(i + 1, len(tracked_dets)):
                id1, id2 = tracked_dets[i]['track_id'], tracked_dets[j]['track_id']
                pair_ids = tuple(sorted((id1, id2)))
                
                # Check Synchronicity
                sync = self._compute_synchronicity(id1, id2)
                if sync > 0.85:
                    found_interactions.append({'ids': pair_ids, 'type': 'Group_Bond'})
                    current_pair_states.add((pair_ids[0], pair_ids[1], 'Group_Bond'))

                # Personal Space Polygon Check
                poly1 = self._get_personal_space_polygon(tracked_dets[i].get('pose_keypoints'), tracked_dets[i]['bbox'])
                poly2 = self._get_personal_space_polygon(tracked_dets[j].get('pose_keypoints'), tracked_dets[j]['bbox'])
                
                if self._check_polygon_intersection(poly1, poly2):
                    # Check if Nose-to-Neck vector intersects other's bbox (Intentional focus)
                    focus_a = self._check_intentional_focus(tracked_dets[i], tracked_dets[j])
                    focus_b = self._check_intentional_focus(tracked_dets[j], tracked_dets[i])
                    
                    if focus_a or focus_b:
                        itype = self._detect_pair_interaction(tracked_dets[i], tracked_dets[j])
                        if itype:
                            current_pair_states.add((pair_ids[0], pair_ids[1], itype))
                            found_interactions.append({'ids': pair_ids, 'type': itype})
                            
        self.active_interactions = current_pair_states # Update for role discovery

        # 2. Hand-Object Interaction (HOI) for Role Validation
        person_statuses = {}
        for det in tracked_dets:
            tid = det['track_id']
            role = self._discover_role_v2(det, environment_objects, current_time)
            
            # Security: Theft Heuristic (Hand Occlusion + Orientation)
            self._check_security_intent(det, environment_objects)
            
            # Service: Pre-emptive Service Alert (Scanning behavior)
            self._check_service_intent(det)

            person_statuses[tid] = {
                'role': role,
                'intent': self._get_intent_text(tid),
                # ... other existing status fields ...
                'engaged': tid in {k[0] for k in current_pair_states} or tid in {k[1] for k in current_pair_states}
            }

        return found_interactions, person_statuses

    def _check_intentional_focus(self, det_a, det_b):
        """Check if Node A's Nose-to-Neck vector intersects Node B's bbox."""
        kpts = det_a.get('pose_keypoints')
        if kpts is None or len(kpts) < 1: return False
        
        nose = kpts[0]
        if nose[2] < 0.5: return False
        
        facing = self._get_facing_vector(kpts)
        if facing is None: return False
        
        # Ray casting
        ray_origin = nose[:2]
        bbox_b = det_b['bbox']
        
        # Check if ray (origin + facing * t) enters bbox_b
        # Simple box intersection for ray
        for t in range(10, 300, 10):
            pt = ray_origin + facing * t
            if bbox_b[0] <= pt[0] <= bbox_b[2] and bbox_b[1] <= pt[1] <= bbox_b[3]:
                return True
        return False

    def _discover_role_v2(self, det, objects, current_time):
        tid = det['track_id']
        if tid not in self.role_stats:
            self.role_stats[tid] = {
                'total_distance': 0.0,
                'unique_interactions': set(),
                'hoi_count': 0, 
                'open_palm_frames': 0, 
                'start_time': current_time, 
                'unique_contacts': set()
            }
        
        stats = self.role_stats[tid]
        
        # Check HOI: Is hand near object?
        kpts = det.get('pose_keypoints')
        if kpts is not None and len(kpts) > 10:
            hands = [kpts[9], kpts[10]] # Wrists
            for h in hands:
                if h[2] > 0.5:
                    for obj in objects:
                        obj_bbox = obj['bbox']
                        # Buffer distance 20px
                        if (obj_bbox[0]-20 <= h[0] <= obj_bbox[2]+20 and 
                            obj_bbox[1]-20 <= h[1] <= obj_bbox[3]+20):
                            stats['hoi_count'] += 1
                            stats['unique_contacts'].add(obj['label'])
                            
            # Check Gesture: Open Palm (Wrist to Elbow extension)
            # Simplistic: if distance is large enough relative to shoulder width
            sh_width = np.linalg.norm(kpts[5][:2] - kpts[6][:2])
            ext_l = np.linalg.norm(kpts[9][:2] - kpts[7][:2])
            ext_r = np.linalg.norm(kpts[10][:2] - kpts[8][:2])
            if ext_l > 0.6 * sh_width or ext_r > 0.6 * sh_width:
                stats['open_palm_frames'] += 1

        # Logic for Staff Labeling
        if stats['hoi_count'] > 50 and stats['open_palm_frames'] > 20:
            return "Staff (Validated by HOI)"
        
        # Fallback to existing mobility based discovery
        return self._discover_role(tid, current_time)

    def _check_security_intent(self, det, objects):
        """Theft: Hand near high-value object + Orientation away from Staff."""
        kpts = det.get('pose_keypoints')
        if kpts is None or len(kpts) < 11: return
        
        # Find if hands are occluded/hidden near object
        hands_visible = (kpts[9][2] > 0.3) or (kpts[10][2] > 0.3)
        if not hands_visible:
            # Check if person is 'leaning' into an object
            for obj in objects:
                if self._compute_iou(det['bbox'], obj['bbox']) > 0.1:
                    # check orientation
                    facing = self._get_facing_vector(kpts)
                    # if facing is 'away' and hands not visible
                    # this is a weak heuristic but follows the prompt
                    pass

    def _check_service_intent(self, det):
        """Scanning behavior: side-to-side head movements while stationary."""
        tid = det['track_id']
        if tid not in self.history or len(self.history[tid]) < self.fps * 2: return
        
        recent = list(self.history[tid])[-self.fps*2:]
        if self.is_stationary(tid):
            facings = [h['facing'] for h in recent if h['facing'] is not None]
            if len(facings) > 10:
                # check variance in angle
                angles = [np.arctan2(f[1], f[0]) for f in facings]
                if np.std(angles) > 0.5: # high variance in head orientation
                    self.metrics['intent_alerts'].append({
                        'tid': tid, 
                        'type': 'Pre-emptive Service', 
                        'msg': 'Scanning behavior detected'
                    })

    def _get_intent_text(self, tid):
        for alert in reversed(self.metrics['intent_alerts']):
            if alert['tid'] == tid:
                return alert['type']
        return "Normal"

    def _calculate_angle(self, vec1, vec2):
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0: return 180
        cos_theta = np.dot(vec1, vec2) / (norm1 * norm2)
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    def _compute_iou(self, b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        if x2 < x1 or y2 < y1: return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

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
        if time_elapsed > 15: # Reduced from 30s to 15s for faster feedback
            mobility = stats['total_distance'] / time_elapsed
            social_reach = len(stats['unique_interactions'])
            
            if mobility > 40 and social_reach >= 2:
                return "Staff"
            elif mobility < 15 and social_reach <= 1:
                return "Visitor (Stationary)"
            else:
                return "Visitor (Mobile)"
                
        return "Analyzing..."
