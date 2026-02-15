import datetime

class SceneDescriber:
    def __init__(self, log_file="scene_log.txt"):
        self.log_file = log_file
        # Initialize/Clear log file
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"Scene Log Started: {datetime.datetime.now()}\n")
            f.write("--------------------------------------------------\n")

    def describe(self, detections, frame_idx, interactions=None):
        """
        Generate a text simple description of the scene based on detections.
        """
        descriptions = []
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # 1. Individual Status
        for det in detections:
            track_id = det.get('track_id', -1)
            if track_id == -1: continue
                
            desc = f"ID {track_id}: "
            if 'emotion' in det and det['emotion']:
                desc += f"FER={det['emotion']} "
            if 'age' in det and det['age']:
                desc += f"Age={det['age']} "
            if 'gender' in det and det['gender']:
                desc += f"Gen={det['gender']} "
            
            # New attributes from social analyzer
            if 'posture' in det:
                desc += f"Pos={det['posture']} "
            if 'activity' in det:
                desc += f"Act={det['activity']} "
            
            # Spatial Position (Normalized % of frame)
            x1, y1, x2, y2 = det['bbox']
            cx, cy = (x1+x2)/2, (y1+y2)/2
            # Assuming we don't have frame width/height here, 
            # we can just use raw or pass them. 
            # For now, let's keep it simple as the user might just want the posture.
            
            if 'pose_keypoints' in det and det['pose_keypoints'] is not None:
                desc += "Pose=Tracked "
            descriptions.append(desc.strip())
            
        # 2. Social Interactions
        if interactions:
            for inter in interactions:
                ids = inter['ids']
                itype = inter['type']
                descriptions.append(f"Interaction({ids[0]}&{ids[1]}): {itype}")

        if not descriptions:
            return None
            
        full_text = f"[{frame_idx}] {timestamp} | " + " | ".join(descriptions)
        return full_text

    def save_log(self, text):
        if text:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(text + "\n")
