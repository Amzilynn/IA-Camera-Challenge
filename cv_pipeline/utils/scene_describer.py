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
                desc += f"Emotion={det['emotion']} "
            if 'pose_keypoints' in det and det['pose_keypoints'] is not None:
                desc += "Pose=Tracked "
            descriptions.append(desc)
            
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
