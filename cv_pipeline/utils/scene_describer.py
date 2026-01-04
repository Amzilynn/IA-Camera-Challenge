import datetime

class SceneDescriber:
    def __init__(self, log_file="scene_log.txt"):
        self.log_file = log_file
        # Initialize/Clear log file
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"Scene Log Started: {datetime.datetime.now()}\n")
            f.write("--------------------------------------------------\n")

    def describe(self, detections, frame_idx):
        """
        Generate a text simple description of the scene based on detections.
        """
        descriptions = []
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        for det in detections:
            track_id = det.get('track_id', -1)
            # Skip untracked objects if you only want persistent IDs
            if track_id == -1: 
                continue
                
            desc = f"[Frame {frame_idx}] ID {track_id}: "
            
            # Emotion
            if 'emotion' in det and det['emotion']:
                desc += f"Emotion={det['emotion']} "
            
            # Pose (simplified check)
            if 'pose_keypoints' in det and det['pose_keypoints'] is not None:
                # We could add logic here to determine "Standing", "Arms Up", etc.
                # For now, just indicate pose is tracked
                desc += "Pose=Tracked "
                
            descriptions.append(desc)
            
        if not descriptions:
            return None
            
        full_text = f"{timestamp} | " + " | ".join(descriptions)
        return full_text

    def save_log(self, text):
        if text:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(text + "\n")
