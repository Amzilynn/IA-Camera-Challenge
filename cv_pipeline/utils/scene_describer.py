import datetime
import json
import os

class SceneDescriber:
    def __init__(self, log_file="scene_log.json"):
        self.log_file = log_file
        # Initialize/Clear log file
        with open(self.log_file, "w", encoding="utf-8") as f:
            # We'll use line-delimited JSON for robust logging
            pass

    def describe(self, detections, frame_idx, interactions=None):
        """
        Generate a data dictionary of the scene based on detections.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        frame_data = {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "persons": [],
            "interactions": []
        }
        
        # 1. Individual Status
        for det in detections:
            track_id = det.get('track_id', -1)
            if track_id == -1: continue
                
            person = {
                "id": track_id,
                "bbox": [round(float(x), 2) for x in det['bbox']],
                "attributes": {}
            }
            
            # Extract attributes
            attrs = person["attributes"]
            if 'emotion' in det and det['emotion']: attrs['emotion'] = det['emotion']
            if 'age' in det and det['age']: attrs['age'] = det['age']
            if 'gender' in det and det['gender']: attrs['gender'] = det['gender']
            if 'posture' in det: attrs['posture'] = det['posture']
            if 'activity' in det: attrs['activity'] = det['activity']
            if 'role' in det: attrs['role'] = det['role']
            if 'mood_trend' in det: attrs['mood_trend'] = det['mood_trend']
            if 'group_id' in det and det['group_id'] != -1: attrs['group_id'] = det['group_id']
            if det.get('space_violated'): attrs['space_violation'] = True
            
            if 'pose_keypoints' in det and det['pose_keypoints'] is not None:
                attrs['pose_tracked'] = True
                
            frame_data["persons"].append(person)
            
        # 2. Social Interactions
        if interactions:
            for inter in interactions:
                frame_data["interactions"].append({
                    "ids": inter['ids'],
                    "type": inter['type']
                })

        return frame_data

    def save_log(self, data):
        """
        Save the data dictionary to the log file in JSONL format.
        """
        if data:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data) + "\n")
            except Exception as e:
                print(f"E: Failed to save scene log: {e}")
