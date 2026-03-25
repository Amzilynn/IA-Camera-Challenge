import numpy as np
from collections import defaultdict

class StabilityEvaluator:
    """
    Measures the technical quality of the pipeline outputs without manual video checking.
    """
    def __init__(self):
        self.track_data = defaultdict(list) # {id: [keypoints]}
        self.id_switches = 0
        self.frame_count = 0

    def add_frame_data(self, detections):
        self.frame_count += 1
        for det in detections:
            tid = det.get('track_id_display', det.get('track_id'))
            if tid == -1: continue
            
            if 'pose_keypoints' in det and det['pose_keypoints'] is not None:
                self.track_data[tid].append(det['pose_keypoints'])

    def calculate_metrics(self):
        results = {}
        
        # 1. Pose Jitter (Lower is Better)
        # Low variance (stable keypoints) vs high variance (shaking skeleton)
        jitters = []
        for tid, kpts_list in self.track_data.items():
            if len(kpts_list) < 10: continue
            
            # Convert list of arrays to 3D array [N, 17, 3]
            arr = np.array(kpts_list)
            # Calculate variance per keypoint across time, then average
            var = np.var(arr[:, :, :2], axis=0) # variance of x,y
            avg_jitter = np.mean(np.sqrt(var)) 
            jitters.append(avg_jitter)
            
        results['avg_pose_jitter_pixels'] = np.mean(jitters) if jitters else 0
        
        # 2. Tracking Continuity
        # How many frames does each ID last on average?
        lifespans = [len(v) for v in self.track_data.values()]
        results['avg_id_longevity_frames'] = np.mean(lifespans) if lifespans else 0
        results['unique_ids_count'] = len(self.track_data)
        
        return results

if __name__ == "__main__":
    print("This evaluator is designed to be integrated into run_full_pipeline.py")
    print("Example Usage:")
    print("evaluator = StabilityEvaluator()")
    print("evaluator.add_frame_data(detections)")
    print("print(evaluator.calculate_metrics())")
