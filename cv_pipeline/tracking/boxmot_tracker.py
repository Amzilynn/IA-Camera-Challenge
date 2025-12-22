import numpy as np
from boxmot import DeepOcSort  
from pathlib import Path
import cv2
import time

class PersonTracker:
    """
    Wrapper for BoxMOT tracker to assign persistent IDs to detected persons.
    """

    def __init__(self,
                 tracker_type='deepocsort',
                 reid_weights=Path('osnet_x0_25_msmt17.pt'),
                 device='cpu',
                 fp16=True,
                ):

        self.tracker_type = tracker_type.lower()
        self.device = device

        # Initialize tracker - updated to match required parameters
        if self.tracker_type == 'deepocsort':
            self.tracker = DeepOcSort(
                reid_weights=reid_weights,  # First required positional argument
                half=fp16,                  # Second required positional argument (renamed from fp16)
                device=device,
            )
        else:
            raise ValueError(
                f"Tracker '{tracker_type}' not supported. Use 'deepocsort'."
            )

        # Color palette for consistent track visualization
        self.id_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
            (0, 255, 128), (128, 0, 255), (0, 128, 255), (192, 192, 192), (128, 128, 128)
        ]
        
        # Track history for trajectories
        self.track_history = {}  # {track_id: [positions]}
        self.max_trajectory_len = 30  # Maximum length of trajectory history

        print(f"✓ PersonTracker initialized with {self.tracker_type.upper()} on {device}")

    def update(self, frame, detections):
        """
        Update tracker with new detections and return tracked objects with IDs.
        """
        if len(detections) == 0:
            # Handle empty detections case
            self.tracker.update(np.empty((0, 6)), frame)
            return detections

        # Convert detections to format expected by BoxMOT (xyxy, conf, class)
        dets_array = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['conf']
            class_id = 0  # Person class
            dets_array.append([x1, y1, x2, y2, conf, class_id])

        dets_array = np.array(dets_array, dtype=np.float32)

        # Update tracker with new detections
        tracks = self.tracker.update(dets_array, frame)

        # Match tracks to detections using IoU
        if tracks.shape[0] > 0:
            for det in detections:
                det_bbox = np.array(det['bbox'])

                best_idx = -1
                best_iou = 0

                for j, trk in enumerate(tracks):
                    track_bbox = trk[:4]
                    iou = self._compute_iou(det_bbox, track_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j

                if best_idx >= 0 and best_iou > 0.3:
                    # Assign track ID to detection
                    track_id = int(tracks[best_idx, 4])
                    det['track_id'] = track_id
                    det['track_color'] = self.id_colors[track_id % len(self.id_colors)]
                    
                    # Update track history for trajectory visualization
                    center_x = (det_bbox[0] + det_bbox[2]) / 2
                    center_y = (det_bbox[1] + det_bbox[3]) / 2
                    
                    if track_id not in self.track_history:
                        self.track_history[track_id] = []
                    
                    self.track_history[track_id].append((center_x, center_y))
                    
                    # Keep track history within max length
                    if len(self.track_history[track_id]) > self.max_trajectory_len:
                        self.track_history[track_id] = self.track_history[track_id][-self.max_trajectory_len:]
                    
                    # Add trajectory to detection
                    det['trajectory'] = self.track_history[track_id].copy()
                else:
                    # No matching track
                    det['track_id'] = -1
                    det['track_color'] = (128, 128, 128)
                    det['trajectory'] = []
        else:
            # No tracks
            for det in detections:
                det['track_id'] = -1
                det['track_color'] = (128, 128, 128)
                det['trajectory'] = []

        return detections

    def reset(self):
            """Reset tracker state."""
            self.tracker = DeepOcSort(
                reid_weights=Path('osnet_x0_25_msmt17.pt'),
                half=True,                # Required parameter (not "fp16")
                device=self.device
            )
            self.track_history = {}
            print("✓ Tracker reset")
    @staticmethod
    def _compute_iou(b1, b2):
        """
        Compute IoU between two bounding boxes.
        """
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)

        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

        union = area1 + area2 - inter

        if union <= 0:
            return 0

        return inter / union