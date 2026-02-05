import numpy as np
from boxmot import BoostTrack
from pathlib import Path


class PersonTracker:
    """
    Wrapper for BoxMOT tracker to assign persistent IDs to detected persons.
    """

    def __init__(self,
                 tracker_type='deepocsort',
                 reid_weights=Path('osnet_x0_25_msmt17.pt'),
                 device='cuda',
                 fp16=True,
):

        self.tracker_type = tracker_type.lower()
        self.device = device

        import logging
        # Suppress boxmot/BoostTrack verbose logging
        logging.getLogger("boxmot").setLevel(logging.ERROR)

        if self.tracker_type == 'deepocsort':
            self.tracker = BoostTrack(
                reid_weights=reid_weights,
                device=device,
                half=fp16,
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

        print(f"I: PersonTracker initialized with {self.tracker_type.upper()} on {device}")

    def update(self, frame, detections):
        """
        Update tracker with new detections and return tracked objects with IDs.
        """

        if len(detections) == 0:
            self.tracker.update(np.empty((0, 6)), frame)
            return detections

        dets_array = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['conf']
            class_id = 0
            dets_array.append([x1, y1, x2, y2, conf, class_id])

        dets_array = np.array(dets_array, dtype=np.float32)

        tracks = self.tracker.update(dets_array, frame)

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
                    track_id = int(tracks[best_idx, 4])
                    det['track_id'] = track_id
                    det['track_color'] = self.id_colors[track_id % len(self.id_colors)]
                    
                    # Store features if available in the tracker
                    # BoxMOT trackers usually store features in trk[...] if configured
                    # but here we might need to rely on the fact that BoostTrack uses ReID internally.
                    # As a fallback, we'll signal that this track is 'active'
                else:
                    det['track_id'] = -1
                    det['track_color'] = (128, 128, 128)
        else:
            for det in detections:
                det['track_id'] = -1
                det['track_color'] = (128, 128, 128)

        return detections

    def reset(self):
        """Reset tracker state."""
        self.tracker = BoostTrack(
            reid_weights=Path('osnet_x0_25_msmt17.pt'),
            device=self.device,
            half=True
        )
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
