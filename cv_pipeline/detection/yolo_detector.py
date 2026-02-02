from ultralytics import YOLO
import cv2
import numpy as np
import torch

# Configuration constants

MIN_HEIGHT_RATIO = 0.2
CONF_HUMAN = 0.35
CONF_POSE = 0.30
CONF_FACE = 0.40
IOU_HUMAN = 0.45
IMG_SIZE = 800

# YOLO Pose skeleton connections - BODY ONLY (excluding face keypoints)
# 0: nose, 1-2: eyes, 3-4: ears, 5-6: shoulders, 7-8: elbows, 
# 9-10: wrists, 11-12: hips, 13-14: knees, 15-16: ankles
POSE_CONNECTIONS = [
    # Body skeleton only (no face connections for stability)
    (5, 6),          # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10), # right arm
    (5, 11), (6, 12),# shoulders to hips
    (11, 12),        # hips
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16)  # right leg
]

class YOLODetector:
    def __init__(self,
                 human_model_path="yolov8m.pt",
                 pose_model_path="yolov8m-pose.pt",
                 face_model_path="C:/Users/Dr.console/Desktop/IA-Camera-Challenge/cv_pipeline/models/yolov8n-face.pt"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.human_model = YOLO(human_model_path)
        self.pose_model = YOLO(pose_model_path)

        try:
            self.face_model = YOLO(face_model_path)
        except:
            print("I: YOLO face model not found. Using DeepFace fallback for analysis.")
            self.face_model = None

    @staticmethod
    def _crop_region(frame, bbox, margin=0.1):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        mx = int(bw * margin)
        my = int(bh * margin)
        cx1 = int(max(0, x1 - mx))
        cy1 = int(max(0, y1 - my))
        cx2 = int(min(w, x2 + mx))
        cy2 = int(min(h, y2 + my))
        return frame[cy1:cy2, cx1:cx2], (cx1, cy1)

    def detect(self, frame):
        h, w = frame.shape[:2]

        # 1️⃣ HUMAN DETECTION (GPU FORCED)
        human_results = self.human_model(
            frame,
            verbose=False,
            imgsz=640,
            device=self.device,
            conf=CONF_HUMAN,
            iou=IOU_HUMAN
        )[0]

        detections = []

        if human_results.boxes:
            for box in human_results.boxes:
                if int(box.cls[0]) != 0: 
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                if (y2 - y1) < MIN_HEIGHT_RATIO * h:
                    continue

                det = {
                    "bbox": (x1, y1, x2, y2),
                    "conf": float(box.conf[0]),
                    "pose_keypoints": None,
                    "faces": []
                }

                # 2️⃣ POSE ESTIMATION (GPU FORCED)
                human_crop, offset = self._crop_region(frame, det["bbox"], margin=0.05)
                if human_crop.size > 0:
                    pose_res = self.pose_model(
                        human_crop,
                        verbose=False,
                        imgsz=192,
                        device=self.device,
                        conf=0.25
                    )[0]

                    if len(pose_res.boxes) > 0 and pose_res.keypoints is not None:
                        kpts_xy = pose_res.keypoints.xy[0].cpu().numpy()

                        if pose_res.keypoints.conf is not None:
                            kpts_conf = pose_res.keypoints.conf[0].cpu().numpy()
                        else:
                            kpts_conf = np.ones((kpts_xy.shape[0],))

                        kpts_xy[:, 0] += offset[0]
                        kpts_xy[:, 1] += offset[1]

                        det["pose_keypoints"] = np.column_stack((kpts_xy, kpts_conf))

                # 3️⃣ FACE DETECTION (GPU FORCED)
                if self.face_model and human_crop.size > 0:
                    face_res = self.face_model(
                        human_crop,
                        verbose=False,
                        imgsz=256,
                        device=self.device,
                        conf=CONF_FACE
                    )[0]

                    for fbox in face_res.boxes:
                        fx1, fy1, fx2, fy2 = fbox.xyxy[0].cpu().numpy()
                        fx1 += offset[0]
                        fy1 += offset[1]
                        fx2 += offset[0]
                        fy2 += offset[1]

                        det["faces"].append({
                            "bbox": (fx1, fy1, fx2, fy2),
                            "conf": float(fbox.conf[0])
                        })

                detections.append(det)

        return detections

    def draw(self, frame, detections, draw_skeleton=True, draw_faces=True):
        out = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            conf = det["conf"]

            # Determine color and label based on tracking
            if 'track_id' in det and det['track_id'] >= 0:
                # Use track color if available
                color = det.get('track_color', (0, 255, 0))
                label = f"ID:{det['track_id']} {conf:.2f}"
            else:
                # Default green for untracked
                color = (0, 255, 0)
                label = f"Person {conf:.2f}"

            # human box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, label, (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # skeleton
            if draw_skeleton and det["pose_keypoints"] is not None:
                kpts = det["pose_keypoints"]
                
                # Draw bones (connections)
                for start_idx, end_idx in POSE_CONNECTIONS:
                    if start_idx < len(kpts) and end_idx < len(kpts):
                        x1, y1, v1 = kpts[start_idx]
                        x2, y2, v2 = kpts[end_idx]
                        if v1 > 0.5 and v2 > 0.5:  # Higher confidence for stability
                            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), 
                                   (0, 255, 255), 2, cv2.LINE_AA)
                
                # Draw body keypoints only (indices 5-16: shoulders to ankles)
                for i, (px, py, v) in enumerate(kpts):
                    if i >= 5 and v > 0.5:  # Only body keypoints, skip face (0-4)
                        cv2.circle(out, (int(px), int(py)), 4, (0, 0, 255), -1)
                        cv2.circle(out, (int(px), int(py)), 5, (255, 255, 255), 1)

            # faces
            if draw_faces:
                for f in det["faces"]:
                    fx1, fy1, fx2, fy2 = map(int, f["bbox"])
                    fconf = f["conf"]
                    cv2.rectangle(out, (fx1, fy1), (fx2, fy2), (255,0,0), 2)
                    cv2.putText(out, f"Face {fconf:.2f}", (fx1, fy1-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        return out

    def count_people(self, detections):
        return len(detections)
