import cv2
import time
import argparse
import sys
from pathlib import Path
import numpy as np


# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cv_pipeline.detection.yolo_detector import YOLODetector
from cv_pipeline.tracking.boxmot_tracker import PersonTracker
from cv_pipeline.pose_estimation.mediapipe_pose import MediaPipePoseEstimator


def visualize(video_path, draw_skeleton=True, draw_faces=True, enable_tracking=True):
    """Run the hybrid YOLO detector on a video and display results.

    Args:
        video_path (str): Path to the input video file.
        draw_skeleton (bool): Whether to overlay pose keypoints.
        draw_faces (bool): Whether to overlay detected face boxes.
        enable_tracking (bool): Whether to enable person tracking with IDs.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: video {video_path} not found or cannot be opened.")
        return

    # Initialize detector
    print("Initializing YOLOv8 detector...")
    detector = YOLODetector(model_path='yolov8n.pt', device='cuda')
    
    # Initialize tracker if requested
    tracker = None
    if enable_tracking:
        try:
            print("Initializing BoxMOT tracker...")
            tracker = PersonTracker(tracker_type='deepocsort', device=detector.device)
            print("✓ Tracking enabled with DeepOCSORT")
        except Exception as e:
            print(f"⚠️ Failed to initialize tracker: {e}")
            print("Continuing without tracking...")
            enable_tracking = False
    
    # Initialize pose estimator
    pose_estimator = None
    if draw_skeleton:
        try:
            print("Initializing MediaPipe pose estimator...")
            pose_estimator = MediaPipePoseEstimator(model_complexity=1)
            print("✓ Pose estimation enabled with MediaPipe")
        except Exception as e:
            print(f"⚠️ Failed to initialize pose estimator: {e}")
            print("Continuing without pose estimation...")
            draw_skeleton = False
    
    title = "Security Camera Pipeline"
    print(f"\n--- {title} ---\n")

    frame_count = 0
    processing_times = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start = time.time()
        
        # 1. Detect persons using YOLO
        detections = detector.detect(frame)
        
        # 2. Track persons if enabled
        if enable_tracking and tracker:
            detections = tracker.update(frame, detections)
        
        # 3. Estimate pose if enabled
        if draw_skeleton and pose_estimator:
            detections = pose_estimator.estimate_pose(frame, detections)
        
        # 4. Draw results
        drawn = detector.draw(frame, detections, draw_skeleton=False, draw_faces=draw_faces)
        
        # Draw trajectories if tracking is enabled
        if enable_tracking and tracker:
            for det in detections:
                if 'track_id' in det and det['track_id'] != -1 and 'trajectory' in det:
                    color = det.get('track_color', (0, 255, 0))
                    trajectory = det['trajectory']
                    if len(trajectory) >= 2:
                        points = np.array(trajectory, dtype=np.int32)
                        cv2.polylines(drawn, [points], False, color, 2)
        
        # Draw skeletons if pose estimation is enabled
        if draw_skeleton and pose_estimator:
            for det in detections:
                drawn = pose_estimator.draw_pose(drawn, det)
        
        # Calculate and display FPS
        frame_time = time.time() - start
        processing_times.append(frame_time)
        if len(processing_times) > 30:
            processing_times.pop(0)
            
        fps = 1 / (sum(processing_times) / len(processing_times))
        cv2.putText(drawn, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Display person count
        cv2.putText(drawn, f"Persons: {len(detections)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Resize and display
        display = cv2.resize(drawn, (1280, 720))
        cv2.imshow(title, display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    
    # Print performance summary
    if frame_count > 0:
        avg_fps = frame_count / sum(processing_times)
        print(f"\nProcessed {frame_count} frames")
        print(f"Average FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize hybrid YOLO detection on a video.")
    parser.add_argument("video", nargs='?', default="vd2.mp4", help="Path to the video file (default: vd2.mp4)")
    parser.add_argument("--no-skeleton", action="store_true", help="Disable pose skeleton drawing.")
    parser.add_argument("--no-faces", action="store_true", help="Disable face box drawing.")
    parser.add_argument("--no-track", action="store_true", help="Disable person tracking with IDs.")
    args = parser.parse_args()
    
    visualize(
        args.video, 
        draw_skeleton=not args.no_skeleton, 
        draw_faces=not args.no_faces, 
        enable_tracking=not args.no_track
    )