# pipeline.py
import cv2
import time
import numpy as np
import argparse
from pathlib import Path
import sys

# Import components
from cv_pipeline.detection.yolo_detector import YOLODetector
from cv_pipeline.tracking.boxmot_tracker import PersonTracker
from cv_pipeline.pose_estimation.mediapipe_pose import MediaPipePoseEstimator

class SecurityCameraPipeline:
    """
    End-to-end pipeline for security camera processing:
    1. Object detection with YOLOv8
    2. Person tracking with BoxMOT
    3. Pose estimation with MediaPipe BlazePose
    """
    
    def __init__(self,
                 yolo_model='yolov8n.pt',
                 device='cuda',
                 enable_tracking=True,
                 enable_pose=True,
                 conf_threshold=0.5,
                 model_complexity=1):
        
        self.enable_tracking = enable_tracking
        self.enable_pose = enable_pose
        
        # Initialize detector
        print("\n--- Loading Models ---")
        self.detector = YOLODetector(
            model_path=yolo_model,
            device=device,
            conf_threshold=conf_threshold,
            classes=[0]  # Only detect persons
        )
        
        # Initialize tracker if enabled
        self.tracker = None
        if self.enable_tracking:
            try:
                self.tracker = PersonTracker(
                    tracker_type='deepocsort',
                    device=device
                )
            except Exception as e:
                print(f"⚠️ Failed to initialize tracker: {e}")
                print("Continuing without tracking...")
                self.enable_tracking = False
        
        # Initialize pose estimator if enabled
        self.pose_estimator = None
        if self.enable_pose:
            try:
                self.pose_estimator = MediaPipePoseEstimator(
                    model_complexity=model_complexity,
                    static_image_mode=False,
                    smooth_landmarks=True
                )
            except Exception as e:
                print(f"⚠️ Failed to initialize pose estimator: {e}")
                print("Continuing without pose estimation...")
                self.enable_pose = False
                
        print("\n--- Pipeline Ready ---")
        
    def process_frame(self, frame):
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Processed frame and detections
        """
        if frame is None:
            return None, []
            
        # 1. Object Detection
        start_time = time.time()
        detections = self.detector.detect(frame)
        
        # 2. Object Tracking
        if self.enable_tracking and self.tracker is not None and len(detections) > 0:
            detections = self.tracker.update(frame, detections)
            
        # 3. Pose Estimation
        if self.enable_pose and self.pose_estimator is not None and len(detections) > 0:
            detections = self.pose_estimator.estimate_pose(frame, detections)
            
        # 4. Visualization
        output_frame = self._visualize_results(frame, detections)
            
        # Calculate FPS
        fps = 1 / (time.time() - start_time + 1e-6)
        
        return output_frame, detections, fps
    
    def _visualize_results(self, frame, detections):
        """
        Visualize the results of detection, tracking, and pose estimation.
        """
        # Draw bounding boxes and tracking IDs
        output = self.detector.draw(frame, detections, 
                                    draw_skeleton=False, 
                                    draw_faces=True)
        
        # Draw trajectories if tracking is enabled
        if self.enable_tracking and self.tracker is not None:
            for det in detections:
                if 'track_id' in det and det['track_id'] != -1 and 'trajectory' in det:
                    color = det.get('track_color', (0, 255, 0))
                    trajectory = det['trajectory']
                    if len(trajectory) >= 2:
                        points = np.array(trajectory, dtype=np.int32)
                        cv2.polylines(output, [points], False, color, 2)
        
        # Draw pose skeletons if pose estimation is enabled
        if self.enable_pose and self.pose_estimator is not None:
            for det in detections:
                if 'keypoints' in det and det['keypoints'] is not None:
                    output = self.pose_estimator.draw_pose(output, det)
        
        return output
    
    def process_video(self, video_path=0, output_path=None, display=True):
        """
        Process a video file or camera stream.
        
        Args:
            video_path: Path to video or camera index (default: 0 for webcam)
            output_path: Path to save output video (optional)
            display: Whether to display the processed video
            
        Returns:
            None
        """
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_path}")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Set up video writer if output path is specified
        writer = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Saving output to {output_path}")
        
        # Process frames
        frame_count = 0
        start_time = time.time()
        processing_times = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_start = time.time()
            
            # Process the frame
            output_frame, detections, frame_fps = self.process_frame(frame)
            
            # Calculate processing time
            frame_time = time.time() - frame_start
            processing_times.append(frame_time)
            
            # Add FPS counter
            avg_fps = 1.0 / (sum(processing_times[-30:]) / min(len(processing_times), 30))
            cv2.putText(output_frame, f"FPS: {avg_fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Add person count
            cv2.putText(output_frame, f"Persons: {len(detections)}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Write output frame if saving
            if writer is not None:
                writer.write(output_frame)
            
            # Display output frame if requested
            if display:
                cv2.imshow("Security Camera Pipeline", output_frame)
                
                # Exit on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
        
        # Clean up
        total_time = time.time() - start_time
        if frame_count > 0:
            print(f"\nProcessed {frame_count} frames in {total_time:.2f} seconds")
            print(f"Average FPS: {frame_count / total_time:.2f}")
        
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Security Camera Pipeline Demo")
    
    # Input/output options
    parser.add_argument("--input", type=str, default="0", 
                      help="Path to video file or camera index (default: 0 for webcam)")
    parser.add_argument("--output", type=str, default=None,
                      help="Path to save output video (optional)")
    
    # Model options
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt",
                      help="YOLOv8 model to use (default: yolov8n.pt)")
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to run models on ('cuda' or 'cpu')")
    parser.add_argument("--conf-threshold", type=float, default=0.5,
                      help="Confidence threshold for detection (default: 0.5)")
    parser.add_argument("--model-complexity", type=int, default=1,
                      help="MediaPipe pose model complexity (0, 1, or 2)")
    
    # Feature flags
    parser.add_argument("--no-tracking", action="store_true",
                      help="Disable person tracking")
    parser.add_argument("--no-pose", action="store_true",
                      help="Disable pose estimation")
    
    args = parser.parse_args()
    
    # Convert string camera index to integer
    input_src = args.input
    if input_src.isdigit():
        input_src = int(input_src)
    
    # Initialize the pipeline
    pipeline = SecurityCameraPipeline(
        yolo_model=args.yolo_model,
        device=args.device,
        enable_tracking=not args.no_tracking,
        enable_pose=not args.no_pose,
        conf_threshold=args.conf_threshold,
        model_complexity=args.model_complexity
    )
    
    # Process the video
    pipeline.process_video(
        video_path=input_src,
        output_path=args.output,
        display=True
    )


if __name__ == "__main__":
    main()