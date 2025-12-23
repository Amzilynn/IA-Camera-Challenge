import cv2
import time
import numpy as np
from pathlib import Path
import sys
from cv_pipeline.detection.yolo_detector import YOLODetector
from cv_pipeline.tracking.boxmot_tracker import PersonTracker
from cv_pipeline.pose_estimation.mediapipe_pose import MediaPipePoseEstimator
from cv_pipeline.emotion.emotion_analyzer import EmotionAnalyzer

class SecurityCameraPipeline:
    """
    Complete pipeline: Detection → Tracking → Pose → Emotion Analysis
    """
    
    def __init__(self,
                 yolo_model='yolov8n.pt',
                 device='cpu',
                 enable_tracking=True,
                 enable_pose=True,
                 enable_emotion=True,
                 conf_threshold=0.5,
                 model_complexity=1):
        
        self.enable_tracking = enable_tracking
        self.enable_pose = enable_pose
        self.enable_emotion = enable_emotion
        
        print("\n" + "="*60)
        print("🚀 Security Camera Pipeline with Emotion Analysis")
        print("="*60)
        
        # Initialize detector
        self.detector = YOLODetector(
            model_path=yolo_model,
            device=device,
            conf_threshold=conf_threshold,
            classes=[0]
        )
        
        # Initialize tracker
        self.tracker = None
        if self.enable_tracking:
            try:
                self.tracker = PersonTracker(
                    tracker_type='deepocsort',
                    device=device
                )
            except Exception as e:
                print(f"⚠️ Tracker failed: {e}")
                self.enable_tracking = False
        
        # Initialize pose estimator
        self.pose_estimator = None
        if self.enable_pose:
            try:
                self.pose_estimator = MediaPipePoseEstimator(
                    model_complexity=model_complexity
                )
            except Exception as e:
                print(f"⚠️ Pose estimator failed: {e}")
                self.enable_pose = False
        
        # Initialize emotion analyzer
        self.emotion_analyzer = None
        if self.enable_emotion:
            try:
                self.emotion_analyzer = EmotionAnalyzer(
                    detector_backend='opencv',
                    enforce_detection=False,
                    history_length=30
                )
            except Exception as e:
                print(f"⚠️ Emotion analyzer failed: {e}")
                self.enable_emotion = False
        
        print("="*60)
        print("✓ Pipeline Ready!")
        print("="*60 + "\n")
        
    def process_frame(self, frame):
        """Process single frame through complete pipeline."""
        if frame is None:
            return None, [], 0
            
        start_time = time.time()
        
        # 1. Detection
        detections = self.detector.detect(frame)
        
        # 2. Tracking
        if self.enable_tracking and self.tracker and len(detections) > 0:
            detections = self.tracker.update(frame, detections)
        
        # 3. Pose Estimation
        if self.enable_pose and self.pose_estimator and len(detections) > 0:
            detections = self.pose_estimator.estimate_pose(frame, detections)
        
        # 4. Emotion Analysis
        if self.enable_emotion and self.emotion_analyzer and len(detections) > 0:
            detections = self.emotion_analyzer.update_emotions(frame, detections)
        
        # 5. Visualization
        output_frame = self._visualize(frame, detections)
        
        fps = 1 / (time.time() - start_time + 1e-6)
        
        return output_frame, detections, fps
    
    def _visualize(self, frame, detections):
        """Complete visualization with all features."""
        output = frame.copy()
        
        # Draw bounding boxes and IDs
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            conf = det['conf']
            track_id = det.get('track_id', -1)
            
            # Get color
            if track_id != -1:
                color = det.get('track_color', (0, 255, 0))
            else:
                color = (128, 128, 128)
            
            # Draw bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID and confidence
            if track_id != -1:
                label = f"ID:{track_id} {conf:.2f}"
                cv2.putText(output, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw trajectories
        if self.enable_tracking and self.tracker:
            for det in detections:
                if 'trajectory' in det and len(det['trajectory']) >= 2:
                    color = det.get('track_color', (0, 255, 0))
                    points = np.array(det['trajectory'], dtype=np.int32)
                    cv2.polylines(output, [points], False, color, 2)
        
        # Draw pose skeletons
        if self.enable_pose and self.pose_estimator:
            for det in detections:
                output = self.pose_estimator.draw_pose(output, det)
        
        # Draw emotions
        if self.enable_emotion and self.emotion_analyzer:
            for det in detections:
                output = self.emotion_analyzer.draw_emotion(output, det)
        
        return output
    
    def process_video(self, video_path=0, output_path=None, display=True):
        """Process video with complete pipeline."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processing_times = []
        
        print(f"\n🎬 Processing video: {video_path}")
        print(f"📐 Resolution: {width}x{height} @ {fps} FPS\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            output_frame, detections, frame_fps = self.process_frame(frame)
            processing_times.append(1 / frame_fps)
            
            # Add stats overlay
            avg_fps = 1.0 / (sum(processing_times[-30:]) / min(len(processing_times), 30))
            
            # Info panel
            cv2.rectangle(output_frame, (10, 10), (300, 120), (0, 0, 0), -1)
            cv2.rectangle(output_frame, (10, 10), (300, 120), (255, 255, 255), 2)
            
            cv2.putText(output_frame, f"FPS: {avg_fps:.1f}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(output_frame, f"Persons: {len(detections)}", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(output_frame, f"Frame: {frame_count}", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show emotions summary
            if self.enable_emotion and len(detections) > 0:
                emotions = [d.get('emotion', 'N/A') for d in detections if 'emotion' in d]
                if emotions:
                    emotion_text = f"Emotions: {', '.join(emotions)}"
                    cv2.putText(output_frame, emotion_text, (20, height - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            if writer:
                writer.write(output_frame)
            
            if display:
                cv2.imshow("Security Camera - Full Pipeline", output_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⏹️  Stopped by user")
                    break
            
            frame_count += 1
            
            # Progress indicator
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames | FPS: {avg_fps:.1f}", end='\r')
        
        # Cleanup
        total_time = sum(processing_times)
        print(f"\n\n{'='*60}")
        print(f"📊 Processing Complete")
        print(f"{'='*60}")
        print(f"Total Frames: {frame_count}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Average FPS: {frame_count / total_time:.2f}")
        print(f"{'='*60}\n")
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()