"""
Visualization script for demonstrating BoxMOT person tracking with persistent IDs.
"""
import cv2
import time
import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cv_pipeline.detection.yolo_detector import YOLODetector
from cv_pipeline.tracking.boxmot_tracker import PersonTracker


def visualize_tracking(video_path, draw_skeleton=True, draw_faces=True):
    """
    Run person tracking with persistent ID assignment on a video.
    
    Args:
        video_path (str): Path to the input video file.
        draw_skeleton (bool): Whether to overlay pose keypoints.
        draw_faces (bool): Whether to overlay detected face boxes.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: video '{video_path}' not found or cannot be opened.")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("\n" + "="*60)
    print("BoxMOT Person Tracking Demo")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"FPS: {fps:.1f} | Total Frames: {total_frames}")
    print("="*60 + "\n")
    
    # Initialize detector and tracker
    detector = YOLODetector()
    tracker = PersonTracker(tracker_type='deepocsort', device=detector.device)
    
    # Tracking statistics
    frame_count = 0
    total_fps = 0
    active_track_ids = set()
    max_simultaneous_tracks = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        start = time.time()
        
        # Detect persons
        detections = detector.detect(frame)
        
        # Update tracker with detections
        detections = tracker.update(frame, detections)
        
        # Draw results
        drawn = detector.draw(frame, detections, 
                            draw_skeleton=draw_skeleton, 
                            draw_faces=draw_faces)
        
        # Calculate FPS
        elapsed = time.time() - start
        current_fps = 1 / (elapsed + 1e-6)
        total_fps += current_fps
        
        # Update statistics
        current_track_ids = {det['track_id'] for det in detections if det.get('track_id', -1) >= 0}
        active_track_ids.update(current_track_ids)
        max_simultaneous_tracks = max(max_simultaneous_tracks, len(current_track_ids))
        
        # Display info overlay
        info_y = 30
        cv2.putText(drawn, f"FPS: {current_fps:.1f}", (20, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.putText(drawn, f"Frame: {frame_count}/{total_frames}", (20, info_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(drawn, f"Active Tracks: {len(current_track_ids)}", (20, info_y + 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(drawn, f"Total IDs Seen: {len(active_track_ids)}", (20, info_y + 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display
        display = cv2.resize(drawn, (1280, 720))
        cv2.imshow("BoxMOT Person Tracking", display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⏹️  Stopped by user")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print("\n" + "="*60)
    print("📊 Tracking Statistics")
    print("="*60)
    print(f"Total Frames Processed: {frame_count}")
    print(f"Average FPS: {total_fps / max(frame_count, 1):.2f}")
    print(f"Total Unique IDs Tracked: {len(active_track_ids)}")
    print(f"Max Simultaneous Tracks: {max_simultaneous_tracks}")
    print(f"Active Track IDs: {sorted(active_track_ids) if active_track_ids else 'None'}")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize BoxMOT person tracking with persistent IDs."
    )
    parser.add_argument("video", nargs='?', default="vd2.mp4", 
                       help="Path to the video file (default: vd2.mp4)")
    parser.add_argument("--no-skeleton", action="store_true", 
                       help="Disable pose skeleton drawing.")
    parser.add_argument("--no-faces", action="store_true", 
                       help="Disable face box drawing.")
    
    args = parser.parse_args()
    
    visualize_tracking(args.video, 
                      draw_skeleton=not args.no_skeleton, 
                      draw_faces=not args.no_faces)
