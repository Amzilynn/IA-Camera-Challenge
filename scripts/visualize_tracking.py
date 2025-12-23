# scripts/visualize_tracking.py
import cv2
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
    
from cv_pipeline.pipeline import SecurityCameraPipeline

def main():
    parser = argparse.ArgumentParser(
        description="Security Camera Pipeline with Emotion Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with webcam and all features enabled
  python scripts/visualize_tracking.py
  
  # Run with video file
  python scripts/visualize_tracking.py --input vd2.mp4 --output output.mp4
  
  # Run without emotion analysis for better performance
  python scripts/visualize_tracking.py --no-emotion
  
  # Run with CPU only
  python scripts/visualize_tracking.py --device cpu
  
  # Run with minimal features for maximum speed
  python scripts/visualize_tracking.py --no-pose --no-emotion
        """
    )
    
    # Input/Output
    parser.add_argument(
        "video", 
        nargs='?', 
        default="vd2.mp4",
        help="Path to video file or camera index (default: vd2.mp4)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Path to save output video (optional)"
    )
    
    # Model Options
    parser.add_argument(
        "--yolo-model", 
        type=str, 
        default="yolov8x.pt",
        help="YOLO model path (default: yolov8x.pt)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu",
        choices=['cpu', 'cuda'],
        help="Device to use (default: cpu)"
    )
    parser.add_argument(
        "--conf-threshold", 
        type=float, 
        default=0.5,
        help="Detection confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--model-complexity", 
        type=int, 
        default=1,
        choices=[0, 1, 2],
        help="MediaPipe pose model complexity: 0=fastest, 2=most accurate (default: 1)"
    )
    
    # Feature Toggles
    parser.add_argument(
        "--no-tracking", 
        action="store_true",
        help="Disable person tracking"
    )
    parser.add_argument(
        "--no-pose", 
        action="store_true",
        help="Disable pose estimation (skeleton)"
    )
    parser.add_argument(
        "--no-emotion", 
        action="store_true",
        help="Disable emotion analysis"
    )
    
    # Display Options
    parser.add_argument(
        "--no-display", 
        action="store_true",
        help="Disable video display (useful for headless processing)"
    )
    
    args = parser.parse_args()
    
    # Convert camera index if numeric
    video_input = args.video
    if video_input.isdigit():
        video_input = int(video_input)
    
    # Display configuration
    print("\n" + "="*60)
    print("🎯 Configuration")
    print("="*60)
    print(f"Input: {video_input}")
    print(f"Output: {args.output if args.output else 'None (display only)'}")
    print(f"Device: {args.device.upper()}")
    print(f"YOLO Model: {args.yolo_model}")
    print(f"Features:")
    print(f"  - Detection: ✓ (always enabled)")
    print(f"  - Tracking: {'✓' if not args.no_tracking else '✗'}")
    print(f"  - Pose Skeleton: {'✓' if not args.no_pose else '✗'}")
    print(f"  - Emotion Analysis: {'✓' if not args.no_emotion else '✗'}")
    print("="*60)
    
    # Initialize pipeline
    pipeline = SecurityCameraPipeline(
        yolo_model=args.yolo_model,
        device=args.device,
        enable_tracking=not args.no_tracking,
        enable_pose=not args.no_pose,
        enable_emotion=not args.no_emotion,
        conf_threshold=args.conf_threshold,
        model_complexity=args.model_complexity
    )
    
    # Process video
    pipeline.process_video(
        video_path=video_input,
        output_path=args.output,
        display=not args.no_display
    )


if __name__ == "__main__":
    main()