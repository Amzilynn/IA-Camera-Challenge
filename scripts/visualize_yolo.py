import cv2
import time
import argparse
from cv_pipeline.detection.yolo_detector import YOLODetector


def visualize(video_path, draw_skeleton=True, draw_faces=True):
    """Run the hybrid YOLO detector on a video and display results.

    Args:
        video_path (str): Path to the input video file.
        draw_skeleton (bool): Whether to overlay pose keypoints.
        draw_faces (bool): Whether to overlay detected face boxes.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: video not found or cannot be opened.")
        return

    detector = YOLODetector()
    print("\n--- Hybrid YOLO Detection (Human + Pose + Face) ---\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start = time.time()
        detections = detector.detect(frame)
        drawn = detector.draw(frame, detections, draw_skeleton=draw_skeleton, draw_faces=draw_faces)
        fps = 1 / (time.time() - start + 1e-6)
        cv2.putText(drawn, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        display = cv2.resize(drawn, (1280, 720))
        cv2.imshow("Hybrid YOLO Detection", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize hybrid YOLO detection on a video.")
    parser.add_argument("video", nargs='?', default="vd2.mp4", help="Path to the video file (default: vd2.mp4)")
    parser.add_argument("--no-skeleton", action="store_true", help="Disable pose skeleton drawing.")
    parser.add_argument("--no-faces", action="store_true", help="Disable face box drawing.")
    args = parser.parse_args()
    visualize(args.video, draw_skeleton=not args.no_skeleton, draw_faces=not args.no_faces)
