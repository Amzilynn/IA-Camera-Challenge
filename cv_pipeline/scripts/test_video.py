from cv_pipeline.detection.yolo_detector import YOLODetector
from cv_pipeline.utils.video_reader import VideoReader

detector = YOLODetector()


reader = VideoReader("vd2.mp4")

for frame in reader:
    dets = detector.detect(frame)
    print(dets[:3])  

