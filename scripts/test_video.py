from cv_pipeline.utils.video_reader import VideoReader

vr = VideoReader("C:/Users/User/Desktop/IA-Camera-Challenge/vd1.mp4")  

print(f"FPS: {vr.get_fps()}")
print(f"Total frames: {vr.get_frame_count()}")

for i, frame in enumerate(vr):
    if i % 30 == 0:
        print(f"Read frame {i}")
