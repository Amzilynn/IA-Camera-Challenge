import cv2
import numpy as np
import os

def test_write(codec):
    filename = f'test_{codec}.mp4'
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
        if not out.isOpened():
            print(f"Codec {codec}: FAILED to open")
            return
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw something to be sure
        cv2.putText(frame, "TEST", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        for _ in range(20):
            out.write(frame)
        out.release()
        size = os.path.getsize(filename)
        print(f"Codec {codec}: SUCCESS, Size: {size}")
    except Exception as e:
        print(f"Codec {codec}: ERROR {str(e)}")

if __name__ == "__main__":
    for c in ['avc1', 'mp4v', 'H264', 'XVID']:
        test_write(c)
