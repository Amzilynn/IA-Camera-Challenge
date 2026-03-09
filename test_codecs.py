import cv2
import os

def check_codecs():
    codecs = ['avc1', 'mp4v', 'XVID', 'MJPG', 'H264']
    filename = 'test_codec.mp4'
    img = bytearray([0] * (640 * 480 * 3))
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    for c in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*c)
            out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
            if out.isOpened():
                print(f"Codec {c}: SUCCESS")
                out.release()
            else:
                print(f"Codec {c}: FAILED (not opened)")
        except Exception as e:
            print(f"Codec {c}: ERROR ({str(e)})")
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    check_codecs()
