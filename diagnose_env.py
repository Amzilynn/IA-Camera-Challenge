import sys
import os

print(f"Python: {sys.version}")
print(f"Path: {sys.path}")

print("-" * 20)
try:
    import tensorflow as tf
    print(f"TensorFlow imported. Type: {type(tf)}")
    print(f"TF Dir: {dir(tf)}")
    if hasattr(tf, '__version__'):
        print(f"TF Version: {tf.__version__}")
    elif hasattr(tf, 'version'):
        print(f"TF Version (submodule): {tf.version.VERSION}")
    else:
        print("TF has NO version attribute.")
        print(f"TF File: {tf.__file__}")
except ImportError as e:
    print(f"TF Import Error: {e}")
except Exception as e:
    print(f"TF Error: {e}")

print("-" * 20)
try:
    import mediapipe
    print(f"MediaPipe imported. Type: {type(mediapipe)}")
    print(f"MediaPipe File: {mediapipe.__file__}")
    print(f"MediaPipe Dir: {dir(mediapipe)}")
    
    try:
        import mediapipe.python
        print("mediapipe.python imported")
    except ImportError as e:
        print(f"mediapipe.python FAILED: {e}")

except ImportError as e:
    print(f"MediaPipe Import Error: {e}")
