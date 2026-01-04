import traceback
try:
    import mediapipe as mp
    print(f"MediaPipe Version: {mp.__version__}")
    
    # Try explicit import which is often masked
    import mediapipe.python.solutions as solutions
    print("Explicit import of solutions SUCCESS")
    print(f"Solutions: {solutions}")
    
    if hasattr(mp, 'solutions'):
        print("mp.solutions is available")
    else:
        print("mp.solutions is MISSING (despite explicit import success?)")

except ImportError as e:
    print("\n!!! IMPORT ERROR !!!")
    traceback.print_exc()
except AttributeError as e:
    print("\n!!! ATTRIBUTE ERROR !!!")
    traceback.print_exc()
except Exception as e:
    print("\n!!! UNKNOWN ERROR !!!")
    traceback.print_exc()
