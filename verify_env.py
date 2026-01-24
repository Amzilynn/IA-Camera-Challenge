import sys
import importlib

packages = [
    "ultralytics",
    "boxmot",
    "mediapipe",
    "deepface",
    "cv2", # opencv-python
    "numpy",
    "filterpy",
    "lap",
    "scipy",
    "termcolor",
    "gdown",
    "onetick_py", # Note the underscore usage usually for imports
    "torch",
    "tensorflow"
]

print(f"Python: {sys.version}")
print("-" * 20)

all_good = True
for pkg in packages:
    try:
        module = importlib.import_module(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f"[OK] {pkg} installed (version: {version})")
    except ImportError as e:
        print(f"[FAIL] {pkg} NOT installed/importable. Error: {e}")
        all_good = False
    except Exception as e:
        print(f"[ERROR] {pkg} caused an error: {e}")
        all_good = False

if all_good:
    print("\nSUCCESS: All packages verified.")
    sys.exit(0)
else:
    print("\nFAILURE: Some packages are missing or broken.")
    sys.exit(1)
