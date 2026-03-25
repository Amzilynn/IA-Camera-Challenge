# IA Camera Challenge - Scene Intelligence Pipeline

## 1. Project Overview
The **Scene Intelligence Pipeline** is a sophisticated, high-performance computer vision system designed for human-centric scene understanding and interaction analysis. It processes video feeds to extract, classify, and track human behavior, social dynamics, and service efficiency in complex environments like retail stores, offices, and public spaces. 

### Real-World Use Case
Businesses often deploy cameras for security, but this project turns those cameras into **intelligent sensors**. It can tell you how many people are in a room, distinguish between staff and visitors, track the general "satisfaction" or mood of the room, and identify who is interacting with whom. This provides actionable insights into customer service quality, space utilization, and overall customer experience.

### Main Features
- **Human & Face Detection**: Pinpoints individuals and their faces even in crowded scenes.
- **Emotion Recognition**: Analyzes facial expressions to gauge sentiment (Happiness, Fear, Neutral, etc.).
- **Body Pose Tracking**: Extracts 17-point skeletal keypoints to analyze body language and orientation.
- **Unsupervised Role Classification**: Distinguishes "Staff" from "Visitors" based on their movement patterns (mobility and "anchoring") and interactions.
- **Temporal Stabilization**: Ensures steady tracking and consistent dashboard metrics even when detections drop out for a few frames.
- **Cinematic HUD Rendering**: Overlays a professional, cyber-agent style dashboard on the video output to visualize relationships, roles, and aggregate statistics in real time.

---

## 2. System Architecture

The pipeline processes video frame-by-frame, passing data through a sequence of specialized modules.

### Data Flow
`Input Video` → `YOLOv8 Detection` → `RTMPose Estimation` → `BoxMOT Tracking` → `Face & Emotion Analysis` → `Social Analyzer (Role/Intent)` → `HUD Metrics Tracker (Smoothing)` → `UI Processor (Rendering)` → `Final Video Output` & `JSON Logs`.

### Module Breakdown

- **Human Detection (`yolo_detector.py`)**
  Uses state-of-the-art YOLOv8 models to identify bounding boxes for every person in the frame.
- **Face Detection & Emotion Recognition (`emotion_analyzer.py`)**
  Detects faces within the human bounding boxes and uses the high-performance `HSEmotion` model (or DeepFace) to extract the dominant emotion and confidence scores.
- **Body/Pose Tracking (`rtmpose` / `yolo_detector.py`)**
  Estimates the 17-point human skeleton. This is crucial for determining which way a person is facing and where they are looking.
- **Multi-Object Tracking (`boxmot_tracker.py`)**
  Uses algorithms like DeepOCSORT or ByteTrack to assign a unique, persistent ID to each person across frames. **Crucially**, it predicts ("coasts") where a person should be if YOLO misses them for a frame, preventing ID flickering.
- **Staff vs Customer Classification (`social_analyzer.py`)**
  Monitors the velocity and position of tracked IDs. An individual who remains stationary (average speed < 100px/s) for over 15 seconds is "Anchored" and classified as `Staff`. Once anchored, the role "sticks" to the ID to prevent rapid flipping.
- **HUD/Dashboard Renderer (`ui_processor.py` & `metrics_tracker.py`)**
  Aggregates the raw, frame-by-frame data. The `HUDMetricsTracker` applies a 30-frame median filter to counts to erase jitter, and an Exponential Moving Average (EMA) to the Satisfaction Index. The `UIProcessor` draws the Glassmorphism bounding boxes, glowing energy links for interactions, and the top-right tracking dashboard.

---

## 3. Technologies & Dependencies

- **Python 3.11+**: Core programming language.
- **OpenCV (`cv2`)**: Used for video reading, writing, and low-level image drawing operations.
- **PyTorch & Torchvision**: The underlying deep learning framework powering most of the models.
- **Ultralytics (YOLOv8)**: Provides lightning-fast, highly accurate object detection.
- **BoxMOT**: A plug-and-play multi-object tracking library used to track YOLO detections over time.
- **HSEmotion**: State-of-the-art facial expression recognition library optimized for high speed and accuracy.
- **DeepFace**: Alternative lightweight face analysis library.
- **NumPy & SciPy**: Used for matrix operations, velocity calculations, and spatial-temporal filtering.

---

## 4. Installation Guide

Follow these steps to set up the environment on a Windows or Linux machine. An NVIDIA GPU with CUDA 12.1+ is highly recommended.

### 1. Environment Setup
Create and activate a virtual environment to avoid dependency conflicts:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Install Dependencies
Install all required libraries using the provided `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Model Weights
The pipeline uses auto-downloading for PyTorch models (like HSEmotion and BoxMOT trackers). However, ensure you have the YOLO weights in the `cv_pipeline/models/` directory or they will be downloaded on the first run.
*(Note: If you run into `libopenh264` codec errors when writing MP4 files, OpenCV will automatically attempt to bypass it or prompt you to install the OpenH264 DLL. Ensure `ffmpeg` is available on your system path).*

---

## 5. How to Run the Project

### Standard Pipeline Execution
To process a video, run the main entry point script from the root directory:

```powershell
python cv_pipeline/scripts/run_full_pipeline.py InputVideo.mp4 --output OutputVideo.mp4 --log scene_log.json
```

### Configurable Parameters
- `input`: The path to the source video (e.g., `VDC.mp4`).
- `--output`: The path where the annotated video will be saved (e.g., `VDC_processed.mp4`).
- `--log`: The path to save the detailed JSON sequence of events and detections (e.g., `scene_log_vdc.json`).
- `--headless`: (Optional) Run the pipeline without displaying the live `cv2.imshow` window. This significantly speeds up processing time on servers.
- `--model`: (Hardcoded in script but adjustable) Swap between `yolov8n.pt`, `yolov8s.pt`, or `yolov8m.pt` depending on your required accuracy/speed tradeoff.

---

## 6. How to Test the System (VERY IMPORTANT)

Testing computer vision pipelines requires structured verification to ensure temporary occlusions or rapid movements don't break the logic.

### 1. Input Videos to Use
Use videos with the following characteristics:
- **Stationary vs Moving People**: Scenarios where one person stands behind a counter (Staff) while others walk past (Visitors).
- **Occlusions**: People walking behind pillars or other people to test tracking persistence.
- **Clear Faces**: High resolution (1080p+) where faces are visible to test the emotion recognition.

### 2. Scenarios to Test & Verify
Run the pipeline on your test video, generating both the `.mp4` and the `.json` log. Verify the following **Success Criteria**:

- **Stable Dashboard Values**: Watch the upper-right UI box. The "Total People", "Staff", and "Visitors" counts should transition cleanly. They should **never** flicker (e.g., rapidly jumping 4 → 3 → 4 in the span of 3 frames).
  - *Verification*: If flickering occurs, verify that `metrics_tracker.py` is initialized with `window_size=30`.
- **Correct Tracking Across Frames**: An individual should maintain the same ID (e.g., `ID: 2`) throughout their presence in the video. If YOLO drops them for a frame, the bounding box should "coast" via BoxMOT predictions.
- **Accurate Staff vs Customer**: Ensure a stationary individual turns from `Visitor` to `Staff` after lingering in the same spot, and that they do not revert back to `Visitor` randomly.
- **Coherent Satisfaction Index**: The "Satisfaction" percentage should hover around 80% (Neutral) and smoothly drift up or down based on the dominant emotions detected, without chaotic jumps.

### Debugging Tips
- If the HUD flashes or ghost bounding boxes appear, check `boxmot_tracker.py`'s input structure. Ensure that `empty_detections` are being passed to `tracker.update()` when YOLO finds nothing, so that the Kalman filters can coast.
- If the video plays back "too fast" locally, remember that processing speed is tied to your GPU. Use `--headless` for batched processing, or rely on the written `.mp4` output which plays at exactly the source FPS.

---

## 7. Output Explanation

### Visual Overlays
- **Bounding Boxes**: Green boxes typically denote "Visitors" while Blue/Purple denote "Staff". Included is the `ID` and the dominant `Emotion`.
- **Lines/Links**: Glowing lines connecting two bounding boxes indicate a detected `Interaction` (e.g., Proximity + Facing each other).

### The "System Vibe" Dashboard (Top Right)
- **Total People / Staff / Visitors**: The temporally smoothed, stabilized count of tracked individuals. It strictly enforces the logical rule: `Total = Staff + Visitors`.
- **Active Interactions**: The number of unique individuals currently engaging in a social link.
- **Satisfaction Index**: A moving average out of 100%. Computed by assigning scores to emotions (Happiness = 100%, Neutral = 80%, Fear/Anger = 50%) and averaging them across all visible faces, smoothed over time.
- **System Status**: Displays `NOMINAL` when operations are standard, or `CRITICAL` if satisfaction heavily dips or negative interactions spike.

---

## 8. Project Structure

```text
IA-Camera-Challenge/
├── cv_pipeline/
│   ├── detection/          
│   │   └── yolo_detector.py        # Core YOLOv8 and RTMPose inference
│   ├── tracking/           
│   │   └── boxmot_tracker.py       # BoxMOT wrapper w/ coasting logic
│   ├── emotion_analysis/   
│   │   └── emotion_analyzer.py     # HSEmotion/DeepFace integration
│   ├── social_interaction/ 
│   │   └── social_analyzer.py      # Role Discovery (Anchor) & Interaction logic
│   ├── utils/              
│   │   ├── metrics_tracker.py      # HUD smoothing & Median filter logic
│   │   ├── ui_processor.py         # Glassmorphism & HUD rendering
│   │   └── scene_describer.py      # JSONL log generation
│   ├── models/                     # Auto-downloaded model weight directory
│   └── scripts/                
│       └── run_full_pipeline.py    # The main execution loop
├── requirements.txt                # Python package dependencies
├── venv/                           # Virtual environment (user-created)
└── README.md                       # This documentation
```

---

## 9. Known Limitations & Improvements

### Current Limitations
- **Hardware Heavy**: Processing heavy YOLOv8, Pose estimation, BoxMOT, and Emotion recognition models sequentially on a single thread will not achieve real-time (30 FPS) performance without a high-end dedicated GPU.
- **Emotion Accuracy at Distance**: HSEmotion struggles if faces are blurred or strictly in profile; it requires >30x30px clear facial crops.
- **Occlusion Breakage**: While tracking handles 1-5 frames of occlusion via coasting, long occlusions (e.g., someone walking behind a wall for 3 seconds) will result in a new ID being assigned when they emerge.

### Future Improvements
- **Asynchronous Processing**: Decouple the UI rendering and the Video capture into separate threads/queues to vastly improve framerate.
- **Tracking by Detection**: Run YOLO only every 3rd frame, relying purely on BoxMOT optical flow/kalman filters for the intermediate frames to speed up processing.
- **Re-ID Model Integration**: Integrate an OSNet or Re-ID embedding model into the tracker to remember IDs even after 10+ seconds of complete occlusion.
