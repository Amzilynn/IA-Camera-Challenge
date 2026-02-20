# IA Camera Challenge - Computer Vision Pipeline

A comprehensive CV pipeline for security camera analysis with real-time person detection, tracking, pose estimation, emotion analysis, demographic estimation, and social interaction detection.

## 🚀 Features

### Core Modules
- **Object Detection**: YOLOv8m for high-accuracy human detection.
- **Multi-Object Tracking**: BoxMOT (DeepOCSORT) for persistent ID assignment and cross-scene re-identification.
- **Pose Estimation**: YOLOv8m-pose for stable, high-speed body tracking (Optimized for crowded scenes).
- **Emotion Analysis**: DeepFace for facial emotion recognition and sentiment trend analysis.
- **Demographic Analysis**: MiVOLO (Face + Body) for accurate age and gender estimation.
- **Social Interaction (STAS)**: Advanced geometry-based detection of social behaviors (Talking, Approaching, Walking together, Service recognition, Space violation).
- **Scene Logging**: Robust JSONL logging with frame-by-frame data for easy downstream analysis.

## 📁 Project Structure

```bash
IA-Camera-Challenge/
├── cv_pipeline/
│   ├── detection/          # YOLOv8 detection module
│   ├── tracking/           # BoxMOT tracking module
│   ├── pose_estimation/    # Pose estimator (YOLO-based)
│   ├── emotion_analysis/   # DeepFace & MiVOLO modules
│   ├── social_interaction/ # STAS Interaction Analyzer & Role Inference
│   ├── service_behavior/   # Specialized service/satisfaction analysis
│   └── utils/              # Scene describer (JSON) & utilities
├── scripts/
│   ├── run_full_pipeline.py  # Main entry point
│   ├── download_model.py     # Model downloader helper
│   └── visualize_tracking.py # Debugging visuals
├── models/                 # Model weights (YOLO, MiVOLO, etc.)
├── final_output.mp4        # Annotated high-res video output
└── scene_log.json          # Structured frame-by-frame event data (JSONL)
```

## 🛠 Quick Start

### Prerequisites
- **Python**: 3.11+
- **GPU**: NVIDIA GPU (GTX 1650 or better) with CUDA 12.1+
- **OS**: Windows (Recommended for performance)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Amzilynn/IA-Camera-Challenge.git
   cd IA-Camera-Challenge
   ```

2. **Setup Virtual Environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   # Install PyTorch with CUDA 12.1 support first
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

   # Install remaining requirements
   pip install -r requirements.txt

   # Install tf-keras for TensorFlow 2.15+ compatibility
   pip install tf-keras
   ```

4. **Download Models**
   Place `yolov8m.pt` and `yolov8m-pose.pt` in the root directory.
   Download MiVOLO weights (`model_imdb_cross_person_4.24_99.46.pth.tar`) from [MiVOLO Releases](https://github.com/WildChlamydia/MiVOLO/releases) and place them in `models/mivolo_imbd.pth.tar`.

### Usage

**Run the full pipeline:**
```bash
# Process default video (vd2.mp4)
python scripts/run_full_pipeline.py

# Process specific video file
python scripts/run_full_pipeline.py path/to/video.mp4

# Run on live webcam (Device index 0)
python scripts/run_full_pipeline.py 0
```

## 📊 Outputs

- **`final_output.mp4`**: Annotated video with bounding boxes, skeletons, emotion labels, and interaction tags.
- **`scene_log.json`**: Structured JSONL file containing:
  - `frame_idx`: Current frame number.
  - `persons`: List of detected individuals with ID, BBox, and attributes (Age, Gender, Emotion, Role, Posture).
  - `interactions`: List of social behaviors detected between persons.

## 🧠 Technical Details

| Module | Implementation | Model/Algorithm |
| :--- | :--- | :--- |
| **Detection** | YOLOv8 | `yolov8m.pt` |
| **Tracking** | BoxMOT | DeepOCSORT / BoostTrack |
| **Pose** | YOLO-Pose | `yolov8m-pose.pt` |
| **Demographics** | MiVOLO | Face + Body multi-modal analysis |
| **Emotion** | DeepFace | OpenCV backend for speed |
| **Social** | STAS | Custom Spatio-Temporal Interaction Logic |

## 👥 Authors
- **Amzilynn** - [GitHub Profile](https://github.com/Amzilynn)

## 🙏 Acknowledgments
- **Ultralytics** for YOLOv8.
- **mikel-brostrom** for BoxMOT.
- **MiVOLO** for demographic estimation.
- **DeepFace** for comprehensive facial analysis.
