# IA Camera Challenge - Scene Intelligence Pipeline

A high-performance computer vision pipeline for human-centric scene understanding, featuring Scene Graph Generation (SGG), real-time social interaction analysis, and a premium HUD visualization.

## Key Features

### Scene Intelligence
- **REACT++ Integration**: Leverages state-of-the-art Scene Graph Generation for real-time object and relationship detection.
- **Social Interaction Index (STAS)**: Advanced temporal analysis of human-to-human relationships, including talking, physical contact, and social proximity.
- **Automated Role Discovery**: Distinguishes between Staff and Visitors based on mobility, social centrality, and Hand-Object Interaction (HOI) patterns.
- **Temporal Smoothing**: Implements a 10-frame majority-voting buffer to eliminate semantic flickering and ensure stable interaction labels.

### Premium HUD Visualization
- **Glassmorphism Interface**: Semi-transparent, dark-themed intelligence cards for roles and satisfaction metrics.
- **Anti-Aliased Typography**: High-quality PIL-based rendering for professional-grade text and graphics.
- **Glow-Effect Relationships**: Cyan-glowing "energy cords" visualize social dynamics between individuals.
- **Customer Satisfaction Index**: Real-time dashboard calculating aggregate visitor happiness from emotion trends.

### Robust CV Core
- **Multi-Object Tracking**: DEEPOCSORT (via BoxMOT) for persistent person identification.
- **HSEmotion Analysis**: State-of-the-art emotion detection for real-time satisfaction monitoring.
- **MiVOLO Demographics**: Precise age and gender estimation using face and body features.
- **Structured Logging**: Frame-by-frame JSONL output for downstream data science and analytics.

## Repository Structure

```bash
IA-Camera-Challenge/
├── cv_pipeline/
│   ├── detection/          # REACT++ SGG & Human Detection
│   ├── tracking/           # BoxMOT tracking module
│   ├── pose_estimation/    # RTMPose & Pose Analysis
│   ├── emotion_analysis/   # HSEmotion & MiVOLO
│   ├── social_interaction/ # SocialAnalyzer & Role Discovery
│   └── utils/              # UIProcessor HUD & Scene Describer
├── scripts/
│   ├── run_full_pipeline.py  # Main entry point
│   └── download_model.py     # Model setup helper
├── models/                 # Model weights (REACT++, RTMPose, Face)
└── scene_log.json          # Structured frame-by-frame data
```

## Quick Start

### Prerequisites
- Python 3.11+
- NVIDIA GPU with CUDA 12.1+ support

### Installation
1. Setup virtual environment: `python -m venv venv`
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure model weights are placed in the models/ directory as specified in the configuration.

### Usage
```bash
python cv_pipeline/scripts/run_full_pipeline.py [video_path] --output [output_name.mp4]
```

## Outputs
- **Annotated Video**: High-quality MP4 with ID, Role, Satisfaction, and Interaction HUD.
- **Intelligence Log**: JSONL structured data file for high-level scene metrics and behavioral analysis.

---
**Maintained by**: Amzilynn | **Engine**: REACT++ SGG + Custom Scene Intelligence
