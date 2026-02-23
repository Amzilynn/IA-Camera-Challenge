from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import sys

app = FastAPI(title="IA Camera Challenge API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "online", "message": "IA Camera Challenge backend is running"}

@app.post("/process-video")
async def process_video(video_path: str, background_tasks: BackgroundTasks):
    """
    Trigger the CV pipeline for a given video file.
    Runs as a background task to avoid blocking the API response.
    """
    # Resolve path relative to project root if not absolute
    full_path = video_path
    if not os.path.isabs(video_path):
        # backend is in /backend, videos are in /
        root_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
        full_path = os.path.join(root_path, video_path)

    if not os.path.exists(full_path):
        return {"error": f"Video file not found: {video_path} at {full_path}"}
    
    video_path = full_path # Use the absolute path for the pipeline

    def run_pipeline():
        # Path to the script relative to project root
        script_path = os.path.join(os.getcwd(), "cv_pipeline", "scripts", "run_full_pipeline.py")
        try:
            # Run the existing pipeline script
            subprocess.run([sys.executable, script_path, video_path], check=True)
            print(f"I: Finished processing {video_path}")
        except Exception as e:
            print(f"E: Pipeline failed for {video_path}: {e}")

    background_tasks.add_task(run_pipeline)
    return {"message": "Processing started", "file": video_path}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
