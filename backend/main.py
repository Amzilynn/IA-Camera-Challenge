from fastapi import FastAPI, BackgroundTasks, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
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

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a video file and automatically start processing.
    """
    root_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    file_path = os.path.join(root_path, file.filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Trigger processing
    await process_video(file.filename, background_tasks)
    
    return {"message": "Upload successful and processing started", "filename": file.filename}

@app.get("/stats/summary")
async def get_stats_summary():
    """
    Calculates high-level statistics from scene_log.json.
    """
    log_path = os.path.join(os.getcwd(), "..", "scene_log.json")
    if not os.path.exists(log_path):
        return {"total_frames": 0, "unique_persons": 0, "emotions": {}, "interactions": 0}
    
    stats = {
        "total_frames": 0,
        "unique_persons": set(),
        "emotions": {},
        "interaction_count": 0,
        "interaction_types": {}
    }
    
    try:
        with open(log_path, "r") as f:
            for line in f:
                if not line.strip(): continue
                frame = json.loads(line)
                stats["total_frames"] += 1
                
                for p in frame.get("persons", []):
                    stats["unique_persons"].add(p["id"])
                    emo = p.get("attributes", {}).get("emotion")
                    if emo:
                        stats["emotions"][emo] = stats["emotions"].get(emo, 0) + 1
                
                for inter in frame.get("interactions", []):
                    stats["interaction_count"] += 1
                    itype = inter.get("type")
                    stats["interaction_types"][itype] = stats["interaction_types"].get(itype, 0) + 1
    except Exception as e:
        return {"error": str(e)}
    
    return {
        "total_frames": stats["total_frames"],
        "unique_persons_count": len(stats["unique_persons"]),
        "emotions_breakdown": stats["emotions"],
        "total_interactions": stats["interaction_count"],
        "interaction_types": stats["interaction_types"]
    }

@app.get("/scene-data")
async def get_scene_data():
    """
    Returns the current contents of scene_log.json.
    """
    log_path = os.path.join(os.getcwd(), "..", "scene_log.json")
    if not os.path.exists(log_path):
        return {"error": "Scene log not found", "data": []}
    
    data = []
    try:
        with open(log_path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except Exception as e:
        return {"error": str(e), "data": []}
    
    return {"data": data}

from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    log_path = os.path.join(os.getcwd(), "..", "scene_log.json")
    last_line_sent = 0
    
    try:
        while True:
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    lines = f.readlines()
                    if len(lines) > last_line_sent:
                        # Send new lines
                        for line in lines[last_line_sent:]:
                            if line.strip():
                                await websocket.send_text(line)
                        last_line_sent = len(lines)
            
            await asyncio.sleep(0.5) # Poll for file changes every 500ms
    except WebSocketDisconnect:
        print("Client disconnected from WebSocket")
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
