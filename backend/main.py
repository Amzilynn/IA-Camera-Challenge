from fastapi import FastAPI, BackgroundTasks, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import subprocess
import os
import sys
import json
import asyncio
import re
from collections import Counter

app = FastAPI(title="VisionIntel Pro API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve project root (one level up from /backend)
ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), ".."))
LOG_PATH = os.path.join(ROOT_PATH, "scene_log.json")
OUTPUT_VIDEO_PATH = os.path.join(ROOT_PATH, "final_output.mp4")

# Global pipeline state
pipeline_state = {
    "status": "idle",         # idle | processing | done | error
    "progress": 0,
    "total_frames": 0,
    "current_file": None,
    "video_url": None,
    "output_video": None,
    "error": None
}
pipeline_process = None

# Mount root as static to serve raw uploads
app.mount("/videos", StaticFiles(directory=ROOT_PATH), name="videos")


@app.get("/health")
async def health_check():
    return {"status": "online", "message": "VisionIntel Pro backend running"}


@app.get("/status")
async def get_status():
    """Returns the current pipeline processing status."""
    return pipeline_state


@app.get("/output-video")
async def get_output_video():
    """Serve the processed output video."""
    if not os.path.exists(OUTPUT_VIDEO_PATH):
        return {"error": "No output video available yet."}
    return FileResponse(
        OUTPUT_VIDEO_PATH,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"}
    )


@app.post("/reset")
async def reset_state():
    """Clear all processed data so the frontend starts fresh on reload."""
    global pipeline_state
    if pipeline_state["status"] in ["done", "error", "idle"]:
        pipeline_state.update({
            "status": "idle",
            "progress": 0,
            "total_frames": 0,
            "current_file": None,
            "video_url": None,
            "output_video": None,
            "error": None
        })
        if os.path.exists(LOG_PATH):
            try: os.remove(LOG_PATH)
            except: pass
        if os.path.exists(OUTPUT_VIDEO_PATH):
            try: os.remove(OUTPUT_VIDEO_PATH)
            except: pass
        if os.path.exists(OUTPUT_VIDEO_PATH + ".raw.mp4"):
            try: os.remove(OUTPUT_VIDEO_PATH + ".raw.mp4")
            except: pass
    return {"status": "ok"}


@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a video file and automatically start processing."""
    global pipeline_process, pipeline_state
    
    # 1. Clear old data if a pipeline is already running or done
    if pipeline_process:
        try:
            pipeline_process.terminate()
        except:
            pass
        pipeline_process = None

    file_path = os.path.join(ROOT_PATH, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Reset and clear old logs/videos
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    if os.path.exists(OUTPUT_VIDEO_PATH):
        try:
            os.remove(OUTPUT_VIDEO_PATH)
        except:
            pass

    # Reset state for new run
    pipeline_state.update({
        "status": "processing",
        "progress": 0,
        "total_frames": 0,
        "current_file": file.filename,
        "video_url": f"/videos/{file.filename}",
        "output_video": None,
        "error": None
    })

    def run_pipeline():
        global pipeline_process, pipeline_state
        script_path = os.path.join(ROOT_PATH, "cv_pipeline", "scripts", "run_full_pipeline.py")
        output_path = OUTPUT_VIDEO_PATH
        
        try:
            # Create an empty log file to start with
            with open(LOG_PATH, "w") as f:
                f.write("")

            pipeline_process = subprocess.Popen(
                [sys.executable, script_path, file_path, "--headless", "--log", LOG_PATH, "--output", output_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=ROOT_PATH
            )
            
            total_re = re.compile(r"(\d+)/(\d+)")
            for line in pipeline_process.stdout:
                m = total_re.search(line)
                if m:
                    curr, total = int(m.group(1)), int(m.group(2))
                    pipeline_state["progress"] = curr
                    pipeline_state["total_frames"] = total

            pipeline_process.wait()
            
            if pipeline_process.returncode == 0:
                pipeline_state["progress"] = pipeline_state["total_frames"]
                
                # TRANSCODE TO WEB-SAFE H.264
                # OpenCV's avc1 is often missing the moov atom or uses incorrect encoding settings for web.
                # We use imageio (which uses a bundled ffmpeg) to safely transcode it.
                tmp_output = output_path + ".raw.mp4"
                if os.path.exists(output_path):
                    import shutil
                    shutil.move(output_path, tmp_output)
                    
                    try:
                        import imageio
                        # Read the raw opencv output and write it strictly as web-safe h264
                        reader = imageio.get_reader(tmp_output)
                        fps = reader.get_meta_data()['fps']
                        writer = imageio.get_writer(output_path, format='FFMPEG', mode='I', fps=fps, codec='libx264', macro_block_size=None)
                        
                        for i, frame in enumerate(reader):
                            writer.append_data(frame)
                            
                        writer.close()
                        reader.close()
                        os.remove(tmp_output)
                        
                    except Exception as transcode_err:
                        print(f"Transcoding failed (falling back to raw): {transcode_err}")
                        if os.path.exists(tmp_output) and not os.path.exists(output_path):
                            shutil.move(tmp_output, output_path)

                pipeline_state["status"] = "done"
                pipeline_state["output_video"] = "/output-video"
            else:
                pipeline_state["status"] = "error"
                pipeline_state["error"] = f"Pipeline exited with code {pipeline_process.returncode}"
        except Exception as e:
            pipeline_state["status"] = "error"
            pipeline_state["error"] = str(e)
        finally:
            pipeline_process = None
    
    background_tasks.add_task(run_pipeline)
    
    return {
        "message": "Upload successful. Processing started.",
        "filename": file.filename,
        "video_url": f"/videos/{file.filename}"
    }


@app.get("/stats/summary")
async def get_stats_summary():
    """Calculates high-level statistics from scene_log.json."""
    if not os.path.exists(LOG_PATH):
        return {
            "total_frames": 0, "unique_persons_count": 0,
            "emotions_breakdown": {}, "total_interactions": 0,
            "interaction_types": {}, "roles_breakdown": {},
            "intents_breakdown": {}, "avg_persons_per_frame": 0
        }
    
    unique_persons = set()
    emotions = Counter()
    roles = Counter()
    intents = Counter()
    interaction_types = Counter()
    total_interactions = 0
    total_persons_sum = 0
    total_frames = 0
    
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    frame = json.loads(line)
                    total_frames += 1
                    persons = frame.get("persons", [])
                    total_persons_sum += len(persons)
                    
                    for p in persons:
                        unique_persons.add(p["id"])
                        attrs = p.get("attributes", {})
                        if attrs.get("emotion"):
                            emotions[attrs["emotion"]] += 1
                        if attrs.get("role"):
                            roles[attrs["role"]] += 1
                        if attrs.get("intent"):
                            intents[attrs["intent"]] += 1
                    
                    for inter in frame.get("interactions", []):
                        total_interactions += 1
                        interaction_types[inter.get("type", "Unknown")] += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return {"error": str(e)}
        
    # --- Synthetic Retail Metrics Generation ---
    # To demonstrate business value, we generate realistic retail analytics 
    # based on the true underlying detections and frames.
    
    # 1. Avg Dwell Time (synthetic calculation based on frames and persons)
    # Assume 30fps default if not perfectly accurate, add slight variance based on person count
    base_dwell = total_frames / 30.0 
    avg_dwell_time_seconds = round(base_dwell * 0.4 + (len(unique_persons) * 1.5), 1) if unique_persons else 0.0

    # 2. Engagement Score (synthetic percentage based on active interactions)
    # The more interactions we detect relative to the crowd size, the higher the engagement.
    engagement_base = (total_interactions / max(len(unique_persons), 1)) * 25
    engagement_score = min(max(round(30 + engagement_base, 1), 0), 98.5) if unique_persons else 0.0
    
    # 3. Zone Activity (synthetic distribution of traffic across physical areas)
    # We use the unique_persons count and total_interactions as a seed for stable synthetic generation
    seed_val = len(unique_persons) + total_interactions
    zone_activity = {
        "Store Entrance": int(len(unique_persons) * 0.9 + (seed_val % 3)),
        "Main Aisle": int(len(unique_persons) * 0.6 + (seed_val % 4)),
        "Promo Display": int(len(unique_persons) * 0.4 + total_interactions),
        "Checkout": int(len(unique_persons) * 0.5 + (seed_val % 2))
    } if unique_persons else {}
    
    return {
        "total_frames": total_frames,
        "unique_persons_count": len(unique_persons),
        "emotions_breakdown": dict(emotions),
        "total_interactions": total_interactions,
        "interaction_types": dict(interaction_types),
        "roles_breakdown": dict(roles),
        "intents_breakdown": dict(intents),
        "avg_persons_per_frame": round(total_persons_sum / max(total_frames, 1), 2),
        "avg_dwell_time_seconds": avg_dwell_time_seconds,
        "engagement_score": engagement_score,
        "zone_activity": zone_activity
    }


@app.get("/scene-data")
async def get_scene_data():
    """Returns the contents of scene_log.json."""
    if not os.path.exists(LOG_PATH):
        return {"error": "Scene log not found", "data": []}
    
    data = []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        return {"error": str(e), "data": []}
    
    return {"data": data}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    pos = 0

    try:
        while True:
            # Send pipeline state on every tick
            try:
                await websocket.send_text(json.dumps({"__type": "status", **pipeline_state}))
            except:
                break
            
            if os.path.exists(LOG_PATH):
                # If file was truncated or replaced, reset position
                if os.path.getsize(LOG_PATH) < pos:
                    pos = 0
                
                with open(LOG_PATH, "r", encoding="utf-8") as f:
                    f.seek(pos)
                    lines = f.readlines()
                    if lines:
                        for line in lines:
                            if line.strip():
                                try:
                                    # Validate it's proper JSON before sending
                                    json.loads(line)
                                    await websocket.send_text(line)
                                except:
                                    continue
                        pos = f.tell()
            
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, asyncio.CancelledError, ConnectionResetError):
        pass
    except Exception as e:
        # Ignore standard Windows connection reset errors commonly thrown by asyncio/uvicorn
        if "10054" in str(e) or "10053" in str(e):
            pass
        else:
            print(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
