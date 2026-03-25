import os
import requests
from tqdm import tqdm

# Common base models used across sub-modules
BASE_MODELS = {
    "cv_pipeline/models/yolov8x_person_face.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt",
    "cv_pipeline/models/yolov8n-face.pt": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/yolov8n-face.pt",
    "cv_pipeline/models/yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
    "cv_pipeline/models/yolov8m-pose.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-pose.pt"
}

# Specific models for this module
MODELS = {
    "cv_pipeline/models/rtmpose-m.onnx": "https://github.com/Tau-J/rtmlib/releases/download/v0.0.1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx",
    "cv_pipeline/models/mivolo_imbd.pth.tar": "https://github.com/WildChlamydia/MiVOLO/releases/download/v1.0/model_imdb_cross_person_4.24_99.46.pth.tar",
}

def download_model(name, url):
    dest = name
    model_dir = os.path.dirname(dest)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    
    # Check current size
    curr_size = os.path.getsize(dest) if os.path.exists(dest) else 0
    if curr_size < 1000 and os.path.exists(dest):
        print(f"I: Deleting invalid/small file {name} ({curr_size} bytes)")
        os.remove(dest)
        
    if not os.path.exists(dest):
        print(f"Downloading {name} from {url}...")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        try:
            response = requests.get(url, stream=True, headers=headers, allow_redirects=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            if total_size < 1000 and total_size != 0:
                print(f"W: Server returned small content-length ({total_size}) for {name}. This might be an error page.")
            
            with open(dest, 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True, desc=name
            ) as pbar:
                for data in response.iter_content(chunk_size=8192):
                    size = f.write(data)
                    pbar.update(size)
            print(f"Successfully downloaded {name}")
        except Exception as e:
            print(f"Error downloading {name}: {e}")
            if os.path.exists(dest): os.remove(dest)
    else:
        print(f"{name} already exists and looks valid ({os.path.getsize(dest)//1024//1024} MB).")

if __name__ == "__main__":
    # Download base models
    for name, url in BASE_MODELS.items():
        download_model(name, url)
        
    # Download specific models
    for name, url in MODELS.items():
        download_model(name, url)
