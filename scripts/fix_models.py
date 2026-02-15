import os
import requests
from tqdm import tqdm

MODELS = {
    "rtmpose-m.onnx": "https://github.com/Tau-J/rtmlib/releases/download/v0.0.1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx",
    "mivolo_imbd.pth.tar": "https://github.com/WildChlamydia/MiVOLO/releases/download/v1.0/model_imdb_cross_person_4.24_99.46.pth.tar",
    "yolov8n-face.pt": "https://huggingface.co/ElenaRyumina/MASAI_models/resolve/main/yolov8n-face.pt"
}

def download_model(name, url):
    dest = os.path.join("models", name)
    os.makedirs("models", exist_ok=True)
    
    print(f"Checking {name}...")
    
    # If file exists and is too small, delete it
    if os.path.exists(dest) and os.path.getsize(dest) < 1000:
        print(f"File {name} is corrupted (too small). Deleting and re-downloading...")
        os.remove(dest)
        
    if not os.path.exists(dest):
        print(f"Downloading {name} from {url}...")
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest, 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True, desc=name
            ) as pbar:
                for data in response.iter_content(1024):
                    size = f.write(data)
                    pbar.update(size)
            print(f"Successfully downloaded {name}")
        except Exception as e:
            print(f"Error downloading {name}: {e}")
    else:
        print(f"{name} already exists and looks valid ({os.path.getsize(dest)} bytes).")

if __name__ == "__main__":
    for name, url in MODELS.items():
        download_model(name, url)
