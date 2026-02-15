import os
import requests
import zipfile
import io
from tqdm import tqdm

def download_and_extract_rtmpose():
    url = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
    dest_path = "models/rtmpose-m.onnx"
    os.makedirs("models", exist_ok=True)
    
    print(f"Downloading RTMPose-m ZIP from {url}...")
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        zip_data = io.BytesIO()
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc="RTMPose ZIP") as pbar:
            for data in response.iter_content(1024):
                size = zip_data.write(data)
                pbar.update(size)
        
        print("Extracting ONNX model...")
        with zipfile.ZipFile(zip_data) as z:
            # Find the onnx file in the zip
            onnx_files = [f for f in z.namelist() if f.endswith('.onnx')]
            if onnx_files:
                with z.open(onnx_files[0]) as source, open(dest_path, 'wb') as target:
                    target.write(source.read())
                print(f"Successfully extracted {onnx_files[0]} to {dest_path}")
            else:
                print("No .onnx file found in the ZIP.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    download_and_extract_rtmpose()
