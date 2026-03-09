import requests
import zipfile
import os

url = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
dest = "cv_pipeline/models/rtmpose_pkg.zip"
os.makedirs("cv_pipeline/models", exist_ok=True)

print("Downloading RTMPose package...")
r = requests.get(url)
with open(dest, 'wb') as f:
    f.write(r.content)

print("Extracting...")
with zipfile.ZipFile(dest, 'r') as zip_ref:
    zip_ref.extractall("cv_pipeline/models/rtmpose_extracted")

# Find the .onnx file and move it
for root, dirs, files in os.walk("cv_pipeline/models/rtmpose_extracted"):
    for file in files:
        if file.endswith(".onnx"):
            os.rename(os.path.join(root, file), "cv_pipeline/models/rtmpose-m.onnx")
            print(f"Restored: {file}")

print("Cleanup...")
os.remove(dest)
import shutil
shutil.rmtree("cv_pipeline/models/rtmpose_extracted")
