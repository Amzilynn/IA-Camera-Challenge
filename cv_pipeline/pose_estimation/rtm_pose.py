import cv2
import numpy as np
import onnxruntime as ort
import os
import requests
from tqdm import tqdm

class RTMPoseEstimator:
    def __init__(self, model_path="models/rtmpose-m.onnx", device="cuda"):
        self.model_path = model_path
        self.device = device
        
        if not os.path.exists(model_path):
            self._download_model()
            
        # Smarter provider selection to avoid DLL load errors
        available_providers = ort.get_available_providers()
        print(f"I: Available ONNX providers: {available_providers}")
        
        if device == "cuda" and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
            
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            print(f"W: Failed to load ONNX with {providers}: {e}. Falling back to CPU.")
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        
        # RTMPose-m default input size
        self.input_size = (192, 256) # (w, h)
        
    def _download_model(self):
        print(f"I: Downloading RTMPose model to {self.model_path}...")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        # Using a reliable mirror for RTMPose-m ONNX
        url = "https://github.com/cst-pku/RTMPose-ONNX/releases/download/v1.0/rtmpose-m.onnx"
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(self.model_path, "wb") as f, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc="RTMPose"
        ) as pbar:
            for data in response.iter_content(1024):
                f.write(data)
                pbar.update(len(data))

    def _preprocess(self, img, bbox):
        """
        Crop and resize person for RTMPose.
        """
        x1, y1, x2, y2 = map(int, bbox)
        w, h = x2 - x1, y2 - y1
        
        # Pad bbox to maintain aspect ratio 3:4
        center = np.array([x1 + w/2, y1 + h/2])
        if w / h > 0.75:
            h = w / 0.75
        else:
            w = h * 0.75
            
        nx1 = int(center[0] - w/2)
        ny1 = int(center[1] - h/2)
        nx2 = int(center[0] + w/2)
        ny2 = int(center[1] + h/2)
        
        # Crop with padding
        img_h, img_w = img.shape[:2]
        pad_x1 = max(0, -nx1)
        pad_y1 = max(0, -ny1)
        pad_x2 = max(0, nx2 - img_w)
        pad_y2 = max(0, ny2 - img_h)
        
        crop = img[max(0, ny1):min(img_h, ny2), max(0, nx1):min(img_w, nx2)]
        crop = cv2.copyMakeBorder(crop, pad_y1, pad_y2, pad_x1, pad_x2, cv2.BORDER_CONSTANT, value=(0,0,0))
        
        # Resize to 192x256
        resized = cv2.resize(crop, self.input_size)
        
        # Prepare for ONNX
        input_tensor = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        input_tensor = input_tensor.astype(np.float32) / 255.0
        input_tensor = input_tensor[None, ...] # Add batch dim
        
        return input_tensor, (nx1, ny1, w, h)

    def estimate(self, frame, bbox):
        """
        Run inference on a single person bbox.
        """
        input_tensor, (ox, oy, box_w, box_h) = self._preprocess(frame, bbox)
        
        outputs = self.session.run(None, {self.input_name: input_tensor})
        simcc_x, simcc_y = outputs[0], outputs[1]
        
        # Decode SimCC to keypoints
        # RTMPose-m uses 17 keypoints (COCO format)
        kpts_x = np.argmax(simcc_x, axis=2)[0] / (simcc_x.shape[2] / self.input_size[0])
        kpts_y = np.argmax(simcc_y, axis=2)[0] / (simcc_y.shape[2] / self.input_size[1])
        conf = np.max(simcc_x, axis=2)[0] # Simplified confidence
        
        # Map back to global coordinates
        global_kpts = []
        for i in range(len(kpts_x)):
            gx = ox + (kpts_x[i] / self.input_size[0]) * box_w
            gy = oy + (kpts_y[i] / self.input_size[1]) * box_h
            global_kpts.append([gx, gy, float(conf[i])])
            
        return np.array(global_kpts)
