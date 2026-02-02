import torch
import numpy as np
import cv2
import os
import sys

# Add project root to path to ensure mivolo can be imported
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mivolo.model.mi_volo import MiVOLO
from mivolo.data.misc import prepare_classification_images

class MivoloAnalyzer:
    def __init__(self, checkpoint_path="models/mivolo_imbd.pth.tar", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        if not os.path.exists(checkpoint_path):
            print(f"W: MiVOLO checkpoint not found at {checkpoint_path}. Analyzer will return None.")
            self.model = None
            return

        try:
            # MiVOLO handles its own meta loading and model creation
            self.model = MiVOLO(
                ckpt_path=checkpoint_path,
                device=self.device,
                half=(self.device == "cuda"),
                use_persons=True,
                disable_faces=False,
                verbose=False
            )
            print(f"I: MiVOLO initialized on {self.device}")
        except Exception as e:
            print(f"E: Failed to initialize MiVOLO: {e}")
            self.model = None

    def analyze(self, frame, person_bbox, face_bbox=None):
        """
        Analyze age and gender using MiVOLO.
        Args:
            frame: Full video frame (numpy array)
            person_bbox: (x1, y1, x2, y2)
            face_bbox: (x1, y1, x2, y2) or None
        Returns:
            dict: {'age': float, 'gender': str} or None
        """
        if self.model is None:
            return None

        h, w = frame.shape[:2]
        
        # 1. Prepare Person Crop
        px1, py1, px2, py2 = map(int, person_bbox)
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(w, px2), min(h, py2)
        
        if px2 <= px1 or py2 <= py1:
            return None
            
        person_crop = frame[py1:py2, px1:px2]
        
        # 2. Prepare Face Crop
        face_crop = None
        if face_bbox is not None:
            fx1, fy1, fx2, fy2 = map(int, face_bbox)
            fx1, fy1 = max(0, fx1), max(0, fy1)
            fx2, fy2 = min(w, fx2), min(h, fy2)
            if fx2 > fx1 and fy2 > fy1:
                face_crop = frame[fy1:fy2, fx1:fx2]

        try:
            # MiVOLO expects lists of crops
            # Even if face is missing, pass None and prepare_classification_images handles it
            target_size = self.model.input_size
            
            # Use MiVOLO's own preprocessing
            faces_input = prepare_classification_images(
                [face_crop], 
                target_size, 
                self.model.data_config["mean"], 
                self.model.data_config["std"], 
                device=self.model.device
            )
            
            person_input = prepare_classification_images(
                [person_crop], 
                target_size, 
                self.model.data_config["mean"], 
                self.model.data_config["std"], 
                device=self.model.device
            )

            if self.model.meta.with_persons_model:
                model_input = torch.cat((faces_input, person_input), dim=1)
            else:
                model_input = faces_input

            # Inference
            output = self.model.inference(model_input)
            
            # Parse output
            # MiVOLO output format: [gender_male, gender_female, age_normalized]
            gender_output = output[:, :2].softmax(-1)
            gender_indx = gender_output.argmax(-1).item()
            gender = "male" if gender_indx == 0 else "female"
            
            age_norm = output[:, 2].item()
            age = age_norm * (self.model.meta.max_age - self.model.meta.min_age) + self.model.meta.avg_age
            age = round(float(age), 1)

            return {
                'age': age,
                'gender': gender
            }
            
        except Exception as e:
            # print(f"MiVOLO analysis failed: {e}")
            return None
