import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from src.training.metrics import soft_argmax_2d


class LandmarkDetector:
    def __init__(self, model, device="cuda", img_size=224):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.img_size = img_size
    
    @torch.no_grad()
    def predict(self, image_path_or_array, return_heatmaps=False):
        # Load and preprocess image
        if isinstance(image_path_or_array, (str, Path)):
            img = cv2.imread(str(image_path_or_array), cv2.IMREAD_GRAYSCALE)
        else:
            img = image_path_or_array
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        original_h, original_w = img.shape
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Convert to tensor
        img_tensor = torch.tensor(img_normalized).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        img_tensor = img_tensor.repeat(1, 3, 1, 1)  # (1, 3, H, W) for Swin
        img_tensor = img_tensor.to(self.device)
        
        # Predict
        outputs = self.model(img_tensor, return_aux=False)
        pred_heatmaps = outputs if not isinstance(outputs, tuple) else outputs[0]
        
        # Extract coordinates
        pred_coords = soft_argmax_2d(pred_heatmaps)  # (1, K, 2)
        pred_coords = pred_coords[0].cpu().numpy()  # (K, 2)
        
        # Scale coordinates back to original image size
        pred_coords[:, 0] = pred_coords[:, 0] * (original_w / self.img_size)
        pred_coords[:, 1] = pred_coords[:, 1] * (original_h / self.img_size)
        
        if return_heatmaps:
            heatmaps = pred_heatmaps[0].cpu().numpy()  # (K, H, W)
            return pred_coords, heatmaps
        return pred_coords
    
    def predict_batch(self, image_paths, return_heatmaps=False):
        results = []
        for img_path in image_paths:
            result = self.predict(img_path, return_heatmaps=return_heatmaps)
            results.append(result)
        return results


def load_model(checkpoint_path, model_class, num_landmarks=19, device="cuda"):
    model = model_class(num_landmarks=num_landmarks)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model

