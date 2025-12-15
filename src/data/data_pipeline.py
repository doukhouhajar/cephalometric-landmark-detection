import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from .heatmap_generator import generate_landmark_heatmaps 

def get_transforms(img_size=512, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.HorizontalFlip(p=0.5),
            ToTensorV2()
        ], is_check_shapes=False)  # Disable shape checking to avoid warnings
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            ToTensorV2()
        ], is_check_shapes=False)

class CephaloDataset(Dataset):
    def __init__(self, csv_path, img_root, img_size=512, sigma=5, transforms=None):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.img_size = img_size
        self.sigma = sigma
        self.transforms = transforms 
        self.K = (self.df.shape[1] - 1) // 2 # number of landmarks

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        image_path = os.path.join(self.img_root, row["image_path"])
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        original_h, original_w = img.shape

        # extract and scale coordinates
        coords = []
        for i in range(1, self.K + 1):
            x = row[f"{i}_x"]
            y = row[f"{i}_y"]

            x_scaled = x * (self.img_size / original_w)
            y_scaled = y * (self.img_size / original_h)

            coords.append((x_scaled, y_scaled))

        # apply augmentations 
        if self.transforms is not None:
            transformed = self.transforms(image=img)
            img_tensor = transformed["image"]  # (1, H, W)
        else:
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            img_tensor = torch.tensor(img).unsqueeze(0)

        # Convert grayscale -> 3-channel (for Swin)
        img_tensor = img_tensor.repeat(3, 1, 1)

        heatmaps = generate_landmark_heatmaps(
            self.img_size, self.img_size, coords, sigma=self.sigma
        )

        heatmaps_tensor = torch.tensor(heatmaps)

        return {
            "image": img_tensor.float(),          # (1, H, W)
            "heatmaps": heatmaps_tensor.float(),  # (K, H, W)
            "coords": torch.tensor(coords).float(),
            "img_path": image_path
        }


def get_dataloader(csv_path, img_root, batch_size=4, img_size=512, sigma=5, train=True, device="cuda"):

    transforms = get_transforms(img_size=img_size, is_train=train)

    dataset = CephaloDataset(
        csv_path=csv_path,
        img_root=img_root,
        img_size=img_size,
        sigma=sigma,
        transforms=transforms
    )

    # pin_memory only works with CUDA, not MPS (Mac GPU)
    pin_memory = device == "cuda" and torch.cuda.is_available()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=pin_memory
    )

    return loader
