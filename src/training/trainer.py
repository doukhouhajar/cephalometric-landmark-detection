import torch
from torch import nn
import os
from tqdm import tqdm

from .metrics import soft_argmax_2d, mean_radial_error, success_detection_rate


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device="cuda"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.train_loader):
            images = batch["image"].to(self.device)
            heatmaps = batch["heatmaps"].to(self.device)

            self.optimizer.zero_grad()

            preds = self.model(images)

            loss = self.criterion(preds, heatmaps)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        mre = 0
        sdr_2mm = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                heatmaps = batch["heatmaps"].to(self.device)
                gt_coords = batch["coords"].to(self.device)

                preds = self.model(images)

                loss = self.criterion(preds, heatmaps)
                total_loss += loss.item()

                pred_coords = soft_argmax_2d(preds)

                mre += mean_radial_error(pred_coords, gt_coords)
                sdr_2mm += success_detection_rate(pred_coords, gt_coords, threshold=2.0)

        return {
            "loss": total_loss / len(self.val_loader),
            "mre": mre / len(self.val_loader),
            "sdr_2mm": sdr_2mm / len(self.val_loader),
        }

    def fit(self, epochs, ckpt_path="outputs/checkpoints/model.pth"):
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_stats = self.validate()

            print(f"\nEpoch {epoch}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_stats['loss']:.4f}")
            print(f"MRE:        {val_stats['mre']:.2f} px")
            print(f"SDR(2mm):   {val_stats['sdr_2mm']*100:.2f}%")

            if val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(self.model.state_dict(), ckpt_path)
                print("Saved.\n")
