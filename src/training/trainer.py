import torch
from torch import nn
import os
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path

from .metrics import (
    soft_argmax_2d, 
    mean_radial_error, 
    success_detection_rate,
    compute_comprehensive_metrics
)


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 device="cuda", log_dir="outputs/logs", use_tensorboard=True):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard if requested
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                print("TensorBoard not available, skipping...")
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_mre": [],
            "val_sdr": {}
        }

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.train_loader):
            images = batch["image"].to(self.device)
            heatmaps = batch["heatmaps"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with multi-resolution outputs
            outputs = self.model(images, return_aux=True)
            if isinstance(outputs, tuple):
                preds, aux2, aux3, aux4 = outputs
                aux_outputs = [aux2, aux3, aux4]
            else:
                preds = outputs
                aux_outputs = None

            loss = self.criterion(preds, heatmaps, aux_outputs=aux_outputs)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_pred_coords = []
        all_gt_coords = []

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                heatmaps = batch["heatmaps"].to(self.device)
                gt_coords = batch["coords"].to(self.device)

                # Forward pass (no aux outputs needed for validation)
                outputs = self.model(images, return_aux=False)
                preds = outputs if not isinstance(outputs, tuple) else outputs[0]

                loss = self.criterion(preds, heatmaps)
                total_loss += loss.item()

                pred_coords = soft_argmax_2d(preds)
                all_pred_coords.append(pred_coords.cpu())
                all_gt_coords.append(gt_coords.cpu())

        # Concatenate all predictions
        all_pred_coords = torch.cat(all_pred_coords, dim=0)
        all_gt_coords = torch.cat(all_gt_coords, dim=0)
        
        # Compute comprehensive metrics
        metrics = compute_comprehensive_metrics(
            all_pred_coords, 
            all_gt_coords, 
            thresholds=[2.0, 2.5, 3.0, 4.0]
        )

        return {
            "loss": total_loss / len(self.val_loader),
            "mre": metrics["mre"],
            "sdr_2mm": metrics["sdr"][2.0],
            "sdr_2p5mm": metrics["sdr"][2.5],
            "sdr_3mm": metrics["sdr"][3.0],
            "sdr_4mm": metrics["sdr"][4.0],
            "comprehensive": metrics,  # For detailed analysis
        }

    def fit(self, epochs, ckpt_path="outputs/checkpoints/model.pth"):
        best_val_loss = float("inf")
        best_val_mre = float("inf")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_stats = self.validate()

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_stats["loss"])
            self.history["val_mre"].append(val_stats["mre"])
            # Update history with correct key names
            threshold_map = {2.0: "sdr_2mm", 2.5: "sdr_2p5mm", 3.0: "sdr_3mm", 4.0: "sdr_4mm"}
            for threshold, key in threshold_map.items():
                if key not in self.history["val_sdr"]:
                    self.history["val_sdr"][key] = []
                self.history["val_sdr"][key].append(val_stats[key])

            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar("Loss/Train", train_loss, epoch)
                self.writer.add_scalar("Loss/Val", val_stats["loss"], epoch)
                self.writer.add_scalar("Metrics/MRE", val_stats["mre"], epoch)
                # Map thresholds to correct key names
                threshold_map = {2.0: "sdr_2mm", 2.5: "sdr_2p5mm", 3.0: "sdr_3mm", 4.0: "sdr_4mm"}
                for threshold, key in threshold_map.items():
                    self.writer.add_scalar(
                        f"Metrics/SDR_{threshold}mm", 
                        val_stats[key], 
                        epoch
                    )

            print(f"\nEpoch {epoch}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_stats['loss']:.4f}")
            print(f"MRE:        {val_stats['mre']:.2f} px")
            print(f"SDR(2mm):   {val_stats['sdr_2mm']*100:.2f}%")
            print(f"SDR(2.5mm): {val_stats['sdr_2p5mm']*100:.2f}%")
            print(f"SDR(3mm):   {val_stats['sdr_3mm']*100:.2f}%")
            print(f"SDR(4mm):   {val_stats['sdr_4mm']*100:.2f}%")

            # Save best model
            if val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]
                best_val_mre = val_stats["mre"]
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_stats["loss"],
                    "val_mre": val_stats["mre"],
                }, ckpt_path)
                print("Saved.\n")
        
        # Save training history
        history_path = self.log_dir / "training_history.json"
        # Convert numpy arrays to lists for JSON serialization
        history_json = {}
        for key, value in self.history.items():
            if key == "val_sdr":
                history_json[key] = {k: [float(vv) for vv in v] for k, v in value.items()}
            else:
                history_json[key] = [float(v) for v in value]
        
        with open(history_path, "w") as f:
            json.dump(history_json, f, indent=2)
        
        if self.writer:
            self.writer.close()
        
        print(f"\nTraining completed!")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"Best Val MRE: {best_val_mre:.2f} px")
        print(f"Training history saved to: {history_path}")
