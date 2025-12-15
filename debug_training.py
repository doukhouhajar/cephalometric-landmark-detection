import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.data.data_pipeline import get_dataloader
from src.models.la_unet import LA_UNet
from src.training.metrics import soft_argmax_2d
from src.training.utils import load_yaml


def visualize_predictions(model, dataloader, device, num_samples=3, save_dir="outputs/debug"):
    """Visualize model predictions to debug training."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
                
            images = batch["image"].to(device)
            heatmaps_gt = batch["heatmaps"].to(device)
            gt_coords = batch["coords"].to(device)
            
            # Get predictions
            preds = model(images, return_aux=False)
            if isinstance(preds, tuple):
                preds = preds[0]
            
            pred_coords = soft_argmax_2d(preds)
            
            # Process first sample in batch
            img = images[0, 0].cpu().numpy()  # First channel
            pred_hm = preds[0].cpu().numpy()  # (K, H, W)
            gt_hm = heatmaps_gt[0].cpu().numpy()
            pred_c = pred_coords[0].cpu().numpy()  # (K, 2)
            gt_c = gt_coords[0].cpu().numpy()
            
            K = pred_hm.shape[0]
            
            # Create figure
            fig, axes = plt.subplots(3, min(K, 6), figsize=(18, 10))
            
            for i in range(min(K, 6)):
                # Row 1: Image with landmarks
                axes[0, i].imshow(img, cmap='gray')
                axes[0, i].scatter(gt_c[i, 0], gt_c[i, 1], c='green', s=50, marker='o', label='GT')
                axes[0, i].scatter(pred_c[i, 0], pred_c[i, 1], c='red', s=50, marker='x', label='Pred')
                axes[0, i].set_title(f'L{i+1} err={np.linalg.norm(pred_c[i]-gt_c[i]):.1f}px')
                axes[0, i].axis('off')
                
                # Row 2: Ground truth heatmap
                axes[1, i].imshow(gt_hm[i], cmap='hot')
                axes[1, i].set_title(f'GT max={gt_hm[i].max():.3f}')
                axes[1, i].axis('off')
                
                # Row 3: Predicted heatmap
                axes[2, i].imshow(pred_hm[i], cmap='hot')
                axes[2, i].set_title(f'Pred max={pred_hm[i].max():.3f}')
                axes[2, i].axis('off')
            
            axes[0, 0].legend()
            plt.suptitle(f'Sample {batch_idx+1} - Green=GT, Red=Pred', fontsize=14)
            plt.tight_layout()
            plt.savefig(save_dir / f'debug_sample_{batch_idx}.png', dpi=150)
            plt.close()
            
            # Print stats
            print(f"\nSample {batch_idx+1}:")
            print(f"  Pred heatmap: min={pred_hm.min():.4f}, max={pred_hm.max():.4f}, mean={pred_hm.mean():.4f}")
            print(f"  GT heatmap:   min={gt_hm.min():.4f}, max={gt_hm.max():.4f}, mean={gt_hm.mean():.4f}")
            print(f"  Mean error:   {np.linalg.norm(pred_c - gt_c, axis=1).mean():.2f} px")
    
    print(f"\nDebug images saved to: {save_dir}")


def main():
    # Load config
    model_cfg = load_yaml("configs/model_config.yaml")
    train_cfg = load_yaml("configs/training_config.yaml")
    
    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Load data
    data_cfg = train_cfg["data"]
    val_loader = get_dataloader(
        csv_path=data_cfg["val_csv"],
        img_root=data_cfg["img_root"],
        batch_size=1,
        img_size=data_cfg["img_size"],
        sigma=data_cfg["heatmap_sigma"],
        train=False,
        device=device
    )
    
    # Load model
    model = LA_UNet(num_landmarks=model_cfg["model"]["num_landmarks"])
    
    checkpoint_path = train_cfg["outputs"]["checkpoint_path"]
    if Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("No checkpoint found, using random weights")
    
    model = model.to(device)
    
    # Visualize
    visualize_predictions(model, val_loader, device, num_samples=5)


if __name__ == "__main__":
    main()

