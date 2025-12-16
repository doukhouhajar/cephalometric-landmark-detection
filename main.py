import os
import torch
import torch.optim as optim
import warnings

# Suppress albumentations version check warnings
os.environ["ALBUMENTATIONS_CHECK_VERSION"] = "0"
warnings.filterwarnings("ignore", message=".*albumentations.*")

from src.data.data_pipeline import get_dataloader
from src.models.la_unet import LA_UNet
from src.training.losses import HeatmapLoss
from src.training.trainer import Trainer
from src.training.utils import load_yaml


def main():

    # LOAD CONFIGS
    model_cfg = load_yaml("configs/model_config.yaml")
    train_cfg = load_yaml("configs/training_config.yaml")

    # Device
    device_cfg = train_cfg["training"]["device"]
    if device_cfg == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"  # Mac GPU
        else:
            device = "cpu"
    else:
        device = device_cfg

    # Data
    data_cfg = train_cfg["data"]

    train_loader = get_dataloader(
        csv_path=data_cfg["train_csv"],
        img_root=data_cfg["img_root"],
        batch_size=train_cfg["training"]["batch_size"],
        img_size=data_cfg["img_size"],
        sigma=data_cfg["heatmap_sigma"],
        train=True,
        device=device
    )

    val_loader = get_dataloader(
        csv_path=data_cfg["val_csv"],
        img_root=data_cfg["img_root"],
        batch_size=train_cfg["training"]["batch_size"],
        img_size=data_cfg["img_size"],
        sigma=data_cfg["heatmap_sigma"],
        train=False,
        device=device
    )

    # Model
    model = LA_UNet(
        num_landmarks=model_cfg["model"]["num_landmarks"]
    )

    # Loss - Combined weighted heatmap + coordinate supervision
    loss_cfg = train_cfg["training"]["loss"]
    criterion = HeatmapLoss(
        coord_weight=loss_cfg.get("coord_weight", 5.0),  # Strong coordinate supervision
        use_deep_supervision=loss_cfg.get("use_deep_supervision", True),
        deep_supervision_weights=loss_cfg.get("deep_supervision_weights", [1.0, 0.4, 0.2, 0.1]),
        loss_type=loss_cfg.get("type", "combined")
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg["training"]["learning_rate"],
        weight_decay=train_cfg["training"]["weight_decay"]
    )

    # Learning rate scheduler
    scheduler = None
    scheduler_cfg = train_cfg["training"].get("scheduler", {})
    if scheduler_cfg.get("use", False):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_cfg["training"]["epochs"] - scheduler_cfg.get("warmup_epochs", 5),
            eta_min=scheduler_cfg.get("min_lr", 1e-6)
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )

    # Train
    trainer.fit(
        epochs=train_cfg["training"]["epochs"],
        ckpt_path=train_cfg["outputs"]["checkpoint_path"]
    )


if __name__ == "__main__":
    main()
