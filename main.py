import torch
import torch.optim as optim

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
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
        train=True
    )

    val_loader = get_dataloader(
        csv_path=data_cfg["val_csv"],
        img_root=data_cfg["img_root"],
        batch_size=train_cfg["training"]["batch_size"],
        img_size=data_cfg["img_size"],
        sigma=data_cfg["heatmap_sigma"],
        train=False
    )

    # Model
    model = LA_UNet(
        num_landmarks=model_cfg["model"]["num_landmarks"]
    )

    # Loss
    criterion = HeatmapLoss(
        ssim_weight=train_cfg["training"]["loss"]["ssim_weight"]
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg["training"]["learning_rate"],
        weight_decay=train_cfg["training"]["weight_decay"]
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )

    # Train
    trainer.fit(
        epochs=train_cfg["training"]["epochs"],
        ckpt_path=train_cfg["outputs"]["checkpoint_path"]
    )


if __name__ == "__main__":
    main()
