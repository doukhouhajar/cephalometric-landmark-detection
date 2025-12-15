"""
Analysis script for generating comprehensive plots and comparisons for academic report.
"""
import torch
import numpy as np
import json
from pathlib import Path
import argparse
from tqdm import tqdm

from src.data.data_pipeline import get_dataloader
from src.models.la_unet import LA_UNet
from src.models.baseline_unet import BaselineUNet
from src.inference.inference import LandmarkDetector, load_model
from src.training.metrics import compute_comprehensive_metrics, soft_argmax_2d
from src.visualization.visualizer import (
    plot_training_curves,
    plot_error_distribution,
    plot_per_landmark_metrics,
    plot_model_comparison,
    visualize_heatmaps_with_predictions,
    create_sample_overlay
)


def evaluate_model(model, dataloader, device, model_name="Model"):
    """Evaluate a model and return comprehensive metrics."""
    model.eval()
    all_pred_coords = []
    all_gt_coords = []
    sample_predictions = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {model_name}")):
            images = batch["image"].to(device)
            heatmaps = batch["heatmaps"].to(device)
            gt_coords = batch["coords"].to(device)
            
            outputs = model(images, return_aux=False)
            pred_heatmaps = outputs if not isinstance(outputs, tuple) else outputs[0]
            pred_coords = soft_argmax_2d(pred_heatmaps)
            
            all_pred_coords.append(pred_coords.cpu())
            all_gt_coords.append(gt_coords.cpu())
            
            # Store first few samples for visualization
            if i < 3:
                sample_predictions.append({
                    'image': images[0].cpu().numpy()[0],  # Grayscale channel
                    'heatmaps': pred_heatmaps[0].cpu().numpy(),
                    'pred_coords': pred_coords[0].cpu().numpy(),
                    'gt_coords': gt_coords[0].cpu().numpy(),
                })
    
    all_pred_coords = torch.cat(all_pred_coords, dim=0)
    all_gt_coords = torch.cat(all_gt_coords, dim=0)
    
    metrics = compute_comprehensive_metrics(
        all_pred_coords, 
        all_gt_coords, 
        thresholds=[2.0, 2.5, 3.0, 4.0]
    )
    
    return metrics, sample_predictions


def main():
    parser = argparse.ArgumentParser(description="Analyze model results and generate plots")
    parser.add_argument("--checkpoint", type=str, 
                       default="outputs/checkpoints/la_unet_swin_cbam.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--baseline-checkpoint", type=str, default=None,
                       help="Path to baseline model checkpoint (optional)")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to training config")
    parser.add_argument("--output-dir", type=str, default="outputs/analysis",
                       help="Output directory for plots")
    parser.add_argument("--test-csv", type=str, default="datasets/raw/test1_senior.csv",
                       help="Path to test CSV")
    parser.add_argument("--img-root", type=str, default="datasets/raw/cepha400/cepha400",
                       help="Image root directory")
    parser.add_argument("--num-samples", type=int, default=10,
                       help="Number of sample visualizations to generate")
    
    args = parser.parse_args()
    
    # Load config
    from src.training.utils import load_yaml
    config = load_yaml(args.config)
    data_cfg = config["data"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test dataloader
    test_loader = get_dataloader(
        csv_path=args.test_csv,
        img_root=args.img_root,
        batch_size=4,
        img_size=data_cfg["img_size"],
        sigma=data_cfg["heatmap_sigma"],
        train=False
    )
    
    results_dict = {}
    
    # Evaluate main model
    print("Evaluating LA-UNet with Swin Transformer...")
    model = load_model(args.checkpoint, LA_UNet, num_landmarks=19, device=device)
    metrics, samples = evaluate_model(model, test_loader, device, "LA-UNet-Swin")
    results_dict["LA-UNet-Swin"] = metrics
    
    # Save sample visualizations
    print(f"Generating sample visualizations...")
    for i, sample in enumerate(samples):
        fig = visualize_heatmaps_with_predictions(
            sample['image'],
            sample['heatmaps'],
            sample['pred_coords'],
            sample['gt_coords'],
            save_path=output_dir / f"sample_{i}_heatmaps.png"
        )
        plt.close(fig)
        
        fig = create_sample_overlay(
            sample['image'],
            sample['pred_coords'],
            sample['gt_coords'],
            save_path=output_dir / f"sample_{i}_overlay.png"
        )
        plt.close(fig)
    
    # Evaluate baseline if provided
    if args.baseline_checkpoint and Path(args.baseline_checkpoint).exists():
        print("Evaluating Baseline U-Net...")
        baseline_model = load_model(args.baseline_checkpoint, BaselineUNet, 
                                   num_landmarks=19, device=device)
        baseline_metrics, _ = evaluate_model(baseline_model, test_loader, device, "Baseline-UNet")
        results_dict["Baseline-UNet"] = baseline_metrics
    
    # Generate plots
    print("Generating analysis plots...")
    
    # 1. Error distribution
    fig = plot_error_distribution(
        metrics['all_errors'],
        save_path=output_dir / "error_distribution.png"
    )
    plt.close(fig)
    
    # 2. Per-landmark metrics
    fig = plot_per_landmark_metrics(
        metrics['per_landmark_mre'],
        metrics['per_landmark_sdr'],
        save_path=output_dir / "per_landmark_metrics.png"
    )
    plt.close(fig)
    
    # 3. Model comparison (if baseline available)
    if len(results_dict) > 1:
        # Convert metrics format for comparison plot
        comparison_dict = {}
        for name, m in results_dict.items():
            comparison_dict[name] = {
                'mre': m['mre'],
                'sdr': m['sdr']
            }
        fig = plot_model_comparison(
            comparison_dict,
            save_path=output_dir / "model_comparison.png"
        )
        plt.close(fig)
    
    # 4. Training curves (if available)
    history_path = Path("outputs/logs/training_history.json")
    if history_path.exists():
        fig = plot_training_curves(
            history_path,
            save_path=output_dir / "training_curves.png"
        )
        plt.close(fig)
    
    # Save metrics to JSON
    metrics_json = {}
    for name, m in results_dict.items():
        metrics_json[name] = {
            'mre': float(m['mre']),
            'per_landmark_mre': m['per_landmark_mre'].tolist(),
            'sdr': {k: float(v) for k, v in m['sdr'].items()},
            'per_landmark_sdr': {k: v.tolist() for k, v in m['per_landmark_sdr'].items()},
        }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for name, m in results_dict.items():
        print(f"\n{name}:")
        print(f"  MRE: {m['mre']:.2f} px")
        print(f"  SDR @ 2mm: {m['sdr'][2.0]*100:.2f}%")
        print(f"  SDR @ 2.5mm: {m['sdr'][2.5]*100:.2f}%")
        print(f"  SDR @ 3mm: {m['sdr'][3.0]*100:.2f}%")
        print(f"  SDR @ 4mm: {m['sdr'][4.0]*100:.2f}%")
    print("\n" + "="*60)
    print(f"Plots saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

