"""
Visualization utilities for cephalometric landmark detection analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from matplotlib.colors import LinearSegmentedColormap
# import seaborn as sns
from pathlib import Path
import torch
import json

from src.training.metrics import compute_comprehensive_metrics


def visualize_heatmaps_with_predictions(image, heatmaps, pred_coords, gt_coords, 
                                       landmark_names=None, save_path=None, 
                                       num_cols=5, figsize=(20, 16)):
    """
    Visualize input image with predicted and ground truth landmarks overlaid on heatmaps.
    
    Args:
        image: (H, W) numpy array
        heatmaps: (K, H, W) numpy array of predicted heatmaps
        pred_coords: (K, 2) numpy array of predicted coordinates
        gt_coords: (K, 2) numpy array of ground truth coordinates
        landmark_names: List of landmark names (optional)
        save_path: Path to save figure
        num_cols: Number of columns in grid
        figsize: Figure size
    """
    K = heatmaps.shape[0]
    num_rows = (K + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten() if K > 1 else [axes]
    
    for i in range(K):
        ax = axes[i]
        
        # Overlay heatmap on grayscale image
        ax.imshow(image, cmap='gray', alpha=0.5)
        heatmap = heatmaps[i]
        ax.imshow(heatmap, cmap='hot', alpha=0.6, interpolation='bilinear')
        
        # Ground truth (green)
        if gt_coords[i, 0] >= 0 and gt_coords[i, 1] >= 0:
            ax.plot(gt_coords[i, 0], gt_coords[i, 1], 'go', markersize=10, 
                   label='GT', markeredgecolor='darkgreen', markeredgewidth=2)
        
        # Prediction (red)
        ax.plot(pred_coords[i, 0], pred_coords[i, 1], 'rx', markersize=12, 
               linewidth=3, label='Pred')
        
        # Error line
        if gt_coords[i, 0] >= 0 and gt_coords[i, 1] >= 0:
            error = np.linalg.norm(pred_coords[i] - gt_coords[i])
            ax.plot([pred_coords[i, 0], gt_coords[i, 0]], 
                   [pred_coords[i, 1], gt_coords[i, 1]], 
                   'b--', linewidth=1, alpha=0.5)
            ax.text(pred_coords[i, 0] + 5, pred_coords[i, 1] - 5, 
                   f'{error:.1f}px', color='blue', fontsize=8, weight='bold')
        
        if landmark_names:
            ax.set_title(f'{landmark_names[i] if i < len(landmark_names) else f"L{i+1}"}', 
                        fontsize=10, weight='bold')
        else:
            ax.set_title(f'Landmark {i+1}', fontsize=10, weight='bold')
        ax.axis('off')
    
    # Hide extra subplots
    for i in range(K, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_error_distribution(all_errors, save_path=None, bins=50):
    """Plot distribution of radial errors."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(all_errors, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.mean(all_errors), color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {np.mean(all_errors):.2f} px')
    ax.axvline(np.median(all_errors), color='green', linestyle='--', linewidth=2, 
              label=f'Median: {np.median(all_errors):.2f} px')
    
    ax.set_xlabel('Radial Error (pixels)', fontsize=12, weight='bold')
    ax.set_ylabel('Frequency', fontsize=12, weight='bold')
    ax.set_title('Distribution of Radial Errors', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_per_landmark_metrics(per_landmark_mre, per_landmark_sdr=None, 
                              landmark_names=None, thresholds=[2.0, 2.5, 3.0, 4.0],
                              save_path=None):
    """Plot per-landmark MRE and SDR."""
    K = len(per_landmark_mre)
    if landmark_names is None:
        landmark_names = [f'L{i+1}' for i in range(K)]
    
    fig, axes = plt.subplots(2, 1, figsize=(max(12, K*0.6), 10))
    
    # MRE plot
    ax1 = axes[0]
    bars = ax1.bar(range(K), per_landmark_mre, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axhline(np.mean(per_landmark_mre), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(per_landmark_mre):.2f} px')
    ax1.set_xlabel('Landmark', fontsize=12, weight='bold')
    ax1.set_ylabel('Mean Radial Error (px)', fontsize=12, weight='bold')
    ax1.set_title('Per-Landmark Mean Radial Error', fontsize=14, weight='bold')
    ax1.set_xticks(range(K))
    ax1.set_xticklabels(landmark_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # SDR plot
    if per_landmark_sdr is not None:
        ax2 = axes[1]
        x = np.arange(K)
        width = 0.2
        
        for i, threshold in enumerate(thresholds):
            offset = (i - len(thresholds)/2 + 0.5) * width
            ax2.bar(x + offset, per_landmark_sdr[threshold] * 100, width, 
                   label=f'{threshold}mm', alpha=0.8, edgecolor='black')
        
        ax2.set_xlabel('Landmark', fontsize=12, weight='bold')
        ax2.set_ylabel('Success Detection Rate (%)', fontsize=12, weight='bold')
        ax2.set_title('Per-Landmark Success Detection Rate', fontsize=14, weight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(landmark_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 105])
    else:
        axes[1].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_training_curves(history_path, save_path=None):
    """Plot training curves from history JSON file."""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, weight='bold')
    ax1.set_ylabel('Loss', fontsize=12, weight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=14, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MRE curve
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['val_mre'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, weight='bold')
    ax2.set_ylabel('Mean Radial Error (px)', fontsize=12, weight='bold')
    ax2.set_title('Validation Mean Radial Error', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3)
    
    # SDR curves
    ax3 = axes[1, 0]
    for key in sorted(history['val_sdr'].keys()):
        label = key.replace('sdr_', '').replace('p', '.').upper()
        ax3.plot(epochs, [v*100 for v in history['val_sdr'][key]], 
                label=label, linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12, weight='bold')
    ax3.set_ylabel('Success Detection Rate (%)', fontsize=12, weight='bold')
    ax3.set_title('Validation Success Detection Rate', fontsize=14, weight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 105])
    
    # Learning curve comparison
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    line2 = ax4_twin.plot(epochs, history['val_mre'], 'g-', label='MRE', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12, weight='bold')
    ax4.set_ylabel('Validation Loss', fontsize=12, weight='bold', color='r')
    ax4_twin.set_ylabel('Mean Radial Error (px)', fontsize=12, weight='bold', color='g')
    ax4.set_title('Loss vs MRE Correlation', fontsize=14, weight='bold')
    ax4.tick_params(axis='y', labelcolor='r')
    ax4_twin.tick_params(axis='y', labelcolor='g')
    ax4.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_model_comparison(results_dict, save_path=None):
    """
    Compare multiple models.
    
    Args:
        results_dict: Dict mapping model_name -> {'mre': float, 'sdr': {2.0: float, ...}}
        save_path: Path to save figure
    """
    model_names = list(results_dict.keys())
    mres = [results_dict[m]['mre'] for m in model_names]
    sdr_2mm = [results_dict[m]['sdr'].get(2.0, 0) * 100 for m in model_names]
    sdr_4mm = [results_dict[m]['sdr'].get(4.0, 0) * 100 for m in model_names]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # MRE comparison
    ax1 = axes[0]
    bars1 = ax1.bar(model_names, mres, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Mean Radial Error (px)', fontsize=12, weight='bold')
    ax1.set_title('MRE Comparison', fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    # SDR 2mm comparison
    ax2 = axes[1]
    bars2 = ax2.bar(model_names, sdr_2mm, color='green', alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Success Detection Rate (%)', fontsize=12, weight='bold')
    ax2.set_title('SDR @ 2mm Threshold', fontsize=14, weight='bold')
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3, axis='y')
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, weight='bold')
    
    # SDR 4mm comparison
    ax3 = axes[2]
    bars3 = ax3.bar(model_names, sdr_4mm, color='orange', alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Success Detection Rate (%)', fontsize=12, weight='bold')
    ax3.set_title('SDR @ 4mm Threshold', fontsize=14, weight='bold')
    ax3.set_ylim([0, 105])
    ax3.grid(True, alpha=0.3, axis='y')
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, weight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def create_sample_overlay(image, pred_coords, gt_coords, landmark_names=None, 
                         save_path=None, error_threshold=4.0):
    """
    Create a single image overlay showing all landmarks.
    
    Args:
        error_threshold: Color landmarks based on error threshold
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    ax.imshow(image, cmap='gray')
    
    errors = np.linalg.norm(pred_coords - gt_coords, axis=1)
    
    for i, (pred, gt, err) in enumerate(zip(pred_coords, gt_coords, errors)):
        if gt[0] < 0 or gt[1] < 0:  # Skip missing landmarks
            continue
            
        # Color based on error
        if err <= error_threshold:
            color = 'green' if err <= 2.0 else 'yellow' if err <= 3.0 else 'orange'
        else:
            color = 'red'
        
        # Ground truth
        ax.plot(gt[0], gt[1], 'o', color=color, markersize=8, 
               markeredgecolor='white', markeredgewidth=1.5)
        # Prediction
        ax.plot(pred[0], pred[1], 'x', color=color, markersize=10, 
               linewidth=2.5, markeredgecolor='white', markeredgewidth=1)
        # Error line
        ax.plot([pred[0], gt[0]], [pred[1], gt[1]], 
               color=color, linestyle='--', linewidth=1, alpha=0.6)
        
        # Label
        if landmark_names and i < len(landmark_names):
            label = f"{landmark_names[i]}\n{err:.1f}px"
        else:
            label = f"L{i+1}\n{err:.1f}px"
        ax.text(pred[0] + 5, pred[1] - 5, label, 
               fontsize=7, color=color, weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.axis('off')
    ax.set_title('Landmark Detection Results\n(Green≤2mm, Yellow≤3mm, Orange≤4mm, Red>4mm)', 
                fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig

