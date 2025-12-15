import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


def soft_argmax_2d(heatmaps):
    """
    Extract landmark coordinates from heatmaps using soft argmax.
    
    Args:
        heatmaps: (B, K, H, W) tensor of heatmaps
        
    Returns:
        coords: (B, K, 2) tensor of [x, y] coordinates
    """
    B, K, H, W = heatmaps.shape

    heatmaps = heatmaps.reshape(B, K, -1)
    heatmaps = F.softmax(heatmaps, dim=-1)

    xs = torch.linspace(0, W - 1, W).to(heatmaps.device)
    ys = torch.linspace(0, H - 1, H).to(heatmaps.device)

    xs = xs.repeat(H, 1).reshape(-1)
    ys = ys.repeat_interleave(W)

    x = torch.sum(heatmaps * xs, dim=-1)
    y = torch.sum(heatmaps * ys, dim=-1)

    coords = torch.stack([x, y], dim=-1)
    return coords


def compute_radial_errors(pred_coords, gt_coords):
    """
    Compute radial errors for each landmark.
    
    Args:
        pred_coords: (B, K, 2) predicted coordinates
        gt_coords: (B, K, 2) ground truth coordinates
        
    Returns:
        errors: (B, K) tensor of radial errors in pixels
    """
    diff = pred_coords - gt_coords
    errors = torch.sqrt((diff ** 2).sum(dim=-1))  # (B, K)
    return errors


def mean_radial_error(pred_coords, gt_coords):
    """Mean radial error across all landmarks and samples."""
    errors = compute_radial_errors(pred_coords, gt_coords)
    return errors.mean().item()


def per_landmark_mre(pred_coords, gt_coords):
    """
    Mean radial error per landmark.
    
    Returns:
        mre_per_landmark: (K,) numpy array
    """
    errors = compute_radial_errors(pred_coords, gt_coords)  # (B, K)
    return errors.mean(dim=0).cpu().numpy()


def success_detection_rate(pred_coords, gt_coords, threshold=2.0):
    errors = compute_radial_errors(pred_coords, gt_coords)  # (B, K)
    success = (errors < threshold).float().mean().item()
    return success


def per_landmark_sdr(pred_coords, gt_coords, threshold=2.0):
    errors = compute_radial_errors(pred_coords, gt_coords)  # (B, K)
    success = (errors < threshold).float()
    return success.mean(dim=0).cpu().numpy()


def compute_comprehensive_metrics(pred_coords, gt_coords, thresholds=[2.0, 2.5, 3.0, 4.0]):
    errors = compute_radial_errors(pred_coords, gt_coords)  # (B, K)
    
    metrics = {
        "mre": errors.mean().item(),
        "per_landmark_mre": errors.mean(dim=0).cpu().numpy(),
        "sdr": {},
        "per_landmark_sdr": {},
        "all_errors": errors.flatten().cpu().numpy(),
    }
    
    for threshold in thresholds:
        metrics["sdr"][threshold] = (errors < threshold).float().mean().item()
        metrics["per_landmark_sdr"][threshold] = (errors < threshold).float().mean(dim=0).cpu().numpy()
    
    return metrics
