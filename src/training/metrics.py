import torch
import torch.nn.functional as F
import numpy as np


def soft_argmax_2d(heatmaps):
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


def mean_radial_error(pred_coords, gt_coords):
    diff = pred_coords - gt_coords
    dist = torch.sqrt((diff ** 2).sum(dim=-1)) # (B, K)
    return dist.mean().item()


def success_detection_rate(pred_coords, gt_coords, threshold=2.0):
    diff = pred_coords - gt_coords
    dist = torch.sqrt((diff ** 2).sum(dim=-1)) # (B, K)
    success = (dist < threshold).float().mean().item()
    return success
