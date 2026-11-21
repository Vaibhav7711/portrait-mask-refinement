from __future__ import annotations

import torch
import torch.nn.functional as F


def boundary_band(alpha: torch.Tensor, width: int = 5) -> torch.Tensor:
    """Fixed-width band around the ground-truth 0.5 contour."""
    binary = (alpha >= 0.5).float()
    kernel = width * 2 + 1
    dilated = F.max_pool2d(binary, kernel, stride=1, padding=width)
    eroded = 1 - F.max_pool2d(1 - binary, kernel, stride=1, padding=width)
    return (dilated - eroded).clamp(0, 1)


def alpha_errors(pred: torch.Tensor, target: torch.Tensor, width: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    absolute = (pred - target).abs()
    band = boundary_band(target, width)
    full_mae = absolute.flatten(1).mean(1)
    boundary_mae = (absolute * band).flatten(1).sum(1) / band.flatten(1).sum(1).clamp_min(1)
    return full_mae, boundary_mae


def boundary_weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    width: int = 5,
    boundary_weight: float = 4.0,
) -> torch.Tensor:
    full, boundary = alpha_errors(pred, target, width)
    return (full + boundary_weight * boundary).mean()
