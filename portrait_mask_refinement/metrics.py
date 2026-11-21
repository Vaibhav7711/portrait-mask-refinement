from __future__ import annotations

import torch
import torch.nn.functional as F


def square_max_pool(value: torch.Tensor, width: int) -> torch.Tensor:
    """Exactly the square max filter, factored into two cheaper 1D passes."""
    kernel = width * 2 + 1
    horizontal = F.max_pool2d(value, (1, kernel), stride=1, padding=(0, width))
    return F.max_pool2d(horizontal, (kernel, 1), stride=1, padding=(width, 0))


def boundary_band(alpha: torch.Tensor, width: int = 5) -> torch.Tensor:
    """Fixed-width band around the ground-truth 0.5 contour."""
    binary = (alpha >= 0.5).float()
    dilated = square_max_pool(binary, width)
    eroded = 1 - square_max_pool(1 - binary, width)
    return (dilated - eroded).clamp(0, 1)


def alpha_errors(pred: torch.Tensor, target: torch.Tensor, width: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    absolute = (pred - target).abs()
    band = boundary_band(target, width)
    full_mae = absolute.flatten(1).mean(1)
    boundary_mae = (absolute * band).flatten(1).sum(1) / band.flatten(1).sum(1).clamp_min(1)
    return full_mae, boundary_mae


def diagnostic_errors(
    pred: torch.Tensor,
    rough: torch.Tensor,
    target: torch.Tensor,
    width: int = 5,
    correct_tolerance: float = 0.05,
    damage_margin: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Region-specific errors and damage to pixels the rough mask already got right."""
    absolute = (pred - target).abs()
    rough_absolute = (rough - target).abs()
    gt_boundary = boundary_band(target, width).bool()
    fg = (target >= 0.95) & ~gt_boundary
    bg = (target <= 0.05) & ~gt_boundary
    correct = rough_absolute <= correct_tolerance

    def mean_in(mask: torch.Tensor) -> torch.Tensor:
        numerator = (absolute * mask).flatten(1).sum(1)
        denominator = mask.flatten(1).sum(1).clamp_min(1)
        return numerator / denominator

    def rate_in(mask: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        numerator = (event & mask).flatten(1).sum(1).float()
        denominator = mask.flatten(1).sum(1).clamp_min(1)
        return numerator / denominator

    full, boundary = alpha_errors(pred, target, width)
    correct_fg = correct & fg
    correct_bg = correct & bg
    damaged = absolute > rough_absolute + damage_margin
    changed = (pred - rough).abs() > (1 / 255)
    return {
        "full_mae": full,
        "boundary_mae": boundary,
        "fg_interior_mae": mean_in(fg),
        "bg_exterior_mae": mean_in(bg),
        "correct_fg_mae": mean_in(correct_fg),
        "correct_bg_mae": mean_in(correct_bg),
        "correct_fg_damage_rate": rate_in(correct_fg, damaged),
        "correct_bg_damage_rate": rate_in(correct_bg, damaged),
        "correct_pixel_change_rate": rate_in(correct & (fg | bg), changed),
    }


def boundary_weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    width: int = 5,
    boundary_weight: float = 4.0,
) -> torch.Tensor:
    full, boundary = alpha_errors(pred, target, width)
    return (full + boundary_weight * boundary).mean()
