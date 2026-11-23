from __future__ import annotations

import torch
import torch.nn.functional as F


def editable_region(rough: torch.Tensor, width: int) -> torch.Tensor:
    """Return an erosion/dilation band using only the inference-time rough mask."""
    if rough.ndim != 4 or rough.shape[1] != 1:
        raise ValueError("rough must have shape Bx1xHxW")
    if width < 0:
        raise ValueError("width must be non-negative")
    if width == 0:
        return torch.zeros_like(rough)
    binary = (rough >= 0.5).to(rough.dtype)
    kernel = 2 * width + 1
    dilated = F.max_pool2d(binary, kernel, stride=1, padding=width)
    eroded = 1 - F.max_pool2d(1 - binary, kernel, stride=1, padding=width)
    return (dilated - eroded).clamp(0, 1)


def constrain_alpha(prediction: torch.Tensor, rough: torch.Tensor, width: int) -> torch.Tensor:
    """Use CNN alpha inside the editable band and preserve rough pixels elsewhere."""
    if prediction.shape != rough.shape:
        raise ValueError("prediction and rough must have identical shapes")
    editable = editable_region(rough, width)
    return prediction * editable + rough * (1 - editable)
