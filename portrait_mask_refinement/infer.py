from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .evaluate import comparison_grid, feather
from .train import load_model
from .utils import select_device


@torch.no_grad()
def run_inference(
    checkpoint: str | Path,
    image_path: str | Path,
    mask_path: str | Path,
    alpha_output: str | Path,
    comparison_output: str | Path,
    device_name: str = "auto",
) -> None:
    device = select_device(device_name)
    model, payload = load_model(checkpoint, device)
    mode = payload["input_mode"]
    resolution = payload["data_config"]["resolution"]
    with Image.open(image_path) as im:
        original_size = im.size
        image_np = np.asarray(im.convert("RGB").resize((resolution, resolution), Image.Resampling.LANCZOS), dtype=np.float32) / 255
    with Image.open(mask_path) as im:
        mask_np = np.asarray(im.convert("L").resize((resolution, resolution), Image.Resampling.NEAREST), dtype=np.float32) / 255
    image = torch.from_numpy(image_np).permute(2, 0, 1).float()[None].to(device)
    rough = torch.from_numpy(mask_np).float()[None, None].to(device)
    model_input = rough if mode == "mask" else torch.cat([image, rough], dim=1)
    refined = model(model_input)
    alpha = _alpha_pil(refined).resize(original_size, Image.Resampling.LANCZOS)
    Path(alpha_output).parent.mkdir(parents=True, exist_ok=True)
    Path(comparison_output).parent.mkdir(parents=True, exist_ok=True)
    alpha.save(alpha_output)
    comparison_grid(image[0].cpu(), rough[0].cpu(), feather(rough.cpu(), 2), refined.cpu()).save(comparison_output)


def _alpha_pil(alpha: torch.Tensor) -> Image.Image:
    arr = (alpha[0, 0].detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, "L")
