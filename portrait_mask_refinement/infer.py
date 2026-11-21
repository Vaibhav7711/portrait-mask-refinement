from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .constraint import constrain_alpha
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
    editable_width: int | None = None,
) -> dict[str, object]:
    device = select_device(device_name)
    model, payload = load_model(checkpoint, device)
    mode = payload["input_mode"]
    resolution = payload["data_config"]["resolution"]
    with Image.open(image_path) as im:
        original_size = im.size
        image_np = np.asarray(im.convert("RGB").resize((resolution, resolution), Image.Resampling.LANCZOS), dtype=np.float32) / 255
    with Image.open(mask_path) as im:
        original_mask_image = im.convert("L")
        if original_mask_image.size != original_size:
            raise ValueError("Image and supplied mask dimensions must match")
        original_mask_np = np.asarray(original_mask_image, dtype=np.float32) / 255
        mask_np = np.asarray(original_mask_image.resize((resolution, resolution), Image.Resampling.NEAREST), dtype=np.float32) / 255
    image = torch.from_numpy(image_np).permute(2, 0, 1).float()[None].to(device)
    rough = torch.from_numpy(mask_np).float()[None, None].to(device)
    model_input = rough if mode == "mask" else torch.cat([image, rough], dim=1)
    raw = model(model_input)
    training_config = payload.get("train_config", {})
    trained_constrained = bool(training_config.get("constrained", False))
    model_width = editable_width if editable_width is not None else int(training_config.get("editable_width", 0))
    raw_original = _alpha_pil(raw).resize(original_size, Image.Resampling.LANCZOS)
    raw_original_tensor = torch.from_numpy(np.asarray(raw_original, dtype=np.float32) / 255)[None, None]
    rough_original_tensor = torch.from_numpy(original_mask_np).float()[None, None]
    use_constraint = editable_width is not None or trained_constrained
    if use_constraint:
        if model_width < 0:
            raise ValueError("editable_width must be nonnegative")
        scaled_width = 0 if model_width == 0 else max(1, round(model_width * min(original_size) / resolution))
        refined_original = constrain_alpha(raw_original_tensor, rough_original_tensor, scaled_width)
    else:
        scaled_width = 0
        refined_original = raw_original_tensor
    alpha = _alpha_pil(refined_original)
    Path(alpha_output).parent.mkdir(parents=True, exist_ok=True)
    Path(comparison_output).parent.mkdir(parents=True, exist_ok=True)
    alpha.save(alpha_output)
    display_refined = constrain_alpha(raw, rough, model_width) if use_constraint else raw
    comparison_grid(image[0].cpu(), rough[0].cpu(), feather(rough.cpu(), 2), display_refined.cpu()).save(comparison_output)
    return {
        "original_size": list(original_size),
        "output_size": list(alpha.size),
        "constrained": use_constraint,
        "editable_width_model_pixels": model_width,
        "editable_width_output_pixels": scaled_width,
    }


def _alpha_pil(alpha: torch.Tensor) -> Image.Image:
    arr = np.rint(alpha[0, 0].detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, "L")
