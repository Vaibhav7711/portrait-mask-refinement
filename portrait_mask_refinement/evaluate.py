from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from .data import DataConfig, PortraitCompositeDataset
from .metrics import alpha_errors
from .train import load_model
from .utils import select_device, write_json


def feather(mask: torch.Tensor, radius: float) -> torch.Tensor:
    arr = (mask.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
    result = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=radius))
    return torch.from_numpy(np.asarray(result, dtype=np.float32) / 255.0).view(1, 1, *arr.shape)


def exposure_edit(image: torch.Tensor, alpha: torch.Tensor, stops: float = 0.75) -> torch.Tensor:
    gain = 2**stops
    lifted = torch.clamp(image * gain, 0, 1)
    return image * (1 - alpha) + lifted * alpha


def _to_image(tensor: torch.Tensor, mode: str = "RGB") -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0, 1)
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.shape[0] == 1:
        arr = (tensor[0].numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr, "L").convert(mode)
    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def comparison_grid(image: torch.Tensor, rough: torch.Tensor, feathered: torch.Tensor, refined: torch.Tensor | None = None) -> Image.Image:
    panels: list[tuple[str, Image.Image]] = [
        ("RGB composite", _to_image(image)),
        ("Rough mask", _to_image(rough)),
        ("Rough edit", _to_image(exposure_edit(image[None], rough[None]))),
        ("Feathered edit", _to_image(exposure_edit(image[None], feathered))),
    ]
    if refined is not None:
        panels.extend([
            ("Refined alpha", _to_image(refined)),
            ("Refined edit", _to_image(exposure_edit(image[None], refined))),
        ])
    w, h = panels[0][1].size
    scale = max(1, math.ceil(160 / w))
    if scale > 1:
        panels = [(label, panel.resize((w * scale, h * scale), Image.Resampling.NEAREST if "mask" in label.lower() or "alpha" in label.lower() else Image.Resampling.LANCZOS)) for label, panel in panels]
        w, h = w * scale, h * scale
    header = 30
    cols = 3 if len(panels) > 4 else 2
    rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * (h + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, panel) in enumerate(panels):
        x, y = (i % cols) * w, (i // cols) * (h + header)
        draw.text((x + 7, y + 7), label, fill="black")
        canvas.paste(panel, (x, y + header))
    return canvas


def boundary_crop_grid(
    image: torch.Tensor,
    rough: torch.Tensor,
    feathered: torch.Tensor,
    mask_refined: torch.Tensor,
    rgb_refined: torch.Tensor,
) -> Image.Image:
    """Show all alpha methods and equal-exposure hair/shoulder crops."""
    mattes = [
        ("Original RGB", _to_image(image)),
        ("Rough mask", _to_image(rough)),
        ("Feathered alpha", _to_image(feathered)),
        ("Mask U-Net alpha", _to_image(mask_refined)),
        ("RGB+mask U-Net alpha", _to_image(rgb_refined)),
    ]
    edits = [
        ("Original RGB boundary crop", image[None]),
        ("Rough edit boundary crop", exposure_edit(image[None], rough[None])),
        ("Feathered edit boundary crop", exposure_edit(image[None], feathered)),
        ("Mask U-Net edit boundary crop", exposure_edit(image[None], mask_refined)),
        ("RGB+mask edit boundary crop", exposure_edit(image[None], rgb_refined)),
    ]
    w, h = mattes[0][1].size
    # Upper-left region crosses the synthetic hair silhouette and background.
    crop_box = (int(w * 0.18), int(h * 0.06), int(w * 0.55), int(h * 0.62))
    crops = [(label, _to_image(value).crop(crop_box).resize((w, h), Image.Resampling.NEAREST)) for label, value in edits]
    panels = mattes + crops
    scale = max(1, math.ceil(160 / w))
    w2, h2 = w * scale, h * scale
    header = 30
    canvas = Image.new("RGB", (5 * w2, 2 * (h2 + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, panel) in enumerate(panels):
        panel = panel.resize((w2, h2), Image.Resampling.NEAREST if i >= 5 else Image.Resampling.LANCZOS)
        x, y = (i % 5) * w2, (i // 5) * (h2 + header)
        draw.text((x + 6, y + 8), label, fill="black")
        canvas.paste(panel, (x, y + header))
    return canvas


@torch.no_grad()
def tune_feather_radius(dataset: PortraitCompositeDataset, radii: list[float], width: int) -> tuple[float, dict[str, float]]:
    scores = {}
    for radius in radii:
        values = []
        for sample in dataset:
            pred = feather(sample["rough"], radius)
            _, boundary = alpha_errors(pred, sample["alpha"][None], width)
            values.append(boundary.item())
        scores[str(radius)] = sum(values) / len(values)
    best = min(radii, key=lambda r: scores[str(r)])
    return best, scores


@torch.no_grad()
def evaluate_models(
    mask_checkpoint: str | Path,
    rgb_checkpoint: str | Path,
    data_config: DataConfig,
    output_dir: str | Path,
    device_name: str = "auto",
    boundary_width: int = 5,
) -> dict:
    device = select_device(device_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_model, _ = load_model(mask_checkpoint, device)
    rgb_model, _ = load_model(rgb_checkpoint, device)
    val_set = PortraitCompositeDataset("val", data_config)
    best_radius, tuning = tune_feather_radius(val_set, [0.5, 1.0, 2.0, 3.0, 5.0], boundary_width)
    test_set = PortraitCompositeDataset("test", data_config)
    rows = []
    visual_saved = False
    for sample in test_set:
        image = sample["image"][None].to(device)
        rough = sample["rough"][None].to(device)
        alpha = sample["alpha"][None].to(device)
        predictions = {
            "rough": rough,
            "feathered": feather(rough, best_radius).to(device),
            "mask_unet": mask_model(rough),
            "rgb_mask_unet": rgb_model(torch.cat([image, rough], dim=1)),
        }
        for method, pred in predictions.items():
            full, boundary = alpha_errors(pred, alpha, boundary_width)
            rows.append({
                "identity": sample["identity"],
                "version": sample["version"],
                "method": method,
                "full_mae": full.item(),
                "boundary_mae": boundary.item(),
            })
        if not visual_saved:
            comparison_grid(sample["image"], sample["rough"], predictions["feathered"].cpu(), predictions["rgb_mask_unet"].cpu()).save(output_dir / "four-way-visual.png")
            boundary_crop_grid(
                sample["image"],
                sample["rough"],
                predictions["feathered"].cpu(),
                predictions["mask_unet"].cpu(),
                predictions["rgb_mask_unet"].cpu(),
            ).save(output_dir / "boundary-crops.png")
            visual_saved = True
    with (output_dir / "per-image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for method in ("rough", "feathered", "mask_unet", "rgb_mask_unet"):
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            "full_mae": sum(row["full_mae"] for row in selected) / len(selected),
            "boundary_mae": sum(row["boundary_mae"] for row in selected) / len(selected),
        }
    result = {"fixture_only": not bool(data_config.portraits_dir), "feather_radius": best_radius, "feather_validation": tuning, "summary": summary}
    write_json(output_dir / "summary.json", result)
    return result
