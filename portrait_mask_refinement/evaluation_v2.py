from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from .constraint import constrain_alpha, editable_region
from .evaluate import _to_image, exposure_edit, feather
from .metrics import diagnostic_errors
from .train import load_model
from .utils import select_device, write_json

METRICS = (
    "full_mae",
    "boundary_mae",
    "fg_interior_mae",
    "bg_exterior_mae",
    "correct_fg_mae",
    "correct_bg_mae",
    "correct_fg_damage_rate",
    "correct_bg_damage_rate",
    "correct_pixel_change_rate",
)


def _model_prediction(model: torch.nn.Module, mode: str, image: torch.Tensor, rough: torch.Tensor) -> torch.Tensor:
    inputs = rough if mode == "mask" else torch.cat([image, rough], dim=1)
    return model(inputs)


@torch.no_grad()
def tune_editable_width(
    model: torch.nn.Module,
    mode: str,
    dataset: Any,
    widths: list[int],
    device: torch.device,
    gt_boundary_width: int,
) -> tuple[int, dict[str, dict[str, float]]]:
    totals = {width: {"boundary": [], "coverage": []} for width in widths}
    for sample in dataset:
        image = sample["image"][None].to(device)
        rough = sample["rough"][None].to(device)
        target = sample["alpha"][None].to(device)
        raw = _model_prediction(model, mode, image, rough)
        error_pixels = (rough - target).abs() > 0.05
        for width in widths:
            pred = constrain_alpha(raw, rough, width)
            metrics = diagnostic_errors(pred, rough, target, gt_boundary_width)
            totals[width]["boundary"].append(metrics["boundary_mae"].item())
            band = editable_region(rough, width).bool()
            coverage = (band & error_pixels).sum().float() / error_pixels.sum().clamp_min(1)
            totals[width]["coverage"].append(coverage.item())
    summary = {
        str(width): {
            "boundary_mae": float(np.mean(values["boundary"])),
            "rough_error_coverage": float(np.mean(values["coverage"])),
        }
        for width, values in totals.items()
    }
    best = min(widths, key=lambda width: summary[str(width)]["boundary_mae"])
    return best, summary


def _bootstrap_identity_ci(rows: list[dict[str, Any]], method: str, metric: str, seed: int = 17) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] == method:
            grouped[row["identity"]].append(float(row[metric]))
    values = np.array([np.mean(items) for items in grouped.values()], dtype=np.float64)
    if len(values) < 2:
        return [float(values.mean()), float(values.mean())]
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _visual(
    sample: dict[str, Any],
    predictions: dict[str, torch.Tensor],
    output: Path,
    title: str,
) -> None:
    keys = ["rough", "feathered", "rgb_unconstrained", "rgb_posthoc", "mask_constrained", "rgb_constrained"]
    image = sample["image"]
    target = sample["alpha"]
    change = (predictions["rgb_unconstrained"] - predictions["rough"]).abs()
    panels = [("RGB", _to_image(image)), ("Reference", _to_image(target)), ("Unconstrained change", _to_image(change))]
    for key in keys:
        label = key.replace("_", " ")
        panels.append((label, _to_image(predictions[key])))
    crop_box = (int(image.shape[2] * 0.12), int(image.shape[1] * 0.05), int(image.shape[2] * 0.62), int(image.shape[1] * 0.62))
    for key in ("rough", "feathered", "rgb_unconstrained", "rgb_constrained"):
        edit = _to_image(exposure_edit(image[None], predictions[key]))
        panels.append((f"{key.replace('_', ' ')} crop", edit.crop(crop_box).resize(_to_image(image).size, Image.Resampling.NEAREST)))
    w, h = 192, 192
    header = 28
    cols = 4
    rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, 34 + rows * (h + header)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), title, fill="black")
    for index, (label, panel) in enumerate(panels):
        x = (index % cols) * w
        y = 34 + (index // cols) * (h + header)
        draw.text((x + 6, y + 7), label, fill="black")
        canvas.paste(panel.resize((w, h), Image.Resampling.LANCZOS), (x, y + header))
    canvas.save(output)


@torch.no_grad()
def evaluate_boundary_constrained(
    val_dataset: Any,
    test_dataset: Any,
    original_rgb_checkpoint: str | Path,
    constrained_mask_checkpoint: str | Path,
    constrained_rgb_checkpoint: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    gt_boundary_width: int = 3,
    candidate_widths: list[int] | None = None,
) -> dict[str, Any]:
    device = select_device(device_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original_rgb, original_payload = load_model(original_rgb_checkpoint, device)
    mask_model, mask_payload = load_model(constrained_mask_checkpoint, device)
    rgb_model, rgb_payload = load_model(constrained_rgb_checkpoint, device)
    candidate_widths = candidate_widths or [1, 2, 3, 5, 8]
    best_width, band_tuning = tune_editable_width(
        original_rgb,
        original_payload["input_mode"],
        val_dataset,
        candidate_widths,
        device,
        gt_boundary_width,
    )

    radii = [0.5, 1.0, 2.0, 3.0, 5.0]
    radius_scores: dict[str, list[float]] = {str(radius): [] for radius in radii}
    for sample in val_dataset:
        target = sample["alpha"][None]
        rough = sample["rough"][None]
        for radius in radii:
            metrics = diagnostic_errors(feather(rough, radius), rough, target, gt_boundary_width)
            radius_scores[str(radius)].append(metrics["boundary_mae"].item())
    feather_tuning = {key: float(np.mean(values)) for key, values in radius_scores.items()}
    best_radius = min(radii, key=lambda radius: feather_tuning[str(radius)])

    trained_mask_width = int(mask_payload["train_config"].get("editable_width", best_width))
    trained_rgb_width = int(rgb_payload["train_config"].get("editable_width", best_width))
    rows: list[dict[str, Any]] = []
    visuals: list[tuple[float, dict[str, Any], dict[str, torch.Tensor]]] = []
    for sample in test_dataset:
        image = sample["image"][None].to(device)
        rough = sample["rough"][None].to(device)
        target = sample["alpha"][None].to(device)
        original_raw = _model_prediction(original_rgb, original_payload["input_mode"], image, rough)
        mask_raw = _model_prediction(mask_model, mask_payload["input_mode"], image, rough)
        rgb_raw = _model_prediction(rgb_model, rgb_payload["input_mode"], image, rough)
        predictions = {
            "rough": rough,
            "feathered": feather(rough.cpu(), best_radius).to(device),
            "rgb_unconstrained": original_raw,
            "rgb_posthoc": constrain_alpha(original_raw, rough, best_width),
            "mask_constrained": constrain_alpha(mask_raw, rough, trained_mask_width),
            "rgb_constrained": constrain_alpha(rgb_raw, rough, trained_rgb_width),
        }
        sample_rows = {}
        for method, prediction in predictions.items():
            values = diagnostic_errors(prediction, rough, target, gt_boundary_width)
            row = {"identity": sample["identity"], "version": sample["version"], "method": method}
            row.update({key: float(value.item()) for key, value in values.items()})
            rows.append(row)
            sample_rows[method] = row
        damage_delta = sample_rows["rgb_unconstrained"]["full_mae"] - sample_rows["rough"]["full_mae"]
        cpu_predictions = {name: value.cpu() for name, value in predictions.items()}
        visuals.append((damage_delta, sample, cpu_predictions))

    rough_by_pair = {(row["identity"], row["version"]): row for row in rows if row["method"] == "rough"}
    for row in rows:
        rough_row = rough_by_pair[(row["identity"], row["version"])]
        row["paired_full_delta_vs_rough"] = row["full_mae"] - rough_row["full_mae"]
        row["paired_boundary_delta_vs_rough"] = row["boundary_mae"] - rough_row["boundary_mae"]

    with (output_dir / "per-image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    methods = list(dict.fromkeys(row["method"] for row in rows))
    summary = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {metric: float(np.mean([row[metric] for row in selected])) for metric in METRICS}
        summary[method]["boundary_mae_identity_bootstrap_95ci"] = _bootstrap_identity_ci(rows, method, "boundary_mae")
        summary[method]["paired_full_delta_vs_rough"] = float(np.mean([row["paired_full_delta_vs_rough"] for row in selected]))
        summary[method]["paired_boundary_delta_vs_rough"] = float(np.mean([row["paired_boundary_delta_vs_rough"] for row in selected]))

    visuals.sort(key=lambda item: item[0])
    typical = visuals[len(visuals) // 2]
    failure = visuals[-1]
    _visual(typical[1], typical[2], output_dir / "typical-comparison.png", "Typical held-out example")
    _visual(failure[1], failure[2], output_dir / "failure-comparison.png", "Largest unconstrained full-image regression")
    result = {
        "protocol": {
            "validation_foregrounds": len(val_dataset.identities),
            "test_foregrounds": len(test_dataset.identities),
            "test_composites_or_conditions": len(test_dataset),
            "gt_boundary_width": gt_boundary_width,
            "selected_editable_width": best_width,
            "selected_feather_radius": best_radius,
        },
        "editable_width_validation": band_tuning,
        "feather_validation": feather_tuning,
        "summary": summary,
    }
    write_json(output_dir / "summary.json", result)
    return result
