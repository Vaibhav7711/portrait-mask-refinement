from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .constraint import constrain_alpha
from .train import load_model
from .utils import select_device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def benchmark_checkpoint(
    checkpoint: str | Path,
    image_path: str | Path,
    mask_path: str | Path,
    device_name: str = "auto",
    warmup: int = 10,
    runs: int = 50,
    editable_width: int | None = None,
) -> dict[str, object]:
    """Measure decode, preprocessing, network, constraint, and output resizing."""
    device = select_device(device_name)
    model, payload = load_model(checkpoint, device)
    resolution = int(payload["data_config"]["resolution"])
    mode = payload["input_mode"]
    config = payload.get("train_config", {})
    width = editable_width if editable_width is not None else int(config.get("editable_width", 0))
    constrained = editable_width is not None or bool(config.get("constrained", False))

    def one_run() -> None:
        with Image.open(image_path) as source:
            original_size = source.size
            image_np = np.asarray(source.convert("RGB").resize((resolution, resolution), Image.Resampling.LANCZOS), dtype=np.float32) / 255
        with Image.open(mask_path) as source:
            original_mask = source.convert("L").resize(original_size, Image.Resampling.NEAREST)
            mask_np = np.asarray(original_mask.resize((resolution, resolution), Image.Resampling.NEAREST), dtype=np.float32) / 255
            original_mask_np = np.asarray(original_mask, dtype=np.float32) / 255
        image = torch.from_numpy(image_np).permute(2, 0, 1).float()[None].to(device)
        rough = torch.from_numpy(mask_np).float()[None, None].to(device)
        inputs = rough if mode == "mask" else torch.cat([image, rough], dim=1)
        prediction = model(inputs)
        if constrained:
            prediction = constrain_alpha(prediction, rough, width)
        output = Image.fromarray((prediction[0, 0].cpu().numpy() * 255).astype(np.uint8), "L").resize(original_size, Image.Resampling.LANCZOS)
        if constrained:
            output_tensor = torch.from_numpy(np.asarray(output, dtype=np.float32) / 255)[None, None]
            original_rough = torch.from_numpy(original_mask_np).float()[None, None]
            scaled_width = max(1, round(width * min(original_size) / resolution))
            constrain_alpha(output_tensor, original_rough, scaled_width)
        _synchronize(device)

    for _ in range(warmup):
        one_run()
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        one_run()
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "device": str(device),
        "device_label": "Apple MPS" if device.type == "mps" else str(device),
        "resolution": resolution,
        "warmup_runs": warmup,
        "measured_runs": runs,
        "includes": ["image_decode", "preprocessing", "network", "constraint", "output_resize"],
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
    }
