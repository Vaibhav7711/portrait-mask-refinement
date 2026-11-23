"""Generate binary rough masks with the official MODNet photographic checkpoint.

MODNet code and weights are third-party dependencies and are intentionally not
vendored. See docs/DATASETS.md for the exact official download commands.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modnet-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    repo = Path(args.modnet_repo).resolve()
    sys.path.insert(0, str(repo))
    from src.models.modnet import MODNet  # type: ignore[import-not-found]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = MODNet(backbone_pretrained=False).to(device)
    weights = torch.load(args.checkpoint, map_location=device, weights_only=False)
    weights = {key.removeprefix("module."): value for key, value in weights.items()}
    model.load_state_dict(weights)
    model.eval()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(path for path in Path(args.images_dir).glob("*") if path.is_file()):
        with Image.open(image_path) as source:
            original_size = source.size
            rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        tensor = (tensor - 0.5) / 0.5
        height, width = tensor.shape[1:]
        if width >= height:
            resized_height, resized_width = 512, int(width / height * 512)
        else:
            resized_width, resized_height = 512, int(height / width * 512)
        resized_height -= resized_height % 32
        resized_width -= resized_width % 32
        tensor = F.interpolate(tensor[None], (resized_height, resized_width), mode="area").to(device)
        with torch.no_grad():
            _, _, matte = model(tensor, True)
        matte = F.interpolate(matte, (height, width), mode="bilinear", align_corners=False)[0, 0]
        binary = (matte >= args.threshold).to(torch.uint8).cpu().numpy() * 255
        Image.fromarray(binary, "L").resize(original_size, Image.Resampling.NEAREST).save(output_dir / f"{image_path.stem}.png")


if __name__ == "__main__":
    main()
