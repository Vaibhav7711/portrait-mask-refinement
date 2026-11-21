"""Export measured tables and regenerate cleared README crops; run from repo root."""
import csv
import json
import shutil
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from portrait_mask_refinement.constraint import constrain_alpha
from portrait_mask_refinement.data import DataConfig, PortraitCompositeDataset
from portrait_mask_refinement.evaluate import _to_image, exposure_edit, feather
from portrait_mask_refinement.train import load_model


@torch.no_grad()
def main():
    destination = Path("reports/phase1")
    destination.mkdir(parents=True, exist_ok=True)
    for label, source in {
        "ppm-controlled-256": "outputs/256/eval-ppm100-seed17",
        "ppm-modnet-64": "outputs/constrained-width2/eval-ppm100-modnet",
        "fixture-256": "outputs/256/eval-fixture",
    }.items():
        for name in ("summary.json", "per-image.csv"):
            shutil.copyfile(Path(source) / name, destination / f"{label}-{name}")
    shutil.copyfile("outputs/256/seed-stability.json", destination / "seed-stability.json")
    data = PortraitCompositeDataset("test", DataConfig(resolution=256, fixture_identities=36))
    rows = list(csv.DictReader(open("outputs/256/eval-fixture/per-image.csv")))
    values = {(r["identity"], r["version"], r["method"]): float(r["boundary_mae"]) for r in rows}
    ranked = sorted(range(len(data)), key=lambda i: values[(data[i]["identity"], data[i]["version"], "rgb_constrained")] - values[(data[i]["identity"], data[i]["version"], "feathered")])
    indices = [ranked[len(ranked) // 2], ranked[-1]]
    model, _ = load_model("demo/checkpoints/portrait-refiner-fixture-256.pt", torch.device("cpu"))
    canvas = Image.new("RGB", (768, 500), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=15)
    metadata = []
    for row, index in enumerate(indices):
        sample = data[index]
        rgb, rough = sample["image"][None], sample["rough"][None]
        pred = constrain_alpha(model(torch.cat([rgb, rough], 1)), rough, 4)
        delta = values[(sample["identity"], sample["version"], "rgb_constrained")] - values[(sample["identity"], sample["version"], "feathered")]
        label = "Median" if row == 0 else "Worst vs feather"
        draw.text((6, row * 250 + 4), f"{label}: {sample['identity']} / {sample['version']} | boundary MAE delta {delta:+.4f}", fill="black", font=font)
        panels = [("Input RGB", rgb), ("Rough edit", exposure_edit(rgb, rough)), ("Gaussian edit", exposure_edit(rgb, feather(rough, 1))), ("Constrained edit", exposure_edit(rgb, pred))]
        for col, (title, tensor) in enumerate(panels):
            draw.text((col * 192 + 6, row * 250 + 30), title, fill="black", font=font)
            panel = _to_image(tensor).crop((36, 40, 132, 136)).resize((192, 192), Image.Resampling.NEAREST)
            canvas.paste(panel, (col * 192, row * 250 + 54))
        metadata.append({"identity": sample["identity"], "version": sample["version"], "boundary_delta_vs_feather": delta})
    canvas.save("docs/assets/sharing-crops.png")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
