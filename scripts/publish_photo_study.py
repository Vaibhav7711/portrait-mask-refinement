"""Summarize measured runs; publish numeric plots, keep portrait crops local."""
import csv
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from portrait_mask_refinement.constraint import constrain_alpha
from portrait_mask_refinement.evaluate import _to_image, exposure_edit, feather
from portrait_mask_refinement.metrics import boundary_band
from portrait_mask_refinement.photographic import PhotographicMatteDataset
from portrait_mask_refinement.train import load_model
from portrait_mask_refinement.utils import write_json
from scripts.evaluate_frozen import sha256, write_csv


@torch.no_grad()
def local_figures(root, main):
    rows = list(csv.DictReader((root / "modnet-256/per-image.csv").open()))
    values = {(row["identity"], row["method"]): float(row["boundary_mae"]) for row in rows}
    ids = sorted({row["identity"] for row in rows})
    improved = sorted([i for i in ids if values[i, "photographic_seed_mean"] < values[i, "procedural_seed_mean"]],
                      key=lambda i: values[i, "photographic_seed_mean"] - values[i, "procedural_seed_mean"])
    failures = sorted([i for i in ids if values[i, "photographic_seed_mean"] > values[i, "feathered"]],
                      key=lambda i: values[i, "photographic_seed_mean"] - values[i, "feathered"])
    choices = []
    if improved:
        choices.append(("representative-improvement", improved[len(improved) // 2]))
    if failures:
        choices.append(("representative-failure", failures[len(failures) // 2]))
    data = PhotographicMatteDataset("data/ppm100/extracted/PPM-100/image", "data/ppm100/extracted/PPM-100/matte", "test", roughs_dir="outputs/modnet-roughs")
    models = {domain: load_model(f"outputs/photo-study/{domain}-17/best.pt", torch.device("cpu"))[0]
              for domain in ("procedural", "photographic")}
    output = Path("outputs/photo-study/figures")
    output.mkdir(parents=True, exist_ok=True)
    metadata = []
    for title, identity in choices:
        sample = data[data.identities.index(identity)]
        image, rough, target = [sample[key][None] for key in ("image", "rough", "alpha")]
        predictions = {"Raw MODNet": rough, "Gaussian": feather(rough, main["selected_radius"])}
        for domain, model in models.items():
            predictions[domain.title()] = constrain_alpha(model(torch.cat([image, rough], 1)), rough, 4)
        predictions["Ground truth"] = target
        panels = [("Input RGB", image)] + [(name, exposure_edit(image, alpha, .75)) for name, alpha in predictions.items()]
        # Focus on an upper-half contour point; all crops cover exactly 96x96 px.
        band = boundary_band(target, 2)[0, 0].numpy()
        candidates = np.argwhere(band[24:144] > 0)
        if len(candidates):
            point = candidates[np.argmin((candidates[:, 0] + 24 - 80) ** 2 + (candidates[:, 1] - 80) ** 2)] + [24, 0]
            top, left = np.clip(point - 48, 0, 160).astype(int)
        else:
            top, left = 32, 32
        box = (int(left), int(top), int(left + 96), int(top + 96))
        canvas = Image.new("RGB", (1152, 460), "white")
        draw, font = ImageDraw.Draw(canvas), ImageFont.load_default(size=15)
        delta = values[identity, "photographic_seed_mean"] - values[identity, "procedural_seed_mean"]
        draw.text((6, 4), f"PPM-100 / {identity} | {title} | seed-mean photo minus procedural: {delta:+.4f}", fill="black", font=font)
        draw.text((6, 27), "Displayed CNNs: seed 17 | all edits +0.75 stops | lower row: identical 96x96 crops, enlarged 2x", fill="black", font=font)
        for column, (name, tensor) in enumerate(panels):
            panel = _to_image(tensor)
            draw.text((column * 192 + 6, 51), name, fill="black", font=font)
            full = panel.resize((192, 192), Image.Resampling.LANCZOS)
            outline = ImageDraw.Draw(full)
            outline.rectangle(tuple(round(v * .75) for v in box), outline="#ff00ff", width=2)
            canvas.paste(full, (column * 192, 74))
            canvas.paste(panel.crop(box).resize((192, 192), Image.Resampling.NEAREST), (column * 192, 268))
        canvas.save(output / f"{title}.png")
        metadata.append({"case": title, "identity": identity, "crop_box_at_256": box,
                         "photo_minus_procedural_seed_mean": delta,
                         "photo_minus_feather_seed_mean": values[identity, "photographic_seed_mean"] - values[identity, "feathered"],
                         "selection": "median within improved-vs-procedural or worse-than-feather subset; images stay local"})
    write_json(root / "figure-selection.json", metadata)


def main():
    torch.set_num_threads(4)
    root = Path("reports/photographic-study")
    summaries = {condition: json.loads((root / f"{condition}-256/summary.json").read_text()) for condition in ("modnet", "controlled")}
    history_dir = root / "training"
    history_dir.mkdir(exist_ok=True)
    for path in Path("outputs/photo-study").glob("*/history.json"):
        shutil.copyfile(path, history_dir / f"{path.parent.name}.json")
    seed_stats = {}
    for condition, summary in summaries.items():
        seed_stats[condition] = {}
        for domain in ("procedural", "photographic"):
            values = [summary["metrics"][f"{domain}{seed}"]["boundary_mae"] for seed in (17, 23, 41)]
            seed_stats[condition][domain] = {"boundary_mae": values, "mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))}
    write_json(root / "seed-statistics.json", seed_stats)
    weights = Path("data/tools/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt")
    write_json(root / "third-party-provenance.json", {"modnet_checkpoint_sha256": sha256(weights),
        "modnet_code_commit": subprocess.check_output(["git", "-C", "data/tools/MODNet", "rev-parse", "HEAD"], text=True).strip(),
        "photomatte85_archive_sha256": sha256("data/photomatte85/PhotoMatte85.zip")})
    write_csv(root / "photographic-modnet-mask-hashes.csv", [{"file": p.name, "sha256": sha256(p)} for p in sorted(Path("data/photomatte85/modnet-masks").glob("*.png"))])
    pairs = list(csv.DictReader((root / "modnet-256/paired-differences.csv").open()))
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for ax, comparison, title in zip(axes, ["photographic_seed_mean-minus-procedural_seed_mean", "photographic_seed_mean-minus-feathered"], ["Photo training − procedural training", "Photo training − Gaussian feathering"]):
        deltas = sorted(float(row["boundary_delta"]) for row in pairs if row["comparison"] == comparison)
        ax.bar(range(1, len(deltas) + 1), deltas, color=["#267c78" if v < 0 else "#c36c40" for v in deltas], width=1)
        ax.axhline(0, color="black", linewidth=.7)
        ax.set(title=title, xlabel="81 foregrounds (sorted separately)", ylabel="Boundary MAE difference")
        ax.title.set_fontsize(11)
        ax.xaxis.label.set_fontsize(10)
        ax.yaxis.label.set_fontsize(10)
    fig.suptitle("Real MODNet masks · 3 matched seeds · PhotoMatte85 pilot", fontsize=13)
    fig.savefig("docs/assets/photographic-paired-differences.png", dpi=150)
    plt.close(fig)
    local_figures(root, summaries["modnet"])
    print(json.dumps(seed_stats, indent=2))


if __name__ == "__main__":
    main()
