"""Evaluate fixed checkpoints on the frozen PPM-100 masks, never train or tune on test.

Run from repository root. Repeated --checkpoint NAME=PATH arguments support
matched training seeds; names should include the seed. All alphas are unquantized
network outputs except the historical PIL Gaussian baseline (8-bit).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from portrait_mask_refinement.constraint import constrain_alpha
from portrait_mask_refinement.evaluate import feather
from portrait_mask_refinement.metrics import boundary_band, diagnostic_errors
from portrait_mask_refinement.photographic import PhotographicMatteDataset
from portrait_mask_refinement.train import load_model
from portrait_mask_refinement.utils import select_device, write_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_interval(values, seed=17):
    """Input must contain one paired delta per foreground, not per corruption."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        raise ValueError("Cannot bootstrap an empty comparison")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (5000, len(values)), replace=True).mean(1)
    return {"mean": float(values.mean()), "foregrounds": len(values),
            "bootstrap_95ci": np.quantile(means, [.025, .975]).tolist(),
            "improved_foregrounds": int((values < 0).sum())}


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(args):
    torch.set_num_threads(4)
    device = select_device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    datasets = {split: PhotographicMatteDataset(args.images_dir, args.mattes_dir, split,
                resolution=256, seed=17, perturbations=3,
                roughs_dir=args.roughs_dir) for split in ("val", "test")}
    manifest = []
    for split, dataset in datasets.items():
        for image, matte in dataset.pairs:
            row = {"foreground": image.stem, "split": split,
                   "image_sha256": sha256(image), "matte_sha256": sha256(matte)}
            if args.roughs_dir:
                row["rough_sha256"] = sha256(Path(args.roughs_dir) / f"{image.stem}.png")
            manifest.append(row)
    write_csv(output / "frozen-manifest.csv", manifest)
    print("Caching fixed evaluation inputs", flush=True)
    cached = {split: list(dataset) for split, dataset in datasets.items()}
    radii = [.5, 1., 2., 3., 5.]
    tuning = {}
    for radius in radii:
        scores = []
        for sample in cached["val"]:
            target, rough = sample["alpha"][None], sample["rough"][None]
            scores.append(diagnostic_errors(feather(rough, radius), rough, target, 12)["boundary_mae"].item())
        tuning[str(radius)] = float(np.mean(scores))
    radius = min(radii, key=lambda r: tuning[str(r)])
    print(f"Validation selected Gaussian radius {radius}", flush=True)
    checkpoint_paths = dict(item.split("=", 1) for item in args.checkpoint)
    models = {name: load_model(path, device) for name, path in checkpoint_paths.items()}
    rows = []
    loader = DataLoader(cached["test"], batch_size=4)
    for batch in loader:
        image, rough = batch["image"].to(device), batch["rough"].to(device)
        target = batch["alpha"]
        predictions = {"rough": batch["rough"],
                       "feathered": torch.cat([feather(mask, radius) for mask in batch["rough"]])}
        for name, (model, payload) in models.items():
            model_input = rough if payload["input_mode"] == "mask" else torch.cat([image, rough], 1)
            prediction = model(model_input)
            config = payload["train_config"]
            if config.get("constrained"):
                prediction = constrain_alpha(prediction, rough, config["editable_width"])
            predictions[name] = prediction.cpu()
        interior = ~boundary_band(target, 12).bool()
        correct = (batch["rough"] - target).abs() <= .05
        counts = {"fg": ((target >= .95) & interior & correct).flatten(1).sum(1),
                  "bg": ((target <= .05) & interior & correct).flatten(1).sum(1)}
        for name, prediction in predictions.items():
            metrics = diagnostic_errors(prediction, batch["rough"], target, 12)
            for i in range(len(image)):
                row = {"identity": batch["identity"][i], "version": batch["version"][i], "method": name}
                row.update({key: float(value[i]) for key, value in metrics.items()})
                for region in ("fg", "bg"):
                    row[f"correct_{region}_pixels"] = int(counts[region][i])
                    if counts[region][i] == 0:
                        row[f"correct_{region}_damage_rate"] = ""
                        row[f"correct_{region}_mae"] = ""
                rows.append(row)
    # These are seed-averaged errors, NOT predictions from an ensemble.
    for specification in args.group:
        name, members = specification.split("=", 1)
        members = members.split(",")
        for sample in cached["test"]:
            selected = [r for r in rows if r["identity"] == sample["identity"]
                        and r["version"] == sample["version"] and r["method"] in members]
            if len(selected) != len(members):
                raise ValueError("Every seed must supply every paired condition")
            row = {"identity": sample["identity"], "version": sample["version"], "method": name}
            for key in selected[0]:
                if key not in row:
                    row[key] = float(np.mean([r[key] for r in selected])) if selected[0][key] != "" else ""
            rows.append(row)
    write_csv(output / "per-image.csv", rows)
    summaries = {}
    for method in sorted({row["method"] for row in rows}):
        selected = [r for r in rows if r["method"] == method]
        summaries[method] = {key: float(np.mean([r[key] for r in selected if r[key] != ""]))
                             for key in metrics if any(r[key] != "" for r in selected)}
        for region in ("fg", "bg"):
            summaries[method][f"conditions_with_correct_{region}"] = sum(r[f"correct_{region}_pixels"] > 0 for r in selected)
    # Pair by foreground/version before averaging repeated corruptions.
    lookup = {(r["identity"], r["version"], r["method"]): r for r in rows}
    paired_rows, intervals = [], {}
    comparisons = [tuple(pair.split(":")) for pair in args.compare]
    comparisons += [(name, "feathered") for name in checkpoint_paths]
    for left, right in comparisons:
        deltas = []
        for identity in datasets["test"].identities:
            condition_deltas = []
            for version in range(datasets["test"].perturbations):
                a, b = lookup[(identity, str(version), left)], lookup[(identity, str(version), right)]
                delta = a["boundary_mae"] - b["boundary_mae"]
                condition_deltas.append(delta)
                paired_rows.append({"identity": identity, "version": version,
                    "comparison": f"{left}-minus-{right}", "boundary_delta": delta,
                    "full_delta": a["full_mae"] - b["full_mae"]})
            deltas.append(float(np.mean(condition_deltas)))
        intervals[f"{left}-minus-{right}"] = paired_interval(deltas)
    if paired_rows:
        write_csv(output / "paired-differences.csv", paired_rows)
    result = {"resolution": 256, "evaluation_half_width": 12, "data_seed": 17,
        "validation_foregrounds": len(datasets["val"].pairs),
        "test_foregrounds": len(datasets["test"].pairs), "test_conditions": len(datasets["test"]),
        "device": str(device), "torch": torch.__version__, "feather_validation": tuning,
        "selected_radius": radius, "checkpoint_sha256": {k: sha256(v) for k, v in checkpoint_paths.items()},
        "seed_metric_groups": args.group,
        "metrics": summaries, "paired_foreground_intervals": intervals}
    write_json(output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--mattes-dir", required=True)
    parser.add_argument("--roughs-dir")
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--compare", action="append", default=[], help="LEFT:RIGHT checkpoint names")
    parser.add_argument("--group", action="append", default=[], help="NAME=SEED1,SEED2,...; average errors, not predictions")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
