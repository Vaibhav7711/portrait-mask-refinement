"""Audit source overlap and freeze PhotoMatte85 source-image train/validation split."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from portrait_mask_refinement.photographic import _fit_pair
from portrait_mask_refinement.utils import write_json
from scripts.evaluate_frozen import sha256, write_csv


def canonical(rgb, alpha):
    return Image.fromarray(np.rint((rgb * alpha[..., None] + .5 * (1 - alpha[..., None])) * 255).astype(np.uint8))


def phash(image):
    pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    basis = np.cos(np.pi / 32 * np.arange(8)[:, None] * (np.arange(32)[None, :] + .5))
    transformed = (basis @ pixels @ basis.T).ravel()
    return transformed > np.median(transformed[1:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--ppm-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    cache, report = Path(args.cache_dir), Path(args.report_dir)
    cache.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    ppm, photos = [], []
    mattes = {p.stem: p for p in (Path(args.ppm_dir) / "matte").iterdir() if p.is_file()}
    for path in sorted((Path(args.ppm_dir) / "image").glob("*.jpg")):
        with Image.open(path) as image, Image.open(mattes[path.stem]) as matte:
            rgb, alpha = _fit_pair(image, matte, 256)
        normal = canonical(rgb, alpha)
        ppm.append({"identity": path.stem, "sha256": sha256(path), "canonical": normal,
                    "hash": phash(normal), "flipped_hash": phash(normal.transpose(Image.Transpose.FLIP_LEFT_RIGHT))})
    for path in sorted(Path(args.source_dir).glob("*.png")):
        with Image.open(path) as image:
            if image.mode != "RGBA":
                raise ValueError(f"Expected ground-truth RGBA: {path}")
            original_size = list(image.size)
            rgb, alpha = _fit_pair(image, image.getchannel("A"), 256)
        rgba = np.concatenate([rgb, alpha[..., None]], -1)
        Image.fromarray(np.rint(rgba * 255).astype(np.uint8)).save(cache / path.name)
        normal = canonical(rgb, alpha)
        photo = {"identity": path.stem, "sha256": sha256(path), "original_size": original_size,
                 "canonical": normal, "hash": phash(normal)}
        photos.append(photo)
    audit = []
    for photo in photos:
        distances = [min(int(np.count_nonzero(photo["hash"] != item["hash"])),
                         int(np.count_nonzero(photo["hash"] != item["flipped_hash"]))) for item in ppm]
        nearest = int(np.argmin(distances))
        exact = any(photo["sha256"] == item["sha256"] for item in ppm)
        canonical_exact = any(photo["canonical"].tobytes() == item["canonical"].tobytes() for item in ppm)
        # Conservative exclusion threshold; manual candidates are retained as a local sheet.
        audit.append({"identity": photo["identity"], "nearest_ppm": ppm[nearest]["identity"],
                      "foreground_phash_distance": distances[nearest], "exact_file_match": exact,
                      "exact_canonical_match": canonical_exact,
                      "excluded": exact or canonical_exact or distances[nearest] <= 8})
    write_csv(report / "source-overlap-audit.csv", audit)
    ranked = sorted(zip(photos, audit), key=lambda pair: pair[1]["foreground_phash_distance"])[:10]
    sheet = Image.new("RGB", (512, 10 * 160), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (photo, row) in enumerate(ranked):
        other = next(item for item in ppm if item["identity"] == row["nearest_ppm"])
        for j, item in enumerate((photo, other)):
            sheet.paste(item["canonical"].resize((140, 140)), (j * 256, i * 160 + 20))
        draw.text((0, i * 160 + 2), f"{row['identity']} vs {row['nearest_ppm']} d={row['foreground_phash_distance']}", fill="black")
    sheet.save(cache / "overlap-nearest.png")
    excluded = {row["identity"] for row in audit if row["excluded"]}
    eligible = [photo for photo in photos if photo["identity"] not in excluded]
    eligible.sort(key=lambda item: hashlib.sha256(f"photo-study-17:{item['identity']}".encode()).digest())
    records = [{"identity": photo["identity"], "split": "val" if i < 20 else "train",
                "source_sha256": photo["sha256"], "cached_sha256": sha256(cache / f"{photo['identity']}.png"),
                "original_size": photo["original_size"]} for i, photo in enumerate(eligible)]
    write_json(report / "photo-split.json", records)
    summary = {"source_images": len(photos), "ppm_images_audited": len(ppm), "excluded": sorted(excluded),
               "training_images": sum(r["split"] == "train" for r in records), "validation_images": 20,
               "identity_overlap": "Not established: neither source provides reliable person IDs; no face recognition performed.",
               "audit_limit": "Exact file/canonical comparisons and 64-bit perceptual hashes of alpha-composited center crops, including mirrored PPM crops, cannot prove source disjointness under arbitrary edits."}
    write_json(report / "data-audit.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
