from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .data import perturb_alpha


def _stable_partition(name: str, validation_fraction: float = 0.2) -> str:
    value = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big") / 2**64
    return "val" if value < validation_fraction else "test"


def _fit_pair(image: Image.Image, matte: Image.Image, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Center-crop an aligned image/matte pair and resize together."""
    w, h = image.size
    crop = min(w, h)
    left, top = (w - crop) // 2, (h - crop) // 2
    box = (left, top, left + crop, top + crop)
    rgb = image.convert("RGB").crop(box).resize((resolution, resolution), Image.Resampling.LANCZOS)
    alpha = matte.convert("L").crop(box).resize((resolution, resolution), Image.Resampling.LANCZOS)
    return np.asarray(rgb, dtype=np.float32) / 255, np.asarray(alpha, dtype=np.float32) / 255


class PhotographicMatteDataset(Dataset):
    """Evaluation-only adapter for paired natural images and ground-truth mattes."""

    def __init__(
        self,
        images_dir: str | Path,
        mattes_dir: str | Path,
        split: str,
        resolution: int = 256,
        seed: int = 17,
        perturbations: int = 3,
        roughs_dir: str | Path | None = None,
    ):
        self.images_dir = Path(images_dir)
        self.mattes_dir = Path(mattes_dir)
        self.split = split
        self.resolution = resolution
        self.seed = seed
        self.roughs_dir = Path(roughs_dir) if roughs_dir else None
        self.rough_by_stem = {path.stem: path for path in self.roughs_dir.glob("*") if path.is_file()} if self.roughs_dir else {}
        self.perturbations = 1 if self.roughs_dir else perturbations
        matte_by_stem = {path.stem: path for path in self.mattes_dir.glob("*") if path.is_file()}
        pairs = []
        for image_path in sorted(path for path in self.images_dir.glob("*") if path.is_file()):
            if image_path.stem in matte_by_stem and _stable_partition(image_path.stem) == split:
                pairs.append((image_path, matte_by_stem[image_path.stem]))
        if not pairs:
            raise ValueError(f"No paired photographic samples found for split {split}")
        self.pairs = pairs
        self._cached_pair = None

    @property
    def identities(self) -> list[str]:
        return [image.stem for image, _ in self.pairs]

    def __len__(self) -> int:
        return len(self.pairs) * self.perturbations

    def _load_pair(self, pair_index: int):
        if self._cached_pair is not None and self._cached_pair[0] == pair_index:
            return self._cached_pair[1]
        image_path, matte_path = self.pairs[pair_index]
        with Image.open(image_path) as image, Image.open(matte_path) as matte:
            if image.size != matte.size:
                raise ValueError("Photographic image and matte dimensions must match")
            fitted = _fit_pair(image, matte, self.resolution)
        self._cached_pair = (pair_index, fitted)
        return fitted

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, _matte_path = self.pairs[index // self.perturbations]
        version = index % self.perturbations
        rgb, alpha = self._load_pair(index // self.perturbations)
        alpha_tensor = torch.from_numpy(alpha).float().unsqueeze(0)
        if self.roughs_dir:
            if image_path.stem not in self.rough_by_stem:
                raise ValueError(f"Missing rough mask for {image_path.stem}")
            with Image.open(image_path) as image, Image.open(self.rough_by_stem[image_path.stem]) as supplied_rough:
                _, rough_np = _fit_pair(image, supplied_rough, self.resolution)
            rough = torch.from_numpy((rough_np >= 0.5).astype(np.float32)).unsqueeze(0)
        else:
            digest = hashlib.sha256(f"{self.seed}:{self.split}:{image_path.stem}:{version}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            rough = perturb_alpha(alpha_tensor[0], version % 3, rng)
        return {
            "image": torch.from_numpy(rgb).permute(2, 0, 1).float(),
            "alpha": alpha_tensor,
            "rough": rough,
            "identity": image_path.stem,
            "version": str(version),
        }
