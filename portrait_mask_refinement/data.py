from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset


def identity_split(identity: str, seed: int = 17) -> str:
    """Stable 70/15/15 split based only on original portrait identity."""
    digest = hashlib.sha256(f"{seed}:{identity}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    if bucket < 0.70:
        return "train"
    if bucket < 0.85:
        return "val"
    return "test"


def split_identities(identities: Iterable[str], seed: int = 17) -> dict[str, list[str]]:
    result = {"train": [], "val": [], "test": []}
    for identity in sorted(identities):
        result[identity_split(identity, seed)].append(identity)
    return result


def _rng_for(*values: object) -> np.random.Generator:
    text = ":".join(map(str, values))
    value = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    return np.random.default_rng(value)


def _procedural_background(size: int, rng: np.random.Generator) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    c0 = rng.uniform(0.05, 0.65, 3)
    c1 = rng.uniform(0.25, 0.95, 3)
    angle = rng.uniform(0, math.tau)
    ramp = (x * math.cos(angle) + y * math.sin(angle)) / (size * 1.4) + 0.35
    ramp = np.clip(ramp, 0, 1)[..., None]
    bg = c0 * (1 - ramp) + c1 * ramp
    for _ in range(5):
        cx, cy = rng.uniform(0, size, 2)
        radius = rng.uniform(size * 0.08, size * 0.28)
        glow = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * radius**2))[..., None]
        bg += glow * rng.uniform(-0.10, 0.14, 3)
    return np.clip(bg, 0, 1).astype(np.float32)


def _procedural_portrait(identity: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Make a redistributable portrait-like RGB foreground and soft alpha matte."""
    rng = _rng_for("portrait", identity)
    scale = 4
    s = size * scale
    alpha = Image.new("L", (s, s), 0)
    draw = ImageDraw.Draw(alpha)
    cx = s * (0.5 + rng.uniform(-0.035, 0.035))
    head_w = s * rng.uniform(0.27, 0.34)
    head_h = s * rng.uniform(0.40, 0.48)
    top = s * rng.uniform(0.12, 0.18)
    draw.ellipse((cx - head_w / 2, top, cx + head_w / 2, top + head_h), fill=255)
    shoulder_y = s * rng.uniform(0.60, 0.68)
    draw.ellipse((cx - s * 0.42, shoulder_y, cx + s * 0.42, s * 1.14), fill=255)
    draw.rectangle((cx - s * 0.09, top + head_h * 0.75, cx + s * 0.09, shoulder_y + s * 0.08), fill=255)
    # Wispy hair silhouettes make the boundary question visible even in the fixture.
    hair_color = 255
    for side in (-1, 1):
        for strand in range(8):
            x0 = cx + side * (head_w * (0.32 + strand * 0.018))
            y0 = top + head_h * rng.uniform(0.05, 0.35)
            x1 = x0 + side * s * rng.uniform(0.015, 0.07)
            y1 = y0 + s * rng.uniform(0.18, 0.42)
            width = max(2, int(s * rng.uniform(0.006, 0.014)))
            draw.line((x0, y0, x1, y1), fill=hair_color, width=width)
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=scale * 0.7)).resize((size, size), Image.Resampling.LANCZOS)
    a = np.asarray(alpha, dtype=np.float32) / 255.0

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    skin = rng.uniform([0.55, 0.28, 0.18], [0.95, 0.72, 0.55])
    shirt = rng.uniform(0.05, 0.75, 3)
    face = np.exp(-(((xx - cx / scale) / (head_w / scale * 0.55)) ** 2 + ((yy - (top + head_h * 0.52) / scale) / (head_h / scale * 0.55)) ** 2))
    fg = shirt[None, None, :] * np.ones((size, size, 3), np.float32)
    fg = fg * (1 - face[..., None]) + skin[None, None, :] * face[..., None]
    light = 0.82 + 0.22 * (1 - xx / size) + 0.05 * np.sin(yy / size * math.tau)
    fg *= light[..., None]
    # Dark hair cap and side regions.
    hair = ((yy < (top + head_h * 0.32) / scale) & (np.abs(xx - cx / scale) < head_w / scale * 0.6))
    hair |= ((yy < (top + head_h * 0.78) / scale) & (np.abs(xx - cx / scale) > head_w / scale * 0.38) & (np.abs(xx - cx / scale) < head_w / scale * 0.64))
    fg[hair] = rng.uniform(0.025, 0.18, 3)
    return np.clip(fg, 0, 1), a


def _load_and_fit(path: Path, size: int, mode: str) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert(mode)
        w, h = im.size
        crop = min(w, h)
        left, top = (w - crop) // 2, (h - crop) // 2
        im = im.crop((left, top, left + crop, top + crop)).resize((size, size), Image.Resampling.LANCZOS)
        return np.asarray(im, dtype=np.float32) / 255.0


def perturb_alpha(alpha: torch.Tensor, strength: int, rng: np.random.Generator) -> torch.Tensor:
    """Create a binary rough mask with scale loss, boundary shift, and holes."""
    if alpha.ndim != 2:
        raise ValueError("alpha must be HxW")
    h, w = alpha.shape
    factor = [8, 16, 24][min(strength, 2)]
    small_h, small_w = max(2, h // factor), max(2, w // factor)
    mask = (alpha >= 0.5).float()[None, None]
    mask = F.interpolate(mask, size=(small_h, small_w), mode="bilinear", align_corners=False)
    mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
    mask = (mask >= 0.5).float()
    radius = strength + 1
    kernel = radius * 2 + 1
    if rng.random() < 0.5:
        mask = F.max_pool2d(mask, kernel, stride=1, padding=radius)
    else:
        mask = 1 - F.max_pool2d(1 - mask, kernel, stride=1, padding=radius)
    # Occasional small missing regions, biased toward the foreground.
    if rng.random() < 0.65:
        count = strength + 1
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        for _ in range(count):
            cy = int(rng.uniform(h * 0.25, h * 0.8))
            cx = int(rng.uniform(w * 0.25, w * 0.75))
            rr = int(rng.uniform(max(2, h * 0.015), max(3, h * (0.025 + 0.012 * strength))))
            hole = (yy - cy) ** 2 + (xx - cx) ** 2 <= rr**2
            mask[0, 0, hole] = 0
    return mask[0]


@dataclass(frozen=True)
class DataConfig:
    resolution: int = 256
    seed: int = 17
    samples_per_identity: int = 3
    fixture_identities: int = 48
    portraits_dir: str | None = None
    backgrounds_dir: str | None = None

    def __post_init__(self) -> None:
        if self.resolution <= 0 or self.resolution % 8:
            raise ValueError("resolution must be a positive multiple of 8 for the three-level U-Net")
        if self.samples_per_identity <= 0:
            raise ValueError("samples_per_identity must be positive")


class PortraitCompositeDataset(Dataset):
    """On-demand composites; no full dataset is held in RAM."""

    def __init__(self, split: str, config: DataConfig):
        self.split = split
        self.config = config
        self.epoch = 0
        if config.portraits_dir:
            portrait_paths = sorted(Path(config.portraits_dir).glob("*.png"))
            self.portrait_paths = {p.stem: p for p in portrait_paths}
            identities = list(self.portrait_paths)
        else:
            self.portrait_paths = {}
            identities = [f"fixture-{i:04d}" for i in range(config.fixture_identities)]
        self.splits = split_identities(identities, config.seed)
        self.identities = self.splits[split]
        if not self.identities:
            raise ValueError(f"No identities assigned to {split}; add more portraits")
        self.backgrounds = sorted(Path(config.backgrounds_dir).glob("*")) if config.backgrounds_dir else []
        if config.backgrounds_dir and not any(path.is_file() for path in self.backgrounds):
            raise ValueError(f"No background files found in {config.backgrounds_dir}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.identities) * self.config.samples_per_identity

    def _portrait(self, identity: str) -> tuple[np.ndarray, np.ndarray]:
        if identity not in self.portrait_paths:
            return _procedural_portrait(identity, self.config.resolution)
        with Image.open(self.portrait_paths[identity]) as im:
            if im.mode != "RGBA":
                raise ValueError(f"{im.filename} must be an RGBA PNG with ground-truth alpha")
            im = im.resize((self.config.resolution, self.config.resolution), Image.Resampling.LANCZOS)
            arr = np.asarray(im, dtype=np.float32) / 255.0
            return arr[..., :3], arr[..., 3]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        identity = self.identities[index // self.config.samples_per_identity]
        version = index % self.config.samples_per_identity
        epoch_key = self.epoch if self.split == "train" else 0
        rng = _rng_for(self.config.seed, self.split, identity, version, epoch_key)
        fg, alpha = self._portrait(identity)
        if self.backgrounds:
            candidates = [p for p in self.backgrounds if p.is_file()]
            bg = _load_and_fit(candidates[int(rng.integers(len(candidates)))], self.config.resolution, "RGB")
        else:
            bg = _procedural_background(self.config.resolution, rng)
        image = fg * alpha[..., None] + bg * (1 - alpha[..., None])
        alpha_t = torch.from_numpy(alpha).float().unsqueeze(0)
        strength = version % 3
        rough = perturb_alpha(alpha_t[0], strength, rng)
        return {
            "image": torch.from_numpy(image).permute(2, 0, 1).float(),
            "alpha": alpha_t,
            "rough": rough,
            "identity": identity,
            "version": str(version),
        }
