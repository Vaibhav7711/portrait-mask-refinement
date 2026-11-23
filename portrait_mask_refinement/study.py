"""Matched photographic/procedural pilot, independent of frozen PPM evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .data import _procedural_background, _procedural_portrait, _rng_for, perturb_alpha


class MatchedPortraitDataset(Dataset):
    """Same backgrounds and corruption draws across domains; source-level split."""

    def __init__(self, manifest, cache_dir, split, domain="photographic", roughs_dir=None):
        self.records = [r for r in json.loads(Path(manifest).read_text()) if r["split"] == split]
        if not self.records or domain not in ("procedural", "photographic"):
            raise ValueError("Expected a nonempty split and known domain")
        self.split, self.domain, self.epoch = split, domain, 0
        self.roughs_dir = Path(roughs_dir) if roughs_dir else None
        self.portraits = []
        for record in self.records:
            if domain == "photographic":
                with Image.open(Path(cache_dir) / f"{record['identity']}.png") as image:
                    rgba = np.array(image.convert("RGBA"), dtype=np.float32) / 255
                fg, alpha = rgba[..., :3], rgba[..., 3]
            else:
                fg, alpha = _procedural_portrait(f"matched-{record['identity']}", 256)
            self.portraits.append((fg, alpha))
        self.composites = {}

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.records) * 3

    def __getitem__(self, index):
        record, version = self.records[index // 3], index % 3
        if index not in self.composites:
            fg, alpha = self.portraits[index // 3]
            background = _procedural_background(256, _rng_for(17, "study-background", record["identity"], version))
            image = fg * alpha[..., None] + background * (1 - alpha[..., None])
            self.composites[index] = torch.from_numpy(image).permute(2, 0, 1).float()
        alpha = torch.from_numpy(self.portraits[index // 3][1]).float()[None]
        if self.roughs_dir:
            with Image.open(self.roughs_dir / f"{record['identity']}-{version}.png") as mask:
                rough = torch.from_numpy(np.array(mask.convert("L"), dtype=np.float32) / 255)[None]
        else:
            epoch = self.epoch if self.split == "train" else 0
            rng = _rng_for(17, "study-corruption", self.split, record["identity"], version, epoch)
            rough = perturb_alpha(alpha[0], version, rng)
        return {"image": self.composites[index], "alpha": alpha, "rough": rough,
                "identity": record["identity"], "version": str(version)}
