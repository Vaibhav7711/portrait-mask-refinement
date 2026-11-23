import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from portrait_mask_refinement.metrics import square_max_pool
from portrait_mask_refinement.statistics import paired_interval
from portrait_mask_refinement.study import MatchedPortraitDataset


def test_paired_foreground_interval_is_deterministic():
    first = paired_interval([-.02, -.01, .01])
    assert first == paired_interval([-.02, -.01, .01])
    assert first["foregrounds"] == 3
    assert first["improved_foregrounds"] == 2
    assert np.isclose(first["mean"], -.02 / 3)
    assert first["bootstrap_95ci"][0] <= first["mean"] <= first["bootstrap_95ci"][1]


def test_paired_interval_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        paired_interval([])


@pytest.mark.parametrize("width", [0, 1, 4, 12, 32])
def test_separable_morphology_matches_original_exactly(width):
    value = torch.rand(2, 1, 39, 53)
    expected = F.max_pool2d(value, 2 * width + 1, stride=1, padding=width)
    assert torch.equal(square_max_pool(value, width), expected)


def test_matched_domains_share_corruption_draws_and_fixed_validation(tmp_path):
    records = [{"identity": "example", "split": "train"}, {"identity": "heldout", "split": "val"}]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(records))
    rgba = np.zeros((256, 256, 4), dtype=np.uint8)
    rgba[..., :3] = 127
    rgba[30:230, 50:210, 3] = 255
    for record in records:
        Image.fromarray(rgba).save(tmp_path / f"{record['identity']}.png")
    photo = MatchedPortraitDataset(manifest, tmp_path, "train")
    procedural = MatchedPortraitDataset(manifest, tmp_path, "train", "procedural")
    # Give both arms the same foreground to isolate RNG/loader behavior.
    procedural.portraits = photo.portraits
    for epoch in (0, 1):
        for dataset in (photo, procedural):
            dataset.set_epoch(epoch)
        for index in range(3):
            assert torch.equal(photo[index]["rough"], procedural[index]["rough"])
            assert torch.equal(photo[index]["image"], procedural[index]["image"])
    validation = MatchedPortraitDataset(manifest, tmp_path, "val")
    before = validation[0]
    validation.set_epoch(9)
    assert torch.equal(before["rough"], validation[0]["rough"])
