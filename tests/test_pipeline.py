import torch
from PIL import Image

from portrait_mask_refinement.data import DataConfig, PortraitCompositeDataset, split_identities
from portrait_mask_refinement.metrics import alpha_errors, boundary_band
from portrait_mask_refinement.model import SmallUNet
from portrait_mask_refinement.infer import run_inference


def test_identity_splits_are_disjoint_and_complete():
    identities = [f"person-{i}" for i in range(100)]
    splits = split_identities(identities, seed=17)
    sets = {name: set(values) for name, values in splits.items()}
    assert not (sets["train"] & sets["val"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["val"] & sets["test"])
    assert set().union(*sets.values()) == set(identities)


def test_validation_fixture_is_reproducible():
    config = DataConfig(resolution=32, fixture_identities=30, samples_per_identity=2)
    first = PortraitCompositeDataset("val", config)[0]
    second = PortraitCompositeDataset("val", config)[0]
    assert torch.equal(first["image"], second["image"])
    assert torch.equal(first["rough"], second["rough"])


def test_model_shapes_and_metric_band():
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 8:24, 8:24] = 1
    band = boundary_band(target, width=2)
    assert 0 < band.sum() < band.numel()
    for channels in (1, 4):
        model = SmallUNet(channels, base_channels=8)
        pred = model(torch.randn(2, channels, 32, 32))
        assert pred.shape == target.shape
        full, boundary = alpha_errors(pred, target, width=2)
        assert full.shape == boundary.shape == (2,)
        assert torch.all((pred >= 0) & (pred <= 1))


def test_inference_uses_only_photo_and_mask(tmp_path):
    resolution = 32
    config = DataConfig(resolution=resolution, fixture_identities=30, samples_per_identity=1)
    sample = PortraitCompositeDataset("test", config)[0]
    image_path = tmp_path / "photo.png"
    mask_path = tmp_path / "mask.png"
    alpha_path = tmp_path / "alpha.png"
    comparison_path = tmp_path / "comparison.png"
    Image.fromarray((sample["image"].permute(1, 2, 0).numpy() * 255).astype("uint8")).save(image_path)
    Image.fromarray((sample["rough"][0].numpy() * 255).astype("uint8")).save(mask_path)
    model = SmallUNet(4, base_channels=8)
    checkpoint = tmp_path / "model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "input_mode": "rgb-mask",
        "data_config": config.__dict__,
        "train_config": {"base_channels": 8},
    }, checkpoint)
    run_inference(checkpoint, image_path, mask_path, alpha_path, comparison_path, "cpu")
    assert Image.open(alpha_path).size == (resolution, resolution)
    assert comparison_path.exists()
