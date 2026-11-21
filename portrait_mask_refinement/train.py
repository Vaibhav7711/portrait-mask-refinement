from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .constraint import constrain_alpha
from .data import DataConfig, PortraitCompositeDataset
from .metrics import alpha_errors, boundary_weighted_loss
from .model import SmallUNet
from .utils import seed_everything, select_device, write_json


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 12
    batch_size: int = 8
    learning_rate: float = 2e-3
    base_channels: int = 16
    boundary_width: int = 5
    boundary_weight: float = 4.0
    seed: int = 17
    device: str = "auto"
    num_workers: int = 0
    constrained: bool = False
    editable_width: int = 5


def _predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    rough: torch.Tensor,
    input_mode: str,
    constrained: bool,
    editable_width: int,
) -> torch.Tensor:
    model_input = rough if input_mode == "mask" else torch.cat([image, rough], dim=1)
    prediction = model(model_input)
    return constrain_alpha(prediction, rough, editable_width) if constrained else prediction


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device, input_mode: str, config: TrainConfig) -> dict[str, float]:
    model.eval()
    full_values, boundary_values = [], []
    for batch in loader:
        image = batch["image"].to(device)
        rough = batch["rough"].to(device)
        alpha = batch["alpha"].to(device)
        prediction = _predict(model, image, rough, input_mode, config.constrained, config.editable_width)
        full, boundary = alpha_errors(prediction, alpha, config.boundary_width)
        full_values.extend(full.cpu().tolist())
        boundary_values.extend(boundary.cpu().tolist())
    return {"full_mae": sum(full_values) / len(full_values), "boundary_mae": sum(boundary_values) / len(boundary_values)}


def train_model(
    input_mode: str,
    data_config: DataConfig,
    train_config: TrainConfig,
    output_dir: str | Path,
) -> Path:
    if input_mode not in {"mask", "rgb-mask"}:
        raise ValueError("input_mode must be 'mask' or 'rgb-mask'")
    seed_everything(train_config.seed)
    device = select_device(train_config.device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_set = PortraitCompositeDataset("train", data_config)
    val_set = PortraitCompositeDataset("val", data_config)
    generator = torch.Generator().manual_seed(train_config.seed)
    train_loader = DataLoader(train_set, batch_size=train_config.batch_size, shuffle=True, num_workers=train_config.num_workers, generator=generator)
    val_loader = DataLoader(val_set, batch_size=train_config.batch_size, shuffle=False, num_workers=train_config.num_workers)
    model = SmallUNet(1 if input_mode == "mask" else 4, train_config.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=1e-4)
    best = float("inf")
    suffix = "-constrained" if train_config.constrained else ""
    checkpoint_path = output_dir / f"best-{input_mode}{suffix}.pt"
    history = []
    for epoch in range(train_config.epochs):
        train_set.set_epoch(epoch)
        model.train()
        running = 0.0
        for batch in train_loader:
            image = batch["image"].to(device)
            rough = batch["rough"].to(device)
            alpha = batch["alpha"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = _predict(model, image, rough, input_mode, train_config.constrained, train_config.editable_width)
            loss = boundary_weighted_loss(pred, alpha, train_config.boundary_width, train_config.boundary_weight)
            loss.backward()
            optimizer.step()
            running += loss.item() * image.shape[0]
        metrics = validate(model, val_loader, device, input_mode, train_config)
        row = {"epoch": epoch + 1, "train_loss": running / len(train_set), **metrics}
        history.append(row)
        print(f"{input_mode} epoch {epoch + 1:02d}/{train_config.epochs}: loss={row['train_loss']:.4f} val_boundary={row['boundary_mae']:.4f}", flush=True)
        if metrics["boundary_mae"] < best:
            best = metrics["boundary_mae"]
            torch.save({
                "model_state": model.state_dict(),
                "input_mode": input_mode,
                "data_config": asdict(data_config),
                "train_config": asdict(train_config),
                "best_val_boundary_mae": best,
            }, checkpoint_path)
    write_json(output_dir / f"history-{input_mode}{suffix}.json", history)
    return checkpoint_path


def load_model(checkpoint: str | Path, device: torch.device) -> tuple[SmallUNet, dict]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    mode = payload["input_mode"]
    base = payload["train_config"]["base_channels"]
    model = SmallUNet(1 if mode == "mask" else 4, base).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload
