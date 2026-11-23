"""Train matched arms with shared photographic validation and fixed six-epoch budget."""
import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from portrait_mask_refinement.metrics import boundary_weighted_loss
from portrait_mask_refinement.model import SmallUNet
from portrait_mask_refinement.study import MatchedPortraitDataset
from portrait_mask_refinement.train import TrainConfig, _predict, validate
from portrait_mask_refinement.utils import seed_everything, select_device, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--domain", choices=["procedural", "photographic"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--roughs-dir", help="Optional MODNet masks for these fixed composites, both splits")
    args = parser.parse_args()
    if args.roughs_dir and args.domain != "photographic":
        parser.error("MODNet variant is photographic only")
    torch.set_num_threads(4)
    seed_everything(args.seed)
    device = select_device(args.device)
    cfg = TrainConfig(epochs=6, batch_size=4, base_channels=8, boundary_width=12,
                      constrained=True, editable_width=4, seed=args.seed, device=str(device))
    training = MatchedPortraitDataset(args.manifest, args.cache_dir, "train", args.domain, args.roughs_dir)
    # Shared validation domain eliminates validation-domain changes as a confound.
    validation = MatchedPortraitDataset(args.manifest, args.cache_dir, "val", "photographic", args.roughs_dir)
    loader = DataLoader(training, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(validation, batch_size=4)
    model = SmallUNet(4, 8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=1e-4)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best, steps, history, started = float("inf"), 0, [], time.monotonic()
    for epoch in range(6):
        training.set_epoch(epoch)
        model.train()
        total = 0.
        for batch in loader:
            image, rough, alpha = [batch[k].to(device) for k in ("image", "rough", "alpha")]
            optimizer.zero_grad(set_to_none=True)
            pred = _predict(model, image, rough, "rgb-mask", True, 4)
            loss = boundary_weighted_loss(pred, alpha, 12, 4.)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(image)
            steps += 1
        metrics = validate(model, val_loader, device, "rgb-mask", cfg)
        row = {"epoch": epoch + 1, "steps": steps, "train_loss": total / len(training), **metrics}
        history.append(row)
        print(f"{args.domain} seed {args.seed}: {row}", flush=True)
        if metrics["boundary_mae"] < best:
            best = metrics["boundary_mae"]
            torch.save({"model_state": model.state_dict(), "input_mode": "rgb-mask",
                        "data_config": {"resolution": 256}, "train_config": asdict(cfg),
                        "selected_epoch": epoch + 1, "best_val_boundary_mae": best,
                        "study": vars(args)}, output / "best.pt")
    write_json(output / "history.json", {"arguments": vars(args), "config": asdict(cfg),
               "train_foregrounds": len(training.records), "val_foregrounds": len(validation.records),
               "steps": steps, "seconds": time.monotonic() - started, "torch": torch.__version__, "history": history})


if __name__ == "__main__":
    main()
