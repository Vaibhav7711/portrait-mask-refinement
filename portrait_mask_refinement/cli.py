from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import benchmark_checkpoint
from .data import DataConfig, PortraitCompositeDataset
from .evaluate import comparison_grid, evaluate_models, feather
from .evaluation_v2 import evaluate_boundary_constrained
from .infer import run_inference
from .photographic import PhotographicMatteDataset
from .train import TrainConfig, train_model
from .utils import write_json


def _data_config(args: argparse.Namespace) -> DataConfig:
    return DataConfig(
        resolution=args.resolution,
        seed=args.seed,
        samples_per_identity=args.samples_per_identity,
        fixture_identities=args.fixture_identities,
        portraits_dir=getattr(args, "portraits_dir", None),
        backgrounds_dir=getattr(args, "backgrounds_dir", None),
    )


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples-per-identity", type=int, default=3)
    parser.add_argument("--fixture-identities", type=int, default=48)
    parser.add_argument("--portraits-dir", help="RGBA PNG portraits; filename stem is identity")
    parser.add_argument("--backgrounds-dir", help="Background image folder")


def main() -> None:
    parser = argparse.ArgumentParser(prog="pmr", description="Portrait mask refinement experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    preview = sub.add_parser("fixture-preview", help="Create the requested rough-vs-feathered starting visual")
    add_data_args(preview)
    preview.add_argument("--output-dir", default="outputs/preview")

    prepare = sub.add_parser("prepare", help="Validate assets and write the identity-first split manifest")
    add_data_args(prepare)
    prepare.add_argument("--output", default="outputs/data/splits.json")

    train = sub.add_parser("train", help="Train one or both U-Net variants")
    add_data_args(train)
    train.add_argument("--input-mode", choices=["mask", "rgb-mask", "both"], default="both")
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=2e-3)
    train.add_argument("--base-channels", type=int, default=16)
    train.add_argument("--boundary-width", type=int, default=5)
    train.add_argument("--boundary-weight", type=float, default=4.0)
    train.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    train.add_argument("--train-seed", type=int, help="Initialization/shuffle seed; defaults to --seed")
    train.add_argument("--constrained", action="store_true", help="Train through the rough-mask editable-band constraint")
    train.add_argument("--editable-width", type=int, default=5, help="Editable band half-width in model pixels")
    train.add_argument("--output-dir", default="outputs/train")

    evaluate = sub.add_parser("evaluate", help="Run the four-way held-out comparison")
    add_data_args(evaluate)
    evaluate.add_argument("--mask-checkpoint", required=True)
    evaluate.add_argument("--rgb-checkpoint", required=True)
    evaluate.add_argument("--boundary-width", type=int, default=5)
    evaluate.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    evaluate.add_argument("--output-dir", default="outputs/eval")

    constrained_eval = sub.add_parser("evaluate-constrained", help="Diagnose and compare constrained refiners")
    add_data_args(constrained_eval)
    constrained_eval.add_argument("--dataset", choices=["fixture", "photographic"], default="fixture")
    constrained_eval.add_argument("--images-dir")
    constrained_eval.add_argument("--mattes-dir")
    constrained_eval.add_argument("--roughs-dir", help="Optional pretrained-segmenter masks for a separate condition")
    constrained_eval.add_argument("--original-rgb-checkpoint", required=True)
    constrained_eval.add_argument("--constrained-mask-checkpoint", required=True)
    constrained_eval.add_argument("--constrained-rgb-checkpoint", required=True)
    constrained_eval.add_argument("--boundary-width", type=int, default=5)
    constrained_eval.add_argument("--candidate-widths", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    constrained_eval.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    constrained_eval.add_argument("--output-dir", default="outputs/constrained-eval")

    infer = sub.add_parser("infer", help="Refine one RGB photo and rough mask")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--image", required=True)
    infer.add_argument("--mask", required=True)
    infer.add_argument("--alpha-output", default="outputs/inference/refined-alpha.png")
    infer.add_argument("--comparison-output", default="outputs/inference/comparison.png")
    infer.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    infer.add_argument("--editable-width", type=int, help="Override/enable editable-band constraint")

    benchmark = sub.add_parser("benchmark", help="Benchmark end-to-end inference latency after warmup")
    benchmark.add_argument("--checkpoint", required=True)
    benchmark.add_argument("--image", required=True)
    benchmark.add_argument("--mask", required=True)
    benchmark.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    benchmark.add_argument("--editable-width", type=int)
    benchmark.add_argument("--warmup", type=int, default=10)
    benchmark.add_argument("--runs", type=int, default=50)
    benchmark.add_argument("--output", default="outputs/benchmark.json")

    args = parser.parse_args()
    if args.command == "prepare":
        config = _data_config(args)
        datasets = {name: PortraitCompositeDataset(name, config) for name in ("train", "val", "test")}
        # Decode one complete sample from each split so bad RGBA/background assets fail early.
        for dataset in datasets.values():
            dataset[0]
        manifest = {
            "data_config": config.__dict__,
            "source": "real RGBA portraits" if config.portraits_dir else "procedural fixture",
            "splits": {name: dataset.identities for name, dataset in datasets.items()},
            "split_counts": {name: len(dataset.identities) for name, dataset in datasets.items()},
        }
        write_json(args.output, manifest)
        print(args.output)
    elif args.command == "fixture-preview":
        dataset = PortraitCompositeDataset("test", _data_config(args))
        sample = dataset[0]
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        from .evaluate import _to_image
        _to_image(sample["image"]).save(out / "fixture-rgb.png")
        _to_image(sample["rough"]).save(out / "fixture-rough-mask.png")
        comparison_grid(sample["image"], sample["rough"], feather(sample["rough"], 2)).save(out / "rough-vs-feathered.png")
        print(out / "rough-vs-feathered.png")
    elif args.command == "train":
        data_config = _data_config(args)
        train_config = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            base_channels=args.base_channels,
            boundary_width=args.boundary_width,
            boundary_weight=args.boundary_weight,
            seed=args.train_seed if args.train_seed is not None else args.seed,
            device=args.device,
            constrained=args.constrained,
            editable_width=args.editable_width,
        )
        modes = ["mask", "rgb-mask"] if args.input_mode == "both" else [args.input_mode]
        for mode in modes:
            print(train_model(mode, data_config, train_config, args.output_dir))
    elif args.command == "evaluate":
        result = evaluate_models(args.mask_checkpoint, args.rgb_checkpoint, _data_config(args), args.output_dir, args.device, args.boundary_width)
        print(json.dumps(result, indent=2))
    elif args.command == "evaluate-constrained":
        if args.dataset == "photographic":
            if not args.images_dir or not args.mattes_dir:
                parser.error("photographic evaluation requires --images-dir and --mattes-dir")
            val_dataset = PhotographicMatteDataset(args.images_dir, args.mattes_dir, "val", args.resolution, args.seed, args.samples_per_identity, args.roughs_dir)
            test_dataset = PhotographicMatteDataset(args.images_dir, args.mattes_dir, "test", args.resolution, args.seed, args.samples_per_identity, args.roughs_dir)
        else:
            config = _data_config(args)
            val_dataset = PortraitCompositeDataset("val", config)
            test_dataset = PortraitCompositeDataset("test", config)
        result = evaluate_boundary_constrained(
            val_dataset,
            test_dataset,
            args.original_rgb_checkpoint,
            args.constrained_mask_checkpoint,
            args.constrained_rgb_checkpoint,
            args.output_dir,
            args.device,
            args.boundary_width,
            args.candidate_widths,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "infer":
        result = run_inference(args.checkpoint, args.image, args.mask, args.alpha_output, args.comparison_output, args.device, args.editable_width)
        print(json.dumps(result, indent=2))
    elif args.command == "benchmark":
        result = benchmark_checkpoint(args.checkpoint, args.image, args.mask, args.device, args.warmup, args.runs, args.editable_width)
        write_json(args.output, result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
