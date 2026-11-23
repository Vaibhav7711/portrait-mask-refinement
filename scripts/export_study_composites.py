"""Export only local, fixed photographic composites for the MODNet-mask arm."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from portrait_mask_refinement.study import MatchedPortraitDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for split in ("train", "val"):
        dataset = MatchedPortraitDataset(args.manifest, args.cache_dir, split)
        for sample in dataset:
            array = np.rint(sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(array).save(output / f"{sample['identity']}-{sample['version']}.png")
            count += 1
    print(f"Exported {count} composites; no data may be committed.")


if __name__ == "__main__":
    main()
