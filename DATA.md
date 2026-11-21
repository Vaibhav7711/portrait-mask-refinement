# Data sources and usage terms

The committed repository contains **no third-party portraits, personal photos, or third-party model weights**. Its small demo checkpoint is trained exclusively on the repository's procedural fixture.

## Procedural fixture

The built-in fixture is generated entirely by this repository. It draws portrait-like soft mattes, hair-like strands, simple foreground colors, and synthetic backgrounds. The fixture is released under the repository's MIT license. It is designed for repeatable plumbing and smoke tests, not for claims about photographic portrait quality.

## Supplying real assets

Put subject cutouts in an untracked directory and pass it with `--portraits-dir`. Each file must be an RGBA PNG; its filename stem is treated as the original portrait identity. Put ordinary RGB/JPEG/PNG backgrounds in another untracked directory and pass `--backgrounds-dir`.

Only use assets whose license permits the intended experiment. Record the dataset name, source URL, version, download date, license, and any redistribution limits in your private experiment notes. Do not commit downloaded assets unless their terms explicitly allow redistribution. Candidate public research datasets include Adobe Deep Image Matting and P3M-10k, but their current terms must be reviewed at download time; this repository does not bundle or endorse a copy.

The application benchmark now uses PPM-100 locally. See [docs/DATASETS.md](docs/DATASETS.md) for verified official links, terms, exact setup, the frozen split, and the separate MODNet condition.

Splitting happens from the portrait filename stem before background compositing, rough-mask perturbation, or repetition. Every derived version therefore remains in exactly one split.
