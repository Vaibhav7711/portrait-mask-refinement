# Dataset provenance and setup

## PPM-100 photographic benchmark

- Official project: <https://github.com/ZHKKKe/PPM>
- Official download: Google Drive file `1JUx-EPoV9QAhQgmW0AyOen-xKQUzZia-`
- Retrieved: 2026-11-21
- Archive observed: approximately 262 MB; 100 JPEG images and 100 PNG alpha mattes
- Terms stated by the publisher: source photographs come from Flickr under Creative Commons licenses allowing commercial use and modification; annotated alpha mattes are CC BY-NC-SA 4.0.
- Redistribution decision: no PPM image, matte, derived crop, or model trained on PPM is committed. Only aggregate metrics are published.

PPM-100 is an evaluation benchmark and has no official train/test subdivision. Before tuning, this project froze a SHA-256 filename partition: 19 validation foregrounds and 81 test foregrounds. Subject IDs are not supplied, so unique-subject count and person-disjointness cannot be verified. The models are trained only on the procedural fixture, eliminating train/test foreground leakage. Natural PPM backgrounds are retained; no recompositing occurs.

Exact setup:

```bash
pip install gdown
mkdir -p data/ppm100
gdown 'https://drive.google.com/uc?id=1JUx-EPoV9QAhQgmW0AyOen-xKQUzZia-' -O data/ppm100/PPM-100.zip
unzip data/ppm100/PPM-100.zip -d data/ppm100/extracted
```

## MODNet practical rough-mask check

- Official code: <https://github.com/ZHKKKe/MODNet>
- Official checkpoint folder: <https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR>
- Code/models license: Apache 2.0, as stated by the official repository
- Checkpoint used: `modnet_photographic_portrait_matting.ckpt`
- Use: threshold the predicted matte at 0.5 to form a practical binary rough portrait mask
- Redistribution decision: external code, weights, and generated masks stay ignored.

```bash
git clone --depth 1 https://github.com/ZHKKKe/MODNet.git data/tools/MODNet
gdown 'https://drive.google.com/uc?id=1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz' \
  -O data/tools/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt
python scripts/generate_modnet_masks.py \
  --modnet-repo data/tools/MODNet \
  --checkpoint data/tools/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt \
  --images-dir data/ppm100/extracted/PPM-100/image \
  --output-dir outputs/modnet-roughs --device auto
```

## Procedural fixture

The built-in fixture is generated entirely by repository code and is MIT-licensed. It contains 36 portrait identities for the reported runs, with three deterministic perturbations per identity. It is useful for tests and ablations, not for photographic quality claims.

## P3M-10k alternative

P3M-10k is an official MIT-agreement portrait matting dataset with 9,421 training images, P3M-500-P, and P3M-500-NP. Its official project is <https://github.com/JizhiziLi/P3M>. PPM-100 was selected here because its complete 100-image benchmark was publicly downloadable without approval and practical to evaluate locally. P3M-10k was not downloaded or measured.
