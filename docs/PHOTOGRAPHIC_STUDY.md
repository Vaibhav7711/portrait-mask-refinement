# Photographic-training pilot

## Pre-specified question and controls

Does replacing procedural foregrounds with photographic foregrounds help the same constrained U-Net, and does it beat Gaussian feathering on the frozen PPM-100 MODNet masks? These are separate questions. This is a **PhotoMatte85 pilot**, not training on a large portrait corpus.

The [official Background Matting V2 dataset page](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets) permits commercial and non-commercial dataset use and requests a paper citation for academic work (and a survey for use in commercial products). This repository is a research/portfolio experiment, not a commercial deployment. PhotoMatte85 supplies 85 studio portrait foregrounds and extracted ground-truth alpha mattes. Its larger 13K training collection is not released. Download: official Google Drive ID `1KpHKYW986Dax9-ZIM7I-HyBoWVcLPuaQ` (413 MB archive). Data and trained weights remain ignored. No portrait assets are redistributed in Git.

Citation: Shanchuan Lin, Andrey Ryabtsev, Soumyadip Sengupta, Brian Curless, Steve Seitz, and Ira Kemelmacher-Shlizerman. *Real-Time High-Resolution Background Matting*. CVPR 2021. [Paper](https://arxiv.org/abs/2012.07810).

P3M-10k was also investigated: its official 5.4 GB download is accessible, but its release includes an indemnification agreement. It was **not downloaded or used**; PhotoMatte85 was chosen instead.

- Split source images before compositing; 20 hash-selected photographic validation images, remaining eligible images for training. The committed manifest, not a regenerated random split, is authoritative.
- Check all 85 sources against all 100 PPM images using exact file hashes and exact/perceptual hashes of foreground-on-gray composites, with mirrored comparisons. Conservatively exclude a nearest 64-bit perceptual distance ≤8. Review the ten closest pairs visually. This detects some source duplicates, not arbitrary edits or subject overlap.
- Neither dataset supplies reliable subject IDs. Person-disjointness **cannot be established** within PhotoMatte85 or between it and PPM; no face recognition or identity inference is performed. Counts refer to source foreground images, not proven unique people.
- Photographic foreground RGB and alpha are center-square-cropped and resized to 256×256, then composited onto the same procedural backgrounds used by the procedural arm. Backgrounds and compositions are fixed per source/version. Thus this experiment isolates foreground appearance/shape distribution; it does **not** test training on natural photographic backgrounds.
- Both arms use the same `perturb_alpha` function, matched per-source/version/epoch RNG draws, three corruption strengths, and six epochs. Data seed 17 stays fixed; model/shuffle seeds are 17, 23, 41.
- Identical 4-channel U-Net, base width 8, 256×256, batch 4, AdamW LR .002 / weight decay .0001, loss full MAE +4× boundary MAE, ground-truth half-width 12, editable half-width 4. Same number of training sources and optimizer updates per seed.
- Both arms select the lowest boundary-MAE epoch on the **same photographic validation set** with controlled corruptions. No PPM image selects a CNN checkpoint. The existing 132-update procedural checkpoint is a historical reference, not the matched procedural control.
- Optional mask-matched arm: same photographic composites, but supply binary masks from the official MODNet checkpoint; select on photographic validation MODNet masks. Report its seed count separately; this variant changes both train/validation mask distribution, not foregrounds.

## Frozen evaluation and uncertainty

The existing 19-validation/81-test PPM split and original-resolution binary MODNet PNGs remain unchanged. SHA-256 manifests bind RGB, matte, and MODNet mask bytes. Main evaluation is 81 test foregrounds ×1 MODNet mask at 256×256. Secondary evaluation is the same 81 foregrounds ×3 pre-existing deterministic controlled corruptions (243 conditions). PPM validation selects only the Gaussian radius from `{0.5,1,2,3,5}` separately per mask condition.

"Raw MODNet" here means the frozen **thresholded binary masks**, not MODNet's original soft alpha outputs. This experiment measures refinement of those supplied masks. MODNet is a fixed third-party model; this project cannot independently audit all of its pretraining source images or subject overlap.

Boundary alpha MAE is primary. Full-image alpha MAE and damage to already-correct foreground/background interior pixels are secondary, with the original region definitions in [PROTOCOL.md](PROTOCOL.md). Empty correct-FG/BG regions are omitted from their respective per-image means and their valid sample counts are reported, rather than scored as zero. This differs from historical phase-1 damage-rate aggregation, but not from its full/boundary MAE.

Report per-condition paired differences. Average errors across model seeds and repeated corruptions **within foreground** before bootstrapping 5,000 foreground resamples (seed 17). Intervals describe test-foreground uncertainty conditional on the trained seeds, not full retraining uncertainty or person-cluster uncertainty. Seed SD is reported separately. Seed-mean metrics are not ensemble-prediction metrics.

Hardware: Apple M1, MPS, PyTorch 2.10.0, Python 3.12.1; CUDA unavailable. No cloud compute used. Square morphology is implemented as two separable max-filter passes, tested exactly equal to the original square operation on CPU and MPS.

## Commands

Run from the repository root after `pip install -e '.[dev]'`:

```bash
pip install gdown
mkdir -p data/photomatte85
gdown 'https://drive.google.com/uc?id=1KpHKYW986Dax9-ZIM7I-HyBoWVcLPuaQ' -O data/photomatte85/PhotoMatte85.zip
unzip data/photomatte85/PhotoMatte85.zip -d data/photomatte85/extracted
python -m scripts.prepare_photo_study \
  --source-dir data/photomatte85/extracted \
  --ppm-dir data/ppm100/extracted/PPM-100 \
  --cache-dir data/photomatte85/cache --report-dir reports/photographic-study

for seed in 17 23 41; do
  for domain in procedural photographic; do
    python -m scripts.train_matched --domain "$domain" --seed "$seed" --device mps \
      --manifest reports/photographic-study/photo-split.json \
      --cache-dir data/photomatte85/cache \
      --output-dir "outputs/photo-study/$domain-$seed"
  done
done
```

Optional mask-matched arm and final evaluation:

```bash
python -m scripts.export_study_composites \
  --manifest reports/photographic-study/photo-split.json \
  --cache-dir data/photomatte85/cache --output-dir data/photomatte85/composites
python scripts/generate_modnet_masks.py \
  --modnet-repo data/tools/MODNet \
  --checkpoint data/tools/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt \
  --images-dir data/photomatte85/composites \
  --output-dir data/photomatte85/modnet-masks --device mps
python -m scripts.train_matched --domain photographic --seed 17 --device mps \
  --manifest reports/photographic-study/photo-split.json \
  --cache-dir data/photomatte85/cache --roughs-dir data/photomatte85/modnet-masks \
  --output-dir outputs/photo-study/photographic-modnet-17
bash scripts/evaluate_photo_pilot.sh
pip install matplotlib
python -m scripts.publish_photo_study
```

The official MODNet generator uses min-side 512, dimensions divisible by 32, RGB normalization to [-1,1], bilinear resize back to the image size, and threshold 0.5. Photo composites are quantized to 8-bit only for MODNet input. PPM MODNet masks are reused, never regenerated for this comparison.

The publication script commits only measurements, run histories, provenance hashes, and a numeric plot. Representative photographic edit/crop figures are generated locally under `outputs/photo-study/figures/` and deliberately excluded from Git under the existing no-portrait-redistribution policy. Each figure shows input RGB, binary MODNet, Gaussian, matched procedural CNN, photographic CNN, and ground truth at +0.75 stops; both cases use 96×96 crops enlarged 2×. Case selection uses the median foreground among improvements versus the procedural arm and among failures versus feathering; the displayed CNN images use seed 17, while selection uses three-seed mean errors.

## Data audit and actual budget

- Download SHA-256: `1d3bd0291c41668238596fa0a5a1f5da51ee213eeba26595e21d54621570ff64`.
- All 85 sources audited against all 100 PPM images: zero exact file/canonical matches, zero perceptual candidates at distance ≤8, minimum nearest distance 14. Ten nearest pairs visually reviewed; their source compositions differ. This is evidence against duplicates under the specified checks, **not proof of person-disjointness**.
- Actual split: **65 training / 20 validation source images**, 195/60 fixed composites. Six epochs expose 1,170 training examples and make **294 updates per run**, including each partial last batch of three.
- Main run seeds: 17, 23, 41 for both domains. Optional MODNet-mask training: seed 17 only.
- Training environment: Python 3.12.1, torch 2.10.0, NumPy 2.4.2, Pillow 12.1.1. Fresh-install demo compatibility was checked separately; see [sharing verification](SHARING_VERIFICATION.md).

The experiment's measured results will be appended only after execution, with checkpoint hashes.
