# Photographic-training pilot

## Pre-specified question and controls

Does replacing procedural foregrounds with photographic foregrounds help the same constrained U-Net, and does it beat Gaussian feathering on the frozen PPM-100 MODNet masks? These are separate questions. This is a **PhotoMatte85 pilot**, not training on a large portrait corpus.

The [official Background Matting V2 dataset page](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets) permits commercial and non-commercial dataset use and requests a paper citation for academic work (and a survey for use in commercial products). This repository is a reproducible research implementation, not a commercial deployment. PhotoMatte85 supplies 85 studio portrait foregrounds and extracted ground-truth alpha mattes. Its larger 13K training collection is not released. Download: official Google Drive ID `1KpHKYW986Dax9-ZIM7I-HyBoWVcLPuaQ` (413 MB archive). Data and trained weights remain ignored. No portrait assets are redistributed in Git.

Citation: Shanchuan Lin, Andrey Ryabtsev, Soumyadip Sengupta, Brian Curless, Steve Seitz, and Ira Kemelmacher-Shlizerman. *Real-Time High-Resolution Background Matting*. CVPR 2021. [Paper](https://arxiv.org/abs/2012.07810).

P3M-10k was also considered, but its release requires an additional indemnification agreement. It was **not downloaded or used**; PhotoMatte85 was selected for this study instead.

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

Boundary alpha MAE is primary. Full-image alpha MAE and damage to already-correct foreground/background interior pixels are secondary, with the original region definitions in [PROTOCOL.md](PROTOCOL.md). Empty correct-FG/BG regions are omitted from their respective per-image means and their valid sample counts are reported, rather than scored as zero. This differs from the initial benchmark's damage-rate aggregation, but not from its full/boundary MAE.

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

The publication script prepares measurements, run histories, provenance hashes, and a numeric plot for Git. Representative photographic edit/crop figures are generated locally under `outputs/photo-study/figures/` and deliberately excluded from Git under the existing no-portrait-redistribution policy. Each figure shows input RGB, binary MODNet, Gaussian, matched procedural CNN, photographic CNN, and ground truth at +0.75 stops; both cases use 96×96 crops enlarged 2×. Case selection uses the median foreground among improvements versus the procedural arm and among failures versus feathering; the displayed CNN images use seed 17, while selection uses three-seed mean errors.

## Data audit and actual budget

- Download SHA-256: `1d3bd0291c41668238596fa0a5a1f5da51ee213eeba26595e21d54621570ff64`.
- All 85 sources audited against all 100 PPM images: zero exact file/canonical matches, zero perceptual candidates at distance ≤8, minimum nearest distance 14. Ten nearest pairs visually reviewed; their source compositions differ. This is evidence against duplicates under the specified checks, **not proof of person-disjointness**.
- Actual split: **65 training / 20 validation source images**, 195/60 fixed composites. Six epochs expose 1,170 training examples and make **294 updates per run**, including each partial last batch of three.
- Main run seeds: 17, 23, 41 for both domains. Optional MODNet-mask training: seed 17 only.
- Training environment: Python 3.12.1, torch 2.10.0, NumPy 2.4.2, Pillow 12.1.1. Fresh-install demo compatibility was checked separately; see [reproducibility verification](VERIFICATION.md).

## Main measured result: real MODNet masks

On 81 frozen test foregrounds at 256×256, photographic training with controlled corruption **did not reliably improve** over matched procedural training and **did not beat feathering**. These conclusions are conditional on this small PhotoMatte85 pilot and budget, not claims that photographic data never helps.

| Method | Full MAE | Boundary MAE | Correct FG damage | Correct BG damage |
|---|---:|---:|---:|---:|
| Raw thresholded MODNet | 0.010634 | 0.035353 | 0 | 0 |
| Gaussian, validation radius 0.5 | 0.009604 | 0.031517 | 0.0903% | 0.0260% |
| Existing procedural checkpoint | 0.028721 | 0.103066 | 0.4340% | 0.1265% |
| Matched procedural, mean of 3 seeds | 0.031907 | 0.115822 | 0.3698% | 0.1152% |
| Photographic + controlled corruption, mean of 3 seeds | 0.031550 | 0.115805 | 0.4378% | 0.1064% |
| Photographic + MODNet, seed 17 only | 0.012835 | 0.045920 | 0.2784% | 0.1017% |

Correct-FG damage averages 80 foregrounds with nonempty eligible regions; correct-BG damage averages all 81. Full/boundary MAE always uses all 81.

| Paired boundary difference (left minus right) | Mean | Foreground-bootstrap 95% CI | Improved foregrounds |
|---|---:|---|---:|
| Photographic − procedural, each averaged over 3 seeds | −0.000017 | [−0.005270, +0.005073] | 44/81 |
| Photographic 3-seed mean − feather | +0.084288 | [+0.075723, +0.092626] | 0/81 |
| Photographic MODNet − photographic controlled, matched seed 17 | −0.078136 | [−0.085768, −0.070455] | 81/81 |
| Photographic MODNet seed 17 − feather | +0.014404 | [+0.009158, +0.020075] | 26/81 |

The optional mask-matched variant improves by about 63.0% relative to photographic controlled-corruption seed 17 (0.124056→0.045920), without changing foregrounds. It remains worse than raw MODNet on aggregate, and its feathering deficit is not explained by foreground sampling uncertainty. Do not interpret the one-seed optional result as a three-seed estimate.

### Where the CNN loses

The supplied MODNet boundaries are already accurate (0.03535 boundary MAE). Models trained on much stronger synthetic corruption make unnecessary changes inside the permitted band, raising boundary error to about 0.116 even though interior damage remains small and outside-band preservation is exact. The PhotoMatte validation improvement did not transfer to this mask distribution. Matching the training/selection masks reduces that failure considerably, suggesting mask-error mismatch is important independently of photographic foreground appearance. This is an inference supported by the controlled optional contrast, not proof of a unique cause: natural backgrounds, domain diversity, the small dataset, unknown person overlap, and limited training remain confounded with generalization.

Detailed checkpoint hashes, validation radius scores, per-image measurements, and paired deltas are in [modnet-256](../reports/photographic-study/modnet-256/). The existing-checkpoint/reference metrics and all 100 frozen data/mask hashes match the pre-training reference run.

## Secondary measured result: controlled corruptions

This uses 81 frozen test foregrounds ×3 corruptions (243 conditions), with Gaussian radius 1 selected on 19 validation foregrounds ×3 corruptions. Lower is better:

| Method | Full MAE | Boundary MAE |
|---|---:|---:|
| Raw controlled mask | 0.037549 | 0.138315 |
| Gaussian feather | 0.037139 | 0.136801 |
| Existing procedural checkpoint | 0.038584 | 0.139216 |
| Matched procedural, mean of 3 seeds | 0.039113 | 0.142031 |
| Photographic + controlled corruption, mean of 3 seeds | 0.036318 | 0.133785 |
| Photographic + MODNet, seed 17 only | 0.036407 | 0.133844 |

Photographic minus procedural boundary MAE is **−0.008246**, foreground-bootstrap 95% CI **[−0.013085, −0.003532]**, with 52/81 foregrounds improved after averaging their three corruptions. This is a 5.8% relative improvement. Photographic minus feathering is **−0.003016 [−0.010197, +0.003959]**, with 38/81 improved: a lower point estimate, but no reliable win over feathering. These secondary findings do not reverse the negative main MODNet result.

Correct-FG/BG damage is 1.3758% / 0.0725% for matched procedural, 1.4664% / 0.0584% for photographic controlled, and 0.9147% / 0.0244% for the one-seed MODNet variant. Eligible-region counts are 240/243 for FG and 243/243 for BG. [Complete secondary results](../reports/photographic-study/controlled-256/) include all reference damage metrics and per-condition deltas.

### Across-seed variability

| Condition | Procedural boundary MAE, mean ± sample SD | Photographic controlled, mean ± sample SD |
|---|---:|---:|
| Real MODNet masks | 0.115822 ± 0.002898 | 0.115805 ± 0.007559 |
| Controlled corruptions | 0.142031 ± 0.003551 | 0.133785 ± 0.004183 |

These are three training seeds, not 243 independent training runs. [Seed measurements](../reports/photographic-study/seed-statistics.json) and [training histories](../reports/photographic-study/training/) preserve the individual results and actual budgets.

## Verification and qualitative cases

Both pytest entry points pass all 16 tests, including deterministic foreground bootstrap, matched corruption RNG, inference preservation, and exact separable morphology. Ruff and compileall pass. A clean checkout also passes 16 tests and the documented demo. The main frozen manifest and all three reference methods match the pre-training evaluation exactly; the secondary reference full/boundary metrics reproduce the historical benchmark within floating-point tolerance.

The locally generated improvement is PPM `19551619800_2c7ac67584_o`, a shoulder crop (box `[94,80,190,176]` at 256×256). Its three-seed photographic-minus-procedural boundary difference is −0.016389. The failure is `13179159164_1a4ae8d085_o`, a hair crop (`[38,32,134,128]`), with difference +0.042301 versus procedural and +0.080273 versus feathering. Displayed seed-17 comparisons have the same direction. Both use the same 96×96 crop scale and +0.75-stop exposure; the improved case still loses to feathering. [Selection metadata](../reports/photographic-study/figure-selection.json) records the non-cherry-picked subset-median rule. Run the publication command above to view the photographs locally; only numeric visualizations are published.
