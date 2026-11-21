# Frozen evaluation protocol

## Research question

Can a boundary-constrained CNN refine rough portrait masks around hair and shoulders while preserving correct foreground and background regions?

## Regions

- **Ground-truth boundary:** dilation minus erosion of the thresholded ground-truth alpha. This is used only for evaluation and loss reporting.
- **Foreground interior:** alpha ≥ 0.95 outside the ground-truth boundary.
- **Background exterior:** alpha ≤ 0.05 outside the ground-truth boundary.
- **Already correct:** absolute rough-mask error ≤ 0.05.
- **Damaged:** prediction error exceeds rough-mask error by more than 0.01.
- **Editable region:** dilation minus erosion of the rough input. It never uses ground truth.

The evaluation boundary and editable region are deliberately distinct. Every method sees the same examples and is scored on the same ground-truth regions.

## Photographic benchmark

- PPM-100 foregrounds: 100; known unique subjects: unavailable from metadata.
- Stable filename-hash split: 19 validation, 81 test.
- Controlled condition: three deterministic corruption strengths per foreground; 57 validation and 243 test conditions.
- Practical condition: one thresholded MODNet rough mask per foreground; 19 validation and 81 test conditions.
- Training: procedural identities only. PPM test images are never used for tuning.
- Validation choices: Gaussian radius and editable width.
- Main resolution: 256×256; ground-truth evaluation half-width 12 px; editable half-width candidates `{4, 8, 12, 20, 32}`.
- Main learned comparison seeds: 17, 23, 41. Data split and corruptions stay fixed at seed 17.
- The historical MODNet-mask result is a separate 64×64 pilot with evaluation half-width 3 and trained editable half-width 2; it is not a 256×256 result.
- Uncertainty: 2,000 bootstrap resamples of foreground-level means. CSV rows retain paired per-condition results.

## Methods

1. Rough binary mask.
2. Gaussian feathering, radius tuned on validation.
3. Original unconstrained RGB+mask U-Net.
4. Original RGB+mask U-Net with post-hoc editable-band constraint.
5. Mask-only U-Net trained through the constraint.
6. RGB+mask U-Net trained through the constraint.

## Latency

Model loading and output disk write are excluded. Image decode, preprocessing, network inference, editable-band composition, and output resizing are included. Ten warmup runs precede 50 measured runs. p50 and p95 are reported on the actual selected device.

## Output contract

- Inference accepts only a photo, rough mask, and checkpoint.
- The alpha PNG has the original photo dimensions.
- Alpha is an 8-bit soft matte.
- Outside the scaled editable band, alpha bytes equal the supplied rough-mask bytes.
- Image and mask dimensions must match; mismatches are rejected rather than silently resized. Width zero copies the entire supplied mask.
