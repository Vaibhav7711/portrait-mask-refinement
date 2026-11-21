# Experiment log

All runs below were performed on 2026-11-21 with PyTorch 2.10.0 on Apple Silicon. Unless noted, device was MPS and the data seed was 17.

| Run | Hypothesis / change | Result | Decision |
|---|---|---|---|
| R0: baseline reproduction | Re-run the published 64×64, 6-epoch fixture experiment unchanged. | Exact reproduction: rough 0.1830/0.3901, mask U-Net 0.2766/0.1656, RGB U-Net 0.2271/0.1111 full/boundary MAE. | Treat published fixture values as stable motivation. |
| R1: interior diagnosis | Measure foreground, background, and already-correct pixels instead of inferring the cause from full MAE. | Unconstrained RGB changed 99.94% of correct interior pixels; damage rates were 99.04% foreground and 99.99% background. | Widespread interior damage is real; add an inference-time preservation invariant. |
| R2: 64 px post-hoc band sweep | Restrict the existing model without retraining; widths `{1,2,3,5,8}`. | Width 8 covered 76.9% of validation errors. Test full MAE improved 0.2271→0.1518, but boundary MAE worsened 0.1111→0.2041. | Constraint is a safety/recall trade-off, not a free gain. |
| R3: 64 px photographic pilot | Evaluate on all PPM-100 images using a frozen 19/81 split and three corruptions. | Post-hoc width 2 gave 0.1489 full and 0.3502 boundary MAE; learned width-2 RGB gave 0.1496/0.3523. | Real validation prefers a narrower band; move the final benchmark to 256 px. |
| R4: practical MODNet masks | Replace synthetic corruptions with thresholded official MODNet masks. | On 81 test images: rough 0.0606 boundary MAE, feather 0.0496, post-hoc CNN 0.1397, trained constrained RGB 0.2580. | Do not apply the refiner blindly to already-strong masks; feathering wins this condition. |
| R5: 256 px band sweep | Increase resolution and test widths `{4,8,12,20,32}` on PPM validation. | Width 4 covered 65.5% of errors and scored 0.1341 validation boundary MAE; width 32 covered 98.9% but scored 0.2113. | Freeze editable width 4 and ground-truth evaluation width 12. |
| R6: 256 px main test | Compare all six methods on 81 foregrounds × 3 corruptions. | Feather won boundary MAE at 0.1368. Best constrained RGB scored 0.0386 full / 0.1392 boundary, with 1.48%/0.08% correct FG/BG damage. Unconstrained RGB scored 0.2733/0.2496 and 81.1%/~100% damage. | Evidence supports safety and RGB benefit over constrained mask-only, but not superiority over feathering. |
| R7: seed stability | Train constrained RGB with seeds 17, 23, 41 on fixed data. | Boundary MAE `0.1431 ± 0.0034`; full MAE `0.0397 ± 0.0010`. | Report three seeds; avoid a single-seed claim. |
| R8: latency | Include decode, preprocessing, model, constraint, and resize; 10 warmups + 50 runs. | 256 px: MPS 20.0/25.7 ms p50/p95; CPU 95.0/96.0 ms. | Application path is interactive on available hardware; CUDA remains pending. |

Metric pairs in R0–R6 are full-image MAE / boundary MAE unless otherwise stated. Aggregate JSON, per-image CSVs, downloaded data, and external weights remain under ignored `outputs/` and `data/` directories.

## Failure inspection

- **Entirely missed strand:** if no rough-mask edge surrounds it, a narrow editable band cannot reach it. This appears as uncovered rough error in the band-coverage measurement.
- **Remote hole:** holes separated from the main contour form their own editable boundary only when present in the rough mask; a fully missing isolated region remains frozen. Wider bands improve coverage but measurably damage more correct pixels.
- **Similar foreground/background color:** RGB predictions become uncertain and can move the alpha in the wrong direction. The photographic failure figure is retained locally because PPM assets are not redistributed; the committed procedural failure figure demonstrates the same metric path using cleared visuals.
