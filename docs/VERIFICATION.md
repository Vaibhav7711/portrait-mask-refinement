# Reproducibility verification

Verified 2026-11-21 from clean detached checkouts, using a new virtual environment and the exact installation/demo commands in the README. No downloaded datasets or ignored training checkpoints were available in those checkouts.

- Python 3.12.1 on Apple M1.
- Fresh dependency resolution: torch 2.15.0, NumPy 2.5.5, Pillow 13.0.0, pytest 9.2.0. Historical benchmark measurements used torch 2.10.0; no assertion of identical metrics across library versions.
- Baseline checkout: `pytest -q` reported **8 passed**; the complete photographic-study checkout reported **16 passed**.
- `python -m compileall -q portrait_mask_refinement`: passed.
- Documented `pmr infer` command: passed with the committed procedural checkpoint.
- Input and output: **96×96**. Scaled editable half-width: **2 px**. Compared all **8,144 outside-band pixels**: **zero changed bytes** after converting the supplied mask to its documented grayscale representation.
- Regression tests also cover 48×32 non-square inputs, all 256 soft-mask byte levels, zero-width copying, and dimension-mismatch rejection.
- The published GitHub README was inspected at its normal desktop size. Enlarged hair crops and captions are legible; the lower example is a real constrained-model failure versus feathering (+0.0151 boundary MAE).
- README explicitly separates procedural training from PPM photographic evaluation and identifies feathering's wins on both historical conditions. The MODNet condition is explicitly labeled 64×64, not confused with the 256×256 controlled benchmark.
- Protocol, source terms, aggregate results, and per-image CSVs are linked and published.

The verification supports the repository as a candid, reproducible research project, **not** as a claim of production accuracy or superiority to feathering.
