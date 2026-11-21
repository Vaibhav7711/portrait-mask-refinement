# PR #1 sharing verification

Verified 2026-11-21 from a clean detached checkout of `925c381`, with a new virtual environment and the exact installation/demo commands in the README. No downloaded datasets or ignored training checkpoints were available in that checkout.

- Python 3.12.1 on Apple M1.
- Fresh dependency resolution: torch 2.15.0, NumPy 2.5.5, Pillow 13.0.0, pytest 9.2.0. Historical benchmark measurements used torch 2.10.0; no assertion of identical metrics across library versions.
- `pytest -q`: **8 passed**.
- `python -m compileall -q portrait_mask_refinement`: passed.
- Documented `pmr infer` command: passed with the committed procedural checkpoint.
- Input and output: **96×96**. Scaled editable half-width: **2 px**. Compared all **8,144 outside-band pixels**: **zero changed bytes** after converting the supplied mask to its documented grayscale representation.
- Regression tests also cover 48×32 non-square inputs, all 256 soft-mask byte levels, zero-width copying, and dimension-mismatch rejection.
- Published GitHub README inspected at its normal desktop size. Enlarged hair crops and captions are legible; the lower example is a real constrained-model failure versus feathering (+0.0151 boundary MAE).
- README explicitly separates procedural training from PPM photographic evaluation and identifies feathering's wins on both historical conditions. The MODNet condition is explicitly labeled 64×64, not confused with the 256×256 controlled benchmark.
- Protocol, source terms, aggregate results, and per-image CSVs are linked and published.

PR #1 is ready to merge as a candid, reproducible portfolio experiment, **not** as a claim of production accuracy or superiority to feathering. It was left open and unmerged as requested. The default `main` README remains the previous version until merge authorization.
