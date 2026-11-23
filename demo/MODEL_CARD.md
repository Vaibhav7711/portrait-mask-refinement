# Procedural demo checkpoint

`portrait-refiner-fixture-256.pt` is a small RGB+mask U-Net trained at 256×256 on 36 identities generated entirely by this repository. Training used seed 17, six epochs, base width 8, a 12 px ground-truth loss band, and a 4 px inference editable band.

It is included only to make the inference command immediately reviewable. It is not a production portrait model and should not be presented as trained on PPM-100. PPM-100 was used only for validation/test measurement and is not embedded in the weights.

- Source/fixture license: MIT
- Third-party weights: none
- File size: approximately 492 KB
- SHA-256: `eadf2513e5e72228a86e4f4a3ce63b1d2f84bf2cd2733ed73e6488feb09de4ff`
