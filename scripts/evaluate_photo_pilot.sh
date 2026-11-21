#!/usr/bin/env bash
set -euo pipefail

# Run only after all six matched runs and the optional seed-17 run finish.
checkpoints=(
  --checkpoint existing_procedural=demo/checkpoints/portrait-refiner-fixture-256.pt
  --checkpoint procedural17=outputs/photo-study/procedural-17/best.pt
  --checkpoint procedural23=outputs/photo-study/procedural-23/best.pt
  --checkpoint procedural41=outputs/photo-study/procedural-41/best.pt
  --checkpoint photographic17=outputs/photo-study/photographic-17/best.pt
  --checkpoint photographic23=outputs/photo-study/photographic-23/best.pt
  --checkpoint photographic41=outputs/photo-study/photographic-41/best.pt
  --checkpoint photographic_modnet17=outputs/photo-study/photographic-modnet-17/best.pt
)
comparisons=(
  --group procedural_seed_mean=procedural17,procedural23,procedural41
  --group photographic_seed_mean=photographic17,photographic23,photographic41
  --compare photographic_seed_mean:procedural_seed_mean
  --compare photographic_seed_mean:feathered
  --compare photographic_modnet17:photographic17
)
for condition in modnet controlled; do
  rough_args=()
  if [ "$condition" = modnet ]; then
    rough_args=(--roughs-dir outputs/modnet-roughs)
  fi
  python3 -m scripts.evaluate_frozen \
    --images-dir data/ppm100/extracted/PPM-100/image \
    --mattes-dir data/ppm100/extracted/PPM-100/matte \
    "${rough_args[@]}" "${checkpoints[@]}" "${comparisons[@]}" \
    --device mps --output-dir "reports/photographic-study/$condition-256"
done
