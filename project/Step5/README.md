# Step 5: Real Experimental Pattern Metrics

Step 5 analyzes real Laue detector images and reports practical quality metrics that are useful before full indexing and strain interpretation.

## What it computes

For each input image:
- Intensity statistics (`min`, `max`, `mean`, `median`, `std`, p1/p99)
- Dynamic range (`p99 - p1`)
- Saturation fraction
- Blob-based spot metrics (count, density, blob size, peak intensity)
- Radial distribution metrics relative to detector center
- Spatial anisotropy proxy (`quadrant_imbalance`)
- SNR proxy from detected spot peaks vs robust background

## Quick start

```bash
.venv/bin/python project/Step5/Step5_Analyze_Experimental_Patterns_LaueNN.py \
  --config project/Step5/Step5_config.example.json
```

## Inspect one image (overlay saved by default)

For pre-model QA on a single real image:

```bash
.venv/bin/python project/Step5/Step5_Inspect_Spots_LaueNN.py \
  --input-image lauetoolsnn/examples/GaN_Si/nw1_0000.tif \
  --output-dir project/Ni/step5_experimental_metrics/inspection
```

Outputs:
- `{image_stem}_spots.csv` with `pixel_x`, `pixel_y`, `intensity`, `sigma_px`
- `{image_stem}_overlay.png` (saved by default)
- `{image_stem}_summary.png` (XY scatter + radial histogram)

Use `--no-save-overlay` only if you want to disable overlay writing.

## Config notes

Edit [Step5_config.example.json](Step5_config.example.json):
- `input_path`: folder or single image file
- `image_glob`: pattern (e.g. `*.tif`)
- `blob_detection`: spot extraction sensitivity

## Outputs

Saved under `project/{material}{prefix}/step5_experimental_metrics/`:
- `step5_image_metrics.csv`
- `step5_metrics_summary.png`
- `step5_summary.json`
