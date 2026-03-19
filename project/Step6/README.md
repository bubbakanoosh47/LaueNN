# Step 6: Predict HKL on Real Experimental Images

Step 6 runs true model inference on real detector images:
- detect spots from the image
- convert pixel coordinates to angular coordinates using detector geometry
- build angular descriptors per spot
- predict HKL class and confidence per spot using trained model weights

## Quick start

```bash
.venv/bin/python project/Step6/Step6_Predict_Experimental_HKL_LaueNN.py \
  --config project/Step6/Step6_config.example.json
```

## Inputs

- Trained model files in `project/{material}/` from Step 2:
  - `model_{material}.json`
  - `model_{material}.weights.h5`
  - `MOD_grain_classhkl_angbin.npz`
- Experimental images (`.tif`, `.tiff`, and other supported detector formats)

## Outputs

Saved under `project/{material}/step6_prediction_results/`:
- `{image}_spot_predictions.csv` with pixel XY, 2theta/chi, predicted HKL, confidence
- `{image}_prediction_overlay.png` (if enabled)
- `{image}_predicted.cor` (if enabled)
- `step6_prediction_summary.csv`
- `step6_summary.json`

## Notes

- Step 6 is model prediction/inference.
- Step 5 remains pre-model image/spot quality assessment.
