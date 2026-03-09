# Step 4: Quantitative Validation of Predictions

## Overview

Step 4 provides **quantitative validation** of the trained neural network using held-out test data from Step 1. This is the final quality assurance step before deploying the model to real experimental data.

## What This Step Does

1. **Loads trained model** from Step 2
2. **Loads test dataset** from Step 1 (1200-element feature vectors with ground truth labels)
3. **Runs predictions** on all test patterns
4. **Compares predictions to ground truth labels**
5. **Calculates quantitative metrics**:
   - Overall classification accuracy
   - Per-class accuracy (for each HKL class)
   - Confidence distribution analysis
   - Threshold-based pass rate analysis
6. **Generates visualizations**:
   - Confidence distribution histogram
   - Accuracy vs confidence threshold curve
   - Per-class accuracy bars
   - Summary statistics panel

## Prerequisites

- ✓ **Step 1**: Training/test datasets generated
- ✓ **Step 2**: Neural network trained and saved

## Quick Start

```bash
cd /path/to/LaueNN

# Run with default settings
python project/Step4/Step4_Validate_Predictions_LaueNN.py

# Or specify custom config and output directory
python project/Step4/Step4_Validate_Predictions_LaueNN.py \
  --config project/Step4/Step4_config.example.json \
  --output-root project/
```

## Configuration

Edit `Step4_config.example.json` to customize:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `material_` | string | "Ni" | Primary material (must match Step 2 training) |
| `material1_` | string | "Ni" | Secondary material for two-phase systems |
| `prefix` | string | "" | Output folder suffix |
| `test_batch_size` | int | 50 | Batch size for model predictions |
| `num_test_batches` | int | 10 | Number of test batches to load |
| `confidence_thresholds` | array | [0.5, 0.6, 0.7, 0.8, 0.9] | Thresholds to evaluate |
| `save_detailed_results` | bool | true | Save pickle and text output files |
| `random_seed` | int | 42 | Seed for reproducibility |

## Output Files

All results saved to `project/{material}/`:

| File | Description |
|------|-------------|
| `validation_summary_{material}.png` | 4-panel summary plot with confidence distribution, threshold analysis, per-class accuracy, and statistics |
| `validation_summary_{material}.txt` | Detailed text report with all metrics |
| `validation_results_{material}.pickle` | Full results object for further analysis (numpy arrays, dictionaries) |

## Output Interpretation

### Validation Summary Plot

**Panel 1: Confidence Distribution**
- Shows spread of model confidence scores
- Red dashed line: mean confidence
- Ideal: narrow distribution, high mean (>0.90)

**Panel 2: Threshold Performance**
- Green line: accuracy of patterns above threshold
- Orange line: percentage of patterns passing threshold
- Useful for production tuning: choose threshold balancing accuracy vs coverage
- Example: At threshold 0.80, 85% of patterns pass with 99.2% accuracy

**Panel 3: Per-Class Accuracy**
- Green bars: classes with >80% accuracy
- Orange bars: classes with 50-80% accuracy
- Red bars: classes with <50% accuracy
- Shows which HKL classes the model struggles with

**Panel 4: Summary Statistics**
- Quick reference of key metrics
- Overall accuracy, mean confidence
- Per-class average accuracy

### Text Report

Detailed breakdown including:
- Overall accuracy and confidence statistics
- Per-class accuracy summary
- Threshold performance table with accuracy, pass rate, and confidence for each threshold

Example:
```
======================================================================
STEP 4: VALIDATION RESULTS
======================================================================

OVERALL STATISTICS
Test Patterns: 1778
Overall Accuracy: 0.9781 (97.81%)

CONFIDENCE METRICS
Mean Confidence: 0.976955
Std Dev: 0.080434
Min: 0.245123
Max: 0.999987

PER-CLASS ACCURACY
Mean: 0.9763
Classes with >80% accuracy: 12
Classes with >50% accuracy: 12

THRESHOLD ANALYSIS
Threshold | Count | % Pass | Accuracy | Mean Conf | Std Conf
    0.50 |  1778 | 100.00% |   0.9781 |  0.976955 |  0.080434
    0.60 |  1778 | 100.00% |   0.9781 |  0.976955 |  0.080434
    ...
```

## Workflow Integration

### After Step 4 - Typical Next Steps

1. **If Overall Accuracy < 90%**: Retrain with more data (Step 1) or different hyperparameters (Step 2a)
2. **If Specific Classes < 50%**: Investigate those classes - may need targeted data generation
3. **If Low Confidence Scores**: Training data quality may be an issue - review Step 1 parameters
4. **If Accuracy OK but Uneven Threshold Performance**: Model may be overfitting - try Step 2 regularization
5. **If All Metrics Excellent**: Ready for experimental data validation

### Interpreting Metrics for Model Deployment

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| Overall Accuracy | >95% | 85-95% | <85% |
| Mean Confidence | >0.90 | 0.80-0.90 | <0.80 |
| Std Dev Confidence | <0.10 | 0.10-0.20 | >0.20 |
| Classes >80% Accuracy | All | >80% | <80% |
| Threshold 0.80 Pass Rate | >80% | 50-80% | <50% |

## Two-Phase Systems

For materials with multiple phases, set `material_` and `material1_` differently:

```json
{
  "material_": "Ni",
  "material1_": "Fe",
  "...": "..."
}
```

Results will compare:
- Predictions on Ni test patterns vs Ni ground truth labels
- Predictions on Fe test patterns vs Fe ground truth labels
- Per-class validation metrics for mixed phase systems

## Performance Tips

### Faster Validation
- Reduce `num_test_batches` in config (e.g., 10 → 3) to use fewer test patterns
- Skip detailed plots: set `save_detailed_results: false`

### More Thorough Validation
- Increase `num_test_batches` to use all available test data (e.g., 10 → 20)
- Lower `test_batch_size` to process data in smaller chunks (e.g., 50 → 20)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Model not found" | Ensure Step 2 completed successfully |
| "Test data directory not found" | Ensure Step 1 completed successfully (generates testing_data/ folder) |
| "vali_array" error about missing arguments | Check that MOD_grain_classhkl_angbin.npz exists in model directory |
| Low overall accuracy | Retrain with Step 2a grid search for better hyperparameters |
| Uneven per-class accuracy | Specific classes may need custom data generation in Step 1—try adjusting orientations |
| Memory errors | Reduce `test_batch_size` in config (e.g., 50 → 10) |

## Advanced Usage

### Custom Confidence Thresholds

Use the threshold analysis to set deployment thresholds based on your requirements:
```python
# From validation results:
# If you need 99% accuracy in predictions,
# find the threshold where that accuracy is achieved
# e.g., at 0.85 confidence threshold → 85% of patterns qualify with 99.2% accuracy
```

### Comparing Multiple Models

Run Step 4 with different trained models to compare:
```bash
# Test model A
python project/Step4/Step4_Validate_Predictions_LaueNN.py --output-root project_modelA/

# Test model B
python project/Step4/Step4_Validate_Predictions_LaueNN.py --output-root project_modelB/

# Compare validation_summary_{material}.txt files
```

## See Also

- [Step 3: Prediction & Evaluation](../Step3/README.md)
- [Step 2: Training](../Step2/README.md)
- [Step 1: Dataset Generation](../Step1/README.md)
