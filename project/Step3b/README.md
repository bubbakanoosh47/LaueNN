# Step 3b: Visualize and Assess Simulated Patterns

This step provides comprehensive visualization and statistical analysis of the simulated Laue patterns generated in Step 3a.

## Purpose

After generating simulated Laue patterns in Step 3a, Step 3b allows you to:
- **Visually inspect** the quality and distribution of simulated diffraction spots
- **Analyze statistics** such as spots per pattern, intensity distributions
- **Review ground truth** orientation matrices
- **Generate presentation-ready figures** for reports and publications
- **Quality check** before using data for model training

## Quick Start

### Interactive visualization of first 4 patterns:
```bash
.venv/bin/python project/Step3b/Step3b_Visualize_SimulatedPatterns_LaueNN.py \
  --data-dir project/Ni/simulated_dataset \
  --indices 0 1 2 3
```

### Generate comprehensive assessment report:
```bash
.venv/bin/python project/Step3b/Step3b_Visualize_SimulatedPatterns_LaueNN.py \
  --data-dir project/Ni/simulated_dataset \
  --output project/Ni/assessment.png
```

### Statistics-only mode:
```bash
.venv/bin/python project/Step3b/Step3b_Visualize_SimulatedPatterns_LaueNN.py \
  --data-dir project/Ni/simulated_dataset \
  --stats-only
```

## Using a Config File

Create a custom config (or copy `Step3b_config.example.json`):

```json
{
  "simulated_data_dir": "project/Ni/simulated_dataset",
  "pattern_indices": [0, 1, 2, 3, 4, 5],
  "grid_rows": 2,
  "grid_cols": 3,
  "show_statistics": true,
  "show_ground_truth": true,
  "output_figure": "project/Ni/visualization_report.png",
  "dpi": 300
}
```

Then run:
```bash
.venv/bin/python project/Step3b/Step3b_Visualize_SimulatedPatterns_LaueNN.py \
  --config your_config.json
```

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON configuration file |
| `--data-dir PATH` | Path to simulated dataset directory |
| `--indices N1 N2 ...` | Specific pattern indices to show (e.g., `0 1 2 3`) |
| `--rows N` | Number of rows in grid (default: 2) |
| `--cols N` | Number of columns in grid (default: 2) |
| `--output PATH` | Save figures instead of displaying |
| `--stats-only` | Only show statistics, skip pattern grid |
| `--no-ground-truth` | Don't display ground truth info |
| `--dpi N` | DPI for saved figures (default: 150) |

## Output

### Pattern Grid Visualization
Shows Laue diffraction spots on a 2D detector for multiple patterns:
- **Scatter plot** with spot positions (X, Y pixel coordinates)
- **Color coding** by intensity (hot colormap)
- **Grid overlay** for reference
- **Title** showing pattern name and spot count

### Statistical Analysis
Three plots showing:
1. **Histogram** of spots per pattern
2. **Histogram** of mean intensities
3. **Scatter plot** of spots vs total intensity

Plus printed summary statistics:
- Mean, median, std, min, max spots per pattern
- Mean, median, std for intensities

### Ground Truth Information
Displays:
- Orientation matrix array shape
- Number of patterns and max grains per pattern
- Distribution of grains per pattern
- Example orientation matrices from first pattern

## Typical Workflow

1. **Generate patterns** (Step 3a)
   ```bash
   .venv/bin/python project/Step3a/Step3a_Generate_simulateLPforPrediction_LaueNN.py \
     --config project/Step3a/Step3a_config.example.json
   ```

2. **Quick visual check** (Step 3b)
   ```bash
   .venv/bin/python project/Step3b/Step3b_Visualize_SimulatedPatterns_LaueNN.py \
     --data-dir project/Ni/simulated_dataset \
     --indices 0 1 2 3
   ```

3. **Generate assessment report** (Step 3b)
   ```bash
   .venv/bin/python project/Step3b/Step3b_Visualize_SimulatedPatterns_LaueNN.py \
     --data-dir project/Ni/simulated_dataset \
     --output project/Ni/assessment_report.png \
     --dpi 300
   ```

4. **Proceed to training** (Step 2) or **prediction** (Step 3) if quality looks good

## What to Look For

### Good Patterns
- ✅ Uniform distribution of spots across patterns
- ✅ Reasonable spot counts (typically 50-500 depending on material/energy range)
- ✅ Spots well-distributed across detector (not clustered in corners)
- ✅ Intensity values in reasonable range

### Potential Issues
- ⚠️ Very few spots (<20) may indicate detector/energy range issues
- ⚠️ Extremely high spot counts (>1000) may cause indexing difficulties
- ⚠️ All spots in one region suggests geometry problems
- ⚠️ Zero-intensity spots indicate data generation errors

## Integration with Other Steps

- **After Step 3a**: Validate generated patterns before using for prediction
- **Before Step 3**: Confirm simulated data quality matches training data characteristics
- **When debugging**: Check if model failures correlate with specific pattern features

## Examples

### Compare different materials:
```bash
# Nickel patterns
.venv/bin/python project/Step3b/Step3b_Visualize_SimulatedPatterns_LaueNN.py \
  --data-dir project/Ni/simulated_dataset --output Ni_patterns.png

# Copper patterns (if you generated them)
.venv/bin/python project/Step3b/Step3b_Visualize_SimulatedPatterns_LaueNN.py \
  --data-dir project/Cu/simulated_dataset --output Cu_patterns.png
```

### High-resolution figures for publication:
```bash
.venv/bin/python project/Step3b/Step3b_Visualize_SimulatedPatterns_LaueNN.py \
  --data-dir project/Ni/simulated_dataset \
  --indices 0 1 2 3 4 5 6 7 8 \
  --rows 3 --cols 3 \
  --output publication_figure.png \
  --dpi 600
```

## Tips

- Use `--indices` to focus on specific interesting patterns
- Increase `--dpi` for publication-quality figures
- Use `--stats-only` for quick dataset overview without individual patterns
- Compare statistics before and after parameter changes in Step 3a
- Save assessment reports with timestamps for reproducibility

## Troubleshooting

**No .cor files found:**
- Check that Step 3a completed successfully
- Verify `--data-dir` path is correct

**Plots look empty:**
- Check detector dimensions (`dim1`, `dim2`) match Step 3a settings
- Verify .cor files contain data (not corrupted)

**Ground truth not loading:**
- Ensure `groundtruth_OM.npz` exists in simulated data directory
- Check Step 3a completed without errors
