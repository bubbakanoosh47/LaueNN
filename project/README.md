# LaueNN Refactored Scripts

Config-driven Python scripts for end-to-end Laue diffraction neural network workflow.

## Workflow Overview

The complete workflow consists of 8 steps:

| Step | Purpose | Input | Output |
|------|---------|-------|--------|
| **0** | Detector preflight validation and sensitivity analysis before data generation/training | Detector config, validation/sensitivity settings | Preflight pass/fail report, sensitivity summary, recommended tweak ranges |
| **1** | Generate training/testing datasets from simulated Laue diffraction patterns with random crystal orientations | Detector config, energy range | Training/testing data, HKL class definitions |
| **2** | Train neural network classifier to predict HKL indices from Laue spots | Training data from Step 1 | Trained model weights, training metrics |
| **2a** | Coarse architecture optimization via grid search over activation/dropout/LR/L2 | Training data from Step 1 | Initial ranking of best hyperparameter combinations |
| **2b** | Fine architecture optimization around Step 2a best parameters | Training data + Step 2a best params | Refined hyperparameter ranking for final training |
| **3** | Evaluate trained model on test data; compute accuracy, confusion matrix, and per-class metrics | Model from Step 2, test data from Step 1 | Evaluation metrics, confusion matrix, predictions |
| **3a** | Generate synthetic Laue patterns with known ground-truth orientations for controlled model verification | Detector config, crystal parameters | Simulated patterns with orientation matrices |
| **4** | Quantitative validation by comparing model predictions on synthetic patterns to ground-truth orientations; compute angular errors and success rates | Model from Step 2, simulated data from Step 3a | Validation metrics, confidence analysis, angular error plots |

## Project Files

### Core Scripts
- **Step0/Step0_Detector_Preflight_LaueNN.py** - Validate detector configuration and run parameter sensitivity sweep
- **Step1_Generation_dataset_LaueNN.py** - Generate training and testing datasets
- **Step2_Training_LaueNN.py** - Train neural network on Step 1 data
- **Step2a/Step2a_Optimize_architecture_LaueNN.py** - Coarse grid search for optimal hyperparameters
- **Step2b/Step2b_Fine_grid_search_LaueNN.py** - Fine grid search around Step 2a best parameters
- **Step3/Step3_Prediction_LaueNN.py** - Evaluate model and compute metrics
- **Step3a/Step3a_Generate_simulateLPforPrediction_LaueNN.py** - Generate synthetic data with ground truth
- **Step3b/Step3b_Visualize_SimulatedPatterns_LaueNN.py** - Visualize and inspect simulated patterns
- **Step4/Step4_Validate_Predictions_LaueNN.py** - Quantitative validation vs ground truth

### Configuration Files (9 files)
- **Step0/Step0_config.example.json** - Detector preflight config
- **Step1_config.example.json** - Dataset generation config
- **Step2_config.example.json** - Training config
- **Step2a/Step2a_config.example.json** - Grid search config
- **Step2b/Step2b_config.example.json** - Fine grid search config
- **Step3_config.example.json** - Evaluation config
- **Step3a/Step3a_config.example.json** - Simulation config
- **Step3b/Step3b_config.example.json** - Visualization config
- **Step4/Step4_config.example.json** - Validation config

### Utilities
- **step_utils.py** - Shared utility functions:
  - `load_config()` - Load JSON config with defaults
  - `get_save_directory()` - Create material-based folder structure
  - `get_material_prefix()` - Get material prefix for file naming

## Code Architecture

All scripts follow a consistent pattern with config-driven design:
1. Load JSON configuration with `load_config()` from `step_utils`
2. Retrieve output directory with `get_save_directory()` (handles material naming)
3. Implement step-specific logic in `run_stepN()` function
4. Parse command-line arguments with argparse
5. Call main() entry point

This design eliminates code duplication and ensures consistent file naming and directory structure across all steps.

## Quick Start

### Prerequisites
```bash
cd /path/to/LaueNN
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running Each Step

All scripts default to saving output in `project/` directory. Run with just the config file or override with `--output-root`/`--data-root` if needed.

**Step 0: Detector Preflight (Recommended First)**
```bash
python project/Step0/Step0_Detector_Preflight_LaueNN.py \
  --config project/Step0/Step0_config.example.json
```
Runs detector validation and parameter sensitivity analysis. This step should pass before continuing to Step 1.

**Step 1: Generate Training Dataset**
```bash
python project/Step1_Generation_dataset_LaueNN.py \
  --config project/Step1_config.example.json
```

**Step 2: Train Neural Network**
```bash
python project/Step2_Training_LaueNN.py \
  --config project/Step2_config.example.json
```

Step 2 automatically checks for optimized logs in the material output folder and applies best hyperparameters in this priority:
1. `fine_grid_optimizer_logger_*.txt` (Step 2b)
2. `grid_optimizer_logger_*.txt` (Step 2a)
3. Step 2 config defaults (if no valid log is found)

**Step 2a: Optimize Architecture (Optional)**
```bash
python project/Step2a/Step2a_Optimize_architecture_LaueNN.py \
  --config project/Step2a/Step2a_config.example.json
```
Performs coarse grid search over activation, dropout, learning rate, and L2 regularization to find strong initial hyperparameters.

**Step 2b: Fine Search Around Step 2a Best (Optional but Recommended)**
```bash
python project/Step2b/Step2b_Fine_grid_search_LaueNN.py \
  --config project/Step2b/Step2b_config.example.json
```
Builds a local fine grid around Step 2a best parameters by reading `grid_optimizer_logger_*.txt`.

**Step 3: Evaluate Model**
```bash
python project/Step3_Prediction_LaueNN.py \
  --config project/Step3_config.example.json
```

**Step 3a: Generate Simulated Data (Optional)**
```bash
python project/Step3a_Generate_simulateLPforPrediction_LaueNN.py \
  --config project/Step3a_config.example.json
```
Generates synthetic patterns with ground-truth orientations for controlled model verification and validation data generation.

**Step 3b: Visualize Simulated Patterns (Optional)**
```bash
python project/Step3b/Step3b_Visualize_SimulatedPatterns_LaueNN.py \
  --data-dir project/Ni/simulated_dataset --indices 0 1 2 3
```
Inspects and visualizes simulated patterns with ground truth information.

**Step 4: Quantitative Validation (Final QA)**
```bash
python project/Step4/Step4_Validate_Predictions_LaueNN.py \
  --config project/Step4/Step4_config.example.json
```
Compares predictions on simulated patterns to ground-truth orientations. Provides final quantitative metrics (angular error, success rate, confidence statistics) before deploying to real experimental data.

## Configuration Files

All scripts use JSON configuration files. Start with the `.example.json` files and customize for your setup:

### Common Parameters (All Steps)

| Parameter | Description | Type | Default |
|-----------|-------------|------|---------|
| `material_` | Primary material (must match dict_LaueTools key) | string | "Ni" |
| `material1_` | Secondary material for two-phase systems | string | Same as `material_` |
| `prefix` | Suffix for output folder name | string | "" |
| `nb_grains_per_lp` | Max grains per simulated Laue pattern | int | 5 |
| `grains_nb_simulate` | Total orientations to generate | int | 100 |

### Step 1 Specific Parameters

| Parameter | Description | Type | Default |
|-----------|-------------|------|---------|
| `detectorparameters` | [distance, xc, yc, rot1, rot2] in mm/degrees | array | See example |
| `pixelsize` | Detector pixel size in mm | float | 0.0734 |
| `dim1`, `dim2` | Detector dimensions in pixels | int | 2018, 2016 |
| `emin`, `emax` | Energy range for simulation in keV | float | 5, 22 |

### Step 0 Specific Parameters

| Parameter | Description | Type | Default |
|-----------|-------------|------|---------|
| `use_detector_catalog` | Use built-in detector catalog for baseline detector | bool | true |
| `detector_name` | Catalog detector name (for baseline) | string | "development" |
| `apply_manual_overrides` | Apply detectorparameters/pixelsize/dim overrides from config | bool | true |
| `strict_validation` | Fail preflight when validation has errors | bool | true |
| `fail_on_warnings` | Fail preflight when warnings exist | bool | false |
| `sensitivity_enabled` | Enable ±delta parameter sensitivity sweep | bool | true |
| `sensitivity_deltas` | Per-parameter perturbation values for sweep | object | See example config |
| `tracked_metrics` | Metrics to monitor during sensitivity sweep | array | ["center_offset_norm_mm", "scattering_coverage_deg"] |
| `allowed_tweaks` | Production tweak envelopes around baseline | object | See example config |

### Step 2 Specific Parameters

| Parameter | Description | Type | Default |
|-----------|-------------|------|---------|
| `batch_size` | Batch size during training | int | 50 |
| `epochs` | Number of training epochs | int | 5 |
| `kernel_coeff` | L2 regularization for weights | float | 1e-5 |
| `bias_coeff` | L2 regularization for biases | float | 1e-6 |
| `learning_rate` | Adam optimizer learning rate | float | 0.001 |
| `patience` | Early stopping patience (epochs) | int | 2 |
| `activation` | Activation function (relu, tanh, sigmoid) | string | "relu" |
| `dropout_rate` | Dropout probability (0.0-1.0) | float | 0.3 |

### Step 2a Specific Parameters

| Parameter | Description | Type | Default |
|-----------|-------------|------|---------|
| `grid_batch_size` | Files to use for grid search (smaller = faster) | int | 20 |
| `grid_cv_folds` | Cross-validation folds | int | 5 |
| `activation_functions` | List of activation functions to test | array | ["relu", "tanh"] |
| `dropout_rates` | List of dropout probabilities to test | array | [0.0, 0.3, 0.5] |
| `learning_rates` | List of learning rates to test | array | [0.0001, 0.001, 0.01] |
| `kernel_coeffs` | List of L2 regularization values to test | array | [1e-5, 1e-4, 5e-4] |
| `n_jobs` | Parallel workers (-1 = all CPUs) | int | -1 |

### Step 2b Specific Parameters

| Parameter | Description | Type | Default |
|-----------|-------------|------|---------|
| `grid_batch_size` | Files to use for fine search | int | 40 |
| `grid_cv_folds` | Cross-validation folds | int | 5 |
| `step2a_log_file` | Optional explicit path to Step 2a log file | string | "" |
| `activation_functions` | Optional activations to test (empty = use Step 2a best) | array | [] |
| `learning_rates` | Optional learning rates to test (empty = use Step 2a best) | array | [] |
| `dropout_offsets` | Offsets around Step 2a best dropout | array | [-0.1, 0.0, 0.1] |
| `kernel_multipliers` | Multipliers around Step 2a best kernel coeff | array | [0.5, 1.0, 2.0, 5.0] |
| `n_jobs` | Parallel workers (-1 = all CPUs) | int | -1 |

### Step 3 Specific Parameters

| Parameter | Description | Type | Default |
|-----------|-------------|------|---------|
| `test_batch_size` | Batch size during evaluation | int | 50 |
| `num_test_batches` | Number of test batches to load | int | 5 |
| `symmetry` | Crystal system (cubic, hexagonal, etc.) | string | "cubic" |
| `SG` | Space group number for material_ | int | 225 |
| `softmax_threshold` | Min confidence for predictions | float | 0.80 |

### Step 3a Specific Parameters

| Parameter | Description | Type | Default |
|-----------|-------------|------|---------|
| `grains_max` | Max grains per simulated pattern (randomly 1 to N) | int | 2 |
| `grid_size_x` | Number of simulated patterns in X direction | int | 5 |
| `grid_size_y` | Number of simulated patterns in Y direction | int | 5 |
| `random_seed` | Seed for reproducible pattern generation | int/null | null |

## Output Folder Structure

After running all steps, the structure looks like:

```
project/
├── Ni/                                  # Material folder
│   ├── training_data/                   # Training patches (Step 1)
│   │   ├── training_data_0.pickle
│   │   ├── training_data_1.pickle
│   │   └── ...
│   ├── testing_data/                    # Testing patches (Step 1)
│   │   ├── testing_data_0.pickle
│   │   └── ...
│   ├── simulated_dataset/               # Simulated patterns (Step 3a)
│   │   ├── Ni_0.cor
│   │   ├── Ni_1.cor
│   │   ├── ...
│   │   ├── groundtruth_OM.npz           # Ground truth orientations
│   │   ├── calib.det                    # Detector calibration
│   │   ├── filecreation_stats_Ni_v2.txt # Generation log
│   │   └── config/                      # GUI config files
│   ├── MOD_grain_classhkl_angbin.npz   # HKL class definitions (Step 1)
│   ├── class_weights.pickle             # Class weights (Step 1)
│   ├── model_Ni.json                    # Model architecture (Step 2)
│   ├── model_Ni.weights.h5              # Trained weights (Step 2)
│   ├── best_val_acc_model.h5            # Best checkpoint (Step 2)
│   ├── loss_accuracy_Ni.png             # Training curves (Step 2)
│   ├── loss_accuracy_logger_Ni.txt      # Training log (Step 2)
│   ├── grid_optimizer_logger_Ni.txt     # Grid search results (Step 2a)
│   ├── fine_grid_optimizer_logger_Ni.txt # Fine grid search results (Step 2b)
│   ├── prediction_results_Ni.txt        # Evaluation metrics (Step 3)
│   ├── confusion_matrix_Ni.png          # Confusion matrix (Step 3)
│   ├── confidence_distribution_Ni.png   # Confidence plot (Step 3)
│   ├── predictions_Ni.pickle            # Raw predictions (Step 3)
│   ├── validation_summary_Ni.png        # Validation plots (Step 4)
│   ├── validation_summary_Ni.txt        # Validation report (Step 4)
│   └── validation_results_Ni.pickle     # Detailed validation data (Step 4)
```

## Using Step 2b for Final Hyperparameter Selection

`Step 2b` refines the search locally around Step 2a best results to reduce variance before Step 3 holdout evaluation.

1. Run Step 2a coarse search.
2. Run Step 2b fine search (auto-reads Step 2a log from your material folder).
3. Take Step 2b best parameters and put them into Step 2 config.
4. Train Step 2 model, then evaluate on Step 3 holdout data.

## Using Step 2a Results to Optimize Step 2 Training

**Step 2a** performs grid search to find the best hyperparameters (activation, dropout, learning rate, and L2 regularization) for your specific material and dataset. Here's how to use the results:

### Workflow

1. **Run Step 2a first** (optional but recommended for best results):
   ```bash
   python project/Step2a_Optimize_architecture_LaueNN.py \
     --config project/Step2a_config.example.json
   ```

2. **Review the results** in `project/{material}/grid_optimizer_logger_{material}.txt`:
   ```
   Best mean CV accuracy: 0.918562
   Best parameters: {
     'model__activation': 'relu',
     'model__dropout_rate': 0.3,
     'model__learning_rate': 0.001,
     'model__kernel_coeff': 0.0001
   }
   ```

3. **Update your Step 2 config** with the best parameters:
   ```json
   {
     "material_": "Ni",
     "activation": "relu",
     "dropout_rate": 0.3,
     "learning_rate": 0.001,
     "kernel_coeff": 0.0001,
     ...
   }
   ```

4. **Train with optimized hyperparameters**:
   ```bash
   python project/Step2_Training_LaueNN.py \
     --config project/Step2_config.example.json
   ```

### When to Use Step 2a

- **New material or crystal system**: Different materials may benefit from different activation functions
- **Poor training performance**: If Step 2 shows overfitting or low accuracy
- **First time setup**: To establish optimal hyperparameters for your specific dataset
- **After changing dataset parameters**: If you modify energy range, detector setup, or grain count in Step 1

### Understanding the Results

**Activation functions tested:**
- `relu` - Most common, works well for most materials (default)
- `tanh` - Sometimes better for complex patterns with negative values

**Dropout rates tested:**
- `0.0` - No dropout (may overfit on small datasets)
- `0.3` - Moderate regularization (often optimal, default)
- `0.5` - Heavy regularization (use if overfitting persists)

**Learning rates tested:**
- `0.0001` - Slow but stable convergence
- `0.001` - Standard rate, good default
- `0.01` - Faster learning but may be unstable

**L2 Regularization (kernel_coeff) tested:**
- `1e-5` (0.00001) - Light regularization
- `1e-4` (0.0001) - Moderate regularization  
- `5e-4` (0.0005) - Stronger regularization (original default)

Step 2a tests all combinations (default: 2 × 3 × 3 × 3 = **54 combinations** × 5 CV folds = **270 models**) and ranks them by cross-validation accuracy. Higher accuracy with lower standard deviation indicates more robust hyperparameters.

**Note**: With default settings, expect ~15-30 minutes on a modern CPU with parallel processing.

## Multi-Material (Two-Phase) Systems

For two-phase materials, set different values for `material_` and `material1_`:

```json
{
  "material_": "Ni",
  "material1_": "Fe",
  "...": "..."
}
```

Output will be organized in folder: `project/Ni_Fe{prefix}/`

The model learns to distinguish between both materials' HKL indices simultaneously.

## Performance Tips

### Faster Grid Search (Step 2a)

The default configuration tests **54 combinations** (2 activations × 3 dropout × 3 learning rates × 3 regularizations). To speed up:

**Quick Test (6-9 combinations)**:
- Use only 1 activation: `["relu"]`
- Reduce learning rates: `[0.001]`
- Reduce  kernel_coeffs: `[1e-5, 1e-4]`
- Result: 1 × 3 × 1 × 2 = 6 combinations (~5 min)

**Balanced Search (18 combinations)**:
- Keep 2 activations: `["relu", "tanh"]`
- Reduce learning rates: `[0.001, 0.01]`
- Reduce kernel_coeffs: `[1e-5, 5e-4]`
- Result: 2 × 3 × 2 × 2 = 24 combinations (~10-15 min)

**Additional speedups**:
- Reduce `grid_batch_size`: 20 → 10 (uses less data)
- Reduce `grid_cv_folds`: 5 → 3 (less cross-validation)
- Reduce `n_jobs` if memory limited: -1 → 4

### Faster Training (Step 2)
- Increase `batch_size` (50 → 100) for larger batches
- Reduce `epochs` for initial testing
- Set `patience` to 1 for early stopping

### GPU Usage
If TensorFlow detects a GPU, it will automatically use it. To force CPU:
```bash
export CUDA_VISIBLE_DEVICES=-1
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Missing training data | Run Step 1 first: `python project/Step1_Generation_dataset_LaueNN.py ...` |
| ImportError: lauetoolsnn | Ensure repo root is in Python path (scripts handle this automatically) |
| Grid search too slow | Reduce `grid_batch_size`, `activation_functions` list, or `n_jobs` |
| CUDA out of memory | Use `export CUDA_VISIBLE_DEVICES=0` to select specific GPU or -1 for CPU |
| "File too large" pickle errors | Reduce dataset size in Step 1 or use smaller batch sizes |
| Model weights not found | Ensure Step 2 completed successfully and produced `.weights.h5` file |

## Advanced Usage

### Custom Detector Setup

Detector parameters are crucial for accurate predictions. Format:
```json
"detectorparameters": [
  79.553,   // Sample-detector distance (mm)
  979.32,   // X center (pixel)
  932.31,   // Y center (pixel)
  0.37,     // Rotation angle 1 (degrees)
  0.447     // Rotation angle 2 (degrees)
]
```

To find optimal detector parameters from experimental data, refer to the LaueTools documentation.

### Re-training with Different Parameters

To re-train on the same dataset with different parameters:
```bash
# Modify Step2_config.example.json
cp project/Step2_config.example.json project/Step2_config_custom.json
# Edit Step2_config_custom.json with new parameters
python project/Step2_Training_LaueNN.py --config project/Step2_config_custom.json --data-root project
```

The model will be saved with the same name but weights will be overwritten.

## See Also

- [README_AI_GUIDE.md](../README_AI_GUIDE.md) - Guide for AI assistants working on this project
- [PATCHES.md](../PATCHES.md) - Local modifications to lauetoolsnn package
- [Jupyter Notebooks](../docs/ipynb_lauenn/) - Original tutorial notebooks
- [LaueTools Documentation](https://github.com/BM32ESRF/LaueTools) - Parent project docs
