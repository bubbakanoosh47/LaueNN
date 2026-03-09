# LaueNN Project - AI Assistant Guide

## Overview
This is a refactored LaueNN project with a local copy of `lauetoolsnn` that includes bug fixes and customizations. This guide instructs AI assistants on project conventions and best practices.

## Key Principle
**Always use the local `lauetoolsnn` package first. Do not suggest pip installing the upstream package.**

## Project Structure

```
/
├── lauetoolsnn/              # Local copy with patches applied
│   ├── __init__.py
│   ├── utils_lauenn.py
│   ├── lauetools/            # Core lauetools subpackage
│   │   ├── CrystalParameters.py  (PATCHED)
│   │   ├── dict_LaueTools.py
│   │   └── ...
│   └── ...
├── project/                  # User-facing scripts for Step 1, 2, 3...
│   ├── Step1_Generation_dataset_LaueNN.py
│   ├── Step1_config.example.json
│   └── ...
├── PATCHES.md               # Track all local modifications
└── README_AI_GUIDE.md       # This file
```

## Working with the Local Package

### Import Convention
Scripts in `/project/` should import directly from the local `lauetoolsnn`:

```python
import sys
from pathlib import Path

# Ensure repo root is in path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from lauetoolsnn.utils_lauenn import generate_classHKL, generate_dataset
```

### Never Do This
- ❌ Suggest `pip install lauetoolsnn`
- ❌ Import from external lauetoolsnn package
- ❌ Create local wrapper modules unless absolutely necessary
- ❌ Apply patches without documenting them in PATCHES.md
- ❌ Fix compatibility issues silently — always document the context and version info
- ❌ Modify project scripts without updating path resolution if they move directories

## Applying Patches & Logging Changes

**CRITICAL:** Every modification to code (whether in `/lauetoolsnn/` or `/project/`) must be documented in `PATCHES.md` immediately after implementation.

### Automatic Patch Documentation Workflow

When you encounter and fix:
- **Upstream bugs** in the local `lauetoolsnn` 
- **Compatibility issues** (deprecated APIs, library version changes)
- **Configuration problems** in project scripts
- **Import or path issues** from refactoring

**You must:**
1. Apply the fix to the relevant file(s)
2. **IMMEDIATELY update `PATCHES.md`** with:
   - **File path** - exact location of the change
   - **Line number(s)** - specific lines modified
   - **Issue description** - what was broken and why
   - **Fix applied** - the exact change made
   - **Status** - Applied ✓ / Pending / Testing required
   - **Context** - library versions, deprecation info, etc.
3. Test the fix works
4. Commit/document

### Why Log Everything?
- Future maintainers (human or AI) need to understand why changes exist
- Helps reproduce the exact environment where patches apply
- Prevents "why is this code written this way?" confusion months later
- Critical for debugging when upgrading dependencies

### Example: Recent Compatibility Fix

**What happened:** Step2a failed with `scikeras.wrappers.KerasClassifier` import errors and grid search parameter issues

**What was logged in PATCHES.md:**
```markdown
### 3. Step2a - Modern TensorFlow/scikit-learn Integration
**File:** `project/Step2a/Step2a_Optimize_architecture_LaueNN.py`
**Issues:** 
- Deprecated `keras.wrappers.scikit_learn.KerasClassifier` removed in TensorFlow 2.14+
- Import paths changed from `keras.*` to `tensorflow.keras.*`
- `scikeras` is the official replacement for keras.wrappers
- Grid search hyperparameters need `model__` prefix for scikeras
- Training data must be categorical (one-hot encoded) for categorical_crossentropy loss

**Fixes:**
1. Added `scikeras` dependency to requirements.txt
2. Changed imports to use `tensorflow.keras` instead of standalone `keras`
3. Replaced deprecated `KerasClassifier(build_fn=...)` with `scikeras.wrappers.KerasClassifier(model=...)`
4. Updated hyperparameter grid: `activation=` → `model__activation=`, `dropout_rate=` → `model__dropout_rate=`
5. Changed `tocategorical=False` → `tocategorical=True` when loading training data

**Status:** Applied ✓
**Reference:** TensorFlow 2.14+ compatibility
```

This level of detail in PATCHES.md ensures that:
- Someone upgrading TensorFlow 6 months from now knows exactly what to check
- The next AI assistant understands the full context of these changes
- The reasoning is preserved, not just the code

### Traditional Patch Format (Reference)

For file modifications in `/lauetoolsnn/`:

```markdown
### N. Filename - Brief Description
**File:** `path/to/file.py` (line XXX)
**Issue:** What was broken
**Fix:** What was changed
**Status:** Applied ✓
```

## Workflow for Multi-Step Scripts

When reconstructing Step 2, Step 3, etc., follow this pattern:

1. Create `/project/Step{N}_*.py` file
2. Import directly from local `lauetoolsnn` with path setup
3. Keep config files separate (`Step{N}_config.example.json`)
4. Include `--config` and `--output-root` CLI arguments
5. Document any upstream bugs encountered in `PATCHES.md`

## Current Patches Applied

See [PATCHES.md](../PATCHES.md) for the complete list, including detailed context for each fix.

**Current count:** 3 patches
1. CrystalParameters.py - numpy 2.x compatibility
2. NNmodels.py - metrics list wrapping bug
3. Step2a - Modern TensorFlow 2.14+ / scikeras integration

## Testing Changes

After modifying `/lauetoolsnn/`:

```bash
# Test Step 1 locally
cd /Users/gadejong/Documents/DeJong\ Development/LaueNN
.venv/bin/python project/Step1_Generation_dataset_LaueNN.py --help
.venv/bin/python project/Step1_Generation_dataset_LaueNN.py --config project/Step1_config.example.json --output-root project
```

## Current Status: Complete Validation Workflow ✓

All main workflow steps have been successfully refactored:

- [x] Reconstruct Step 0 (Detector Preflight Validation & Sensitivity) in `/project/` ✓
- [x] Reconstruct Step 1 (Dataset Generation) in `/project/` ✓
- [x] Reconstruct Step 2 (Training) in `/project/` ✓
- [x] Reconstruct Step 3 (Prediction/Evaluation) in `/project/` ✓
- [x] Reconstruct Step 2a (Architecture Optimization) in `/project/` ✓
- [x] Reconstruct Step 3a (Generate Simulated Data) in `/project/` ✓
- [x] Reconstruct Step 3b (Visualize & Assess Simulated Patterns) in `/project/` ✓
- [x] Reconstruct Step 4 (Quantitative Validation) in `/project/` ✓

## Workflow Steps Overview

### Step 0: Detector Preflight Validation & Sensitivity
Validate detector calibration parameters and run local parameter sensitivity analysis before any dataset generation or training. Produces pass/fail preflight report and recommended tweak ranges for production operation.

**Location:** `/project/Step0/`
**Config:** `Step0_config.example.json`
**Output:** `detector_preflight_report.json`, `detector_preflight_summary.md`, recommended tweak envelopes

### Step 1: Dataset Generation
Generate training data from material parameters and detector geometry.

**Location:** `/project/Step1/`
**Config:** `Step1_config.example.json`
**Output:** HDF5 datasets with Laue patterns and labels

### Step 2: Neural Network Training
Train the LaueNN model on generated datasets.

**Location:** `/project/Step2/`
**Config:** `Step2_config.example.json`
**Output:** Trained model weights (.h5 files)

### Step 2a: Architecture Optimization
Hyperparameter tuning for optimal network architecture.

**Location:** `/project/Step2a/`
**Config:** `Step2a_config.example.json`
**Output:** Grid search results, best parameters

### Step 3: Prediction & Evaluation
Apply trained model to experimental or simulated Laue patterns.

**Location:** `/project/Step3/`
**Config:** `Step3_config.example.json`
**Output:** Indexed orientations, match rates, strain analysis

### Step 3a: Generate Simulated Patterns
Create synthetic Laue patterns with known ground truth for validation.

**Location:** `/project/Step3a/`
**Config:** `Step3a_config.example.json`
**Output:** .cor files, ground truth orientation matrices

### Step 3b: Visualize & Assess Patterns
Visual quality assessment and statistical analysis of simulated patterns.

**Location:** `/project/Step3b/`
**Config:** `Step3b_config.example.json`
**Output:** Pattern visualizations, statistics plots

**Quick example:**
```bash
.venv/bin/python project/Step3b/Step3b_Visualize_SimulatedPatterns_LaueNN.py \
  --data-dir project/Ni/simulated_dataset --indices 0 1 2 3
```

### Step 4: Quantitative Validation
Compare neural network predictions on synthetic patterns against ground-truth orientations. Provides final quality assurance metrics before deployment to experimental data.

**Location:** `/project/Step4/`
**Config:** `Step4_config.example.json`
**Output:** Validation metrics, confidence analysis, angular error plots

**Quick example:**
```bash
.venv/bin/python project/Step4/Step4_Validate_Predictions_LaueNN.py \
  --config project/Step4/Step4_config.example.json
```

## Future Enhancements

### Detector Parameters System ✓ STARTED
- [x] Create DetectorConfig class for encapsulation and validation
- [x] Create DetectorCatalog for detector registry and management
- [x] Design comprehensive configuration architecture
- [ ] Implement detector catalog JSON files (production/beamline presets)
- [ ] Add calibration tools (powder diffraction, spot analysis)
- [ ] CLI utilities for parameter discovery and management
- [ ] Full integration with Step 1, 2, 3 scripts
- [ ] Parameter sensitivity analysis framework

### Architecture & Infrastructure
- [x] Centralized default parameters (step_defaults.py)
- [x] Detector configuration classes (detector_config.py, detector_catalog.py)
- [ ] Comprehensive detector parameters reference documentation
- [ ] Detector parameter templates for common beamlines (ESRF BM32, DLS, APS, etc.)
- [ ] Test suite for detector parameter system validation
- [ ] Extended workspace for multi-detector optimization
- [ ] Integration tests for end-to-end workflow
- [ ] Step 4c: Extended validation with strain tensor reconstruction
- [ ] Step 5: Apply model to real experimental Laue patterns


## Questions for the User

When encountering issues or needing clarification:
1. **Check `PATCHES.md` first** — is this a known fix? Is the library version in the patch note relevant?
2. Search the codebase for similar patterns
3. Ask the user about their detector parameters, materials, or expected behavior
4. **Before making changes**, propose the fix with documentation
5. **After implementing**, ask if they want you to document it in PATCHES.md or do it automatically

## Before Making ANY Changes

**Stop and document in PATCHES.md:**
- What library version is causing the issue?
- Which functions/APIs are deprecated?
- Why is the fix needed?
- Will this affect other files?

This ensures the fix is recoverable and understandable months later.

## Notes for Future Work

- The local `lauetoolsnn` is isolated from pip — this is intentional
- `adjustText` warnings are non-critical (visualization only)
- Keep Step scripts lightweight and config-driven
- Document all environment assumptions (numpy 2.x, etc.)
