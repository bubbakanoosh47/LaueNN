# Local Patches Applied to LaueNN

## Bug Fixes

### 1. CrystalParameters.py - Numpy 2.x Compatibility
**File:** `lauetoolsnn/lauetools/CrystalParameters.py` (line 1705)
**Issue:** Deprecated numpy import path breaks with numpy 2.x
**Fix:** Changed `from numpy.linalg.linalg import norm` to `from numpy.linalg import norm`
**Status:** Applied ✓

### 2. NNmodels.py - Metrics List Wrapping Bug
**File:** `lauetoolsnn/NNmodels.py` (lines 145, 147, 215, 267, 269, 307, 309, 578, 580)
**Issue:** `metricsNN` (already a list) was wrapped in another list causing "Expected all entries in the `metrics` list to be metric objects" error
**Fix:** Changed `metrics=[metricsNN]` to `metrics=metricsNN` in all 9 occurrences
**Status:** Applied ✓

## Compatibility Notes

### Keras 3.x Weight File Extension
**Context:** Modern Keras 3.x requires weight files to end with `.weights.h5` instead of `.h5`
**Impact:** Step 2 training script uses `.weights.h5` extension for compatibility
**Note:** This doesn't require patching the upstream code, just awareness in user scripts
### 3. Step2a - Modern TensorFlow/scikit-learn Integration
**File:** `project/Step2a/Step2a_Optimize_architecture_LaueNN.py`
**Issues:** 
- Deprecated `keras.wrappers.scikit_learn.KerasClassifier` removed in TensorFlow 2.14+
- Import paths changed from `keras.*` to `tensorflow.keras.*`
- `scikeras` is the official replacement for keras.wrappers
- Grid search hyperparameters need `model__` prefix for scikeras
- Training data must be categorical (one-hot encoded) for categorical_crossentropy loss

**Fixes:**
1. Added `scikeras` dependency to [requirements.txt](requirements.txt)
2. Changed imports to use `tensorflow.keras` instead of standalone `keras`
3. Replaced deprecated `KerasClassifier(build_fn=...)` with `scikeras.wrappers.KerasClassifier(model=...)`
4. Updated hyperparameter grid: `activation=` → `model__activation=`, `dropout_rate=` → `model__dropout_rate=`
5. Changed `tocategorical=False` → `tocategorical=True` when loading training data

**Status:** Applied ✓
**Reference:** TensorFlow 2.14+ compatibility

## Code Organization & Refactoring

### 4. Centralized Default Parameters (All Step Scripts)
**Files:** 
- Created: `project/step_defaults.py` (new module)
- Modified: `project/Step1/Step1_Generation_dataset_LaueNN.py`
- Modified: `project/Step2/Step2_Training_LaueNN.py`
- Modified: `project/Step2a/Step2a_Optimize_architecture_LaueNN.py`
- Modified: `project/Step2b/Step2b_Fine_grid_search_LaueNN.py`
- Modified: `project/Step3/Step3_Prediction_LaueNN.py`
- Modified: `project/Step3a/Step3a_Generate_simulateLPforPrediction_LaueNN.py`
- Modified: `project/Step3b/Step3b_Visualize_SimulatedPatterns_LaueNN.py`
- Modified: `project/Step4/Step4_Validate_Predictions_LaueNN.py`

**Rationale:** 
Every step script previously defined its own `DEFAULT_PARAMS` dictionary locally. This caused:
- Hard-to-maintain code duplication (~100+ lines of parameters duplicated across 8 scripts)
- Risk of inconsistency when changing parameters (e.g., if detector geometry changes, it needs updating in 5 places)
- Unclear which parameters are invariant vs. step-specific

**Solution:**
Created `project/step_defaults.py` with:
1. **BASE category defaults** (shared across multiple steps):
   - `BASE_MATERIAL`: material_, material1_, prefix, symmetry, SG (common to Steps 1, 3, 3a, 4)
   - `BASE_DETECTOR`: detectorparameters, pixelsize, dim1/2 (common to Steps 1, 3, 3a, 3b)
   - `BASE_ENERGY`: emin, emax (common to Steps 1, 3, 3a)
   - `BASE_GRAIN`: nb_grains_per_lp, grains_nb_simulate (common to Steps 1, 2, 2a, 2b)

2. **Step-specific defaults** (composed from base categories + unique parameters):
   - `STEP1_DEFAULTS` — Dataset generation
   - `STEP2_DEFAULTS` — Neural network training
   - `STEP2A_DEFAULTS` — Architecture optimization (grid search)
   - `STEP2B_DEFAULTS` — Fine grid search
   - `STEP3_DEFAULTS` — Prediction and evaluation
   - `STEP3A_DEFAULTS` — Generate simulated patterns
   - `STEP3B_DEFAULTS` — Visualize simulated patterns
   - `STEP4_DEFAULTS` — Quantitative validation

3. **Master reference**:
   - `ALL_STEPS` — Dictionary of all step defaults
   - `BASE_CATEGORIES` — Reference to base parameter groups

**Benefits:**
- **Single source of truth**: Change detector distance once, it updates all 5 steps that use it
- **Clear parameter ownership**: Visual grouping shows which parameters matter for which workflow
- **Easier maintenance**: Future parameter additions grouped logically
- **Documentation**: Parameters grouped by function with comments
- **Python idiom**: Uses dict unpacking (`**BASE_MATERIAL`) for clean composition

**Migration:**
All step scripts updated to import their defaults:
```python
from step_defaults import STEP1_DEFAULTS  # (or STEP2_DEFAULTS, etc.)

# ... in main function:
config = load_config(args.config, STEP1_DEFAULTS)  # (was: DEFAULT_PARAMS)
```

**Status:** Applied ✓
**Context:** Refactoring to improve maintainability and reduce duplication in step scripts

## New Features

### 5. Detector Parameters Configuration System
**Files Created:**
- `lauetoolsnn/detector_config.py` (new module) — Core DetectorConfig class
- `lauetoolsnn/detector_catalog.py` (new module) — DetectorCatalog registry
- `DETECTOR_PARAMETERS_DESIGN.md` (new documentation) — Comprehensive design guide

**Rationale:**
Detector parameters were previously:
- Hardcoded in scripts individually
- Scattered across multiple example configurations
- No validation or type checking
- No support for development variations or beamline catalogs
- No programmatic discovery/calibration infrastructure

**Solution - Phase 1 Implementation:**

1. **DetectorConfig Class** (`detector_config.py`):
   - Encapsulates geometric calibration (distance, center, angles) + hardware specs
   - Provides validation with clear error messages
   - Supports serialization to/from JSON
   - Calculates derived properties (detector size, scattering angle coverage, etc.)
   - Creates variations for parameter studies
   - Methods:
     - `detectorparameters` property → returns [distance, xcen, ycen, xbet, xgam]
     - `detector_dict` property → CCDcalib-compatible dictionary
     - `validate()` → ValidationResult with physical constraint checks
     - `to_dict()`, `from_dict()` → JSON serialization
     - `apply_parameter_variation(**kwargs)` → Create modified copy
     - `describe(verbose=True)` → Human-readable summary

2. **DetectorCatalog Class** (`detector_catalog.py`):
   - Registry of detector configurations
   - Built-in hardware definitions (sCMOS, MARCCD165, VHR, EIGER variants)
   - Default development calibrations pre-configured
   - Methods:
     - `get_detector(name)` → Retrieve by name
     - `list_detectors()` → Available configurations
     - `register_detector(detector)` → Add new to catalog
     - `create_detector_variant(base, **mods)` → Parameter variations
     - `get_beamline_detectors(id)` → Filter by beamline
     - `load_from_file(path)` / `save_to_file(path)` → JSON I/O
     - `validate_all()` → Check all configurations
     - `get_statistics()` → Summary stats

3. **Validation System:**
   - Physical constraints: distance (30-300 mm), pixel size (0.005-0.5 mm), angles (±0.5 rad)
   - Sanity checks: center location, detector coverage
   - Clear diagnostic messages for each validation failure
   - Returns: `ValidationResult(is_valid, errors[], warnings[])`

4. **Integration Points (Future Steps):**
   - Step scripts will load detector via: `catalog = DetectorCatalog.load_default(); detector = catalog.get_detector("prod_name")`
   - Configuration files will reference detectors by catalog name
   - Reduces hardcoded parameters across all workflow steps

**Benefits:**
- ✓ Separation of concerns: Detector config isolated from business logic
- ✓ Reusability: Single source for each detector definition
- ✓ Validation: Early error detection with clear messages
- ✓ Extensibility: Easy to add new detectors or beamlines
- ✓ Development-friendly: Quick parameter variations without code edits
- ✓ Production-ready: Foundation for version control, change tracking

**Phase 2-4 (Future):**
- Detector catalog JSON files for production/beamlines
- Calibration tools: powder diffraction, spot analysis, parameter optimization
- CLI utilities for detector management
- Full integration with Step scripts
- Parameter sensitivity analysis tools

**Status:** Phase 1 Complete ✓
**Tested:** Both DetectorConfig and DetectorCatalog classes fully functional
**Dependencies:** None (only Python stdlib + existing lauetoolsnn imports)
**Reference:** See `DETECTOR_PARAMETERS_DESIGN.md` for complete architecture

### 6. Step0 - Detector Preflight Validation & Sensitivity Workflow
**Files:**
- Created: `project/Step0/Step0_Detector_Preflight_LaueNN.py`
- Created: `project/Step0/Step0_config.example.json`
- Modified: `project/step_defaults.py`
- Modified: `project/README.md`
- Modified: `README_AI_GUIDE.md`

**Issue / Gap:**
- Workflow started at Step 1 with detector parameters assumed to be valid
- No dedicated preflight gate to verify detector geometry before data generation/training
- No systematic local sensitivity sweep to quantify how small detector tweaks impact key metrics

**Fix Applied:**
1. Added a new **Step 0** script for detector preflight:
   - Resolves detector from catalog (`detector_name`) or manual config
   - Validates detector parameters using `DetectorConfig.validate()`
   - Supports strict gating (`strict_validation`) and optional warning gating (`fail_on_warnings`)
   - Runs configurable ±delta sensitivity sweep for: `distance`, `xcen`, `ycen`, `angle_beta`, `angle_gamma`, `pixelsize`
   - Tracks configurable metrics (default: `center_offset_norm_mm`, `scattering_coverage_deg`)
   - Produces:
     - `detector_preflight_report.json` (machine-readable)
     - `detector_preflight_summary.md` (human summary)
   - Emits recommended production tweak ranges via `allowed_tweaks`

2. Added `STEP0_DEFAULTS` in `project/step_defaults.py` and registered `step0` in `ALL_STEPS`.

3. Added user-facing Step 0 config template:
   - `project/Step0/Step0_config.example.json`

4. Updated workflow docs:
   - `project/README.md` now includes Step 0 in workflow table, scripts/config list, run instructions, and parameter reference
   - `README_AI_GUIDE.md` now includes Step 0 in status checklist and workflow overview

**Status:** Applied ✓
**Context:** Introduces a detector quality gate before Step 1 to reduce downstream failures and make detector tuning explicit/reproducible.

### 7. Step5 - Real Experimental Pattern Metrics Workflow
**Files:**
- Created: `project/Step5/Step5_Analyze_Experimental_Patterns_LaueNN.py` (core logic around lines 152-487)
- Created: `project/Step5/Step5_config.example.json`
- Created: `project/Step5/README.md`
- Modified: `project/step_defaults.py` (added `STEP5_DEFAULTS` around line 271 and `ALL_STEPS["step5"]` around line 318)
- Modified: `project/README.md` (Step 5 workflow/docs additions around lines 33-45, 143-149, 282-287)
- Modified: `README_AI_GUIDE.md` (Step 5 status/workflow updates around lines 169, 241-251, 275)

**Issue / Gap:**
- Workflow previously ended at synthetic validation (Step 4) with no standardized script for real experimental image quality assessment.
- No config-driven path to compute detector image quality metrics and compare against user-provided real-world references.
- No lightweight mechanism to compare model-vs-human Euler angle estimates per image.

**Fix Applied:**
1. Added new **Step 5** script for real image analysis:
   - Reads single image or directory (`input_path`, `image_glob`, `recursive`, `limit_images`)
   - Supports TIFF/EDF/CBF/MCCD through `fabio` with fallback to matplotlib image loading
   - Computes robust per-image intensity metrics (`p1`, `p99`, dynamic range, saturation fraction, median/MAD)
   - Performs blob-based spot extraction (`skimage.feature.blob_log`) and computes spot metrics (count, density, sigma, peak intensity)
   - Computes radial and spatial metrics (mean/std radial spread, quadrant imbalance)
   - Exports:
     - `step5_image_metrics.csv`
     - `step5_summary.json`
     - `step5_metrics_summary.png`

2. Added optional reference metric comparison:
   - `real_world_metrics_csv` support keyed by `filename_column`
   - Writes `step5_reference_metric_deltas.csv` for overlapping numeric fields

3. Added optional angle comparison:
   - `human_angles_csv` and `model_angles_csv` support
   - configurable `angle_columns` (default `euler1,euler2,euler3`)
   - wrapped absolute angular errors (mod 360) and aggregate MAE/RMSE
   - writes `step5_angle_comparison.csv`

4. Added Step 5 defaults and documentation integration:
   - New `STEP5_DEFAULTS` in centralized defaults
   - Step 5 added to workflow and config lists in project docs
   - AI guide updated with Step 5 section and command example

**Status:** Applied ✓
**Context:** Implements first-pass production-ready real experimental image metrics workflow, bridging synthetic validation (Step 4) to real-data QA with optional human/model angle benchmarking.

### 8. Step1 - Prevent Empty Class Set After Frequency Filtering
**File:** `project/Step1/Step1_Generation_dataset_LaueNN.py` (around `run_step1`, after `rmv_freq_class` call)

**Issue:**
- Step 1 can generate valid raw classes in `grain_classhkl_angbin.npz` but then remove all classes in `MOD_grain_classhkl_angbin.npz` when `freq_rmv` is too high for the dataset size.
- This occurred in web run `39b788633b16` where raw classes existed (13), but all class frequencies were `<= 500` (`min=35, median=102, max=217`), so `freq_rmv=500` removed every class.
- Step 2 then fails because `Output classes: 0`.

**Fix Applied:**
1. In Step 1, after `rmv_freq_class(...)`, load `MOD_grain_classhkl_angbin.npz` and check class count.
2. If class count is zero, automatically rerun class filtering with `freq_rmv=0` and `freq_rmv1=0`.
3. If class count is still zero after fallback, raise a hard error indicating class generation itself is invalid.

**Status:** Applied ✓
**Context:** Ensures Step 1 outputs a non-empty class set for Step 2 while preserving configured filtering when feasible.

### 8. Step5 - CSV Template Generator for Human/Reference Data Entry
**Files:**
- Created: `project/Step5/Step5_Generate_CSV_Templates.py`
- Created: `project/Step5/templates/human_angles_template.csv`
- Created: `project/Step5/templates/model_angles_template.csv`
- Created: `project/Step5/templates/real_world_metrics_template.csv`
- Modified: `project/Step5/README.md`
- Modified: `project/Step5/Step5_config.example.json`

**Issue / Gap:**
- Step 5 supports angle/metric comparison CSVs, but users had to hand-build CSV rows for each image filename.
- Manual filename entry is error-prone (typos, missing rows, mismatched keys) and slows real-data onboarding.

**Fix Applied:**
1. Added a small CLI helper that scans image files and auto-generates three template CSVs with consistent filename keys:
   - `human_angles_template.csv`
   - `model_angles_template.csv`
   - `real_world_metrics_template.csv`
2. Added starter template files in `project/Step5/templates/`.
3. Updated Step 5 README with generation command and usage notes.
4. Updated Step 5 config example to point to the template CSV locations by default.

**Status:** Applied ✓
**Context:** Reduces setup friction and key mismatches when importing human-estimated angles and real-world reference metrics for Step 5 comparison reports.

### 9. Step5 - Single Image Spot Inspection Utility (Overlay Default On)
**Files:**
- Created: `project/Step5/Step5_Inspect_Spots_LaueNN.py`
- Modified: `project/Step5/README.md`

**Issue / Gap:**
- Users needed a focused pre-model QA command to inspect a single real `.tif` image and verify extracted spot points before full inference.

**Fix Applied:**
1. Added `Step5_Inspect_Spots_LaueNN.py` to:
   - load one image (`--input-image`)
   - detect spot-like points via LoG blob detection
   - export per-spot CSV (`pixel_x`, `pixel_y`, `intensity`, `sigma_px`)
   - save overlay plot and summary chart
2. Set overlay saving **enabled by default** (`--save-overlay` default true), with optional `--no-save-overlay` toggle.
3. Added usage docs in Step 5 README.

**Status:** Applied ✓
**Context:** Provides a quick visual/data sanity check for real image inputs prior to passing data into prediction/indexation routines.

### 10. Step5 Cleanup + Step6 Real Prediction Split
**Files:**
- Modified: `project/Step5/Step5_Analyze_Experimental_Patterns_LaueNN.py`
- Modified: `project/Step5/Step5_config.example.json`
- Modified: `project/Step5/README.md`
- Modified: `project/step_defaults.py`
- Created: `project/Step6/Step6_Predict_Experimental_HKL_LaueNN.py`
- Created: `project/Step6/Step6_config.example.json`
- Created: `project/Step6/README.md`
- Modified: `project/README.md`
- Modified: `README_AI_GUIDE.md`
- Deleted: `project/Step5/Step5_Generate_CSV_Templates.py`
- Deleted: `project/Step5/templates/human_angles_template.csv`
- Deleted: `project/Step5/templates/model_angles_template.csv`
- Deleted: `project/Step5/templates/real_world_metrics_template.csv`

**Issue / Gap:**
- Step 5 had expanded into optional human-vs-model and reference CSV comparisons, while user intent for Step 5 is pure pre-model spot/image quality metrics.
- No dedicated step existed for true model inference on real experimental images in the refactored workflow.

**Fix Applied:**
1. **Step 5 simplified to metrics-only**:
   - Removed real-world CSV delta comparison and human/model angle comparison logic.
   - Removed related config keys and README sections.
   - Kept core functionality: image statistics, blob-based spot metrics, summary CSV/JSON/plot.

2. **Step 6 added for real prediction**:
   - New script loads trained model artifacts from material directory (`model_*.json`, `model_*.weights.h5`, `MOD_grain_classhkl_angbin.npz`).
   - Detects spots on real images.
   - Converts spot XY to angular coordinates (`2theta`, `chi`) via detector parameters.
   - Builds angular histogram descriptors and predicts HKL class/confidence per spot.
   - Writes per-image prediction CSV, optional overlay, optional `.cor`, plus summary CSV/JSON.

3. Added `STEP6_DEFAULTS` and registered `step6` in `ALL_STEPS`.

4. Updated top-level docs to reflect the split:
   - Step 5 = pre-model QA metrics
   - Step 6 = model-based real-image HKL prediction

**Status:** Applied ✓
**Context:** Aligns workflow semantics with user expectations: Step 5 for quality inspection, Step 6 for inference/indexation outputs on real experimental data.