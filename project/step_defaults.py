#!/usr/bin/env python3
"""Centralized default parameters for all LaueNN Step scripts.

This module contains all default parameter dictionaries used across Step 1, 2, 2a, 2b, 
3, 3a, 3b, and 4 scripts. Consolidating defaults here provides a single source of truth
for all configuration parameters, making maintenance and updates easier.

Parameters are organized by functional category (Detector, Material, Grain, etc.) so that
changing one value (e.g., detector distance) automatically updates all steps that use it.

Each step can import its defaults via:
    from step_defaults import STEP1_DEFAULTS, STEP2_DEFAULTS, etc.

Or access all defaults together:
    from step_defaults import ALL_STEPS

STRUCTURE:
- BASE category defaults (material, detector, energy, grain)
- Step-specific defaults (composed from above + step-unique params)
- ALL_STEPS master dictionary for reference
"""

from typing import Any, Dict

# ==============================================================================
# BASE CATEGORY DEFAULTS (shared across multiple steps)
# ==============================================================================

# Material and crystallography parameters (constant for a given sample)
BASE_MATERIAL: Dict[str, Any] = {
    "material_": "Ni",
    "material1_": "Ni",  # Same as material_ for single-phase
    "prefix": "",  # Optional prefix for output directories
    "symmetry": "cubic",
    "symmetry1": "cubic",
    "SG": 225,  # Space group number
    "SG1": 225,
}

# Detector geometry and setup (constant for a given beamline/detector)
BASE_DETECTOR: Dict[str, Any] = {
    "detectorparameters": [79.553, 979.32, 932.31, 0.37, 0.447],  # [dist, centerX, centerY, angle1, angle2]
    "pixelsize": 0.0734,  # mm/pixel (sCMOS bin2 standard)
    "dim1": 2018,  # Detector width in pixels
    "dim2": 2016,  # Detector height in pixels
}

# X-ray energy range (constant for a given beamline setup)
BASE_ENERGY: Dict[str, Any] = {
    "emin": 5,  # Minimum energy (keV)
    "emax": 22,  # Maximum energy (keV)
}

# Grain simulation parameters (controls dataset generation)
BASE_GRAIN: Dict[str, Any] = {
    "nb_grains_per_lp": 5,  # Number of grains per Laue pattern
    "grains_nb_simulate": 100,  # Number of orientations to simulate (benefits from crystal symmetry)
}


# ==============================================================================
# STEP 0: DETECTOR PREFLIGHT (VALIDATION + SENSITIVITY)
# ==============================================================================
STEP0_DEFAULTS: Dict[str, Any] = {
    **BASE_MATERIAL,
    **BASE_DETECTOR,
    # Detector selection
    "use_detector_catalog": True,
    "detector_name": "development",  # Built-in catalog detector
    "apply_manual_overrides": True,  # Allow config detectorparameters to override catalog detector
    "resolved_detector_name": "preflight_detector",
    "geometry": "Z>0",
    "hardware_label": "sCMOS",
    # Gate behavior
    "strict_validation": True,  # Fail step if detector config is invalid
    "fail_on_warnings": False,
    # Sensitivity sweep
    "sensitivity_enabled": True,
    "sensitivity_deltas": {
        "distance": 1.0,  # mm
        "xcen": 5.0,  # px
        "ycen": 5.0,  # px
        "angle_beta": 0.01,  # rad
        "angle_gamma": 0.01,  # rad
        "pixelsize": 0.001,  # mm/px
    },
    "tracked_metrics": [
        "center_offset_norm_mm",
        "scattering_coverage_deg",
    ],
    # Production tweak envelopes around validated baseline
    "allowed_tweaks": {
        "distance": 2.0,
        "xcen": 20.0,
        "ycen": 20.0,
        "angle_beta": 0.03,
        "angle_gamma": 0.03,
    },
}


# ==============================================================================
# STEP 1: DATASET GENERATION
# ==============================================================================
STEP1_DEFAULTS: Dict[str, Any] = {
    **BASE_MATERIAL,
    **BASE_DETECTOR,
    **BASE_ENERGY,
    # Step 1-specific grain and HKL parameters
    "hkl_max_identify": 5,  # Maximum HKL index for classification
    "hkl_max_identify1": 5,
    "maximum_angle_to_search": 120,  # Angle range for radial distribution (degrees)
    "step_for_binning": 0.1,  # Bin width for angular histogram (degrees)
    "nb_grains_per_lp_mat0": 5,  # Grains for primary material
    "nb_grains_per_lp_mat1": 5,  # Grains for secondary material
    "grains_nb_simulate": 100,
    # Validation and data filtering
    "validation_split_factor": 5,  # 1/N fraction reserved for validation
    "freq_rmv": 500,  # Remove frequencies (primary material)
    "freq_rmv1": 500,  # Remove frequencies (secondary material)
    # Diffraction conditions
    "general_diff_cond": True,
    "general_diff_rules": False,
    "include_scm": False,
    # Misorientation and realism
    "misorientation_angle": 1,
    "modelp": "random",
    "data_realism": True,
    # Preflight gating: require a recent successful Step0 detector preflight
    "require_step0_preflight": True,
    "preflight_max_age_days": 30,
}


# ==============================================================================
# STEP 2: NEURAL NETWORK TRAINING
# ==============================================================================
STEP2_DEFAULTS: Dict[str, Any] = {
    **BASE_MATERIAL,
    **BASE_GRAIN,
    # Training hyperparameters
    "batch_size": 50,
    "epochs": 5,
    "kernel_coeff": 1e-5,  # L2 regularization for kernel weights
    "bias_coeff": 1e-6,  # L2 regularization for bias
    "learning_rate": 1e-3,
    "patience": 2,  # Early stopping patience (epochs without improvement)
    # Neural network architecture
    "activation": "relu",
    "dropout_rate": 0.3,
}


# ==============================================================================
# STEP 2a: ARCHITECTURE OPTIMIZATION (Grid Search)
# ==============================================================================
STEP2A_DEFAULTS: Dict[str, Any] = {
    **BASE_MATERIAL,
    **BASE_GRAIN,
    # Grid search control
    "grid_batch_size": 20,  # Number of files per grid search batch (smaller = faster)
    "grid_cv_folds": 5,  # Cross-validation folds
    # Hyperparameter space to search
    "activation_functions": ["relu", "tanh"],
    "dropout_rates": [0.0, 0.3, 0.5],
    "learning_rates": [0.0001, 0.001, 0.01],
    "kernel_coeffs": [1e-5, 1e-4, 5e-4],
    "n_jobs": -1,  # Parallel workers (-1 = all CPUs)
}


# ==============================================================================
# STEP 2b: FINE GRID SEARCH (Around Step 2a Best Params)
# ==============================================================================
STEP2B_DEFAULTS: Dict[str, Any] = {
    **BASE_MATERIAL,
    **BASE_GRAIN,
    # Fine search control
    "grid_batch_size": 40,
    "grid_cv_folds": 5,
    "n_jobs": -1,
    "step2a_log_file": "",  # Auto-detect when empty; explicit path overrides
    # Fine search neighborhood (leave empty to use Step 2a best values)
    "activation_functions": [],
    "learning_rates": [],
    # Offsets around Step 2a best values
    "dropout_offsets": [-0.1, 0.0, 0.1],  # Range: best ± 0.1
    "kernel_multipliers": [0.5, 1.0, 2.0, 5.0],  # Range: best × [0.5, 2.0, 5.0]
}


# ==============================================================================
# STEP 3: PREDICTION AND EVALUATION
# ==============================================================================
STEP3_DEFAULTS: Dict[str, Any] = {
    **BASE_MATERIAL,
    **BASE_DETECTOR,
    **BASE_ENERGY,
    # Test data parameters
    "test_batch_size": 50,
    "num_test_batches": 5,
    # Indexation/classification thresholds
    "softmax_threshold": 0.80,  # Minimum confidence for accepting prediction
    "match_rate_threshold": 0.90,  # Minimum match rate for indexation
    "strain_calculation": True,  # Calculate strain tensor
    "tolerance_angle": 0.6,  # Angular tolerance (primary material)
    "tolerance_angle1": 0.6,  # Angular tolerance (secondary material)
    # Simulation and grid parameters
    "use_test_data_for_indexation": False,  # Use test data instead of simulated
    "use_simulated_dataset": True,  # Generate synthetic Laue patterns
    "grid_size_x": 5,  # Simulation grid size (smaller for speed)
    "grid_size_y": 5,
    "n_jobs": -1,  # Parallel workers
}


# ==============================================================================
# STEP 3a: GENERATE SIMULATED LAUE PATTERNS
# ==============================================================================
STEP3A_DEFAULTS: Dict[str, Any] = {
    **BASE_MATERIAL,
    **BASE_DETECTOR,
    **BASE_ENERGY,
    # Simulation control
    "grains_max": 2,  # Max grains per pattern (randomly 1 to N)
    "grid_size_x": 5,  # Spatial grid for simulation
    "grid_size_y": 5,
    "random_seed": None,  # Random seed (None = no fixed seed)
}


# ==============================================================================
# STEP 3b: VISUALIZE AND ASSESS SIMULATED PATTERNS
# ==============================================================================
STEP3B_DEFAULTS: Dict[str, Any] = {
    # Data source
    "simulated_data_dir": "project/Ni/simulated_dataset",
    "pattern_indices": None,  # None = all patterns, or [0, 1, 2, 3] for specific indices
    # Visualization parameters
    "grid_rows": 2,
    "grid_cols": 2,
    "show_statistics": True,  # Display pattern statistics
    "show_ground_truth": True,  # Show ground truth orientation
    # Detector dimensions (for visualization context)
    "dim1": 2018,
    "dim2": 2016,
    # Output
    "output_figure": None,  # Path to save figure (None = don't save)
    "dpi": 150,  # Resolution if saved
}


# ==============================================================================
# STEP 4: QUANTITATIVE VALIDATION
# ==============================================================================
STEP4_DEFAULTS: Dict[str, Any] = {
    **BASE_MATERIAL,
    # Test data parameters
    "test_batch_size": 50,
    "num_test_batches": 10,
    # Validation thresholds and metrics
    "confidence_thresholds": [0.5, 0.6, 0.7, 0.8, 0.9],
    "save_detailed_results": True,
    "random_seed": 42,  # Fixed seed for reproducibility
}


# ==============================================================================
# MASTER DICTIONARY FOR ALL STEPS
# ==============================================================================
ALL_STEPS = {
    "step0": STEP0_DEFAULTS,
    "step1": STEP1_DEFAULTS,
    "step2": STEP2_DEFAULTS,
    "step2a": STEP2A_DEFAULTS,
    "step2b": STEP2B_DEFAULTS,
    "step3": STEP3_DEFAULTS,
    "step3a": STEP3A_DEFAULTS,
    "step3b": STEP3B_DEFAULTS,
    "step4": STEP4_DEFAULTS,
}

# Base categories available for reference and composition
BASE_CATEGORIES = {
    "material": BASE_MATERIAL,
    "detector": BASE_DETECTOR,
    "energy": BASE_ENERGY,
    "grain": BASE_GRAIN,
}

