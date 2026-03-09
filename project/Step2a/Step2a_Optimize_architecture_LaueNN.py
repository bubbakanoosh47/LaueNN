#!/usr/bin/env python3
"""Step 2a: Optimize neural network architecture hyperparameters using grid search.

Refactored local script for optimizing the neural network architecture using
sklearn's GridSearchCV on datasets from Step 1.
- Config-driven with sensible defaults
- Performs grid search on activation functions, dropout rates, and layers
- Works with single-phase and two-phase materials
- Logs optimization results to a text file
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Ensure repo root is in path for local lauetoolsnn
repo_root = Path(__file__).resolve().parents[2]  # LaueNN root
project_root = Path(__file__).resolve().parents[1]  # project root
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from step_utils import load_config, get_save_directory
from step_defaults import STEP2A_DEFAULTS

from lauetoolsnn.utils_lauenn import vali_array


def _model_factory(n_bins: int, n_outputs: int, activation: str = "relu", dropout_rate: float = 0.3,
                   learning_rate: float = 0.001, kernel_coeff: float = 0.0005):
    """Factory function to create keras model for grid search."""
    try:
        import tensorflow as tf
        from tensorflow.keras.regularizers import l2
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Activation, Dropout
    except ImportError:
        raise ImportError("TensorFlow/Keras required. Install with: pip install tensorflow")

    # Calculate middle layer size
    if n_outputs >= n_bins:
        param = n_bins
        if param * 15 < (2 * n_outputs):
            param = (n_bins + n_outputs) // 2
    else:
        param = n_outputs * 2

    model = Sequential()
    model.add(tf.keras.Input(shape=(n_bins,)))

    # Hidden layer 1
    model.add(
        Dense(
            n_bins,
            kernel_regularizer=l2(kernel_coeff),
            bias_regularizer=l2(kernel_coeff),
        )
    )
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))

    # Hidden layer 2
    model.add(
        Dense(
            ((param) * 15 + n_bins) // 2,
            kernel_regularizer=l2(kernel_coeff),
            bias_regularizer=l2(kernel_coeff),
        )
    )
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))

    # Hidden layer 3
    model.add(
        Dense(
            (param) * 15,
            kernel_regularizer=l2(kernel_coeff),
            bias_regularizer=l2(kernel_coeff),
        )
    )
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(n_outputs, activation="softmax"))

    # Compile
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model


@contextmanager
def _tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report progress into tqdm."""
    from joblib import parallel

    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def run_step2a(params: Dict[str, Any], data_root: Path | None = None) -> Path:
    """Optimize neural network architecture using grid search."""
    try:
        from sklearn.model_selection import GridSearchCV
        from scikeras.wrappers import KerasClassifier
    except ImportError:
        raise ImportError(
            "scikit-learn, scikeras, and tensorflow required. "
            "Install with: pip install scikit-learn scikeras tensorflow"
        )

    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend

    # Get parameters
    material_ = params["material_"]
    material1_ = params.get("material1_", material_)
    grid_batch_size = int(params.get("grid_batch_size", 20))
    grid_cv_folds = int(params.get("grid_cv_folds", 5))
    n_jobs = int(params.get("n_jobs", -1))

    activation_functions = params.get("activation_functions", ["relu", "tanh"])
    dropout_rates = params.get("dropout_rates", [0.0, 0.3, 0.5])
    learning_rates = params.get("learning_rates", [0.0001, 0.001, 0.01])
    kernel_coeffs = params.get("kernel_coeffs", [1e-5, 1e-4, 5e-4])

    # Get save directory
    save_directory = get_save_directory(params, data_root)
    if not save_directory.exists():
        raise FileNotFoundError(
            f"Training data directory not found: {save_directory}\n"
            f"Please run Step 1 first to generate the dataset."
        )
    print(f"Training data directory: {save_directory}")

    # Load class data
    print("Loading class data...")
    classhkl = np.load(save_directory / "MOD_grain_classhkl_angbin.npz")["arr_0"]
    angbins = np.load(save_directory / "MOD_grain_classhkl_angbin.npz")["arr_1"]
    loc_new = np.load(save_directory / "MOD_grain_classhkl_angbin.npz")["arr_2"]

    n_bins = len(angbins) - 1
    n_outputs = len(classhkl)
    print(f"Input bins: {n_bins}, Output classes: {n_outputs}")

    # Load training data for grid search
    print(f"\nLoading training data for grid search (batch_size={grid_batch_size} files)...")
    x_training, y_training = vali_array(
        str(save_directory / "training_data"),
        grid_batch_size,
        len(classhkl),
        loc_new,
        print,
        tocategorical=True,
    )
    print(f"Loaded {len(x_training)} samples for optimization")
    print(f"Min, Max class ID: {np.min(y_training)}, {np.max(y_training)}")

    # Create Keras classifier wrapper
    print("\nSetting up keras classifier wrapper...")
    
    # Create a wrapper function that scikeras can use
    def make_model(X=None, y=None, activation="relu", dropout_rate=0.3, learning_rate=0.001, kernel_coeff=0.0005):
        return _model_factory(n_bins, n_outputs, activation, dropout_rate, learning_rate, kernel_coeff)
    
    model_ann = KerasClassifier(
        model=make_model,
        epochs=5,
        batch_size=50,
        verbose=0,
    )

    # Define hyperparameter grid
    print("Defining hyperparameter grid for search...")
    hyperparameters = dict(
        model__activation=activation_functions,
        model__dropout_rate=dropout_rates,
        model__learning_rate=learning_rates,
        model__kernel_coeff=kernel_coeffs,
    )

    print("Hyperparameter combinations to test:")
    total_combos = len(activation_functions) * len(dropout_rates) * len(learning_rates) * len(kernel_coeffs)
    print(f"  Activation functions: {activation_functions}")
    print(f"  Dropout rates: {dropout_rates}")
    print(f"  Learning rates: {learning_rates}")
    print(f"  L2 regularization (kernel): {kernel_coeffs}")
    print(f"  Total combinations: {total_combos} × {grid_cv_folds}-fold CV = {total_combos * grid_cv_folds} models")

    # Run grid search
    print(f"\nStarting grid search with {grid_cv_folds}-fold cross-validation...")
    print("This may take several minutes...\n")

    grid = GridSearchCV(
        estimator=model_ann,
        param_grid=hyperparameters,
        cv=grid_cv_folds,
        n_jobs=n_jobs,
        verbose=1,
    )

    total_fits = total_combos * grid_cv_folds
    try:
        from tqdm.auto import tqdm

        print("Showing progress bar for CV fits...")
        with _tqdm_joblib(
            tqdm(total=total_fits, desc="GridSearchCV", unit="fit", dynamic_ncols=True)
        ):
            grid_result = grid.fit(x_training, y_training)
    except ImportError:
        print("tqdm not available; running without progress bar.")
        grid_result = grid.fit(x_training, y_training)

    # Log results
    if material_ != material1_:
        log_file = save_directory / f"grid_optimizer_logger_{material_}_{material1_}.txt"
    else:
        log_file = save_directory / f"grid_optimizer_logger_{material_}.txt"

    print("\n" + "=" * 70)
    print("GRID SEARCH OPTIMIZATION RESULTS")
    print("=" * 70)

    best_score = grid_result.best_score_
    best_params = grid_result.best_params_

    print(f"\nBest mean CV accuracy: {best_score:.6f}")
    print(f"Best parameters: {best_params}")

    # Write results to file
    with open(log_file, "w") as text_file:
        text_file.write("=" * 70 + "\n")
        text_file.write("GRID SEARCH OPTIMIZATION RESULTS\n")
        text_file.write("=" * 70 + "\n\n")

        text_file.write(f"Material: {material_}\n")
        if material_ != material1_:
            text_file.write(f"Material 1: {material1_}\n")
        text_file.write(f"Input bins (features): {n_bins}\n")
        text_file.write(f"Output classes: {n_outputs}\n")
        text_file.write(f"Training samples: {len(x_training)}\n")
        text_file.write(f"Cross-validation folds: {grid_cv_folds}\n\n")

        text_file.write("HYPERPARAMETER GRID:\n")
        text_file.write(f"  Activation functions: {activation_functions}\n")
        text_file.write(f"  Dropout rates: {dropout_rates}\n")
        text_file.write(f"  Total combinations: {total_combos}\n\n")

        text_file.write("=" * 70 + "\n")
        text_file.write(f"Best mean CV accuracy: {best_score:.6f}\n")
        text_file.write(f"Best parameters:\n")
        for key, value in best_params.items():
            text_file.write(f"  {key}: {value}\n")
        text_file.write("=" * 70 + "\n\n")

        text_file.write("ALL RESULTS (sorted by accuracy):\n")
        text_file.write("-" * 70 + "\n")

        means = grid_result.cv_results_["mean_test_score"]
        stds = grid_result.cv_results_["std_test_score"]
        params_list = grid_result.cv_results_["params"]

        # Sort by mean score descending
        results = sorted(
            zip(means, stds, params_list), key=lambda x: x[0], reverse=True
        )

        for idx, (mean, stdev, param) in enumerate(results, 1):
            print(f"{idx}. {mean:.6f} (+/- {stdev:.6f}) with: {param}")
            text_file.write(f"{idx}. {mean:.6f} (+/- {stdev:.6f}) with: {param}\n")

    print("=" * 70)
    print(f"\nResults saved to: {log_file}")

    return save_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 2a for LaueNN: optimize neural network architecture using grid search.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to JSON config overriding defaults.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("project"),
        help="Base directory where Step 1 output dataset folder is located (default: project/).",
    )
    return parser


def main() -> None:
    # Required for multiprocessing in sklearn grid search
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)

    parser = build_parser()
    args = parser.parse_args()

    params = load_config(args.config, STEP2A_DEFAULTS)
    save_directory = run_step2a(params, data_root=args.data_root)
    print(f"\nStep 2a completed. Results written to: {save_directory}")


if __name__ == "__main__":
    main()
