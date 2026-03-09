#!/usr/bin/env python3
"""Step 2b: Fine hyperparameter search around Step 2a best results.

This script reads the best parameters from Step 2a output and builds a local,
finer grid around those values for robust final selection before Step 2 training
and Step 3 holdout evaluation.
"""

from __future__ import annotations

import argparse
import ast
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

from step_utils import get_save_directory, load_config
from step_defaults import STEP2B_DEFAULTS

from lauetoolsnn.utils_lauenn import vali_array


def _model_factory(
    n_bins: int,
    n_outputs: int,
    activation: str = "relu",
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
    kernel_coeff: float = 0.0005,
):
    """Factory function to create keras model for grid search."""
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Activation, Dense, Dropout
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.regularizers import l2
    except ImportError:
        raise ImportError("TensorFlow/Keras required. Install with: pip install tensorflow")

    if n_outputs >= n_bins:
        param = n_bins
        if param * 15 < (2 * n_outputs):
            param = (n_bins + n_outputs) // 2
    else:
        param = n_outputs * 2

    model = Sequential()
    model.add(tf.keras.Input(shape=(n_bins,)))

    model.add(Dense(n_bins, kernel_regularizer=l2(kernel_coeff), bias_regularizer=l2(kernel_coeff)))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))

    model.add(
        Dense(
            ((param) * 15 + n_bins) // 2,
            kernel_regularizer=l2(kernel_coeff),
            bias_regularizer=l2(kernel_coeff),
        )
    )
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))

    model.add(Dense((param) * 15, kernel_regularizer=l2(kernel_coeff), bias_regularizer=l2(kernel_coeff)))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))

    model.add(Dense(n_outputs, activation="softmax"))

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


def _coerce_value(raw: str) -> Any:
    value = raw.strip()
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _parse_best_params_from_log(log_file: Path) -> Dict[str, Any]:
    if not log_file.exists():
        raise FileNotFoundError(f"Step 2a log not found: {log_file}")

    lines = log_file.read_text(encoding="utf-8").splitlines()

    for line in lines:
        if line.startswith("Best parameters:") and "{" in line and "}" in line:
            dict_text = line.split("Best parameters:", 1)[1].strip()
            parsed = ast.literal_eval(dict_text)
            if isinstance(parsed, dict):
                return parsed

    for idx, line in enumerate(lines):
        if line.strip() == "Best parameters:":
            parsed: Dict[str, Any] = {}
            for next_line in lines[idx + 1 :]:
                stripped = next_line.strip()
                if not stripped or stripped.startswith("=") or stripped.startswith("-"):
                    break
                if ":" not in stripped:
                    break
                key, value = stripped.split(":", 1)
                parsed[key.strip()] = _coerce_value(value)
            if parsed:
                return parsed

    raise ValueError(
        f"Could not parse best parameters from Step 2a log: {log_file}. "
        "Expected either one-line dict or 'Best parameters:' block."
    )


def _auto_step2a_log_path(save_directory: Path, material_: str, material1_: str) -> Path:
    if material_ != material1_:
        return save_directory / f"grid_optimizer_logger_{material_}_{material1_}.txt"
    return save_directory / f"grid_optimizer_logger_{material_}.txt"


def _build_fine_grid(params: Dict[str, Any], best_params: Dict[str, Any]) -> Dict[str, list[Any]]:
    best_activation = str(best_params.get("model__activation", "tanh"))
    best_dropout = float(best_params.get("model__dropout_rate", 0.3))
    best_kernel = float(best_params.get("model__kernel_coeff", 1e-4))
    best_lr = float(best_params.get("model__learning_rate", 1e-3))

    activation_functions = params.get("activation_functions", [])
    if not activation_functions:
        activation_functions = [best_activation]

    learning_rates = params.get("learning_rates", [])
    if not learning_rates:
        learning_rates = [best_lr]

    dropout_offsets = params.get("dropout_offsets", [-0.1, 0.0, 0.1])
    dropout_candidates = []
    for offset in dropout_offsets:
        candidate = best_dropout + float(offset)
        if 0.0 <= candidate <= 0.9:
            dropout_candidates.append(round(candidate, 4))
    if best_dropout not in dropout_candidates:
        dropout_candidates.append(best_dropout)
    dropout_rates = sorted(set(dropout_candidates))

    kernel_multipliers = params.get("kernel_multipliers", [0.5, 1.0, 2.0, 5.0])
    kernel_candidates = []
    for multiplier in kernel_multipliers:
        value = best_kernel * float(multiplier)
        if value > 0:
            kernel_candidates.append(float(f"{value:.8g}"))
    if best_kernel not in kernel_candidates:
        kernel_candidates.append(best_kernel)
    kernel_coeffs = sorted(set(kernel_candidates))

    return {
        "model__activation": list(dict.fromkeys(activation_functions)),
        "model__dropout_rate": dropout_rates,
        "model__learning_rate": list(dict.fromkeys(learning_rates)),
        "model__kernel_coeff": kernel_coeffs,
    }


def run_step2b(params: Dict[str, Any], data_root: Path | None = None) -> Path:
    """Run fine grid search centered around Step 2a best parameters."""
    try:
        from scikeras.wrappers import KerasClassifier
        from sklearn.model_selection import GridSearchCV
    except ImportError:
        raise ImportError(
            "scikit-learn, scikeras, and tensorflow required. "
            "Install with: pip install scikit-learn scikeras tensorflow"
        )

    material_ = params["material_"]
    material1_ = params.get("material1_", material_)
    grid_batch_size = int(params.get("grid_batch_size", 40))
    grid_cv_folds = int(params.get("grid_cv_folds", 5))
    n_jobs = int(params.get("n_jobs", -1))

    save_directory = get_save_directory(params, data_root)
    if not save_directory.exists():
        raise FileNotFoundError(
            f"Training data directory not found: {save_directory}\n"
            f"Please run Step 1 first to generate the dataset."
        )
    print(f"Training data directory: {save_directory}")

    step2a_log_override = str(params.get("step2a_log_file", "")).strip()
    if step2a_log_override:
        step2a_log_file = Path(step2a_log_override)
    else:
        step2a_log_file = _auto_step2a_log_path(save_directory, material_, material1_)

    print(f"Reading Step 2a best parameters from: {step2a_log_file}")
    best_params_from_2a = _parse_best_params_from_log(step2a_log_file)
    print(f"Step 2a best parameters: {best_params_from_2a}")

    hyperparameters = _build_fine_grid(params, best_params_from_2a)

    classhkl = np.load(save_directory / "MOD_grain_classhkl_angbin.npz")["arr_0"]
    angbins = np.load(save_directory / "MOD_grain_classhkl_angbin.npz")["arr_1"]
    loc_new = np.load(save_directory / "MOD_grain_classhkl_angbin.npz")["arr_2"]

    n_bins = len(angbins) - 1
    n_outputs = len(classhkl)
    print(f"Input bins: {n_bins}, Output classes: {n_outputs}")

    print(f"\nLoading training data for fine grid search (batch_size={grid_batch_size} files)...")
    x_training, y_training = vali_array(
        str(save_directory / "training_data"),
        grid_batch_size,
        len(classhkl),
        loc_new,
        print,
        tocategorical=True,
    )
    print(f"Loaded {len(x_training)} samples for optimization")

    def make_model(
        X=None,
        y=None,
        activation="relu",
        dropout_rate=0.3,
        learning_rate=0.001,
        kernel_coeff=0.0005,
    ):
        return _model_factory(n_bins, n_outputs, activation, dropout_rate, learning_rate, kernel_coeff)

    model_ann = KerasClassifier(model=make_model, epochs=5, batch_size=50, verbose=0)

    total_combos = 1
    for values in hyperparameters.values():
        total_combos *= len(values)

    print("\nFine grid built around Step 2a best parameters:")
    print(f"  Activations: {hyperparameters['model__activation']}")
    print(f"  Dropout rates: {hyperparameters['model__dropout_rate']}")
    print(f"  Learning rates: {hyperparameters['model__learning_rate']}")
    print(f"  Kernel coeffs: {hyperparameters['model__kernel_coeff']}")
    print(f"  Total combinations: {total_combos} × {grid_cv_folds}-fold CV = {total_combos * grid_cv_folds} models")

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
            tqdm(total=total_fits, desc="Step2b FineSearch", unit="fit", dynamic_ncols=True)
        ):
            grid_result = grid.fit(x_training, y_training)
    except ImportError:
        print("tqdm not available; running without progress bar.")
        grid_result = grid.fit(x_training, y_training)

    if material_ != material1_:
        log_file = save_directory / f"fine_grid_optimizer_logger_{material_}_{material1_}.txt"
    else:
        log_file = save_directory / f"fine_grid_optimizer_logger_{material_}.txt"

    print("\n" + "=" * 70)
    print("STEP 2B FINE GRID SEARCH RESULTS")
    print("=" * 70)
    print(f"Best mean CV accuracy: {grid_result.best_score_:.6f}")
    print(f"Best parameters: {grid_result.best_params_}")

    means = grid_result.cv_results_["mean_test_score"]
    stds = grid_result.cv_results_["std_test_score"]
    params_list = grid_result.cv_results_["params"]
    results = sorted(zip(means, stds, params_list), key=lambda x: x[0], reverse=True)

    with open(log_file, "w", encoding="utf-8") as text_file:
        text_file.write("=" * 70 + "\n")
        text_file.write("STEP 2B FINE GRID SEARCH RESULTS\n")
        text_file.write("=" * 70 + "\n\n")
        text_file.write(f"Material: {material_}\n")
        if material_ != material1_:
            text_file.write(f"Material 1: {material1_}\n")
        text_file.write(f"Source Step 2a log: {step2a_log_file}\n")
        text_file.write(f"Input bins (features): {n_bins}\n")
        text_file.write(f"Output classes: {n_outputs}\n")
        text_file.write(f"Training samples: {len(x_training)}\n")
        text_file.write(f"Cross-validation folds: {grid_cv_folds}\n\n")

        text_file.write("FINE GRID DEFINITION:\n")
        for key, values in hyperparameters.items():
            text_file.write(f"  {key}: {values}\n")
        text_file.write(f"  Total combinations: {total_combos}\n\n")

        text_file.write(f"Best mean CV accuracy: {grid_result.best_score_:.6f}\n")
        text_file.write("Best parameters:\n")
        for key, value in grid_result.best_params_.items():
            text_file.write(f"  {key}: {value}\n")
        text_file.write("\nALL RESULTS (sorted by accuracy):\n")
        text_file.write("-" * 70 + "\n")

        for idx, (mean, stdev, param) in enumerate(results, 1):
            line = f"{idx}. {mean:.6f} (+/- {stdev:.6f}) with: {param}"
            print(line)
            text_file.write(line + "\n")

    print("=" * 70)
    print(f"Results saved to: {log_file}")
    return save_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 2b for LaueNN: fine grid search around Step 2a best parameters.",
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
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)

    parser = build_parser()
    args = parser.parse_args()

    params = load_config(args.config, STEP2B_DEFAULTS)
    save_directory = run_step2b(params, data_root=args.data_root)
    print(f"\nStep 2b completed. Results written to: {save_directory}")


if __name__ == "__main__":
    main()
