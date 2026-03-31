#!/usr/bin/env python3
"""Step 2: Train LaueNN neural network.

Refactored local script for training the neural network on datasets from Step 1.
- Config-driven with sensible defaults
- Works with single-phase and two-phase materials
- Supports different neural network architectures
"""

from __future__ import annotations

import argparse
import ast
import itertools
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import _pickle as cPickle

# Ensure repo root is in path for local lauetoolsnn
repo_root = Path(__file__).resolve().parents[2]  # LaueNN root
project_root = Path(__file__).resolve().parents[1]  # project root
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from step_utils import load_config, get_save_directory, get_input_directory

from lauetoolsnn.utils_lauenn import (
    array_generator,
    array_generator_verify,
    vali_array,
)
from lauetoolsnn.NNmodels import model_arch_general_optimized
from step_defaults import STEP2_DEFAULTS


def _coerce_value(raw: str) -> Any:
    value = raw.strip()
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _parse_best_params_from_log(log_file: Path) -> Dict[str, Any] | None:
    if not log_file.exists():
        return None

    lines = log_file.read_text(encoding="utf-8").splitlines()

    for line in lines:
        if line.startswith("Best parameters:") and "{" in line and "}" in line:
            try:
                parsed = ast.literal_eval(line.split("Best parameters:", 1)[1].strip())
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

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

    return None


def _extract_valid_grid_hyperparameters(best_params: Dict[str, Any]) -> Dict[str, Any] | None:
    activation_raw = best_params.get("model__activation")
    dropout_raw = best_params.get("model__dropout_rate")
    learning_rate_raw = best_params.get("model__learning_rate")
    kernel_coeff_raw = best_params.get("model__kernel_coeff")

    if (
        activation_raw is None
        or dropout_raw is None
        or learning_rate_raw is None
        or kernel_coeff_raw is None
    ):
        return None

    try:
        activation = str(activation_raw)
        dropout_rate = float(dropout_raw)
        learning_rate = float(learning_rate_raw)
        kernel_coeff = float(kernel_coeff_raw)
    except Exception:
        return None

    if not activation:
        return None
    if not (0.0 <= dropout_rate <= 0.9):
        return None
    if not (0.0 < learning_rate <= 1.0):
        return None
    if not (0.0 < kernel_coeff <= 1.0):
        return None

    return {
        "activation": activation,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "kernel_coeff": kernel_coeff,
    }


def _load_optimized_hyperparameters(
    save_directory: Path, material_: str, material1_: str
) -> tuple[Dict[str, Any] | None, Path | None]:
    if material_ != material1_:
        fine_log = save_directory / f"fine_grid_optimizer_logger_{material_}_{material1_}.txt"
        coarse_log = save_directory / f"grid_optimizer_logger_{material_}_{material1_}.txt"
    else:
        fine_log = save_directory / f"fine_grid_optimizer_logger_{material_}.txt"
        coarse_log = save_directory / f"grid_optimizer_logger_{material_}.txt"

    for candidate in (fine_log, coarse_log):
        parsed = _parse_best_params_from_log(candidate)
        if not parsed:
            continue
        validated = _extract_valid_grid_hyperparameters(parsed)
        if validated is not None:
            return validated, candidate

    return None, None



def run_step2(
    params: Dict[str, Any],
    data_root: Path | None = None,
    output_root: Path | None = None,
) -> Path:
    """Train the neural network model."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for plotting
    import matplotlib.pyplot as plt
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from sklearn.metrics import classification_report

    # Get parameters
    material_ = params["material_"]
    material1_ = params.get("material1_", material_)
    nb_grains_per_lp = int(params["nb_grains_per_lp"])
    grains_nb_simulate = int(params["grains_nb_simulate"])
    batch_size = int(params["batch_size"])
    epochs = int(params["epochs"])
    kernel_coeff = float(params["kernel_coeff"])
    bias_coeff = float(params["bias_coeff"])
    learning_rate = float(params["learning_rate"])
    patience = int(params["patience"])
    activation = params.get("activation", "relu")
    dropout_rate = float(params.get("dropout_rate", 0.3))

    # Input directory from Step 1 (must already exist — do NOT create it)
    input_directory = get_input_directory(params, data_root)
    classhkl_sentinel = input_directory / "MOD_grain_classhkl_angbin.npz"
    if not classhkl_sentinel.exists():
        raise FileNotFoundError(
            f"Step 1 output not found at: {input_directory}\n"
            f"Please run Step 1 successfully before running Step 2.\n"
            f"Expected file: MOD_grain_classhkl_angbin.npz"
        )
    print(f"Training data directory: {input_directory}")

    # Output directory for Step 2 artifacts
    output_directory = get_save_directory(params, output_root if output_root is not None else data_root)
    print(f"Step 2 output directory: {output_directory}")

    optimized_params, optimized_source = _load_optimized_hyperparameters(
        output_directory, material_, material1_
    )
    if optimized_params is not None:
        activation = optimized_params["activation"]
        dropout_rate = optimized_params["dropout_rate"]
        learning_rate = optimized_params["learning_rate"]
        kernel_coeff = optimized_params["kernel_coeff"]
        print(f"Using optimized hyperparameters from: {optimized_source}")
    else:
        print("No valid Step 2b/2a optimized log found; using hyperparameters from Step 2 config.")

    # Load class data
    print("Loading class data...")
    classhkl = np.load(input_directory / "MOD_grain_classhkl_angbin.npz")["arr_0"]
    angbins = np.load(input_directory / "MOD_grain_classhkl_angbin.npz")["arr_1"]
    loc_new = np.load(input_directory / "MOD_grain_classhkl_angbin.npz")["arr_2"]

    with open(input_directory / "class_weights.pickle", "rb") as input_file:
        class_weights = cPickle.load(input_file)
    class_weights = class_weights[0]

    n_bins = len(angbins) - 1
    n_outputs = len(classhkl)
    print(f"Input bins: {n_bins}, Output classes: {n_outputs}")

    # Build model
    print("\nBuilding neural network model...")
    print(
        "Hyperparameters: "
        f"activation={activation}, dropout_rate={dropout_rate}, "
        f"learning_rate={learning_rate}, kernel_coeff={kernel_coeff}, bias_coeff={bias_coeff}"
    )
    model = model_arch_general_optimized(
        n_bins,
        n_outputs,
        kernel_coeff=kernel_coeff,
        bias_coeff=bias_coeff,
        lr=learning_rate,
        activation=activation,
        dropout_rate=dropout_rate,
        verbose=1,
        write_to_console=print,
    )

    # Verify batch content
    print(f"\nVerifying batch content (batch_size={batch_size})...")
    trainy_inbatch = array_generator_verify(
        str(input_directory / "training_data"),
        batch_size,
        len(classhkl),
        loc_new,
        print,
    )
    print(f"Number of spots in a batch of {batch_size} files: {len(trainy_inbatch)}")
    print(f"Min, Max class ID: {np.min(trainy_inbatch)}, {np.max(trainy_inbatch)}")

    # Calculate steps per epoch
    nb_grains_per_lp1 = nb_grains_per_lp
    if material_ != material1_:
        nb_grains_list = list(range(nb_grains_per_lp + 1))
        nb_grains1_list = list(range(nb_grains_per_lp1 + 1))
        list_permute = list(itertools.product(nb_grains_list, nb_grains1_list))
        list_permute.pop(0)
        steps_per_epoch = (len(list_permute) * grains_nb_simulate) // batch_size
    else:
        steps_per_epoch = int((nb_grains_per_lp * grains_nb_simulate) / batch_size)

    val_steps_per_epoch = int(steps_per_epoch / 5)
    if steps_per_epoch == 0:
        steps_per_epoch = 1
    if val_steps_per_epoch == 0:
        val_steps_per_epoch = 1

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps per epoch: {val_steps_per_epoch}")

    # Create data generators
    print("\nCreating data generators...")
    training_data_generator = array_generator(
        str(input_directory / "training_data"),
        batch_size,
        len(classhkl),
        loc_new,
        print,
    )
    testing_data_generator = array_generator(
        str(input_directory / "testing_data"),
        batch_size,
        len(classhkl),
        loc_new,
        print,
    )

    # Setup callbacks
    es = EarlyStopping(monitor="val_accuracy", mode="max", patience=patience)
    ms = ModelCheckpoint(
        str(output_directory / "best_val_acc_model.h5"),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
    )

    # Model save path
    if material_ != material1_:
        model_name = output_directory / f"model_{material_}_{material1_}"
    else:
        model_name = output_directory / f"model_{material_}"

    # Train the model
    print(f"\nTraining model for {epochs} epochs...")
    # Note: class_weight is not supported with Python generators in modern Keras
    # The model uses regularization and dropout to handle class imbalance
    stats_model = model.fit(
        training_data_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=testing_data_generator,
        validation_steps=val_steps_per_epoch,
        verbose=1,  # type: ignore
        callbacks=[es, ms],
    )

    # Save model config and weights
    print("\nSaving model...")
    model_json = model.to_json()
    with open(str(model_name) + ".json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights(str(model_name) + ".weights.h5")
    print(f"Model saved to: {model_name}")

    # Print final stats
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Training Accuracy:   {stats_model.history['accuracy'][-1]:.4f}")
    print(f"Training Loss:       {stats_model.history['loss'][-1]:.4f}")
    print(f"Validation Accuracy: {stats_model.history['val_accuracy'][-1]:.4f}")
    print(f"Validation Loss:     {stats_model.history['val_loss'][-1]:.4f}")
    print("=" * 60)

    # Plot accuracy/loss vs epochs
    print("\nGenerating training plots...")
    epochs_range = range(1, len(model.history.history["loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(epochs_range, model.history.history["loss"], "r", label="Training loss")
    ax[0].plot(
        epochs_range,
        model.history.history["val_loss"],
        "r",
        ls="dashed",
        label="Validation loss",
    )
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(
        epochs_range, model.history.history["accuracy"], "g", label="Training Accuracy"
    )
    ax[1].plot(
        epochs_range,
        model.history.history["val_accuracy"],
        "g",
        ls="dashed",
        label="Validation Accuracy",
    )
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    if material_ != material1_:
        plot_path = output_directory / f"loss_accuracy_{material_}_{material1_}.png"
    else:
        plot_path = output_directory / f"loss_accuracy_{material_}.png"

    plt.tight_layout()
    plt.savefig(str(plot_path), bbox_inches="tight", format="png", dpi=300)
    plt.close()
    print(f"Plot saved to: {plot_path}")

    # Save training log
    if material_ != material1_:
        log_path = output_directory / f"loss_accuracy_logger_{material_}_{material1_}.txt"
    else:
        log_path = output_directory / f"loss_accuracy_logger_{material_}.txt"

    with open(log_path, "w") as text_file:
        text_file.write("# EPOCH, LOSS, VAL_LOSS, ACCURACY, VAL_ACCURACY\n")
        for inj in range(len(epochs_range)):
            line = (
                f"{epochs_range[inj]},"
                f"{model.history.history['loss'][inj]},"
                f"{model.history.history['val_loss'][inj]},"
                f"{model.history.history['accuracy'][inj]},"
                f"{model.history.history['val_accuracy'][inj]}\n"
            )
            text_file.write(line)
    print(f"Training log saved to: {log_path}")

    # Generate classification report on test data
    print("\nGenerating classification report on test data...")
    x_test, y_test = vali_array(
        str(input_directory / "testing_data"), 50, len(classhkl), loc_new, print
    )
    y_test_labels = np.argmax(y_test, axis=-1)
    y_pred_labels = np.argmax(model.predict(x_test), axis=-1)

    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred_labels))

    # Copy metadata needed by downstream steps when input/output roots differ
    metadata_candidates = [
        "MOD_grain_classhkl_angbin.npz",
        "grain_classhkl_angbin.npz",
        "class_weights.pickle",
    ]
    for name in metadata_candidates:
        src = input_directory / name
        dst = output_directory / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    for pickle_file in input_directory.glob("classhkl_data*.pickle"):
        dst = output_directory / pickle_file.name
        if not dst.exists():
            shutil.copy2(pickle_file, dst)

    return output_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 2 for LaueNN: train neural network on generated dataset.",
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
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Base directory where Step 2 artifacts are written (default: same as --data-root).",
    )
    return parser


def main() -> None:
    # Required for multiprocessing in Keras
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)

    parser = build_parser()
    args = parser.parse_args()

    params = load_config(args.config, STEP2_DEFAULTS)
    save_directory = run_step2(params, data_root=args.data_root, output_root=args.output_root)
    print(f"\nStep 2 completed. Model files written to: {save_directory}")


if __name__ == "__main__":
    main()
