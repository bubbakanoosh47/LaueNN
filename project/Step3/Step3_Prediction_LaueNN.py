#!/usr/bin/env python3
"""Step 3: Predict and evaluate LaueNN model.

Refactored local script for model prediction and evaluation.
- Load trained model from Step 2
- Evaluate performance on test dataset
- Generate detailed metrics and confusion matrix
- Optional: predict on specific Laue pattern files
"""

from __future__ import annotations

import argparse
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

from step_utils import load_config, get_save_directory
from step_defaults import STEP3_DEFAULTS
from lauetoolsnn.utils_lauenn import (
    vali_array,
    get_material_detail,
    new_MP_function,
)
from lauetoolsnn.NNmodels import read_hdf5


def run_step3(params: Dict[str, Any], data_root: Path | None = None) -> Path:
    """Load trained model and evaluate on test data.
    
    Optionally performs full indexation workflow with orientation matrix calculation.
    """
    from keras.models import model_from_json
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import time
    import datetime
    import itertools
    import multiprocessing
    from multiprocessing import cpu_count

    # Get parameters
    material_ = params["material_"]
    material1_ = params.get("material1_", material_)
    symmetry = params.get("symmetry", "cubic")
    symmetry1 = params.get("symmetry1", "cubic")
    SG = params.get("SG", 225)
    SG1 = params.get("SG1", 225)
    test_batch_size = int(params["test_batch_size"])
    num_test_batches = int(params["num_test_batches"])
    
    # Indexation parameters
    softmax_threshold = float(params.get("softmax_threshold", 0.80))
    match_rate_threshold = float(params.get("match_rate_threshold", 0.90))
    strain_calculation = params.get("strain_calculation", True)
    tolerance_angle = float(params.get("tolerance_angle", 0.6))
    tolerance_angle1 = float(params.get("tolerance_angle1", 0.6))
    
    # Detector parameters
    detectorparameters = params.get("detectorparameters", [79.553, 979.32, 932.31, 0.37, 0.447])
    pixelsize = float(params.get("pixelsize", 0.0734))
    dim1 = int(params.get("dim1", 2018))
    dim2 = int(params.get("dim2", 2016))
    emin = float(params.get("emin", 5))
    emax = float(params.get("emax", 22))
    
    # Data parameters
    use_test_data = params.get("use_test_data_for_indexation", False)
    use_simulated = params.get("use_simulated_dataset", True)
    grid_x = int(params.get("grid_size_x", 5))
    grid_y = int(params.get("grid_size_y", 5))
    n_jobs = int(params.get("n_jobs", -1))
    
    # Get model directory
    model_directory = get_save_directory(params, data_root)
    if not model_directory.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_directory}\n"
            f"Please run Step 2 first to train the model."
        )
    print(f"Model directory: {model_directory}")

    # Load class data
    print("\nLoading class data...")
    classhkl = np.load(model_directory / "MOD_grain_classhkl_angbin.npz")["arr_0"]
    angbins = np.load(model_directory / "MOD_grain_classhkl_angbin.npz")["arr_1"]
    loc_new = np.load(model_directory / "MOD_grain_classhkl_angbin.npz")["arr_2"]

    n_bins = len(angbins) - 1
    n_outputs = len(classhkl)
    print(f"Input bins: {n_bins}, Output classes: {n_outputs}")

    # Load HKL class definitions
    print("Loading HKL class definitions...")
    with open(model_directory / f"classhkl_data_nonpickled_{material_}.pickle", "rb") as f:
        hkl_all_class0 = cPickle.load(f)[0]

    # Display HKL classes (first 5)
    print(f"\nHKL Classes ({len(hkl_all_class0)} total, showing first 5):")
    for i, (key, hkl_list) in enumerate(list(hkl_all_class0.items())[:5]):
        try:
            if isinstance(hkl_list[0], (list, np.ndarray)):
                hkl_str = ", ".join([f"({int(h[0])},{int(h[1])},{int(h[2])})" for h in hkl_list[:3]])
            else:
                hkl_str = f"({int(hkl_list[0])},{int(hkl_list[1])},{int(hkl_list[2])})"
        except:
            hkl_str = str(hkl_list[:3] if len(hkl_list) > 3 else hkl_list)
        
        if len(hkl_list) > 3:
            hkl_str += f" ... (+{len(hkl_list)-3} more)"
        print(f"  Class {key}: {hkl_str}")

    # Load model architecture
    print("\nLoading trained model...")
    if material_ != material1_:
        model_name = f"model_{material_}_{material1_}"
    else:
        model_name = f"model_{material_}"

    json_path = model_directory / f"{model_name}.json"
    weights_path = model_directory / f"{model_name}.weights.h5"

    if not json_path.exists():
        raise FileNotFoundError(f"Model architecture file not found: {json_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights file not found: {weights_path}")

    with open(json_path, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)
    model.load_weights(str(weights_path))
    print(f"Model loaded successfully from: {model_name}")

    # Load test data
    print(f"\nLoading test data (batch_size={test_batch_size}, num_batches={num_test_batches})...")
    test_data_path = str(model_directory / "testing_data")
    x_test, y_test = vali_array(
        test_data_path,
        test_batch_size,
        len(classhkl),
        loc_new,
        print
    )

    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    # Make predictions
    print("\nMaking predictions on test data...")
    y_test_labels = np.argmax(y_test, axis=-1)
    y_pred_proba = model.predict(x_test, verbose=1)
    y_pred_labels = np.argmax(y_pred_proba, axis=-1)

    # Get prediction confidence
    y_pred_confidence = np.max(y_pred_proba, axis=-1)
    avg_confidence = np.mean(y_pred_confidence)
    print(f"\nAverage prediction confidence: {avg_confidence:.4f}")

    # Calculate accuracy
    accuracy = np.mean(y_test_labels == y_pred_labels)
    print(f"Overall Accuracy: {accuracy:.4f}")

    # Classification report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_test_labels, y_pred_labels))

    # Confusion matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_test_labels, y_pred_labels)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[str(i) for i in range(n_outputs)],
        yticklabels=[str(i) for i in range(n_outputs)],
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.title(f'Confusion Matrix - {material_}', fontsize=14)
    plt.tight_layout()

    confusion_matrix_path = model_directory / f"confusion_matrix_{material_}.png"
    plt.savefig(str(confusion_matrix_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {confusion_matrix_path}")

    # Per-class accuracy
    print("\nPer-Class Accuracy (first 10 classes):")
    class_correct = np.diag(cm)
    class_total = np.sum(cm, axis=1)
    class_accuracy = class_correct / class_total
    
    for idx in range(min(10, n_outputs)):
        try:
            hkl_list = hkl_all_class0[idx] if idx < len(hkl_all_class0) else None
            if hkl_list is not None:
                if isinstance(hkl_list[0], (list, np.ndarray)):
                    hkl_sample = hkl_list[0]
                else:
                    hkl_sample = hkl_list
                hkl_str = f"({int(hkl_sample[0])},{int(hkl_sample[1])},{int(hkl_sample[2])})"
            else:
                hkl_str = "unknown"
        except:
            hkl_str = f"class_{idx}"
        
        print(f"  Class {idx} HKL {hkl_str}: "
              f"{class_accuracy[idx]:.4f} ({class_correct[idx]}/{class_total[idx]})")

    # Prediction confidence distribution
    print("\nPrediction Confidence Distribution:")
    confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    for i in range(len(confidence_bins) - 1):
        count = np.sum((y_pred_confidence >= confidence_bins[i]) & 
                      (y_pred_confidence < confidence_bins[i+1]))
        percentage = 100 * count / len(y_pred_confidence)
        print(f"  {confidence_bins[i]:.2f} - {confidence_bins[i+1]:.2f}: "
              f"{count:5d} predictions ({percentage:5.2f}%)")

    # Plot confidence vs accuracy
    print("\nGenerating confidence vs accuracy plot...")
    correct_predictions = (y_test_labels == y_pred_labels)

    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_confidence[correct_predictions], bins=50, alpha=0.6, 
             label='Correct', color='green', density=True)
    plt.hist(y_pred_confidence[~correct_predictions], bins=50, alpha=0.6,
             label='Incorrect', color='red', density=True)
    plt.xlabel('Prediction Confidence', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Prediction Confidence Distribution', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    confidence_plot_path = model_directory / f"confidence_distribution_{material_}.png"
    plt.savefig(str(confidence_plot_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confidence distribution plot saved to: {confidence_plot_path}")

    # Save detailed results
    results_path = model_directory / f"prediction_results_{material_}.txt"
    with open(results_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("LAUENN STEP 3: PREDICTION AND EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Material: {material_}\n")
        if material_ != material1_:
            f.write(f"Material 1: {material1_}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Test samples: {len(y_test_labels)}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(f"Average Confidence: {avg_confidence:.4f}\n\n")
        
        f.write("Per-Class Performance:\n")
        f.write("-"*70 + "\n")
        for idx in range(n_outputs):
            try:
                hkl_list = hkl_all_class0[idx] if idx < len(hkl_all_class0) else None
                if hkl_list is not None:
                    if isinstance(hkl_list[0], (list, np.ndarray)):
                        hkl_sample = hkl_list[0]
                    else:
                        hkl_sample = hkl_list
                    hkl_str = f"({int(hkl_sample[0]):2d},{int(hkl_sample[1]):2d},{int(hkl_sample[2]):2d})"
                else:
                    hkl_str = "unknown"
            except:
                hkl_str = f"class_{idx}"
            
            f.write(f"Class {idx:2d} HKL {hkl_str}: "
                   f"Accuracy={class_accuracy[idx]:.4f}, "
                   f"Correct={class_correct[idx]:.0f}/{class_total[idx]:.0f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n")
        report_str = classification_report(y_test_labels, y_pred_labels)
        if isinstance(report_str, dict):
            # If dict is returned, convert it to string representation
            report_str = str(report_str)
        f.write(report_str)

    print(f"\nDetailed results saved to: {results_path}")

    print("\n" + "="*70)
    print("STEP 3 EVALUATION COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"Model evaluation results saved in: {model_directory}")

    # Save predictions as pickle for further analysis
    pickle_path = model_directory / f"predictions_{material_}.pickle"
    with open(pickle_path, "wb") as f:
        cPickle.dump({
            'y_true': y_test_labels,
            'y_pred': y_pred_labels,
            'confidence': y_pred_confidence,
            'y_pred_proba': y_pred_proba,
            'classes': list(range(n_outputs)),
        }, f)
    print(f"Predictions saved to: {pickle_path}")

    return model_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 3 for LaueNN: predict and evaluate trained model.",
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
        help="Base directory where Step 2 model folder is located (default: project/).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    params = load_config(args.config, STEP3_DEFAULTS)
    model_directory = run_step3(params, data_root=args.data_root)
    print(f"\nStep 3 completed. Results written to: {model_directory}")


if __name__ == "__main__":
    main()
