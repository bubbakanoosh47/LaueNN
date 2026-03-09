#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 4: Quantitative Validation of Trained Model

Validates the trained neural network using test data from Step 1.
Compares predictions to ground truth labels and generates quantitative metrics.

Features:
- Load trained model from Step 2
- Load test dataset from Step 1 (properly formatted feature arrays)
- Run predictions on test data
- Calculate accuracy, confidence metrics, and per-class performance
- Generate statistical plots and detailed validation report
"""

import sys
from pathlib import Path
import json
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm

# Ensure repo root is in path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from keras.models import model_from_json
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))
from step_utils import load_config, get_save_directory, get_material_prefix
from step_defaults import STEP4_DEFAULTS
from lauetoolsnn.utils_lauenn import vali_array


def load_model_and_classes(model_dir, material_, material1_):
    """Load trained model and HKL class definitions.
    
    Args:
        model_dir: Directory containing model files
        material_: Primary material name
        material1_: Secondary material name (same as primary for single phase)
        
    Returns:
        model: Loaded Keras model
        class_HKL: HKL class array
    """
    model_dir = Path(model_dir)
    
    # Determine model and weight file names
    if material_ == material1_:
        model_name = f"model_{material_}.json"
        weights_name = f"model_{material_}.weights.h5"
    else:
        model_name = f"model_{material_}_{material1_}.json"
        weights_name = f"model_{material_}_{material1_}.weights.h5"
    
    model_path = model_dir / model_name
    weights_path = model_dir / weights_name
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model architecture file not found: {model_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights file not found: {weights_path}")
    
    print(f"Loading model from {model_path}")
    with open(model_path, 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights(str(weights_path))
    print("Model loaded successfully")
    
    # Load class HKL
    classHKL_path = model_dir / f"MOD_grain_classhkl_angbin.npz"
    if classHKL_path.exists():
        classHKL_data = np.load(classHKL_path)
        class_HKL = classHKL_data["arr_0"]
        print(f"Loaded {len(class_HKL)} HKL classes")
    else:
        class_HKL = None
        print("Warning: HKL class file not found")
    
    return model, class_HKL


def load_test_data(model_dir, batch_size=50, num_batches=5):
    """Load test dataset from Step 1.
    
    Args:
        model_dir: Directory containing test data (testing_data subfolder)
        batch_size: Batch size for loading
        num_batches: Number of test batches to load
        
    Returns:
        x_test: Test pattern data
        y_test: Test labels (one-hot encoded)
        y_test_labels: Test labels (class indices)
    """
    model_dir = Path(model_dir)
    test_dir = model_dir / "testing_data"
    
    if not test_dir.exists():
        raise FileNotFoundError(f"Test data directory not found: {test_dir}")
    
    # Load class metadata
    classHKL_path = model_dir / "MOD_grain_classhkl_angbin.npz"
    if not classHKL_path.exists():
        raise FileNotFoundError(f"Class metadata file not found: {classHKL_path}")
    
    classHKL_data = np.load(classHKL_path)
    classhkl = classHKL_data["arr_0"]
    loc_new = classHKL_data["arr_2"]
    n_classes = len(classhkl)
    
    print(f"Loading test data from {test_dir}")
    print(f"Number of HKL classes: {n_classes}")
    
    # Load test data using vali_array
    try:
        x_test, y_test = vali_array(
            str(test_dir),
            batch_size,
            n_classes,
            loc_new,
            print
        )
        print(f"Loaded test data: {x_test.shape}")
        
        # Get class labels from one-hot encoding
        y_test_labels = np.argmax(y_test, axis=1)
        print(f"Test labels shape: {y_test_labels.shape}")
        
        return x_test, y_test, y_test_labels
    except Exception as e:
        raise RuntimeError(f"Error loading test data: {e}")


def run_validation(model, x_test, y_test, y_test_labels, batch_size=50, confidence_thresholds=None):
    """Run predictions on test data and compute validation metrics.
    
    Args:
        model: Trained Keras model
        x_test: Test pattern data
        y_test: Test labels (one-hot encoded)
        y_test_labels: Test labels (class indices)
        batch_size: Batch size for prediction
        confidence_thresholds: List of confidence thresholds to evaluate
        
    Returns:
        Dictionary with validation results
    """
    if confidence_thresholds is None:
        confidence_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    print(f"\nRunning validation on {len(x_test)} test patterns...")
    
    # Run predictions
    predictions = model.predict(x_test, batch_size=batch_size, verbose=1)
    pred_labels = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    # Overall accuracy
    overall_accuracy = np.mean(pred_labels == y_test_labels)
    
    # Per-class metrics
    n_classes = np.max(y_test_labels) + 1
    class_correct = np.zeros(n_classes)
    class_total = np.zeros(n_classes)
    
    for i in range(len(y_test_labels)):
        true_class = y_test_labels[i]
        class_total[true_class] += 1
        if pred_labels[i] == true_class:
            class_correct[true_class] += 1
    
    class_accuracy = np.divide(class_correct, class_total, 
                               where=class_total > 0, 
                               out=np.full(n_classes, np.nan))
    
    # Threshold analysis
    threshold_results = {}
    for threshold in confidence_thresholds:
        mask = confidences >= threshold
        count_above = np.sum(mask)
        if count_above > 0:
            accuracy_above = np.mean(pred_labels[mask] == y_test_labels[mask])
            threshold_results[threshold] = {
                "count": count_above,
                "percentage": 100 * count_above / len(confidences),
                "accuracy": accuracy_above,
                "mean_confidence": np.mean(confidences[mask]),
                "std_confidence": np.std(confidences[mask]),
            }
        else:
            threshold_results[threshold] = {
                "count": 0,
                "percentage": 0.0,
                "accuracy": 0.0,
                "mean_confidence": np.nan,
                "std_confidence": np.nan,
            }
    
    results = {
        "num_patterns": len(x_test),
        "overall_accuracy": overall_accuracy,
        "predictions": pred_labels,
        "confidences": confidences,
        "true_labels": y_test_labels,
        "mean_confidence": np.mean(confidences),
        "std_confidence": np.std(confidences),
        "min_confidence": np.min(confidences),
        "max_confidence": np.max(confidences),
        "class_accuracy": class_accuracy,
        "class_correct": class_correct,
        "class_total": class_total,
        "threshold_results": threshold_results,
    }
    
    return results


def plot_validation_results(results, save_dir, material_prefix):
    """Generate validation visualization plots.
    
    Args:
        results: Dictionary of validation results
        save_dir: Directory to save plots
        material_prefix: Material name for file naming
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Confidence distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(results["confidences"], bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax1.axvline(results["mean_confidence"], color="r", linestyle="--", linewidth=2, 
                label=f"Mean: {results['mean_confidence']:.3f}")
    ax1.set_xlabel("Model Confidence")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Prediction Confidence Distribution")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Confidence vs threshold accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    thresholds = sorted(results["threshold_results"].keys())
    accuracies = [results["threshold_results"][t]["accuracy"] for t in thresholds]
    counts = [results["threshold_results"][t]["percentage"] for t in thresholds]
    
    ax2_twin = ax2.twinx()
    ax2.plot(thresholds, accuracies, "o-", linewidth=2, markersize=8, color="green", label="Accuracy")
    ax2_twin.plot(thresholds, counts, "s-", linewidth=2, markersize=8, color="orange", label="% Patterns")
    
    ax2.set_xlabel("Confidence Threshold")
    ax2.set_ylabel("Accuracy", color="green")
    ax2_twin.set_ylabel("Patterns Passing (%)", color="orange")
    ax2.set_title("Threshold Performance Analysis")
    ax2.grid(alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='green')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    ax2.set_xlim((0.4, 1.0))
    
    # 3. Per-class accuracy
    ax3 = fig.add_subplot(gs[1, 0])
    valid_classes = ~np.isnan(results["class_accuracy"])
    if np.any(valid_classes):
        class_indices = np.where(valid_classes)[0][:20]  # Show first 20 classes
        class_acc = results["class_accuracy"][class_indices]
        
        colors = ["green" if acc > 0.8 else "orange" if acc > 0.5 else "red" for acc in class_acc]
        bars = ax3.bar(range(len(class_acc)), class_acc, color=colors, edgecolor="black", alpha=0.7)
        
        ax3.axhline(y=0.8, color="g", linestyle="--", linewidth=1, alpha=0.5, label="80% threshold")
        ax3.axhline(y=0.5, color="orange", linestyle="--", linewidth=1, alpha=0.5, label="50% threshold")
        
        ax3.set_xlabel("HKL Class Index")
        ax3.set_ylabel("Accuracy")
        ax3.set_title(f"Per-Class Accuracy (first {len(class_acc)} classes)")
        ax3.legend()
        ax3.grid(alpha=0.3, axis="y")
        ax3.set_ylim(0, 1.1)
    
    # 4. Summary statistics text
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    
    summary_text = f"""VALIDATION SUMMARY

Test Patterns: {results['num_patterns']}
Overall Accuracy: {results['overall_accuracy']:.4f} ({100*results['overall_accuracy']:.2f}%)

Confidence Metrics:
• Mean: {results['mean_confidence']:.4f}
• Std Dev: {results['std_confidence']:.4f}
• Min: {results['min_confidence']:.4f}
• Max: {results['max_confidence']:.4f}

Per-Class Statistics:
• Avg Class Acc: {np.nanmean(results['class_accuracy']):.4f}
• Classes > 80%: {np.sum(results['class_accuracy'] > 0.8)}
• Classes > 50%: {np.sum(results['class_accuracy'] > 0.5)}
"""
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontfamily="monospace", fontsize=9, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.tight_layout()
    plot_path = save_dir / f"validation_summary_{material_prefix}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved validation plot to {plot_path}")
    plt.close()


def save_detailed_results(results, save_dir, material_prefix):
    """Save detailed validation results to files.
    
    Args:
        results: Dictionary of validation results
        save_dir: Directory to save results
        material_prefix: Material name for file naming
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle for further analysis
    pickle_path = save_dir / f"validation_results_{material_prefix}.pickle"
    with open(pickle_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved pickle results to {pickle_path}")
    
    # Save text summary
    txt_path = save_dir / f"validation_summary_{material_prefix}.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("STEP 4: VALIDATION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Test Patterns: {results['num_patterns']}\n")
        f.write(f"Overall Accuracy: {results['overall_accuracy']:.6f} ({100*results['overall_accuracy']:.2f}%)\n\n")
        
        f.write("CONFIDENCE METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mean Confidence: {results['mean_confidence']:.6f}\n")
        f.write(f"Std Dev: {results['std_confidence']:.6f}\n")
        f.write(f"Min: {results['min_confidence']:.6f}\n")
        f.write(f"Max: {results['max_confidence']:.6f}\n\n")
        
        f.write("PER-CLASS ACCURACY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mean: {np.nanmean(results['class_accuracy']):.6f}\n")
        f.write(f"Classes with >80% accuracy: {np.sum(results['class_accuracy'] > 0.8)}\n")
        f.write(f"Classes with >50% accuracy: {np.sum(results['class_accuracy'] > 0.5)}\n\n")
        
        f.write("THRESHOLD ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write("Threshold | Count | % Pass | Accuracy | Mean Conf | Std Conf\n")
        f.write("-" * 70 + "\n")
        for threshold in sorted(results["threshold_results"].keys()):
            tr = results["threshold_results"][threshold]
            f.write(f"{threshold:9.2f} | {tr['count']:5d} | {tr['percentage']:6.2f}% | "
                   f"{tr['accuracy']:8.4f} | {tr['mean_confidence']:9.6f} | {tr['std_confidence']:8.6f}\n")
    
    print(f"Saved text summary to {txt_path}")


def main(config_file=None, output_root=None, data_root=None):
    """Main validation function.
    
    Args:
        config_file: Path to Step4_config.json (Path or None)
        output_root: Override output directory (Path or None)
        data_root: Override data root directory (Path or None)
    """
    # Convert to Path if needed
    if config_file is not None and not isinstance(config_file, Path):
        config_file = Path(config_file)
    if output_root is not None and not isinstance(output_root, Path):
        output_root = Path(output_root)
    
    # Load configuration
    config = load_config(config_file, STEP4_DEFAULTS)
    
    # Extract parameters from config
    material_ = config["material_"]
    material1_ = config["material1_"]
    prefix = config.get("prefix", "")
    batch_size = config.get("test_batch_size", 50)
    num_batches = config.get("num_test_batches", 10)
    confidence_thresholds = config.get("confidence_thresholds", [0.5, 0.6, 0.7, 0.8, 0.9])
    save_detailed = config.get("save_detailed_results", True)
    
    # Get directories
    if output_root:
        output_root = Path(output_root)
    save_directory = get_save_directory(config, output_root)
    
    print(f"\n{'='*70}")
    print("STEP 4: QUANTITATIVE VALIDATION OF TRAINED MODEL")
    print(f"{'='*70}")
    print(f"Material: {material_}")
    if material_ != material1_:
        print(f"Secondary Material: {material1_}")
    print(f"Output Directory: {save_directory}\n")
    
    # Load model and class definitions
    try:
        model, class_HKL = load_model_and_classes(save_directory, material_, material1_)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run Step 2 first to train the model.")
        return
    
    # Load test data from Step 1
    try:
        x_test, y_test, y_test_labels = load_test_data(save_directory, batch_size, num_batches)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run Step 1 first to generate test data.")
        return
    except RuntimeError as e:
        print(f"Error: {e}")
        return
    
    # Run validation
    results = run_validation(
        model,
        x_test,
        y_test,
        y_test_labels,
        batch_size=batch_size,
        confidence_thresholds=confidence_thresholds,
    )
    
    # Generate plots
    material_prefix = get_material_prefix(config)
    plot_validation_results(results, save_directory, material_prefix)
    
    # Save detailed results
    if save_detailed:
        save_detailed_results(results, save_directory, material_prefix)
    
    # Print summary
    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE")
    print(f"{'='*70}")
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f} ({100*results['overall_accuracy']:.2f}%)")
    print(f"Mean Confidence: {results['mean_confidence']:.6f} ± {results['std_confidence']:.6f}")
    print(f"Per-Class Accuracy: {np.nanmean(results['class_accuracy']):.4f}")
    print(f"\nResults saved to: {save_directory}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    # Get step directory for resolving relative paths
    step_dir = Path(__file__).resolve().parent
    
    parser = argparse.ArgumentParser(
        description="Step 4: Validate predictions against test data ground truth"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to Step 4 config file (default: Step4_config.example.json in script directory)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("project"),
        help="Override output root directory (default: project/)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("project"),
        help="Override data root directory (default: project/)",
    )
    
    args = parser.parse_args()
    
    # Resolve config path: if not provided, look in step directory
    config_path = args.config
    if config_path is None:
        config_path = step_dir / "Step4_config.example.json"
    elif not config_path.is_absolute() and not config_path.exists():
        # If relative path doesn't exist in cwd, try in step directory
        step_config = step_dir / config_path.name
        if step_config.exists():
            config_path = step_config
    
    main(
        config_file=config_path if config_path.exists() else None,
        output_root=args.output_root,
        data_root=args.data_root,
    )
