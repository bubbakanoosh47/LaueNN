#!/usr/bin/env python3
"""Step 3b: Visualize and assess simulated Laue patterns from Step 3a.

This script provides visual inspection and statistical analysis of the simulated
Laue patterns generated in Step 3a, including:
- Interactive visualization of Laue spot patterns
- Ground truth orientation matrix inspection
- Pattern statistics (spots per pattern, intensity distributions)
- Side-by-side comparison of multiple patterns
- Export capabilities for presentations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

# Ensure repo root is in path for local lauetoolsnn
repo_root = Path(__file__).resolve().parents[2]
project_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from step_utils import load_config

from lauetoolsnn.lauetools import IOLaueTools as IOLT
from step_defaults import STEP3B_DEFAULTS


def load_cor_file(cor_file: Path) -> tuple:
    """Load a .cor file and return spot data.
    
    Returns:
        Tuple of (data_pixX, data_pixY, data_I, data_theta, data_chi)
    """
    try:
        result = IOLT.readfile_cor(str(cor_file))
        # readfile_cor returns 7 or 8 values depending on output_CCDparamsdict flag
        # We use the first 7 values: (alldata, data_theta, data_chi, data_pixX, data_pixY, data_I, detParam)
        data_theta = result[1]
        data_chi = result[2]
        data_pixX = result[3]
        data_pixY = result[4]
        data_I = result[5]
        return data_pixX, data_pixY, data_I, data_theta, data_chi
    except Exception as e:
        print(f"Warning: Could not read {cor_file}: {e}")
        return None, None, None, None, None


def plot_single_pattern(cor_file: Path, ax, dim1: int, dim2: int, 
                        show_title: bool = True, show_colorbar: bool = True):
    """Plot a single Laue pattern on given axis.
    
    Args:
        cor_file: Path to .cor file
        ax: Matplotlib axis
        dim1: Detector dimension 1 (pixels)
        dim2: Detector dimension 2 (pixels)
        show_title: Whether to show title
        show_colorbar: Whether to show colorbar
    """
    data_pixX, data_pixY, data_I, data_theta, data_chi = load_cor_file(cor_file)
    
    if data_pixX is None:
        ax.text(0.5, 0.5, 'Failed to load', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_xlim(0, dim2)
        ax.set_ylim(0, dim1)
        return None
    
    # Plot spots
    scatter = ax.scatter(data_pixX, data_pixY, c=data_I, s=50, 
                        cmap='hot', alpha=0.8, edgecolors='none',
                        vmin=0, vmax=np.percentile(data_I, 99))
    
    # Formatting
    ax.set_xlim(0, dim2)
    ax.set_ylim(0, dim1)
    ax.set_xlabel('X (pixels)', fontsize=10)
    ax.set_ylabel('Y (pixels)', fontsize=10)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2, linestyle='--')
    
    if show_title:
        pattern_name = cor_file.stem
        ax.set_title(f'{pattern_name}\n{len(data_pixX)} spots', fontsize=9)
    
    if show_colorbar:
        plt.colorbar(scatter, ax=ax, label='Intensity', fraction=0.046, pad=0.04)
    
    return scatter


def plot_pattern_grid(cor_files: List[Path], params: Dict[str, Any]) -> Figure:
    """Plot multiple patterns in a grid layout.
    
    Args:
        cor_files: List of .cor file paths
        params: Configuration parameters
        
    Returns:
        Figure object
    """
    rows = params.get("grid_rows", 2)
    cols = params.get("grid_cols", 2)
    dim1 = params.get("dim1", 2018)
    dim2 = params.get("dim2", 2016)
    
    n_plots = min(len(cor_files), rows * cols)
    
    fig = plt.figure(figsize=(5*cols, 5*rows))
    
    for i, cor_file in enumerate(cor_files[:n_plots]):
        ax = fig.add_subplot(rows, cols, i+1)
        plot_single_pattern(cor_file, ax, dim1, dim2, 
                          show_title=True, show_colorbar=True)
    
    plt.tight_layout()
    return fig


def plot_pattern_statistics(cor_files: List[Path], params: Dict[str, Any]) -> Figure:
    """Create statistical analysis plots of the patterns.
    
    Args:
        cor_files: List of .cor file paths
        params: Configuration parameters
        
    Returns:
        Figure object
    """
    print("\nComputing pattern statistics...")
    
    spots_per_pattern = []
    mean_intensities = []
    total_intensities = []
    pattern_names = []
    
    for cor_file in cor_files:
        data_pixX, data_pixY, data_I, _, _ = load_cor_file(cor_file)
        if data_pixX is not None:
            spots_per_pattern.append(len(data_pixX))
            mean_intensities.append(np.mean(data_I))
            total_intensities.append(np.sum(data_I))
            pattern_names.append(cor_file.stem)
    
    spots_per_pattern = np.array(spots_per_pattern)
    mean_intensities = np.array(mean_intensities)
    total_intensities = np.array(total_intensities)
    
    # Create figure with statistics
    fig = plt.figure(figsize=(15, 5))
    
    # Histogram of spots per pattern
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.hist(spots_per_pattern, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    mean_spots = float(np.mean(spots_per_pattern))
    ax1.axvline(mean_spots, color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {mean_spots:.1f}')
    ax1.set_xlabel('Number of spots per pattern', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Spots per Pattern', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mean intensity distribution
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.hist(mean_intensities, bins=20, color='coral', alpha=0.7, edgecolor='black')
    mean_intensity = float(np.mean(mean_intensities))
    ax2.axvline(mean_intensity, color='darkred', linestyle='--',
                linewidth=2, label=f'Mean: {mean_intensity:.1f}')
    ax2.set_xlabel('Mean intensity per pattern', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Mean Intensities', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Spots vs Total Intensity scatter
    ax3 = fig.add_subplot(1, 3, 3)
    scatter = ax3.scatter(spots_per_pattern, total_intensities, 
                         c=mean_intensities, cmap='viridis', 
                         s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Number of spots', fontsize=12)
    ax3.set_ylabel('Total intensity', fontsize=12)
    ax3.set_title('Spots vs Total Intensity', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Mean intensity', fontsize=10)
    
    plt.tight_layout()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("PATTERN STATISTICS SUMMARY")
    print("="*70)
    print(f"Total patterns analyzed: {len(spots_per_pattern)}")
    print(f"\nSpots per pattern:")
    print(f"  Mean:   {np.mean(spots_per_pattern):.2f}")
    print(f"  Median: {np.median(spots_per_pattern):.2f}")
    print(f"  Std:    {np.std(spots_per_pattern):.2f}")
    print(f"  Min:    {np.min(spots_per_pattern)}")
    print(f"  Max:    {np.max(spots_per_pattern)}")
    print(f"\nMean intensities:")
    print(f"  Mean:   {np.mean(mean_intensities):.2f}")
    print(f"  Median: {np.median(mean_intensities):.2f}")
    print(f"  Std:    {np.std(mean_intensities):.2f}")
    print("="*70)
    
    return fig


def load_ground_truth(gt_file: Path) -> Optional[np.ndarray]:
    """Load ground truth orientation matrices from .npz file.
    
    Args:
        gt_file: Path to groundtruth_OM.npz file
        
    Returns:
        Array of orientation matrices or None if not found
    """
    try:
        data = np.load(gt_file)
        # Get the first (and typically only) array in the npz file
        key = list(data.keys())[0]
        return data[key]
    except Exception as e:
        print(f"Warning: Could not load ground truth from {gt_file}: {e}")
        return None


def display_ground_truth_info(gt_file: Path):
    """Display information about ground truth orientation matrices.
    
    Args:
        gt_file: Path to groundtruth_OM.npz file
    """
    matrices = load_ground_truth(gt_file)
    
    if matrices is None:
        print("Ground truth orientation matrices not available")
        return
    
    print("\n" + "="*70)
    print("GROUND TRUTH ORIENTATION MATRICES")
    print("="*70)
    print(f"Shape: {matrices.shape}")
    print(f"  {matrices.shape[0]} patterns")
    print(f"  Max {matrices.shape[1]} grains per pattern")
    print(f"  {matrices.shape[2]}×{matrices.shape[3]} matrices")
    
    # Count non-zero matrices
    non_zero_count = np.sum(np.any(matrices != 0, axis=(2, 3)), axis=1)
    print(f"\nGrains per pattern distribution:")
    for n_grains in range(matrices.shape[1] + 1):
        count = np.sum(non_zero_count == n_grains)
        if count > 0:
            print(f"  {n_grains} grain(s): {count} patterns")
    
    # Show first few matrices as examples
    print(f"\nExample orientation matrices (first pattern):")
    for i in range(min(2, matrices.shape[1])):
        if np.any(matrices[0, i] != 0):
            print(f"\nGrain {i}:")
            print(matrices[0, i])
    
    print("="*70)


def run_step3b(params: Dict[str, Any]) -> None:
    """Run Step 3b visualization and assessment."""
    
    # Get parameters
    simulated_dir = Path(params.get("simulated_data_dir", "project/Ni/simulated_dataset"))
    pattern_indices = params.get("pattern_indices", None)
    show_statistics = params.get("show_statistics", True)
    show_ground_truth = params.get("show_ground_truth", True)
    output_figure = params.get("output_figure", None)
    dpi = params.get("dpi", 150)
    
    if not simulated_dir.exists():
        print(f"Error: Simulated data directory not found: {simulated_dir}")
        return
    
    print("=" * 70)
    print("STEP 3b: VISUALIZE & ASSESS SIMULATED PATTERNS")
    print("=" * 70)
    print(f"Data directory: {simulated_dir}")
    
    # Find all .cor files
    cor_files = sorted(list(simulated_dir.glob("*.cor")))
    
    if not cor_files:
        print(f"Error: No .cor files found in {simulated_dir}")
        return
    
    print(f"Found {len(cor_files)} pattern files")
    
    # Filter by indices if specified
    if pattern_indices is not None:
        if isinstance(pattern_indices, (list, tuple)):
            cor_files = [cor_files[i] for i in pattern_indices if i < len(cor_files)]
            print(f"Selected {len(cor_files)} patterns by index")
    
    # Display ground truth info if requested
    if show_ground_truth:
        gt_file = simulated_dir / "groundtruth_OM.npz"
        if gt_file.exists():
            display_ground_truth_info(gt_file)
    
    # Plot pattern grid (skip if rows or cols is 0 - stats-only mode)
    skip_patterns = params.get("grid_rows", 2) == 0 or params.get("grid_cols", 2) == 0
    
    fig_patterns = None
    if not skip_patterns:
        print("\nGenerating pattern visualizations...")
        fig_patterns = plot_pattern_grid(cor_files, params)
    
    # Plot statistics if requested
    fig_stats = None
    if show_statistics:
        fig_stats = plot_pattern_statistics(cor_files, params)
    
    # Save or show
    if output_figure:
        output_path = Path(output_figure)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save patterns
        if not skip_patterns and fig_patterns is not None:
            pattern_file = output_path.parent / f"{output_path.stem}_patterns{output_path.suffix}"
            fig_patterns.savefig(pattern_file, dpi=dpi, bbox_inches='tight')
            print(f"\nSaved pattern visualizations to: {pattern_file}")
        
        # Save statistics
        if fig_stats is not None:
            stats_file = output_path.parent / f"{output_path.stem}_statistics{output_path.suffix}"
            fig_stats.savefig(stats_file, dpi=dpi, bbox_inches='tight')
            print(f"Saved statistics to: {stats_file}")
    else:
        print("\nDisplaying plots... (close windows to continue)")
        plt.show()
    
    print("\n" + "=" * 70)
    print("STEP 3b COMPLETED")
    print("=" * 70)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 3b for LaueNN: Visualize and assess simulated Laue patterns from Step 3a.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize first 4 patterns interactively
  %(prog)s --data-dir project/Ni/simulated_dataset --indices 0 1 2 3
  
  # Generate comprehensive assessment report
  %(prog)s --config Step3b_config.json --output assessment.png
  
  # Quick statistics-only view
  %(prog)s --data-dir project/Ni/simulated_dataset --stats-only
        """
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to JSON config overriding defaults.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to simulated dataset directory (overrides config).",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific pattern indices to visualize (e.g., 0 1 2 3).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Number of rows in visualization grid (default: 2).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Number of columns in visualization grid (default: 2).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save figures to this path instead of displaying.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics, skip pattern grid.",
    )
    parser.add_argument(
        "--no-ground-truth",
        action="store_true",
        help="Don't display ground truth information.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figures (default: 150).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    
    # Load config and apply CLI overrides
    params = load_config(args.config, STEP3B_DEFAULTS)
    
    if args.data_dir:
        params["simulated_data_dir"] = str(args.data_dir)
    if args.indices is not None:
        params["pattern_indices"] = args.indices
    if args.rows is not None:
        params["grid_rows"] = args.rows
    if args.cols is not None:
        params["grid_cols"] = args.cols
    if args.output:
        params["output_figure"] = str(args.output)
    if args.stats_only:
        params["grid_rows"] = 0  # Signal to skip pattern grid
        params["grid_cols"] = 0
    if args.no_ground_truth:
        params["show_ground_truth"] = False
    if args.dpi:
        params["dpi"] = args.dpi
    
    run_step3b(params)


if __name__ == "__main__":
    main()
