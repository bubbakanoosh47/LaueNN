#!/usr/bin/env python3
"""Step 3a: Generate simulated Laue patterns for model verification.

Refactored local script for generating synthetic Laue diffraction patterns with
known orientations. Useful for evaluating trained models on ground-truth data.
- Generates .cor peak position files
- Saves ground truth orientation matrices
- Creates calibration and config files for LaueToolsNN GUI
- Supports single and two-phase materials
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from tqdm import trange

# Ensure repo root is in path for local lauetoolsnn
repo_root = Path(__file__).resolve().parents[2]  # LaueNN root
project_root = Path(__file__).resolve().parents[1]  # project root
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from step_utils import load_config, get_save_directory
from step_defaults import STEP3A_DEFAULTS

from lauetoolsnn.utils_lauenn import get_material_detail, prepare_LP_NB
from lauetoolsnn.lauetools import dict_LaueTools as dictLT
from lauetoolsnn.lauetools import IOLaueTools as IOLT


def run_step3a(params: Dict[str, Any], output_root: Path | None = None) -> Path:
    """Generate simulated Laue patterns for model verification."""
    
    # Get parameters
    material_ = params["material_"]
    material1_ = params.get("material1_", material_)
    symmetry = params.get("symmetry", "cubic")
    symmetry1 = params.get("symmetry1", "cubic")
    SG = params.get("SG", 225)
    SG1 = params.get("SG1", 225)
    
    detectorparameters = params.get("detectorparameters", [79.553, 979.32, 932.31, 0.37, 0.447])
    pixelsize = float(params.get("pixelsize", 0.0734))
    dim1 = int(params.get("dim1", 2018))
    dim2 = int(params.get("dim2", 2016))
    emin = int(params.get("emin", 5))
    emax = int(params.get("emax", 22))
    
    grains_max = int(params.get("grains_max", 2))
    grid_x = int(params.get("grid_size_x", 5))
    grid_y = int(params.get("grid_size_y", 5))
    grid_total = grid_x * grid_y
    
    random_seed = params.get("random_seed", None)
    if random_seed is not None:
        np.random.seed(int(random_seed))

    # Create directories
    save_directory = get_save_directory(params, output_root)
    simulated_data_dir = save_directory / "simulated_dataset"
    simulated_data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("STEP 3a: GENERATE SIMULATED LAUE PATTERNS")
    print("=" * 70)
    print(f"Material: {material_}")
    if material_ != material1_:
        print(f"Material 1: {material1_}")
    print(f"Grid size: {grid_x} × {grid_y} = {grid_total} patterns")
    print(f"Grains per pattern: 1 to {grains_max} (random)")
    print(f"Energy range: {emin} - {emax} keV")
    print(f"Output directory: {simulated_data_dir}")

    # Get material details
    print("\nLoading material parameters...")
    rules, symmetry, lattice_material, crystal, SG, \
        rules1, symmetry1, lattice_material1, crystal1, SG1 = get_material_detail(
            material_, SG, symmetry,
            material1_, SG1, symmetry1
        )
    print(f"Loaded: {material_} ({symmetry}), SG={SG}")

    # Detector setup
    detector_label = "sCMOS"
    ccd_calib = {
        "CCDLabel": detector_label,
        "dd": detectorparameters[0],
        "xcen": detectorparameters[1],
        "ycen": detectorparameters[2],
        "xbet": detectorparameters[3],
        "xgam": detectorparameters[4],
        "pixelsize": pixelsize
    }

    # File prefix
    if material_ != material1_:
        prefix1 = f"{material_}_{material1_}"
    else:
        prefix1 = material_

    # Initialize output arrays
    UBmatrix_GT = np.zeros((grid_total, grains_max, 3, 3))

    # Create status file
    status_file = simulated_data_dir / f"filecreation_stats_{prefix1}_v2.txt"
    print(f"\nGenerating {grid_total} simulated Laue patterns...")
    print(f"Status file: {status_file}")

    with open(status_file, 'w') as text_file:
        text_file.write("# Simulated Laue Pattern Generation Log\n")
        text_file.write(f"# Material: {material_}\n")
        if material_ != material1_:
            text_file.write(f"# Material 1: {material1_}\n")
        text_file.write(f"# Grid: {grid_x} × {grid_y}\n")
        text_file.write(f"# Total patterns: {grid_total}\n\n")

        # Generate patterns
        for ii in trange(grid_total, desc="Generating"):
            # Random number of grains
            if grains_max != 1:
                nbgrains = np.random.randint(1, high=grains_max)
            else:
                nbgrains = np.random.randint(1, high=grains_max + 1)
            
            nbgrains1 = np.random.randint(0, high=grains_max)
            if material_ == material1_:
                nbgrains1 = 0

            seednumber = np.random.randint(int(1e6))

            # Generate Laue pattern
            try:
                result = prepare_LP_NB(
                        nbgrains, nbgrains1,
                        material_, verbose=0,
                        material1_=material1_,
                        seed=seednumber,
                        sortintensity=True,
                        detectorparameters=detectorparameters,
                        pixelsize=pixelsize,
                        dim1=dim1, dim2=dim2,
                        emin=emin, emax=emax,
                        flag=10, noisy_data=False,
                        remove_peaks=False
                    )
                # Handle variable return length from prepare_LP_NB
                if len(result) == 9:
                    tabledistancerandom, hkl_sol, s_posx, s_posy, s_I, s_tth, s_chi, g, g1 = result
                else:
                    tabledistancerandom, hkl_sol, s_posx, s_posy, s_I, s_tth, s_chi = result
                    g, g1 = None, None
            except Exception as e:
                print(f"Warning: Failed to generate pattern {ii}: {e}")
                continue

            # Write .cor file
            cor_file = simulated_data_dir / f"{prefix1}_{ii}"
            IOLT.writefile_cor(
                str(cor_file),
                s_tth, s_chi, s_posx, s_posy, s_I,
                param=ccd_calib,
                sortedexit=0
            )

            # Log to status file
            text_file.write(f"# File: {cor_file}.cor generated\n")
            text_file.write(f"# Phase {material_}: {nbgrains} grains\n")

            grn_cnt = 0
            if g is not None:
                for rm in g:
                    UBmatrix_GT[ii, grn_cnt, :, :] = np.copy(rm)
                    grn_cnt += 1
                    if np.all(rm == 0):
                        continue
                    temp_ = rm.flatten()
                    string1 = (f"[[{temp_[0]},{temp_[1]},{temp_[2]}],"
                              f"[{temp_[3]},{temp_[4]},{temp_[5]}],"
                              f"[{temp_[6]},{temp_[7]},{temp_[8]}]]\n")
                    text_file.write(string1)

            if material_ != material1_:
                text_file.write(f"# Phase {material1_}: {nbgrains1} grains\n")
                if g1 is not None:
                    for rm in g1:
                        UBmatrix_GT[ii, grn_cnt, :, :] = np.copy(rm)
                        grn_cnt += 1
                        if np.all(rm == 0):
                            continue
                        temp_ = rm.flatten()
                        string1 = (f"[[{temp_[0]},{temp_[1]},{temp_[2]}],"
                                  f"[{temp_[3]},{temp_[4]},{temp_[5]}],"
                                  f"[{temp_[6]},{temp_[7]},{temp_[8]}]]\n")
                        text_file.write(string1)

            text_file.write("# ********** \n\n")

    # Save ground truth orientation matrices
    gt_file = simulated_data_dir / "groundtruth_OM.npz"
    np.savez_compressed(gt_file, UBmatrix_GT)
    print(f"Ground truth OM saved to: {gt_file}")

    # Create calibration file
    calib_file = simulated_data_dir / "calib.det"
    _create_calib_file(calib_file, detectorparameters, pixelsize, dim1, dim2)

    # Create GUI config file
    config_dir = simulated_data_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = _create_gui_config(
        config_dir, params, prefix1, material_, material1_,
        symmetry, symmetry1, SG, SG1,
        detectorparameters, pixelsize, dim1, dim2, grid_x, grid_y
    )

    print("\n" + "=" * 70)
    print("STEP 3a COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Generated {grid_total} simulated Laue patterns")
    print(f"\nOutput files:")
    print(f"  .cor files: {simulated_data_dir}/*.cor")
    print(f"  Ground truth: {gt_file}")
    print(f"  Calibration: {calib_file}")
    print(f"  GUI config: {config_file}")
    print(f"  Status log: {status_file}")

    return simulated_data_dir


def _create_calib_file(calib_file: Path, detectorparameters: list, 
                       pixelsize: float, dim1: int, dim2: int) -> None:
    """Create a detector calibration file for LaueTools."""
    with open(calib_file, 'w') as f:
        f.write(f"{detectorparameters[0]}, {detectorparameters[1]}, "
               f"{detectorparameters[2]}, {detectorparameters[3]}, "
               f"{detectorparameters[4]}, {pixelsize}, {dim1}, {dim2}\n")
        f.write("Sample-Detector distance(mm), xO, yO, angle1, angle2, "
               f"pixelsize, dim1, dim2\n")
        f.write("Calibration auto-generated by LaueNN Step 3a\n")
        f.write("Orientation Matrix:\n")
        f.write("[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]\n")
        f.write(f"# Material : {getattr(_create_calib_file, '_material_', 'Unknown')}\n")
        f.write(f"# dd : {detectorparameters[0]}\n")
        f.write(f"# xcen : {detectorparameters[1]}\n")
        f.write(f"# ycen : {detectorparameters[2]}\n")
        f.write(f"# xbet : {detectorparameters[3]}\n")
        f.write(f"# xgam : {detectorparameters[4]}\n")
        f.write(f"# pixelsize : {pixelsize}\n")
        f.write(f"# xpixelsize : {pixelsize}\n")
        f.write(f"# ypixelsize : {pixelsize}\n")
        f.write("# CCDLabel : cor\n")
        f.write(f"# framedim : ({dim1}, {dim2})\n")
        f.write(f"# detectordiameter : {pixelsize * dim1:.5f}\n")
        f.write("# kf_direction : Z>0\n")


def _create_gui_config(config_dir: Path, params: Dict[str, Any], 
                       prefix1: str, material_: str, material1_: str,
                       symmetry: str, symmetry1: str, SG: int, SG1: int,
                       detectorparameters: list, pixelsize: float,
                       dim1: int, dim2: int, grid_x: int, grid_y: int) -> Path:
    """Create a config file for LaueToolsNN GUI."""
    if material_ != material1_:
        config_file = config_dir / f"config_{material_}_{material1_}.txt"
    else:
        config_file = config_dir / f"config_{material_}.txt"

    emin = int(params.get("emin", 5))
    emax = int(params.get("emax", 22))
    prefix = params.get("prefix", "")
    grains_max = params.get("grains_max", 2)

    with open(config_file, 'w') as f:
        f.write("### config file for LaueNeuralNetwork\n")
        f.write("### Auto-generated by Step 3a\n\n")
        
        f.write("[CPU]\n")
        f.write("n_cpu = 8\n\n")
        
        f.write("[GLOBAL_DIRECTORY]\n")
        f.write(f"prefix = {prefix}\n")
        f.write(f"main_directory = {config_dir.parent.parent}\n\n")
        
        f.write("[MATERIAL]\n")
        f.write(f"material = {material_}\n")
        f.write(f"symmetry = {symmetry}\n")
        f.write(f"space_group = {SG}\n")
        f.write("general_diffraction_rules = false\n\n")
        f.write(f"material1 = {material1_}\n")
        f.write(f"symmetry1 = {symmetry1}\n")
        f.write(f"space_group1 = {SG1}\n")
        f.write("general_diffraction_rules1 = false\n\n")
        
        f.write("[DETECTOR]\n")
        f.write("detectorfile = user_input\n")
        f.write(f"params = {detectorparameters[0]},{detectorparameters[1]},"
               f"{detectorparameters[2]},{detectorparameters[3]},"
               f"{detectorparameters[4]},{pixelsize},{dim1},{dim2},cor\n")
        f.write(f"emax = {emax}\n")
        f.write(f"emin = {emin}\n\n")
        
        f.write("[TRAINING]\n")
        f.write("classes_with_frequency_to_remove = 100\n")
        f.write("desired_classes_output = all\n")
        f.write("max_HKL_index = 5\n")
        f.write(f"max_nb_grains = {grains_max}\n")
        f.write("classes_with_frequency_to_remove1 = 100\n")
        f.write("desired_classes_output1 = all\n")
        f.write("max_HKL_index1 = 5\n")
        f.write(f"max_nb_grains1 = {grains_max}\n")
        f.write("max_simulations = 500\n")
        f.write("include_small_misorientation = false\n")
        f.write("angular_distance = 120\n")
        f.write("step_size = 0.1\n")
        f.write("batch_size = 50\n")
        f.write("epochs = 5\n\n")
        
        f.write("[PREDICTION]\n")
        f.write(f"UB_matrix_to_detect = {grains_max}\n")
        f.write("matrix_tolerance = 0.5\n")
        f.write("matrix_tolerance1 = 0.5\n")
        f.write("material0_limit = 1000\n")
        f.write("material1_limit = 1000\n")
        f.write("model_weight_file = none\n")
        f.write("softmax_threshold_global = 0.90\n")
        f.write("mr_threshold_global = 1.00\n")
        f.write("cap_matchrate = 0.01\n")
        f.write("coeff = 0.3\n")
        f.write("coeff_overlap = 0.05\n")
        f.write("mode_spotCycle = graphmode\n")
        f.write("use_previous = false\n\n")
        
        f.write("[EXPERIMENT]\n")
        f.write(f"experiment_directory = {config_dir.parent}\n")
        f.write(f"experiment_file_prefix = {prefix1}_\n")
        f.write(f"image_grid_x = {grid_x}\n")
        f.write(f"image_grid_y = {grid_y}\n\n")
        
        f.write("[PEAKSEARCH]\n")
        f.write("intensity_threshold = 90\n")
        f.write("boxsize = 15\n")
        f.write("fit_peaks_gaussian = 1\n")
        f.write("FitPixelDev = 15\n")
        f.write("NumberMaxofFits = 3000\n\n")
        
        f.write("[STRAINCALCULATION]\n")
        f.write("strain_compute = true\n")
        f.write("tolerance_strain_refinement = 0.5,0.4,0.3,0.2\n")
        f.write("tolerance_strain_refinement1 = 0.5,0.4,0.3,0.2\n")
        f.write("free_parameters = b,c,alpha,beta,gamma\n\n")
        
        f.write("[CALLER]\n")
        f.write("residues_threshold=0.5\n")
        f.write("nb_spots_global_threshold=8\n")
        f.write("option_global = v2\n")
        f.write("nb_spots_consider = 500\n")
        f.write("use_om_user = false\n")
        f.write("path_user_OM = none\n\n")
        
        f.write("[DEVELOPMENT]\n")
        f.write("write_MTEX_file = true\n")

    return config_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 3a for LaueNN: generate simulated Laue patterns for model verification.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to JSON config overriding defaults.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("project"),
        help="Base directory where simulated dataset will be saved (default: project/).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    params = load_config(args.config, STEP3A_DEFAULTS)
    simulated_dir = run_step3a(params, output_root=args.output_root)
    print(f"\nStep 3a completed. Simulated data written to: {simulated_dir}")


if __name__ == "__main__":
    main()
