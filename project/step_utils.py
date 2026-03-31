#!/usr/bin/env python3
"""Common utilities for LaueNN Step scripts.

Shared functions used across Step 1, 2, 2a, 3, and 3a scripts to avoid
code duplication and improve maintainability.

Functions:
    - load_config(): Load JSON config and merge with defaults
    - get_save_directory(): Create material-based output directory
    - get_material_prefix(): Get material prefix string for file naming

Future Extensions:
    - create_parser_factory(): Dynamic argparse builder
    - build_main(): Template main() function for all steps
    - Common plotting utilities
    - Unified result logging
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: Path | None, default_params: Dict[str, Any]) -> Dict[str, Any]:
    """Load and merge configuration from JSON file with defaults.
    
    Args:
        config_path: Path to JSON config file (None = use defaults)
        default_params: Dictionary of default parameters
        
    Returns:
        Merged configuration dictionary with user overrides applied
    """
    params = dict(default_params)
    if config_path is None:
        return params

    with config_path.open("r", encoding="utf-8") as file:
        user_params = json.load(file)

    params.update(user_params)
    return params


def get_save_directory(params: Dict[str, Any], output_root: Path | None) -> Path:
    """Get the directory where step results will be saved.
    
    Creates folder based on material names and optional prefix.
    Supports single-phase and two-phase (multi-material) systems.
    
    Full path includes output_root (typically "project"):
        - Single-phase: "project/{material_}{prefix}/"
        - Two-phase: "project/{material_}_{material1_}{prefix}/"
    
    Args:
        params: Configuration dictionary with material_ and material1_ keys
        output_root: Base output directory (None = current working directory)
        
    Returns:
        Path to the save directory (created if it doesn't exist)
    """
    material_ = params["material_"]
    material1_ = params.get("material1_", material_)
    prefix = params.get("prefix", "")

    if material_ != material1_:
        folder_name = f"{material_}_{material1_}{prefix}"
    else:
        folder_name = f"{material_}{prefix}"

    base = output_root if output_root is not None else Path.cwd()
    save_directory = base / folder_name

    save_directory.mkdir(parents=True, exist_ok=True)
    return save_directory


def get_input_directory(params: Dict[str, Any], input_root: Path | None) -> Path:
    """Resolve the material-based input directory path WITHOUT creating it.

    Use this for directories that must already exist (outputs of a prior step).
    Unlike get_save_directory, this never creates the directory so the caller's
    ``exists()`` check is meaningful.

    Args:
        params: Configuration dictionary with material_ and material1_ keys
        input_root: Base directory to look under

    Returns:
        Path to the expected directory (not guaranteed to exist)
    """
    material_ = params["material_"]
    material1_ = params.get("material1_", material_)
    prefix = params.get("prefix", "")

    if material_ != material1_:
        folder_name = f"{material_}_{material1_}{prefix}"
    else:
        folder_name = f"{material_}{prefix}"

    base = input_root if input_root is not None else Path.cwd()
    return base / folder_name


def get_material_prefix(params: Dict[str, Any]) -> str:
    """Get material prefix string for file naming.
    
    Args:
        params: Configuration dictionary with material_ and material1_ keys
        
    Returns:
        Prefix string like "Ni" for single-phase or "Ni_Fe" for two-phase
    """
    material_ = params["material_"]
    material1_ = params.get("material1_", material_)
    
    if material_ != material1_:
        return f"{material_}_{material1_}"
    else:
        return material_
