#!/usr/bin/env python3
"""Step 1: Generate LaueNN training/testing datasets.

Refactored local script from the legacy notebook export.
- No hardcoded GitHub/Windows paths
- Works from a local clone/environment
- Supports single-phase and two-phase use cases
- Optional JSON config override
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict
import json
from datetime import datetime, timezone

# Ensure repo root is in path for local lauetoolsnn
repo_root = Path(__file__).resolve().parents[2]  # LaueNN root
project_root = Path(__file__).resolve().parents[1]  # project root
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from step_utils import load_config, get_save_directory
from step_defaults import STEP1_DEFAULTS
from lauetoolsnn.utils_lauenn import (
    generate_classHKL,
    generate_dataset,
    get_material_detail,
    rmv_freq_class,
)



def run_step1(params: Dict[str, Any], output_root: Path | None = None) -> Path:
    material_ = params["material_"]
    material1_ = params.get("material1_", material_)

    n = int(params["hkl_max_identify"])
    n1 = int(params.get("hkl_max_identify1", n))

    maximum_angle_to_search = float(params["maximum_angle_to_search"])
    step_for_binning = float(params["step_for_binning"])
    nb_grains_per_lp0 = int(params["nb_grains_per_lp_mat0"])
    nb_grains_per_lp1 = int(params["nb_grains_per_lp_mat1"])
    grains_nb_simulate = int(params["grains_nb_simulate"])

    detectorparameters = params["detectorparameters"]
    pixelsize = float(params["pixelsize"])
    dim1 = int(params["dim1"])
    dim2 = int(params["dim2"])
    emin = int(params["emin"])
    emax = int(params["emax"])

    symm_ = params["symmetry"]
    symm1_ = params["symmetry1"]
    SG = params["SG"]
    SG1 = params["SG1"]

    save_directory = get_save_directory(params, output_root)
    print(f"Save directory: {save_directory}")

    (
        rules,
        symmetry,
        lattice_material,
        crystal,
        SG,
        rules1,
        symmetry1,
        lattice_material1,
        crystal1,
        SG1,
    ) = get_material_detail(material_, SG, symm_, material1_, SG1, symm1_)

    generate_classHKL(
        n,
        rules,
        lattice_material,
        symmetry,
        material_,
        crystal=crystal,
        SG=SG,
        general_diff_cond=bool(params["general_diff_cond"]),
        save_directory=str(save_directory),
        write_to_console=print,
        ang_maxx=maximum_angle_to_search,
        step=step_for_binning,
    )

    if material_ != material1_:
        generate_classHKL(
            n1,
            rules1,
            lattice_material1,
            symmetry1,
            material1_,
            crystal=crystal1,
            SG=SG1,
            general_diff_cond=bool(params["general_diff_cond"]),
            save_directory=str(save_directory),
            write_to_console=print,
            ang_maxx=maximum_angle_to_search,
            step=step_for_binning,
        )

    generate_dataset(
        material_=material_,
        material1_=material1_,
        ang_maxx=maximum_angle_to_search,
        step=step_for_binning,
        mode=0,
        nb_grains=nb_grains_per_lp0,
        nb_grains1=nb_grains_per_lp1,
        grains_nb_simulate=grains_nb_simulate,
        data_realism=bool(params["data_realism"]),
        detectorparameters=detectorparameters,
        pixelsize=pixelsize,
        type_="training_data",
        var0=1,
        dim1=dim1,
        dim2=dim2,
        removeharmonics=1,
        save_directory=str(save_directory),
        write_to_console=print,
        emin=emin,
        emax=emax,
        modelp=params["modelp"],
        misorientation_angle=float(params["misorientation_angle"]),
        general_diff_rules=bool(params["general_diff_rules"]),
        crystal=crystal,
        crystal1=crystal1,
        include_scm=bool(params["include_scm"]),
    )

    factor = int(params["validation_split_factor"])
    generate_dataset(
        material_=material_,
        material1_=material1_,
        ang_maxx=maximum_angle_to_search,
        step=step_for_binning,
        mode=0,
        nb_grains=nb_grains_per_lp0,
        nb_grains1=nb_grains_per_lp1,
        grains_nb_simulate=max(1, grains_nb_simulate // factor),
        data_realism=bool(params["data_realism"]),
        detectorparameters=detectorparameters,
        pixelsize=pixelsize,
        type_="testing_data",
        var0=1,
        dim1=dim1,
        dim2=dim2,
        removeharmonics=1,
        save_directory=str(save_directory),
        write_to_console=print,
        emin=emin,
        emax=emax,
        modelp=params["modelp"],
        misorientation_angle=float(params["misorientation_angle"]),
        general_diff_rules=bool(params["general_diff_rules"]),
        crystal=crystal,
        crystal1=crystal1,
        include_scm=bool(params["include_scm"]),
    )

    rmv_freq_class(
        freq_rmv=int(params["freq_rmv"]),
        freq_rmv1=int(params["freq_rmv1"]),
        save_directory=str(save_directory),
        material_=material_,
        material1_=material1_,
        write_to_console=print,
    )

    return save_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 1 for LaueNN: generate classes and datasets for training.",
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
        help="Base directory where output dataset folder is created (default: project/).",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip Step0 detector preflight check and proceed anyway.",
    )
    parser.add_argument(
        "--preflight-max-age-days",
        type=int,
        default=None,
        help="Override the maximum allowed age (in days) for the Step0 preflight report.",
    )
    return parser


def _check_step0_preflight(material: str, output_root: Path, max_age_days: int) -> tuple[bool, str]:
    """Verify Step0 preflight report exists, is PASS, and is recent enough.

    Returns (ok: bool, message: str).
    """
    report_path = Path(output_root) / material / "detector_preflight" / "detector_preflight_report.json"
    if not report_path.exists():
        return False, f"Preflight report not found: {report_path}"

    try:
        with open(report_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        return False, f"Failed to read preflight report: {exc}"

    # Check validation/status
    status = None
    for key in ("validation", "status", "result"):
        if key in data:
            status = str(data[key])
            break
    if status is None:
        return False, "Preflight report missing validation status"
    if "PASS" not in status.upper():
        return False, f"Preflight validation not PASS (status={status})"

    # Check timestamp recency if present
    ts_keys = ("timestamp", "generated", "run_timestamp", "generated_at")
    ts_val = None
    for k in ts_keys:
        if k in data:
            ts_val = data[k]
            break
    if ts_val is None:
        # No timestamp to validate recency; accept as PASS
        return True, f"Preflight PASS (no timestamp to check)"

    try:
        # Expect ISO 8601
        ts = datetime.fromisoformat(ts_val)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except Exception:
        return True, f"Preflight PASS (unparsed timestamp: {ts_val})"

    age = datetime.now(timezone.utc) - ts
    if age.days > int(max_age_days):
        return False, f"Preflight report too old ({age.days} days > {max_age_days})"

    return True, f"Preflight PASS (age {age.days} days <= {max_age_days})"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    params = load_config(args.config, STEP1_DEFAULTS)
    # Respect CLI override for preflight max age
    if args.preflight_max_age_days is not None:
        params["preflight_max_age_days"] = args.preflight_max_age_days

    # Enforce Step0 preflight unless user explicitly skips
    require_pre = bool(params.get("require_step0_preflight", True))
    if require_pre and not args.skip_preflight:
        material_name = params.get("material_", "Ni")
        max_age = int(params.get("preflight_max_age_days", 30))
        ok, msg = _check_step0_preflight(material_name, args.output_root, max_age)
        if not ok:
            print("Step1 aborted: Step0 preflight requirement not met.")
            print(msg)
            print("If you are sure, re-run with --skip-preflight to bypass this check.")
            sys.exit(2)
        else:
            print("Step0 preflight check: ", msg)
    save_directory = run_step1(params, output_root=args.output_root)
    print(f"Step 1 completed. Files written under: {save_directory}")


if __name__ == "__main__":
    main()
