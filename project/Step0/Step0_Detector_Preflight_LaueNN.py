#!/usr/bin/env python3
"""Step 0: Detector preflight checks for LaueNN workflow.

Purpose:
- Validate detector parameters before running Steps 1-4
- Run a local sensitivity sweep around detector parameters
- Produce a machine-readable report (JSON) and human summary (Markdown)

This script is intentionally lightweight and safe to run repeatedly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

repo_root = Path(__file__).resolve().parents[2]  # LaueNN root
project_root = Path(__file__).resolve().parents[1]  # project root
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from step_defaults import STEP0_DEFAULTS
from step_utils import get_save_directory, load_config
from lauetoolsnn.detector_catalog import DetectorCatalog
from lauetoolsnn.detector_config import DetectorConfig


def _float_list(values: Any) -> list[float]:
    if not isinstance(values, list):
        raise ValueError("detectorparameters must be a list with 5 values")
    if len(values) != 5:
        raise ValueError("detectorparameters must have exactly 5 values: [distance, xcen, ycen, beta, gamma]")
    return [float(v) for v in values]


def _resolve_detector(params: Dict[str, Any]) -> DetectorConfig:
    use_catalog = bool(params.get("use_detector_catalog", True))
    detector_name = params.get("detector_name", "development")

    if use_catalog:
        catalog = DetectorCatalog.load_default()
        try:
            detector = catalog.get_detector(detector_name)
        except KeyError as exc:
            raise ValueError(
                f"Detector '{detector_name}' was not found in built-in catalog. "
                "Set use_detector_catalog=false to provide manual detectorparameters."
            ) from exc
    else:
        detectorparameters = _float_list(params["detectorparameters"])
        detector = DetectorConfig(
            name=str(params.get("manual_detector_name", "manual_detector")),
            distance=detectorparameters[0],
            xcen=detectorparameters[1],
            ycen=detectorparameters[2],
            angle_beta=detectorparameters[3],
            angle_gamma=detectorparameters[4],
            pixelsize=float(params["pixelsize"]),
            dim1=int(params["dim1"]),
            dim2=int(params["dim2"]),
            geometry=str(params.get("geometry", "Z>0")),
            hardware_label=str(params.get("hardware_label", "user_defined")),
            metadata={"source": "manual_step0_config"},
        )

    if bool(params.get("apply_manual_overrides", True)):
        if "detectorparameters" in params:
            detectorparameters = _float_list(params["detectorparameters"])
            detector = detector.apply_parameter_variation(
                distance=detectorparameters[0],
                xcen=detectorparameters[1],
                ycen=detectorparameters[2],
                angle_beta=detectorparameters[3],
                angle_gamma=detectorparameters[4],
            )
        detector = detector.apply_parameter_variation(
            pixelsize=float(params.get("pixelsize", detector.pixelsize)),
            dim1=int(params.get("dim1", detector.dim1)),
            dim2=int(params.get("dim2", detector.dim2)),
            geometry=str(params.get("geometry", detector.geometry)),
            hardware_label=str(params.get("hardware_label", detector.hardware_label or "user_defined")),
            name=str(params.get("resolved_detector_name", detector.name)),
        )

    return detector


def _compute_metrics(detector: DetectorConfig) -> Dict[str, float]:
    center_offset_x_mm, center_offset_y_mm = detector.center_offset_mm
    center_offset_norm_mm = math.sqrt(center_offset_x_mm**2 + center_offset_y_mm**2)

    return {
        "distance_mm": detector.distance,
        "xcen_px": detector.xcen,
        "ycen_px": detector.ycen,
        "angle_beta_rad": detector.angle_beta,
        "angle_gamma_rad": detector.angle_gamma,
        "pixelsize_mm": detector.pixelsize,
        "dim1_px": float(detector.dim1),
        "dim2_px": float(detector.dim2),
        "center_offset_x_mm": center_offset_x_mm,
        "center_offset_y_mm": center_offset_y_mm,
        "center_offset_norm_mm": center_offset_norm_mm,
        "detector_diagonal_mm": detector.detector_diameter_mm,
        "scattering_coverage_deg": detector.scattering_angle_coverage_degrees,
    }


def _safe_percent_change(base: float, new: float) -> float:
    denom = abs(base) if abs(base) > 1e-12 else 1.0
    return 100.0 * (new - base) / denom


def _sensitivity_level(max_percent_change: float) -> str:
    magnitude = abs(max_percent_change)
    if magnitude < 1.0:
        return "low"
    if magnitude < 5.0:
        return "moderate"
    return "high"


def _run_sensitivity(detector: DetectorConfig, params: Dict[str, Any]) -> Dict[str, Any]:
    if not bool(params.get("sensitivity_enabled", True)):
        return {
            "enabled": False,
            "message": "Sensitivity analysis disabled by config",
            "base_metrics": _compute_metrics(detector),
            "parameter_sensitivity": {},
        }

    deltas = params.get("sensitivity_deltas", {})
    if not isinstance(deltas, dict) or not deltas:
        raise ValueError("sensitivity_deltas must be a non-empty object")

    base_metrics = _compute_metrics(detector)
    tracked_metrics = params.get(
        "tracked_metrics",
        ["center_offset_norm_mm", "scattering_coverage_deg"],
    )

    parameter_sensitivity: Dict[str, Any] = {}

    for param_name, delta_value in deltas.items():
        delta = float(delta_value)
        if delta <= 0:
            raise ValueError(f"Sensitivity delta for {param_name} must be > 0")

        if param_name not in {"distance", "xcen", "ycen", "angle_beta", "angle_gamma", "pixelsize"}:
            raise ValueError(
                f"Unsupported sensitivity parameter '{param_name}'. "
                "Supported: distance, xcen, ycen, angle_beta, angle_gamma, pixelsize"
            )

        base_value = getattr(detector, param_name)
        sweep_points = [base_value - delta, base_value + delta]
        samples = []
        max_abs_percent_change = 0.0

        for value in sweep_points:
            variant = detector.apply_parameter_variation(**{param_name: value})
            variant_metrics = _compute_metrics(variant)

            metric_changes: Dict[str, Dict[str, float]] = {}
            for metric_name in tracked_metrics:
                base_m = float(base_metrics[metric_name])
                var_m = float(variant_metrics[metric_name])
                delta_abs = var_m - base_m
                delta_pct = _safe_percent_change(base_m, var_m)
                metric_changes[metric_name] = {
                    "base": base_m,
                    "variant": var_m,
                    "delta": delta_abs,
                    "percent_change": delta_pct,
                }
                max_abs_percent_change = max(max_abs_percent_change, abs(delta_pct))

            samples.append(
                {
                    "parameter_value": value,
                    "metric_changes": metric_changes,
                }
            )

        parameter_sensitivity[param_name] = {
            "delta_applied": delta,
            "base_value": base_value,
            "samples": samples,
            "max_abs_percent_change": max_abs_percent_change,
            "sensitivity_level": _sensitivity_level(max_abs_percent_change),
        }

    return {
        "enabled": True,
        "tracked_metrics": tracked_metrics,
        "base_metrics": base_metrics,
        "parameter_sensitivity": parameter_sensitivity,
    }


def _recommended_tweak_ranges(detector: DetectorConfig, params: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    limits = params.get("allowed_tweaks", {})
    if not isinstance(limits, dict):
        raise ValueError("allowed_tweaks must be an object")

    defaults = {
        "distance": 2.0,
        "xcen": 20.0,
        "ycen": 20.0,
        "angle_beta": 0.03,
        "angle_gamma": 0.03,
    }
    merged_limits = {**defaults, **limits}

    ranges = {}
    for parameter_name, tweak_limit in merged_limits.items():
        base_value = float(getattr(detector, parameter_name))
        span = abs(float(tweak_limit))
        ranges[parameter_name] = {
            "base": base_value,
            "min": base_value - span,
            "max": base_value + span,
            "allowed_plus_minus": span,
        }
    return ranges


def _write_markdown_summary(report: Dict[str, Any], output_path: Path) -> None:
    detector_info = report["detector"]
    validation = report["validation"]
    sensitivity = report["sensitivity"]

    lines = []
    lines.append("# Step 0 Detector Preflight Summary")
    lines.append("")
    lines.append(f"- Generated: {report['generated_at_utc']}")
    lines.append(f"- Detector: {detector_info['name']}")
    lines.append(f"- Geometry: {detector_info['geometry']}")
    lines.append(f"- Hardware: {detector_info['hardware_label']}")
    lines.append(f"- Validation status: {'PASS' if validation['is_valid'] else 'FAIL'}")
    lines.append("")

    if validation["errors"]:
        lines.append("## Validation Errors")
        for msg in validation["errors"]:
            lines.append(f"- {msg}")
        lines.append("")

    if validation["warnings"]:
        lines.append("## Validation Warnings")
        for msg in validation["warnings"]:
            lines.append(f"- {msg}")
        lines.append("")

    lines.append("## Recommended Production Tweak Ranges")
    for param_name, values in report["recommended_tweak_ranges"].items():
        lines.append(
            f"- {param_name}: base={values['base']:.6f}, range=[{values['min']:.6f}, {values['max']:.6f}]"
        )
    lines.append("")

    if sensitivity.get("enabled"):
        lines.append("## Sensitivity Overview")
        for param_name, sensitivity_info in sensitivity["parameter_sensitivity"].items():
            lines.append(
                f"- {param_name}: {sensitivity_info['sensitivity_level']} "
                f"(max |% change| = {sensitivity_info['max_abs_percent_change']:.3f})"
            )
        lines.append("")
    else:
        lines.append("## Sensitivity Overview")
        lines.append(f"- {sensitivity.get('message', 'Disabled')}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_step0(params: Dict[str, Any], output_root: Path | None = None) -> Dict[str, Any]:
    detector = _resolve_detector(params)
    validation_result = detector.validate()
    sensitivity = _run_sensitivity(detector, params)
    recommended_ranges = _recommended_tweak_ranges(detector, params)

    save_directory = get_save_directory(params, output_root)
    preflight_dir = save_directory / "detector_preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "step": "step0",
        "detector": {
            "name": detector.name,
            "geometry": detector.geometry,
            "hardware_label": detector.hardware_label,
            "detectorparameters": detector.detectorparameters,
            "pixelsize": detector.pixelsize,
            "dim1": detector.dim1,
            "dim2": detector.dim2,
        },
        "validation": {
            "is_valid": validation_result.is_valid,
            "errors": validation_result.errors,
            "warnings": validation_result.warnings,
        },
        "sensitivity": sensitivity,
        "recommended_tweak_ranges": recommended_ranges,
        "preflight_dir": str(preflight_dir),
    }

    report_json = preflight_dir / "detector_preflight_report.json"
    report_md = preflight_dir / "detector_preflight_summary.md"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown_summary(report, report_md)

    strict = bool(params.get("strict_validation", True))
    fail_on_warnings = bool(params.get("fail_on_warnings", False))

    if strict and not validation_result.is_valid:
        raise RuntimeError(
            f"Detector preflight failed validation. See report: {report_json}"
        )

    if fail_on_warnings and validation_result.warnings:
        raise RuntimeError(
            f"Detector preflight has warnings and fail_on_warnings=true. See report: {report_json}"
        )

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 0 for LaueNN: detector parameter validation and sensitivity preflight.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to JSON config overriding Step 0 defaults.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("project"),
        help="Base directory where output material folder is created (default: project/).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    params = load_config(args.config, STEP0_DEFAULTS)
    report = run_step0(params, output_root=args.output_root)

    print("Step 0 completed.")
    print(f"Validation: {'PASS' if report['validation']['is_valid'] else 'FAIL'}")
    print(f"Report directory: {report['preflight_dir']}")


if __name__ == "__main__":
    main()
