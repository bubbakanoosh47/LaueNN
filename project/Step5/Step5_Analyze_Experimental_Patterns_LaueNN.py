#!/usr/bin/env python3
"""Step 5: Analyze real experimental Laue images.

This step is designed for real detector images (e.g., TIFF) and computes:
- image quality metrics (dynamic range, saturation fraction, background, SNR proxy)
- diffraction spot metrics from blob detection (count, size, intensity, spatial spread)

Outputs are written under the material output directory in a dedicated Step 5 subfolder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.feature import blob_log
from skimage.filters import gaussian

# Ensure repo root is in path for local lauetoolsnn usage
repo_root = Path(__file__).resolve().parents[2]
project_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from step_utils import load_config, get_save_directory
from step_defaults import STEP5_DEFAULTS


@dataclass
class ImageMetrics:
    filename: str
    path: str
    height: int
    width: int
    min_intensity: float
    max_intensity: float
    mean_intensity: float
    median_intensity: float
    std_intensity: float
    p01_intensity: float
    p99_intensity: float
    dynamic_range_p99_p01: float
    saturation_fraction: float
    background_median: float
    background_mad: float
    snr_proxy: float
    spot_count: int
    spot_density_per_mpix: float
    spot_sigma_mean_px: float
    spot_sigma_std_px: float
    spot_peak_intensity_mean: float
    spot_peak_intensity_std: float
    radial_mean_px: float
    radial_std_px: float
    radial_mean_mm: float
    radial_std_mm: float
    quadrant_imbalance: float


def _resolve_image_files(input_path: Path, image_glob: str, recursive: bool, limit_images: int | None) -> List[Path]:
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.rglob(image_glob) if recursive else input_path.glob(image_glob))

    if limit_images is not None and limit_images > 0:
        files = files[:limit_images]
    return files


def _load_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix in {".tif", ".tiff", ".edf", ".cbf", ".mccd"}:
        try:
            import fabio  # type: ignore
            data = fabio.open(str(path)).data
            return np.asarray(data, dtype=np.float64)
        except Exception:
            pass

    try:
        from matplotlib import pyplot as _plt
        data = _plt.imread(str(path))
        if data.ndim == 3:
            data = np.mean(data, axis=2)
        return np.asarray(data, dtype=np.float64)
    except Exception as exc:
        raise RuntimeError(f"Unable to read image '{path}': {exc}") from exc


def _normalize_for_detection(img: np.ndarray, blur_sigma: float) -> np.ndarray:
    work = np.copy(img)
    work[~np.isfinite(work)] = 0.0
    work = np.clip(work, 0.0, None)

    if blur_sigma > 0:
        work = gaussian(work, sigma=blur_sigma, preserve_range=True)

    p1 = np.percentile(work, 1)
    p99 = np.percentile(work, 99)
    if p99 <= p1:
        return np.zeros_like(work)
    norm = (work - p1) / (p99 - p1)
    return np.clip(norm, 0.0, 1.0)


def _quadrant_imbalance(xs: np.ndarray, ys: np.ndarray, cx: float, cy: float) -> float:
    if len(xs) == 0:
        return 0.0
    q1 = np.sum((xs >= cx) & (ys < cy))
    q2 = np.sum((xs < cx) & (ys < cy))
    q3 = np.sum((xs < cx) & (ys >= cy))
    q4 = np.sum((xs >= cx) & (ys >= cy))
    counts = np.array([q1, q2, q3, q4], dtype=np.float64)
    return float(np.std(counts) / (np.mean(counts) + 1e-12))


def _compute_metrics_for_image(
    img_path: Path,
    pixelsize: float,
    detector_center: Tuple[float, float] | None,
    blob_cfg: Dict[str, Any],
    saturation_value: float | None,
) -> ImageMetrics:
    image = _load_image(img_path)
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image for '{img_path}', got shape {image.shape}")

    h, w = image.shape
    raw = np.copy(image)
    raw[~np.isfinite(raw)] = 0.0

    p01 = float(np.percentile(raw, 1))
    p99 = float(np.percentile(raw, 99))
    median_intensity = float(np.median(raw))
    mad = float(np.median(np.abs(raw - median_intensity)))

    sat_value = float(saturation_value) if saturation_value is not None else float(np.percentile(raw, 99.9))
    sat_fraction = float(np.mean(raw >= sat_value))

    norm = _normalize_for_detection(raw, blur_sigma=float(blob_cfg.get("blur_sigma", 0.8)))

    blobs = blob_log(
        norm,
        min_sigma=int(blob_cfg.get("min_sigma", 2)),
        max_sigma=int(blob_cfg.get("max_sigma", 8)),
        num_sigma=int(blob_cfg.get("num_sigma", 10)),
        threshold=float(blob_cfg.get("threshold", 0.12)),
        overlap=float(blob_cfg.get("overlap", 0.5)),
        exclude_border=bool(blob_cfg.get("exclude_border", True)),
    )

    if blobs.size == 0:
        ys = np.array([], dtype=np.float64)
        xs = np.array([], dtype=np.float64)
        sigmas = np.array([], dtype=np.float64)
        peak_intensities = np.array([], dtype=np.float64)
    else:
        ys = blobs[:, 0]
        xs = blobs[:, 1]
        sigmas = blobs[:, 2] * math.sqrt(2.0)

        yi = np.clip(np.round(ys).astype(int), 0, h - 1)
        xi = np.clip(np.round(xs).astype(int), 0, w - 1)
        peak_intensities = raw[yi, xi]

    if detector_center is None:
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    else:
        cx, cy = detector_center

    if len(xs) > 0:
        rr = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        radial_mean_px = float(np.mean(rr))
        radial_std_px = float(np.std(rr))
    else:
        radial_mean_px = 0.0
        radial_std_px = 0.0

    snr_proxy = float((np.mean(peak_intensities) - median_intensity) / (mad + 1e-12)) if len(peak_intensities) > 0 else 0.0

    spot_density_mpix = float((len(xs) / (h * w)) * 1e6)

    return ImageMetrics(
        filename=img_path.name,
        path=str(img_path),
        height=h,
        width=w,
        min_intensity=float(np.min(raw)),
        max_intensity=float(np.max(raw)),
        mean_intensity=float(np.mean(raw)),
        median_intensity=median_intensity,
        std_intensity=float(np.std(raw)),
        p01_intensity=p01,
        p99_intensity=p99,
        dynamic_range_p99_p01=float(p99 - p01),
        saturation_fraction=sat_fraction,
        background_median=median_intensity,
        background_mad=mad,
        snr_proxy=snr_proxy,
        spot_count=int(len(xs)),
        spot_density_per_mpix=spot_density_mpix,
        spot_sigma_mean_px=float(np.mean(sigmas)) if len(sigmas) > 0 else 0.0,
        spot_sigma_std_px=float(np.std(sigmas)) if len(sigmas) > 0 else 0.0,
        spot_peak_intensity_mean=float(np.mean(peak_intensities)) if len(peak_intensities) > 0 else 0.0,
        spot_peak_intensity_std=float(np.std(peak_intensities)) if len(peak_intensities) > 0 else 0.0,
        radial_mean_px=radial_mean_px,
        radial_std_px=radial_std_px,
        radial_mean_mm=float(radial_mean_px * pixelsize),
        radial_std_mm=float(radial_std_px * pixelsize),
        quadrant_imbalance=_quadrant_imbalance(xs, ys, cx, cy),
    )


def _save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_plots(output_dir: Path, metrics: List[ImageMetrics]) -> None:
    if not metrics:
        return

    spot_counts = np.array([m.spot_count for m in metrics], dtype=float)
    snr = np.array([m.snr_proxy for m in metrics], dtype=float)
    sat = np.array([m.saturation_fraction for m in metrics], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(spot_counts, bins=20, color="steelblue", edgecolor="black", alpha=0.8)
    axes[0].set_title("Spot Count Distribution")
    axes[0].set_xlabel("Spot count")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(snr, bins=20, color="seagreen", edgecolor="black", alpha=0.8)
    axes[1].set_title("SNR Proxy Distribution")
    axes[1].set_xlabel("SNR proxy")

    axes[2].hist(sat, bins=20, color="darkorange", edgecolor="black", alpha=0.8)
    axes[2].set_title("Saturation Fraction")
    axes[2].set_xlabel("Fraction")

    plt.tight_layout()
    plt.savefig(output_dir / "step5_metrics_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_step5(config: Dict[str, Any], output_root: Path | None) -> Path:
    save_root = get_save_directory(config, output_root)
    output_subdir = str(config.get("output_subdir", "step5_experimental_metrics"))
    output_dir = save_root / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(str(config["input_path"]))
    if not input_path.is_absolute():
        input_path = repo_root / input_path

    image_glob = str(config.get("image_glob", "*.tif"))
    recursive = bool(config.get("recursive", False))
    limit_images = config.get("limit_images", None)
    limit_images = int(limit_images) if limit_images not in (None, "") else None

    detector_center = config.get("detector_center", None)
    if detector_center is not None:
        detector_center = (float(detector_center[0]), float(detector_center[1]))

    pixelsize = float(config.get("pixelsize", 0.0734))
    saturation_value = config.get("saturation_value", None)
    saturation_value = float(saturation_value) if saturation_value not in (None, "") else None

    blob_cfg = dict(config.get("blob_detection", {}))

    image_files = _resolve_image_files(input_path, image_glob, recursive, limit_images)
    if not image_files:
        raise FileNotFoundError(f"No images found at '{input_path}' with pattern '{image_glob}'")

    print(f"Found {len(image_files)} image(s) for Step 5 analysis")

    metrics: List[ImageMetrics] = []
    for idx, img_path in enumerate(image_files, start=1):
        print(f"[{idx}/{len(image_files)}] Analyzing {img_path.name}")
        metrics.append(
            _compute_metrics_for_image(
                img_path=img_path,
                pixelsize=pixelsize,
                detector_center=detector_center,
                blob_cfg=blob_cfg,
                saturation_value=saturation_value,
            )
        )

    metric_rows: List[Dict[str, Any]] = [m.__dict__ for m in metrics]
    _save_csv(output_dir / "step5_image_metrics.csv", metric_rows)

    if bool(config.get("save_plots", True)):
        _save_plots(output_dir, metrics)

    summary = {
        "num_images": len(metrics),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "mean_spot_count": float(np.mean([m.spot_count for m in metrics])),
        "mean_snr_proxy": float(np.mean([m.snr_proxy for m in metrics])),
        "mean_saturation_fraction": float(np.mean([m.saturation_fraction for m in metrics])),
    }

    with (output_dir / "step5_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nStep 5 analysis complete")
    print(f"Output directory: {output_dir}")
    print(f"Images analyzed: {summary['num_images']}")
    print(f"Mean spot count: {summary['mean_spot_count']:.2f}")
    print(f"Mean SNR proxy: {summary['mean_snr_proxy']:.2f}")

    return output_dir


def main(config_file: Path | None, output_root: Path | None) -> None:
    config = load_config(config_file, STEP5_DEFAULTS)
    run_step5(config=config, output_root=output_root)


if __name__ == "__main__":
    step_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Step 5: Analyze real experimental Laue images and compute spot/image quality metrics"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to Step 5 config JSON (default: Step5_config.example.json in this folder)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("project"),
        help="Override output root directory (default: project/)",
    )

    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        config_path = step_dir / "Step5_config.example.json"
    elif not config_path.is_absolute() and not config_path.exists():
        step_config = step_dir / config_path.name
        if step_config.exists():
            config_path = step_config

    main(
        config_file=config_path if config_path.exists() else None,
        output_root=args.output_root,
    )
