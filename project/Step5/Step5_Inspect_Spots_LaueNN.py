#!/usr/bin/env python3
"""Step 5 helper: inspect one experimental Laue image before model prediction.

This script extracts spot-like points from one image and writes:
- CSV of derived spot points (pixel_x, pixel_y, intensity, sigma)
- overlay image (raw image + detected spots)
- summary chart (XY scatter + radial histogram)

Overlay saving is enabled by default for quick visual QA.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from skimage.feature import blob_log
from skimage.filters import gaussian


def load_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix in {".tif", ".tiff", ".edf", ".cbf", ".mccd"}:
        try:
            import fabio  # type: ignore
            data = fabio.open(str(path)).data
            return np.asarray(data, dtype=np.float64)
        except Exception:
            pass

    data = plt.imread(str(path))
    if data.ndim == 3:
        data = np.mean(data, axis=2)
    return np.asarray(data, dtype=np.float64)


def normalize_for_detection(img: np.ndarray, blur_sigma: float) -> np.ndarray:
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


def detect_spots(
    image: np.ndarray,
    blur_sigma: float,
    min_sigma: int,
    max_sigma: int,
    num_sigma: int,
    threshold: float,
    overlap: float,
    exclude_border: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    norm = normalize_for_detection(image, blur_sigma=blur_sigma)

    blobs = blob_log(
        norm,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=overlap,
        exclude_border=exclude_border,
    )

    if blobs.size == 0:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    ys = blobs[:, 0]
    xs = blobs[:, 1]
    sigmas = blobs[:, 2] * math.sqrt(2.0)

    h, w = image.shape
    yi = np.clip(np.round(ys).astype(int), 0, h - 1)
    xi = np.clip(np.round(xs).astype(int), 0, w - 1)
    intensities = image[yi, xi]

    return xs, ys, sigmas, intensities


def save_spot_csv(path: Path, xs: np.ndarray, ys: np.ndarray, sigmas: np.ndarray, intensities: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["spot_id", "pixel_x", "pixel_y", "intensity", "sigma_px"],
        )
        writer.writeheader()
        for idx, (x, y, s, inten) in enumerate(zip(xs, ys, sigmas, intensities), start=1):
            writer.writerow(
                {
                    "spot_id": idx,
                    "pixel_x": float(x),
                    "pixel_y": float(y),
                    "intensity": float(inten),
                    "sigma_px": float(s),
                }
            )


def save_overlay(path: Path, image: np.ndarray, xs: np.ndarray, ys: np.ndarray, sigmas: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))

    vmax = np.percentile(image, 99.8)
    vmin = max(float(np.percentile(image, 5)), 1e-6)
    ax.imshow(image, cmap="gray", norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin + 1e-6)))

    if len(xs) > 0:
        ax.scatter(xs, ys, s=np.clip(sigmas * 20.0, 10.0, 180.0), edgecolors="cyan", facecolors="none", linewidths=1.0)

    ax.set_title(f"Detected spots: {len(xs)}")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_summary_chart(path: Path, image: np.ndarray, xs: np.ndarray, ys: np.ndarray, intensities: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    h, w = image.shape
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    rr = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) if len(xs) > 0 else np.array([])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if len(xs) > 0:
        sc = axes[0].scatter(xs, ys, c=intensities, s=16, cmap="viridis")
        fig.colorbar(sc, ax=axes[0], label="Intensity")
    axes[0].set_xlim(0, w)
    axes[0].set_ylim(h, 0)
    axes[0].set_title("Spot XY scatter")
    axes[0].set_xlabel("Pixel X")
    axes[0].set_ylabel("Pixel Y")

    axes[1].hist(rr, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
    axes[1].set_title("Radial distance histogram")
    axes[1].set_xlabel("Radius (pixels)")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one .tif/.tiff image and visualize derived spot points")
    parser.add_argument("--input-image", type=Path, required=True, help="Path to one input image")
    parser.add_argument("--output-dir", type=Path, default=Path("project/Ni/step5_experimental_metrics/inspection"), help="Directory for outputs")

    parser.add_argument("--blur-sigma", type=float, default=0.8)
    parser.add_argument("--min-sigma", type=int, default=2)
    parser.add_argument("--max-sigma", type=int, default=8)
    parser.add_argument("--num-sigma", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--include-border", action="store_true", help="Include border spots (default is exclude border)")

    parser.add_argument("--save-overlay", dest="save_overlay", action="store_true", default=True, help="Save overlay image (default: enabled)")
    parser.add_argument("--no-save-overlay", dest="save_overlay", action="store_false", help="Disable overlay save")

    args = parser.parse_args()

    input_image = args.input_image
    if not input_image.exists():
        raise FileNotFoundError(f"Input image not found: {input_image}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(input_image)
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")

    xs, ys, sigmas, intensities = detect_spots(
        image=image,
        blur_sigma=float(args.blur_sigma),
        min_sigma=int(args.min_sigma),
        max_sigma=int(args.max_sigma),
        num_sigma=int(args.num_sigma),
        threshold=float(args.threshold),
        overlap=float(args.overlap),
        exclude_border=not bool(args.include_border),
    )

    stem = input_image.stem
    csv_path = output_dir / f"{stem}_spots.csv"
    overlay_path = output_dir / f"{stem}_overlay.png"
    summary_path = output_dir / f"{stem}_summary.png"

    save_spot_csv(csv_path, xs, ys, sigmas, intensities)
    if args.save_overlay:
        save_overlay(overlay_path, image, xs, ys, sigmas)
    save_summary_chart(summary_path, image, xs, ys, intensities)

    print(f"Input image: {input_image}")
    print(f"Detected spots: {len(xs)}")
    print(f"Spot CSV: {csv_path}")
    if args.save_overlay:
        print(f"Overlay: {overlay_path}")
    print(f"Summary chart: {summary_path}")


if __name__ == "__main__":
    main()
