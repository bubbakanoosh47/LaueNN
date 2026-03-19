#!/usr/bin/env python3
"""Step 6: Predict HKL classes on real experimental Laue images.

Workflow:
1. Load trained model + class metadata from material directory
2. Detect spots from each experimental image
3. Convert spot XY to (2theta, chi) using detector geometry
4. Build angular-histogram descriptors per spot (same family as training pipeline)
5. Predict HKL class + confidence for each spot
6. Save per-image spot prediction CSV, optional overlays, and summary outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from skimage.feature import blob_log
from skimage.filters import gaussian

repo_root = Path(__file__).resolve().parents[2]
project_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from step_utils import load_config, get_save_directory
from step_defaults import STEP6_DEFAULTS

from keras.models import model_from_json

try:
    import lauetoolsnn.lauetools.LaueGeometry as Lgeo
    import lauetoolsnn.lauetools.generaltools as GT
    import lauetoolsnn.lauetools.IOLaueTools as IOLT
except Exception:
    import lauetoolsnn.lauetools.LaueGeometry as Lgeo
    import lauetoolsnn.lauetools.generaltools as GT
    import lauetoolsnn.lauetools.IOLaueTools as IOLT


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

    data = plt.imread(str(path))
    if data.ndim == 3:
        data = np.mean(data, axis=2)
    return np.asarray(data, dtype=np.float64)


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


def _detect_spots(image: np.ndarray, blob_cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    norm = _normalize_for_detection(image, blur_sigma=float(blob_cfg.get("blur_sigma", 0.8)))
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

    order = np.argsort(intensities)[::-1]
    return xs[order], ys[order], sigmas[order], intensities[order]


def _build_codebars(theta_deg: np.ndarray, chi_deg: np.ndarray, angbins: np.ndarray, nb_spots_consider: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(theta_deg) == 0:
        return np.zeros((0, len(angbins) - 1), dtype=np.float64), np.array([], dtype=int)

    sorted_data = np.transpose(np.array([theta_deg, chi_deg]))
    tabledistancerandom = np.transpose(GT.calculdist_from_thetachi(sorted_data, sorted_data))

    spots_in_center = np.arange(0, len(theta_deg))[: min(nb_spots_consider, len(theta_deg))]
    codebars_all = []

    for i in spots_in_center:
        spotangles = tabledistancerandom[i]
        spotangles = np.delete(spotangles, i)
        codebars = np.histogram(spotangles, bins=angbins)[0].astype(np.float64)
        max_codebars = np.max(codebars) if len(codebars) > 0 else 0.0
        if max_codebars > 0:
            codebars = codebars / max_codebars
        codebars_all.append(codebars)

    return np.array(codebars_all, dtype=np.float64), spots_in_center


def _extract_hkl(classhkl: np.ndarray, class_index: int) -> Tuple[int, int, int]:
    row = np.array(classhkl[class_index]).ravel()
    if row.size >= 3:
        return int(row[0]), int(row[1]), int(row[2])
    if row.size == 2:
        return int(row[0]), int(row[1]), 0
    if row.size == 1:
        return int(row[0]), 0, 0
    return 0, 0, 0


def _write_spot_prediction_csv(
    path: Path,
    xs: np.ndarray,
    ys: np.ndarray,
    intensities: np.ndarray,
    twotheta: np.ndarray,
    chi: np.ndarray,
    pred_classes: np.ndarray,
    confidences: np.ndarray,
    classhkl: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "spot_id",
                "pixel_x",
                "pixel_y",
                "intensity",
                "twotheta_deg",
                "chi_deg",
                "predicted_class",
                "confidence",
                "h",
                "k",
                "l",
            ],
        )
        writer.writeheader()

        for idx in range(len(pred_classes)):
            h, k, l = _extract_hkl(classhkl, int(pred_classes[idx]))
            writer.writerow(
                {
                    "spot_id": idx + 1,
                    "pixel_x": float(xs[idx]),
                    "pixel_y": float(ys[idx]),
                    "intensity": float(intensities[idx]),
                    "twotheta_deg": float(twotheta[idx]),
                    "chi_deg": float(chi[idx]),
                    "predicted_class": int(pred_classes[idx]),
                    "confidence": float(confidences[idx]),
                    "h": h,
                    "k": k,
                    "l": l,
                }
            )


def _save_overlay(path: Path, image: np.ndarray, xs: np.ndarray, ys: np.ndarray, confidences: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    vmax = np.percentile(image, 99.8)
    vmin = max(float(np.percentile(image, 5)), 1e-6)
    ax.imshow(image, cmap="gray", norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin + 1e-6)))

    if len(xs) > 0:
        sc = ax.scatter(xs, ys, c=confidences, cmap="viridis", s=18, edgecolors="none")
        fig.colorbar(sc, ax=ax, label="Prediction confidence")

    ax.set_title(f"Predicted spots: {len(xs)}")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_cor_file(
    path_stem: Path,
    twotheta: np.ndarray,
    chi: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    intensities: np.ndarray,
    detectorparameters: List[float],
    pixelsize: float,
) -> None:
    calib = {
        "CCDLabel": "sCMOS",
        "dd": detectorparameters[0],
        "xcen": detectorparameters[1],
        "ycen": detectorparameters[2],
        "xbet": detectorparameters[3],
        "xgam": detectorparameters[4],
        "pixelsize": pixelsize,
    }
    IOLT.writefile_cor(
        str(path_stem),
        twotheta,
        chi,
        xs,
        ys,
        intensities,
        param=calib,
        sortedexit=0,
    )


def _load_model_assets(model_dir: Path, material_: str, material1_: str) -> Tuple[Any, np.ndarray, np.ndarray]:
    if material_ == material1_:
        model_name = f"model_{material_}.json"
        weights_name = f"model_{material_}.weights.h5"
    else:
        model_name = f"model_{material_}_{material1_}.json"
        weights_name = f"model_{material_}_{material1_}.weights.h5"

    model_path = model_dir / model_name
    weights_path = model_dir / weights_name
    class_path = model_dir / "MOD_grain_classhkl_angbin.npz"

    if not model_path.exists():
        raise FileNotFoundError(f"Model architecture file not found: {model_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights file not found: {weights_path}")
    if not class_path.exists():
        raise FileNotFoundError(f"Class metadata file not found: {class_path}")

    with model_path.open("r", encoding="utf-8") as file:
        model_json = file.read()
    model = model_from_json(model_json)
    model.load_weights(str(weights_path))

    class_data = np.load(class_path, allow_pickle=True)
    classhkl = class_data["arr_0"]
    angbins = class_data["arr_1"]

    return model, classhkl, angbins


def run_step6(config: Dict[str, Any], output_root: Path | None, data_root: Path | None) -> Path:
    material_ = str(config["material_"])
    material1_ = str(config.get("material1_", material_))

    model_dir = get_save_directory(config, data_root)
    output_base = get_save_directory(config, output_root)
    output_dir = output_base / str(config.get("output_subdir", "step6_prediction_results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    model, classhkl, angbins = _load_model_assets(model_dir, material_, material1_)

    input_path = Path(str(config["input_path"]))
    if not input_path.is_absolute():
        input_path = repo_root / input_path

    image_files = _resolve_image_files(
        input_path=input_path,
        image_glob=str(config.get("image_glob", "*.tif")),
        recursive=bool(config.get("recursive", False)),
        limit_images=int(config["limit_images"]) if config.get("limit_images") not in (None, "") else None,
    )
    if not image_files:
        raise FileNotFoundError(f"No images found at '{input_path}'")

    detectorparameters = [float(v) for v in config.get("detectorparameters", [79.553, 979.32, 932.31, 0.37, 0.447])]
    pixelsize = float(config.get("pixelsize", 0.0734))
    geometry = str(config.get("geometry", "Z>0"))
    nb_spots_consider = int(config.get("nb_spots_consider", 100))
    blob_cfg = dict(config.get("blob_detection", {}))

    save_overlay = bool(config.get("save_overlay", True))
    save_cor_file = bool(config.get("save_cor_file", True))

    summary_rows: List[Dict[str, Any]] = []

    print(f"Loaded model from: {model_dir}")
    print(f"Processing {len(image_files)} experimental image(s)")

    for idx, image_path in enumerate(image_files, start=1):
        print(f"[{idx}/{len(image_files)}] Predicting: {image_path.name}")

        image = _load_image(image_path)
        if image.ndim != 2:
            print(f"  Skipping non-2D image: {image_path}")
            continue

        xs_all, ys_all, _, intensities_all = _detect_spots(image, blob_cfg=blob_cfg)
        if len(xs_all) == 0:
            print("  No spots detected")
            summary_rows.append(
                {
                    "filename": image_path.name,
                    "num_detected_spots": 0,
                    "num_predicted_spots": 0,
                    "mean_confidence": None,
                    "median_confidence": None,
                }
            )
            continue

        twotheta_all, chi_all = Lgeo.calc_uflab(
            xs_all,
            ys_all,
            detectorparameters,
            returnAngles=1,
            pixelsize=pixelsize,
            kf_direction=geometry,
        )
        theta_all = twotheta_all / 2.0

        codebars, indices_used = _build_codebars(theta_all, chi_all, angbins=angbins, nb_spots_consider=nb_spots_consider)
        if len(indices_used) == 0:
            print("  Not enough spots to build descriptors")
            continue

        predictions = model.predict(codebars, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)

        xs = xs_all[indices_used]
        ys = ys_all[indices_used]
        intensities = intensities_all[indices_used]
        twotheta = twotheta_all[indices_used]
        chi = chi_all[indices_used]

        stem = image_path.stem
        csv_path = output_dir / f"{stem}_spot_predictions.csv"
        _write_spot_prediction_csv(
            path=csv_path,
            xs=xs,
            ys=ys,
            intensities=intensities,
            twotheta=twotheta,
            chi=chi,
            pred_classes=pred_classes,
            confidences=confidences,
            classhkl=classhkl,
        )

        if save_overlay:
            overlay_path = output_dir / f"{stem}_prediction_overlay.png"
            _save_overlay(overlay_path, image, xs, ys, confidences)

        if save_cor_file:
            _write_cor_file(
                path_stem=output_dir / f"{stem}_predicted",
                twotheta=twotheta,
                chi=chi,
                xs=xs,
                ys=ys,
                intensities=intensities,
                detectorparameters=detectorparameters,
                pixelsize=pixelsize,
            )

        summary_rows.append(
            {
                "filename": image_path.name,
                "num_detected_spots": int(len(xs_all)),
                "num_predicted_spots": int(len(indices_used)),
                "mean_confidence": float(np.mean(confidences)),
                "median_confidence": float(np.median(confidences)),
                "max_confidence": float(np.max(confidences)),
                "min_confidence": float(np.min(confidences)),
            }
        )

    summary_csv = output_dir / "step6_prediction_summary.csv"
    if summary_rows:
        with summary_csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    summary_json = {
        "num_images": len(image_files),
        "num_completed": len(summary_rows),
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "mean_image_confidence": float(np.mean([r["mean_confidence"] for r in summary_rows if r["mean_confidence"] is not None])) if any(r["mean_confidence"] is not None for r in summary_rows) else None,
    }

    with (output_dir / "step6_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary_json, file, indent=2)

    print("\nStep 6 prediction complete")
    print(f"Output directory: {output_dir}")
    print(f"Prediction summary: {summary_csv}")

    return output_dir


def main(config_file: Path | None, output_root: Path | None, data_root: Path | None) -> None:
    config = load_config(config_file, STEP6_DEFAULTS)
    run_step6(config=config, output_root=output_root, data_root=data_root)


if __name__ == "__main__":
    step_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Step 6: Predict HKL classes on real experimental Laue images")
    parser.add_argument("--config", type=Path, default=None, help="Path to Step 6 config JSON")
    parser.add_argument("--output-root", type=Path, default=Path("project"), help="Output root directory")
    parser.add_argument("--data-root", type=Path, default=Path("project"), help="Model/data root directory")
    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        config_path = step_dir / "Step6_config.example.json"
    elif not config_path.is_absolute() and not config_path.exists():
        step_config = step_dir / config_path.name
        if step_config.exists():
            config_path = step_config

    main(
        config_file=config_path if config_path.exists() else None,
        output_root=args.output_root,
        data_root=args.data_root,
    )
