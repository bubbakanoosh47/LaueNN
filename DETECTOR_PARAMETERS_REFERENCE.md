# Detector Parameters Reference

This file lists the precise detector parameters expected by LaueNN / lauetools interfaces.

Required (minimum for LaueNN / LaueTools):
- `name` (string) — Unique configuration name
- `distance` (float, mm) — Sample-to-detector distance
- `xcen` (float, pixels) — Beam center X coordinate
- `ycen` (float, pixels) — Beam center Y coordinate
- `angle_beta` (float, radians) — Rotation around vertical axis
- `angle_gamma` (float, radians) — Rotation around beam axis
- `pixelsize` (float, mm/pixel) — Physical pixel pitch
- `dim1` (int, pixels) — Image width (pixels)
- `dim2` (int, pixels) — Image height (pixels)
- `geometry` (string) — One of "Z>0", "X>0", "X<0"

Recommended (optional but strongly suggested):
- `hardware_label` (string) — Hardware/type identifier (e.g., sCMOS)
- `bit_depth` (int) — e.g., 16
- `saturation` (int) — Maximum meaningful ADU
- `calibration_date` (ISO date string) — e.g., "2025-01-15"
- `calibration_method` (string) — e.g., "powder_diffraction"
- `energy_range` (array of two floats, keV) — [min, max]
- `uncertainty` (object) — {distance_mm, center_px: [dx,dy], angles_rad: [dbeta, dgamma]}
- `metadata` (object) — Free-form notes, operator, equipment IDs
- `mask_file` or `mask` — Path or inline mask definition
- `flip_x`, `flip_y`, `transpose` (bool) — Preprocessing flags
- `binning` (string or object) — e.g., "1x1" or {x:2,y:2}
- `trusted_range` (array) — [minADU, maxADU] to clip/sanitize images

Compact JSON example:
```json
{
  "name": "BM32_sCMOS_latest_2025",
  "distance": 79.553,
  "xcen": 979.32,
  "ycen": 932.31,
  "angle_beta": 0.37,
  "angle_gamma": 0.447,
  "pixelsize": 0.0734,
  "dim1": 2018,
  "dim2": 2016,
  "geometry": "Z>0",
  "hardware_label": "sCMOS",
  "bit_depth": 16,
  "saturation": 65535,
  "calibration_date": "2025-01-15",
  "uncertainty": {"distance_mm": 0.1, "center_px": [1.0, 1.0], "angles_rad": [0.01, 0.01]},
  "metadata": {"notes": "Annual calibration"}
}
```

Notes & conventions:
- Units are explicit: distances in mm, pixels in px, angles in radians.
- Center coordinates may be fractional (sub-pixel) to support refined calibrations.
- `geometry` expresses beam/detector layout and must match LaueTools expectations.
