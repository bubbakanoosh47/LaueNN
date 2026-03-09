# Detector Parameters Quick Start Guide

**For:** Using the new DetectorConfig and DetectorCatalog system in LaueNN step scripts and workflows.

---

## Basic Usage

### 1. Load Default Development Detector

```python
from lauetoolsnn.detector_catalog import DetectorCatalog

# Create catalog with built-in defaults
catalog = DetectorCatalog.load_default()

# Get the development detector
detector = catalog.get_detector("development")

# Access detector parameters (LaueTools format)
detectorparameters = detector.detectorparameters  # [distance, xcen, ycen, xbet, xgam]
pixelsize = detector.pixelsize
dim1 = detector.dim1
dim2 = detector.dim2
geometry = detector.geometry

print(detector.describe())  # Human-readable description
```

### 2. Create Manual Detector Configuration

```python
from lauetoolsnn.detector_config import DetectorConfig

detector = DetectorConfig(
    name="custom_detector",
    distance=79.553,           # mm
    xcen=979.32,               # pixels
    ycen=932.31,               # pixels
    angle_beta=0.37,           # radians
    angle_gamma=0.447,         # radians
    pixelsize=0.0734,          # mm/pixel
    dim1=2018,                 # pixels (width)
    dim2=2016,                 # pixels (height)
    geometry="Z>0",            # Beamline geometry
    hardware_label="sCMOS",    # Detector type
    metadata={
        "calibration_date": "2025-01-15",
        "notes": "Annual calibration"
    }
)

# Validate before use
validation = detector.validate()
if not validation.is_valid:
    print(f"Invalid detector: {validation}")
    sys.exit(1)

print(detector)  # Print summary
```

### 3. Create Detector Variations (Parameter Studies)

```python
catalog = DetectorCatalog.load_default()

# Create variation without registering
test_shorter = catalog.create_detector_variant(
    "development",
    variant_name="test_shorter",
    distance=70.0,
    xcen=980.0
)

# Register variant in catalog for reuse
catalog.register_variant(
    "development",
    variant_name="production_test",
    distance=75.0,
    angle_beta=0.40
)

# Use in analysis
for detector in [test_shorter, catalog.get_detector("production_test")]:
    print(f"{detector.name}: distance={detector.distance} mm")
```

### 4. Validate Detector Configuration

```python
detector = catalog.get_detector("BM32_setup_2025_01")

# Quick check
if detector.is_physically_plausible():
    print("✓ Detector parameters are reasonable")
else:
    print("✗ Problem with detector parameters")

# Detailed validation
result = detector.validate()
print(result)  # Shows errors and warnings
print(result.explain())  # Same thing
```

### 5. Serialize/Load Detector Configurations

```python
# Save to JSON file
detector.to_json("my_detector_config.json")

# Load from JSON file
from lauetoolsnn.detector_config import DetectorConfig
restored = DetectorConfig.from_json("my_detector_config.json")

# Convert to/from dict
config_dict = detector.to_dict()
detector2 = DetectorConfig.from_dict(config_dict)
```

### 6. Work with Detector Catalog Files

```python
# Load catalog from JSON file
catalog = DetectorCatalog.load_from_file("production_detectors.json", name="production")

# List available detectors
print(catalog.list_detectors())

# Query by hardware type
sCMOS_detectors = catalog.get_detectors_by_hardware("sCMOS")

# Get beamline defaults
bm32_detectors = catalog.get_beamline_detectors("BM32")
primary = catalog.get_primary_detector("BM32")

# Save updated catalog
catalog.save_to_file("updated_detectors.json", include_metadata=True)
```

### 7. Get Detector Information

```python
detector = catalog.get_detector("development")

# Individual parameters
print(f"Distance: {detector.distance} mm")
print(f"Center: ({detector.xcen}, {detector.ycen}) pixels")
print(f"Angles: β={detector.angle_beta:.4f} rad, γ={detector.angle_gamma:.4f} rad")

# Derived metrics
print(f"Detector size: {detector.detector_diameter_mm:.1f} mm (diagonal)")
print(f"Center offset: {detector.center_offset_mm} mm from geometric center")
print(f"Scattering coverage: {detector.scattering_angle_coverage_degrees:.1f}°")

# Hardware
print(f"Hardware: {detector.hardware_label}")
print(f"Dimensions: {detector.dim1} × {detector.dim2} pixels")
print(f"Pixel size: {detector.pixelsize} mm/pixel")

# Geometry
print(f"Geometry mode: {detector.geometry}")  # "Z>0", "X>0", or "X<0"
```

---

## Integration with Step Scripts

### Use in Step 1 (Dataset Generation)

```python
from lauetoolsnn.detector_catalog import DetectorCatalog
from lauetoolsnn.utils_lauenn import generate_dataset

# Load detector configuration
catalog = DetectorCatalog.load_default()
detector = catalog.get_detector("development")

# Pass to LaueNN functions
dataset = generate_dataset(
    material="Ni",
    detectorparameters=detector.detectorparameters,
    pixelsize=detector.pixelsize,
    dim1=detector.dim1,
    dim2=detector.dim2,
    # ... other parameters
)
```

### Use in Prediction (Step 3)

```python
from lauetoolsnn.lauetoolsneuralnetwork import index_laue_patterns

detector = catalog.get_detector("production")

results = index_laue_patterns(
    image_files=image_list,
    model=trained_model,
    detectorparameters=detector.detectorparameters,
    pixelsize=detector.pixelsize,
    dim1=detector.dim1,
    dim2=detector.dim2,
    geometry=detector.geometry,
    # ... other parameters
)
```

---

## Parameter Reference

### Geometric Calibration Parameters

| Parameter | LaueTools Key | Unit | Typical Range | Meaning |
|-----------|---------------|------|----------------|---------|
| **Distance** | `distance` | mm | 50-200 | Sample-to-detector distance |
| **X Center** | `xcen` | pixels | 0–dim1 | Beam center X coordinate |
| **Y Center** | `ycen` | pixels | 0–dim2 | Beam center Y coordinate |
| **Beta Angle** | `angle_beta` | radians | ±0.2 | Rotation around vertical (Z) axis |
| **Gamma Angle** | `angle_gamma` | radians | ±0.2 | Rotation around beam (X) axis |

**Note:** Angles in radians, not degrees!

### Hardware Specifications

| Property | Type | sCMOS (2×2) | sCMOS (1×1) | MARCCD165 |
|----------|------|-------------|-------------|-----------|
| `dim1` | int | 2018 | 4036 | 2048 |
| `dim2` | int | 2016 | 4032 | 2048 |
| `pixelsize` | float | 0.0734 | 0.0367 | 0.0791 |
| `bit_depth` | int | 16 | 16 | 16 |
| `saturation` | int | 65535 | 65535 | 65535 |

### Geometry Modes

- **"Z>0"** — Top reflection (2θ=90°, detector above sample)
- **"X>0"** — Transmission (2θ=0°, detector in direct beam)
- **"X<0"** — Back reflection (2θ=180°, detector upstream)

---

## Advanced Usage

### Parameter Sensitivity Analysis

```python
catalog = DetectorCatalog.load_default()
base = catalog.get_detector("development")

# Test sensitivity to distance uncertainty
distances_to_test = [79.5, 79.553, 79.6, 79.8, 80.0]
for dist in distances_to_test:
    variant = base.apply_parameter_variation(distance=dist)
    # Use variant in simulation...
    print(f"Distance: {variant.distance} mm")
```

### Create Beamline Catalog

```python
from lauetoolsnn.detector_catalog import DetectorCatalog
from lauetoolsnn.detector_config import DetectorConfig

# Create new catalog
catalog = DetectorCatalog(name="BM32_ESRF")

# Register multiple detectors for this beamline
stable_2024 = DetectorConfig(
    name="BM32_sCMOS_stable_2024",
    distance=79.500, xcen=978.0, ycen=930.0,
    angle_beta=0.37, angle_gamma=0.45,
    pixelsize=0.0734, dim1=2018, dim2=2016, geometry="Z>0",
    hardware_label="sCMOS",
    metadata={"calibration_date": "2024-06-01"}
)

recent_2025 = stable_2024.apply_parameter_variation(
    name="BM32_sCMOS_latest_2025",
    distance=79.553, xcen=979.32, ycen=932.31,
    angle_beta=0.37, angle_gamma=0.447
)

catalog.register_detector(stable_2024)
catalog.register_detector(recent_2025)

# Save for later use
catalog.save_to_file("BM32_detectors.json")
```

### Batch Validation

```python
catalog = DetectorCatalog.load_from_file("detectors.json")

# Validate all
validation_results = catalog.validate_all()
for detector_name, is_valid in validation_results.items():
    status = "✓" if is_valid else "✗"
    print(f"{status} {detector_name}")

# Get statistics
stats = catalog.get_statistics()
print(f"Total: {stats['total_detectors']}")
print(f"Distance range: {stats['distance_range_mm']}")
print(f"Hardware types: {stats['hardware_types']}")
```

---

## Common Issues

### "Detector not found in catalog"

```python
# Check what's available
print(catalog.list_detectors())

# Add missing detector
detector = DetectorConfig(...)
catalog.register_detector(detector)
```

### "Validation failed - center outside bounds"

```python
# Check bounds
print(f"X bounds: [0, {detector.dim1})")
print(f"Y bounds: [0, {detector.dim2})")

# Fix: ensure xcen < dim1, ycen < dim2
```

### "distance_from_pixels_mm() not working"

```python
# Use the helper method
pixel_distance = 100  # pixels
mm_distance = detector.distance_from_pixels_mm(pixel_distance)

# Or manually
mm_distance = pixel_distance * detector.pixelsize
```

---

## Best Practices

1. **Always validate** detector configurations before use:
   ```python
   if not detector.is_physically_plausible():
       print(detector.validate())
       sys.exit(1)
   ```

2. **Store production detectors** in catalog JSON files:
   ```python
   catalog.save_to_file("production_detectors.json")
   ```

3. **Use catalog variants** for parameter studies:
   ```python
   variant = catalog.create_detector_variant("base", distance=75.0)
   ```

4. **Document calibration metadata**:
   ```python
   detector.metadata["calibration_date"] = "2025-01-15"
   detector.metadata["calibration_method"] = "powder_diffraction"
   ```

5. **Check validation results** for warnings even if valid:
   ```python
   result = detector.validate()
   if result.warnings:
       print(f"Warnings: {result.warnings}")
   ```

---

## See Also

- [DETECTOR_PARAMETERS_DESIGN.md](DETECTOR_PARAMETERS_DESIGN.md) — Full architecture and planned features
- [PATCHES.md](PATCHES.md) — Section "5. Detector Parameters Configuration System"
- [README_AI_GUIDE.md](README_AI_GUIDE.md) — Project conventions and AI assistant instructions

---

**Last Updated:** March 6, 2025  
**Version:** 1.0 (Phase 1 - Core Infrastructure Complete)
