# Migration Example: From Hardcoded to DetectorConfig

This document shows how to migrate an existing LaueNN script from hardcoded detector parameters to the new DetectorConfig system.

---

## Before: Hardcoded Parameters

```python
# Old Step1 script (example from step_defaults)
input_params = {
    "material_": "Ni",
    "material1_": "Ni",
    "prefix": "",
    "symmetry": "cubic",
    "symmetry1": "cubic",
    "SG": 225,
    "SG1": 225,
    # ❌ Detector parameters hardcoded inline
    "detectorparameters": [79.553, 979.32, 932.31, 0.37, 0.447],
    "pixelsize": 0.0734,
    "dim1": 2018,
    "dim2": 2016,
    "emin": 5,
    "emax": 22,
    # ... more parameters
}
```

**Problems:**
- Parameters scattered in multiple scripts
- Hard to change detector across all steps
- No validation of parameter values
- Can't easily test with variations
- No documentation of detector choices

---

## After: Using DetectorConfig

### Option A: Quick Migration (Minimal Changes)

Keep existing config files but add detector loading:

```python
from lauetoolsnn.detector_catalog import DetectorCatalog

# Load detector from catalog (single line!)
catalog = DetectorCatalog.load_default()
detector = catalog.get_detector("development")

# Same config dictionary, but detector params now come from detector object
input_params = {
    "material_": "Ni",
    "material1_": "Ni",
    "prefix": "",
    "symmetry": "cubic",
    "symmetry1": "cubic",
    "SG": 225,
    "SG1": 225,
    # ✓ Get from DetectorConfig object
    "detectorparameters": detector.detectorparameters,  # [distance, xcen, ycen, xbet, xgam]
    "pixelsize": detector.pixelsize,
    "dim1": detector.dim1,
    "dim2": detector.dim2,
    "emin": 5,
    "emax": 22,
    # ... more parameters
}

# Validate before processing
if not detector.is_physically_plausible():
    print(f"Invalid detector config: {detector.validate()}")
    sys.exit(1)
```

### Option B: Full Integration (Better Design)

Create configuration that references detector by name:

```python
import json
from lauetoolsnn.detector_catalog import DetectorCatalog

# Configuration file (JSON) - step_config.json
config_file_content = {
    "detector_name": "development",  # Reference to detector in catalog
    "material": "Ni",
    "symmetry": "cubic",
    "SG": 225,
    "emin": 5,
    "emax": 22,
    # ... step-specific parameters
}

# Load and process in script
def load_step1_config(config_filepath):
    """Load configuration with detector from catalog"""
    with open(config_filepath) as f:
        config = json.load(f)
    
    # Load detector
    catalog = DetectorCatalog.load_default()
    detector_name = config.pop("detector_name", "development")
    detector = catalog.get_detector(detector_name)
    
    # Assemble full parameters
    input_params = {
        **config,  # Material, symmetry, energies, etc.
        "detectorparameters": detector.detectorparameters,
        "pixelsize": detector.pixelsize,
        "dim1": detector.dim1,
        "dim2": detector.dim2,
    }
    
    # Validate
    validation = detector.validate()
    if not validation.is_valid:
        print(f"Detector validation failed: {validation}")
        sys.exit(1)
    
    return input_params
```

---

## Migration Checklist

### Step 1: Update Imports
```python
# Add these imports to your script
from lauetoolsnn.detector_catalog import DetectorCatalog
from lauetoolsnn.detector_config import DetectorConfig
```

### Step 2: Replace Hardcoded Parameters
```python
# OLD: Hardcoded arrays
detectorparameters = [79.553, 979.32, 932.31, 0.37, 0.447]
pixelsize = 0.0734
dim1 = 2018
dim2 = 2016

# NEW: From DetectorConfig
detector = DetectorCatalog.load_default().get_detector("development")
detectorparameters = detector.detectorparameters
pixelsize = detector.pixelsize
dim1 = detector.dim1
dim2 = detector.dim2
```

### Step 3: Add Validation
```python
# Before using detector parameters
if not detector.is_physically_plausible():
    print(detector.validate())
    raise ValueError("Invalid detector configuration")
```

### Step 4: Update Configuration Files
```python
# OLD step_config.json
{
    "material": "Ni",
    "detectorparameters": [79.553, 979.32, 932.31, 0.37, 0.447],
    "pixelsize": 0.0734,
    "dim1": 2018,
    "dim2": 2016
}

# NEW step_config.json
{
    "material": "Ni",
    "detector_name": "development",
    "emin": 5,
    "emax": 22
}
```

### Step 5: Test with Variations
```python
# Easy parameter studies with DetectorConfig
for distance in [70, 75, 79.553, 85, 90]:
    variant = detector.apply_parameter_variation(distance=distance)
    # Run simulation with variant
    print(f"Testing distance={distance} mm")
```

---

## Complete Example: Updated Step1 Script

```python
#!/usr/bin/env python3
"""
Step 1: Generate Training Dataset for LaueNN
Refactored to use DetectorConfig system
"""

import sys
import json
import argparse
from pathlib import Path

# LaueNN imports
from lauetoolsnn.detector_catalog import DetectorCatalog
from lauetoolsnn.utils_lauenn import generate_dataset

def load_config(config_file, detector_catalog):
    """Load configuration and detector parameters"""
    
    with open(config_file) as f:
        config = json.load(f)
    
    # Get detector from catalog
    detector_name = config.get("detector_name", "development")
    detector = detector_catalog.get_detector(detector_name)
    
    # Validate detector
    validation = detector.validate()
    if not validation.is_valid:
        print(f"ERROR: Invalid detector '{detector_name}'")
        print(validation)
        return None
    
    # Add detector parameters to config
    config["detectorparameters"] = detector.detectorparameters
    config["pixelsize"] = detector.pixelsize
    config["dim1"] = detector.dim1
    config["dim2"] = detector.dim2
    config["detector_object"] = detector  # For reference
    
    return config

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate LaueNN training dataset")
    parser.add_argument("--config", required=True, help="Configuration file (JSON)")
    parser.add_argument("--output", default="datasets", help="Output directory")
    parser.add_argument("--detector-catalog", help="Path to detector catalog JSON")
    args = parser.parse_args()
    
    # Load detector catalog
    if args.detector_catalog:
        catalog = DetectorCatalog.load_from_file(args.detector_catalog)
    else:
        catalog = DetectorCatalog.load_default()
    
    # Load configuration with detector
    config = load_config(args.config, catalog)
    if not config:
        sys.exit(1)
    
    # Print detector info
    detector = config.pop("detector_object")
    print(detector.describe())
    
    # Process material parameters
    material = config.get("material_", "Ni")
    print(f"\nGenerating dataset for {material}...")
    
    # Generate dataset
    dataset = generate_dataset(
        material=material,
        detectorparameters=config["detectorparameters"],
        pixelsize=config["pixelsize"],
        dim1=config["dim1"],
        dim2=config["dim2"],
        emin=config.get("emin", 5),
        emax=config.get("emax", 22),
        output_directory=args.output,
        **{k: v for k, v in config.items() 
           if k not in ("detectorparameters", "pixelsize", "dim1", "dim2", 
                       "detector_name", "material_")}
    )
    
    print(f"✓ Dataset saved to {args.output}")

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
# With default detector
python Step1.py --config step1_config.json --output datasets

# With custom detector catalog
python Step1.py --config step1_config.json \
                --detector-catalog production_detectors.json \
                --output datasets

# Command line help
python Step1.py --help
```

---

## Configuration File Evolution

### Generation 1: Monolithic Config (Current)
```json
{
  "material_": "Ni",
  "detectorparameters": [79.553, 979.32, 932.31, 0.37, 0.447],
  "pixelsize": 0.0734,
  "dim1": 2018,
  "dim2": 2016,
  "emin": 5,
  "emax": 22,
  "prefix": "",
  "symmetry": "cubic",
  "SG": 225,
  "hkl_max_identify": 5,
  "nb_grains_per_lp": 5
}
```

### Generation 2: Detector References (With DetectorConfig)
```json
{
  "detector_name": "development",
  "material_": "Ni",
  "emin": 5,
  "emax": 22,
  "prefix": "",
  "symmetry": "cubic",
  "SG": 225,
  "hkl_max_identify": 5,
  "nb_grains_per_lp": 5
}
```
**Advantages:**
- Detector params no longer duplicated
- Can switch detector by name (development → production)
- Validates detector configuration

### Generation 3: Fully Modular (Future)
```json
{
  "detector": "development",
  "material": "Ni",
  "energy_range": [5, 22],
  "crystallography": {
    "symmetry": "cubic",
    "space_group": 225
  },
  "generation": {
    "hkl_max": 5,
    "grains_per_pattern": 5,
    "patterns_to_generate": 100
  }
}
```
**Further advantages:**
- Clearer hierarchy
- Easy to validate each section
- Extensible for new parameters

---

## Rollback / Compatibility

If you need to support both old and new formats:

```python
def load_config_flexible(config_file, catalog=None):
    """Load config with backward compatibility"""
    
    with open(config_file) as f:
        config = json.load(f)
    
    # Check if using new detector reference format
    if "detector_name" in config:
        if catalog is None:
            catalog = DetectorCatalog.load_default()
        detector = catalog.get_detector(config["detector_name"])
        config["detectorparameters"] = detector.detectorparameters
        config["pixelsize"] = detector.pixelsize
        config["dim1"] = detector.dim1
        config["dim2"] = detector.dim2
    elif "detectorparameters" in config:
        # Old format - still supported but not validated
        print("WARNING: Using hardcoded detector parameters")
        print("  Consider migrating to 'detector_name' format")
    
    return config
```

---

## Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Parameter Changes** | Edit 5+ files | Edit detector in one place |
| **Validation** | Manual checking | Automatic via `is_physically_plausible()` |
| **Documentation** | Scattered comments | Centralized in DetectorConfig |
| **Parameter Studies** | Copy, modify, repeat | Use `apply_parameter_variation()` |
| **Multi-beamline** | Separate scripts | One script + catalog selection |
| **Error Detection** | Runtime failures | Early validation warnings |
| **Code Reuse** | Duplicate logic | Shared DetectorConfig class |

---

## Next Steps in Your Project

1. **Pick one Step script** to migrate first (suggest Step 1 or Step 3)
2. **Create a config file** using Generation 2 format
3. **Update the script** following the complete example above
4. **Test thoroughly** with both development and production detectors
5. **Expand to other steps** once first one is solid
6. **Create production catalogs** for your beamlines (Phase 2)

---

## Questions?

Refer to:
- [DETECTOR_QUICK_START.md](DETECTOR_QUICK_START.md) - Usage examples
- [DETECTOR_PARAMETERS_DESIGN.md](DETECTOR_PARAMETERS_DESIGN.md) - Architecture
- [detector_config.py](lauetoolsnn/detector_config.py) - Source code docstrings
- [detector_catalog.py](lauetoolsnn/detector_catalog.py) - Source code docstrings

