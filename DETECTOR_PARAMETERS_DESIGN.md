# Detector Parameters Configuration System - Design Document

## Executive Summary

This document outlines a comprehensive system for managing detector parameters in LaueNN across development, production, and experimental calibration workflows. It identifies current gaps and proposes a structured, extensible solution.

---

## Current State Analysis

### 1. How Detector Parameters Are Currently Used

#### **Core Parameter Structure**
Detectors are defined by two groups of parameters:

**Geometric Calibration** (5 values):
```python
detectorparameters = [
    distance,  # [0] - Sample-detector distance (mm)
    xcen,      # [1] - Detector center X coordinate (pixels)
    ycen,      # [2] - Detector center Y coordinate (pixels)
    xbet,      # [3] - Beta angle - rotation around vertical (lab frame) axis (radians)
    xgam       # [4] - Gamma angle - rotation around beam axis (radians)
]
```

**Hardware Properties**:
- `pixelsize` - Physical size of one detector pixel (mm)
- `dim1`, `dim2` - Detector dimensions in pixels (width × height)
- `ccd_label` - Reference to detector type catalog (e.g., "sCMOS", "MARCCD165")

#### **Current Storage Locations**
1. **detector_config.json** - Hardware specs for ~35 detector models
   - Stores: dimensions, pixel size, saturation value, data format
   - Example: `"sCMOS": [[2018, 2016], 0.0734, 65535, "no", 3828, "uint16", "...", "tif"]`

2. **step_defaults.py** - Single calibration example
   - `BASE_DETECTOR`: One hardcoded detector calibration per property
   - Used as fallback for all steps

3. **Example scripts** - Multiple copies of calibration values
   - Each script has its own inline example parameters
   - No central authority; calibrations may diverge

4. **detector_geometry.json** - Beamline geometry mode
   - Stores default Laue geometry: "Z>0" (top reflection), "X>0" (transmission), "X<0" (back reflection)

### 2. How Parameters Flow Through Applications

```
Configuration File (JSON)
        ↓
Parse into input_params dict
        ↓
Extract detectorparameters array
Extract pixelsize, dim1, dim2
        ↓
Pass to LaueTools functions:
  - generate_dataset()
  - simulate_laue_image()
  - index_laue_patterns()
        ↓
Used in crystallographic calculations (detector geometry transformations)
```

### 3. Current Limitations & Gaps

#### **Problem 1: Parameter Discovery**
- ❌ No programmatic way to calibrate detector parameters from experimental images
- ❌ No tools to extract center coordinates, distance, or angles from images
- ❌ Manual measurement or trial-and-error calibration required
- ❌ No validation that parameters are physically sensible

#### **Problem 2: Configuration Management**
- ❌ No beamline/setup-specific detector catalogs
- ❌ Can't easily switch between different detector configurations
- ❌ No versioning of calibration changes
- ❌ No environment-specific defaults (dev vs. production)

#### **Problem 3: Documentation & Context**
- ❌ Parameter meanings not well documented in code
- ❌ No guidance on typical ranges or constraints
- ❌ Missing documentation on physical meaning of angles
- ❌ No reference for how to measure/verify parameters

#### **Problem 4: Data Integrity**
- ❌ No validation of parameter ranges
- ❌ No type checking before passing to LaueTools
- ❌ Easy to introduce typos in configurations
- ❌ No error messages if parameters are implausible

#### **Problem 5: Extensibility**
- ❌ Adding new detector hardware requires editing JSON files
- ❌ No support for detector-specific preprocessing (flips, crops)
- ❌ No way to define parameter variations (e.g., binning modes)
- ❌ Hard to track experimental parameter adjustments

---

## Proposed Architecture

### 1. Core Component: `DetectorConfig` Class

A Python class to encapsulate all detector-related configuration with validation and utilities.

**Location**: `/lauetoolsnn/detector_config.py`

```python
class DetectorConfig:
    """
    Unified detector configuration with validation and utilities.
    
    Handles geometric calibration, hardware specs, and beamline geometry.
    """
    
    # Constructor
    __init__(self, name, distance, xcen, ycen, angle_beta, angle_gamma,
             pixelsize, dim1, dim2, geometry="Z>0", hardware_label=None,
             metadata=None)
    
    # Properties
    @property
    def detectorparameters(self) -> list[float]
        """Returns [distance, xcen, ycen, angle_beta, angle_gamma]"""
    
    @property
    def detector_dict(self) -> dict
        """Returns dict suitable for CCDcalib structures"""
    
    # Validation
    def validate(self) -> ValidationResult
        """Check parameter ranges and physical constraints"""
    
    def is_physically_plausible(self) -> bool
        """Quick sanity check for typical experimental values"""
    
    # Serialization
    def to_dict(self) -> dict
        """Serialize to dict for JSON storage"""
    
    @classmethod
    def from_dict(cls, config_dict) -> DetectorConfig
        """Deserialize from dict"""
    
    # Comparison & variants
    def apply_parameter_variation(self, **kwargs) -> DetectorConfig
        """Create modified copy (e.g., binning changes)"""
    
    def distance_from_pixels_mm(self, pixel_dist) -> float
        """Convert pixel coordinate difference to mm at detector"""
    
    # Utilities
    def describe(self) -> str
        """Human-readable description with all parameters"""
```

### 2. Detector Catalog System

**Location**: `/lauetoolsnn/detector_catalog.py`

```python
class DetectorCatalog:
    """
    Registry of known detectors and beamline-specific configurations.
    
    Provides:
    - Built-in detector hardware library
    - Beamline-specific calibration catalogs
    - Preset variations (binning, crop, flip modes)
    """
    
    # Built-in hardware definitions
    HARDWARE = {
        "sCMOS": {
            "dim1": 2018,
            "dim2": 2016,
            "pixelsize": 0.0734,
            "description": "sCMOS camera (binned 2×2)"
        },
        "MARCCD165": {
            "dim1": 2048,
            "dim2": 2048,
            "pixelsize": 0.079142,
            "description": "MAR Research 165 mm detector"
        },
        # ... more hardware definitions
    }
    
    # Methods
    def __init__(self, catalog_name="default")
    
    def load_from_json(self, filepath)
        """Load beamline-specific calibrations"""
    
    def get_detector(self, name) -> DetectorConfig
        """Retrieve a named detector configuration"""
    
    def list_detectors(self) -> list[str]
        """List all available configurations"""
    
    def register_detector(self, detector)
        """Add new detector to catalog"""
    
    def create_detector_variant(self, base_name, **modifications) -> DetectorConfig
        """Create variant (e.g., different binning)"""
    
    def get_beamline_defaults(self, beamline_id) -> dict
        """Get default detector, energy, geometry for a beamline"""
```

### 3. Configuration Files (Hierarchical)

**Location**: `/lauetoolsnn/configs/detectors/`

#### **3a. detector_catalog.json** - Hardware Library
```json
{
  "hardware": {
    "sCMOS": {
      "dim1": 2018,
      "dim2": 2016,
      "pixelsize": 0.0734,
      "bit_depth": 16,
      "saturation": 65535,
      "description": "sCMOS (2×2 binned)"
    },
    "sCMOS_1x1": {
      "dim1": 4036,
      "dim2": 4032,
      "pixelsize": 0.0367,
      "bit_depth": 16,
      "saturation": 65535,
      "description": "sCMOS (1×1 unbinned)"
    }
  }
}
```

#### **3b. beamlines/{beamline_name}/calibrations.json** - Production Calibrations
```json
{
  "BM32_setup_2025_01": {
    "name": "BM32_setup_2025_01",
    "hardware": "sCMOS",
    "geometry": "Z>0",
    "distance": 79.553,
    "xcen": 979.32,
    "ycen": 932.31,
    "angle_beta": 0.37,
    "angle_gamma": 0.447,
    "energy_range": [5, 22],
    "calibration_date": "2025-01-15",
    "calibration_method": "powder_diffraction",
    "uncertainty_distance": 0.1,
    "uncertainty_center": [1.0, 1.0],
    "uncertainty_angles": [0.01, 0.01],
    "notes": "Annual recalibration"
  },
  "setup_development": {
    "name": "setup_development",
    "hardware": "sCMOS",
    "geometry": "Z>0",
    "distance": 79.553,
    "xcen": 979.32,
    "ycen": 932.31,
    "angle_beta": 0.37,
    "angle_gamma": 0.447,
    "is_development": true,
    "notes": "For testing and development only"
  }
}
```

#### **3c. development/detector_variants.json** - Development Variations
```json
{
  "variations": {
    "shift_center_right": {
      "base": "BM32_setup_2025_01",
      "modifications": {
        "xcen": 985.0
      },
      "purpose": "Test sensitivity to center shift"
    },
    "shorter_distance": {
      "base": "BM32_setup_2025_01",
      "modifications": {
        "distance": 70.0
      },
      "purpose": "Simulate closer detector"
    }
  }
}
```

### 4. Calibration & Discovery Module

**Location**: `/lauetoolsnn/detector_calibration.py`

```python
class DetectorCalibrator:
    """
    Tools for discovering and refining detector parameters from experimental data.
    
    Supports:
    - Powder diffraction calibration (known materials)
    - Spot centroid analysis
    - Parameter sensitivity analysis
    - Iterative refinement
    """
    
    def calibrate_from_powder_diffraction(
        self,
        image_files: list[str],
        material: str,
        energy_kev: float,
        initial_guess: DetectorConfig,
        **solver_params
    ) -> DetectorConfig:
        """
        Refine detector parameters using powder diffraction rings.
        
        Matches geometric spot patterns to expected lattice geometry.
        Returns refined DetectorConfig.
        """
    
    def calibrate_from_spot_analysis(
        self,
        experimental_image: np.ndarray,
        simulated_image: np.ndarray,
        initial_guess: DetectorConfig
    ) -> DetectorConfig:
        """
        Cross-correlate experimental and simulated patterns to find offset.
        
        Useful for fine-tuning center coordinates and distance.
        """
    
    def estimate_center_of_mass(
        self,
        image: np.ndarray
    ) -> tuple[float, float]:
        """Find geometric center of diffraction pattern"""
    
    def sensitivity_analysis(
        self,
        detector: DetectorConfig,
        material: str,
        energy_kev: float,
        param_name: str,
        variation_range: tuple[float, float],
        steps: int = 10
    ) -> dict:
        """
        Compute how detector parameter changes affect diffraction pattern.
        
        Returns: sensitivity metrics for parameter optimization.
        """
    
    def validate_against_experimental_images(
        self,
        detector: DetectorConfig,
        image_files: list[str],
        tolerance: float = 5.0  # pixels
    ) -> ValidationReport:
        """
        Check if detector parameters produce realistic predictions.
        
        Returns quality metrics and deviation statistics.
        """
```

### 5. Integration with Steps

Each Step script will:
1. Load detector config from catalog/file
2. Get `DetectorConfig` object
3. Extract arrays/dicts as needed for LaueTools
4. Use validation methods to catch errors early

```python
# In Step scripts
from lauetoolsnn.detector_catalog import DetectorCatalog
from lauetoolsnn.detector_config import DetectorConfig

# Load from production catalog
catalog = DetectorCatalog("production")
detector = catalog.get_detector("BM32_setup_2025_01")

# OR create from config file
detector = DetectorConfig.from_dict(config_data["detector"])

# Validate before use
validation_result = detector.validate()
if not validation_result.is_valid:
    print(f"Invalid detector: {validation_result.errors}")
    sys.exit(1)

# Use with LaueTools
detectorparameters = detector.detectorparameters  # [distance, xcen, ycen, beta, gamma]
pixelsize = detector.pixelsize
dim1, dim2 = detector.dim1, detector.dim2
geometry = detector.geometry
```

---

## Implementation Roadmap

### **Phase 1: Core Infrastructure** (Weeks 1-2)
- [ ] Create `DetectorConfig` class with validation
- [ ] Create `DetectorCatalog` with built-in hardware definitions
- [ ] Set up config directory structure
- [ ] Write comprehensive parameter documentation
- [ ] Create `detector_catalog.json` with hardware specs

### **Phase 2: Integration** (Weeks 2-3)
- [ ] Update Step 1, 2, 3 scripts to use `DetectorConfig`
- [ ] Update `step_defaults.py` to reference catalog
- [ ] Create beamline-specific example catalogs
- [ ] Add detector validation to all scripts
- [ ] Update examples with new config format

### **Phase 3: Calibration Tools** (Weeks 3-4)
- [ ] Implement `DetectorCalibrator` class
- [ ] Add powder diffraction calibration routine
- [ ] Add spot analysis calibration
- [ ] Create calibration CLI tool
- [ ] Write calibration guide/tutorial

### **Phase 4: Utilities & Polish** (Week 4+)
- [ ] Parameter sensitivity analysis tools
- [ ] Beamline preset manager
- [ ] GUI elements for detector selection
- [ ] Automated validation tests
- [ ] Comprehensive documentation

---

## File Structure (Proposed)

```
lauetoolsnn/
├── detector_config.py           # Core DetectorConfig class
├── detector_catalog.py          # DetectorCatalog and registry
├── detector_calibration.py      # Calibration tools
├── configs/
│   └── detectors/
│       ├── hardware_definitions.json
│       ├── beamlines/
│       │   ├── BM32/
│       │   │   ├── calibrations.json
│       │   │   └── geometry.json
│       │   ├── DESY/
│       │   └── ...
│       └── development/
│           └── test_variants.json
└── util_scripts/
    └── calibrate_detector.py    # CLI tool for calibration
```

---

## Parameters Reference

### Geometric Calibration Parameters

| Parameter | Symbol | Unit | Typical Range | Physical Meaning |
|-----------|--------|------|----------------|-----------------|
| **Distance** | dd | mm | 50-200 | Sample-to-detector distance along beam |
| **Center X** | xcen | pixels | 0-dim1 | Detector X coordinate of beam center |
| **Center Y** | ycen | pixels | 0-dim2 | Detector Y coordinate of beam center |
| **Beta Angle** | xbet | radians | -0.2 to 0.2 | Rotation around vertical (Z) axis |
| **Gamma Angle** | xgam | radians | -0.2 to 0.2 | Rotation around beam (X) axis |

**Notes:**
- Angles in LaueTools are in **radians**, not degrees
- Center coordinates are measured from top-left corner (pixel [0,0])
- Beta/Gamma define detector misalignment relative to perpendicular orientation

### Hardware Specifications

| Property | Type | Example (sCMOS) | Physical Meaning |
|----------|------|-----------------|-----------------|
| **dim1** | int | 2018 | Detector width in pixels |
| **dim2** | int | 2016 | Detector height in pixels |
| **pixelsize** | float | 0.0734 | Physical size of pixel in mm |
| **bit_depth** | int | 16 | Bits per pixel (8, 16, 32) |
| **saturation** | int | 65535 | Maximum ADC count value |
| **geometry** | str | "Z>0" | Beamline geometry mode |

### Geometry Modes

- **"Z>0"** - Top Reflection (2θ=90°)
  - Camera positioned above sample, detects upward-diffracted beams
  - Most common at synchrotrons
  
- **"X>0"** - Transmission (2θ=0°)
  - Camera in direct beam path downstream of sample
  - Used for transmission Laue
  
- **"X<0"** - Back Reflection (2θ=180°)
  - Camera upstream, detects backscattered beams
  - Less common in Laue practice

---

## Validation Rules

### Physical Constraints

1. **Distance**: Must be positive, typically 50-200 mm
2. **Center coordinates**: 0 ≤ xcen < dim1, 0 ≤ ycen < dim2
3. **Angles**: Typically |angle| < 0.2 radians (~11°)
4. **Pixel size**: Must match hardware specifications
5. **Dimensions**: Must be positive integers

### Sensibility Checks

1. **Distance-based**: 
   - If distance > 150 mm, warn if pixelsize < 0.03 mm (angular resolution)
   
2. **Center location**:
   - Warn if center is very close to edges (< 5% of dimension)
   - Warn if center is far off-center (> 2× symmetric to edges)
   
3. **Angle consistency**:
   - Warn if beta and gamma angles differ by > 0.5 radians (unusual asymmetry)

---

## Configuration Format (Quick Reference)

### Complete Detector Configuration

```json
{
  "detector": {
    "name": "BM32_sCMOS_2025",
    "hardware": "sCMOS",
    "geometry": "Z>0",
    "calibration": {
      "distance": 79.553,
      "xcen": 979.32,
      "ycen": 932.31,
      "angle_beta": 0.37,
      "angle_gamma": 0.447
    },
    "metadata": {
      "calibration_date": "2025-01-15",
      "calibration_method": "powder_diffraction",
      "uncertainty": {
        "distance": 0.1,
        "center": [1.0, 1.0],
        "angles": [0.01, 0.01]
      },
      "notes": "Annual recalibration with Si powder"
    }
  }
}
```

---

## Usage Examples

### Example 1: Use Production Detector

```python
from lauetoolsnn.detector_catalog import DetectorCatalog

# Load production catalog
catalog = DetectorCatalog.load("production")

# Get specific detector
detector = catalog.get_detector("BM32_setup_2025_01")

# Use in generation
detectorparameters = detector.detectorparameters
pixelsize = detector.pixelsize
```

### Example 2: Development with Variations

```python
from lauetoolsnn.detector_catalog import DetectorCatalog

catalog = DetectorCatalog.load("development")

# Try parameter variation
detector_test = catalog.create_detector_variant(
    "BM32_base",
    distance=75.0,  # Shorter distance
    xcen=975.0      # Shifted center
)

# Validate before use
validation = detector_test.validate()
print(detector_test.describe())
```

### Example 3: Manual Configuration

```python
from lauetoolsnn.detector_config import DetectorConfig

detector = DetectorConfig(
    name="custom_setup",
    distance=80.0,
    xcen=1009.0,
    ycen=932.0,
    angle_beta=0.4,
    angle_gamma=0.45,
    pixelsize=0.0734,
    dim1=2018,
    dim2=2016,
    geometry="Z>0",
    hardware_label="sCMOS"
)

# Validate
if detector.is_physically_plausible():
    print("Parameters look reasonable")
else:
    print(detector.validate().explain())
```

### Example 4: Calibration from Experimental Data

```python
from lauetoolsnn.detector_calibration import DetectorCalibrator

calibrator = DetectorCalibrator()

# Refine parameters using experimental powder diffraction
refined_detector = calibrator.calibrate_from_powder_diffraction(
    image_files=["powder_01.tif", "powder_02.tif"],
    material="Si",
    energy_kev=17.5,
    initial_guess=initial_detector,
    max_iterations=100,
    tolerance=0.5  # pixels
)

print(f"Refined distance: {refined_detector.distance}")
print(f"Refined center: ({refined_detector.xcen}, {refined_detector.ycen})")
```

---

## Benefits of This Architecture

✅ **Separation of Concerns**: Detector config isolated from business logic  
✅ **Reusability**: Single source of truth for detector definitions  
✅ **Validation**: Early error detection with clear messages  
✅ **Flexibility**: Easy to add new detectors or beamlines  
✅ **Testability**: Can mock/patch detector configs easily  
✅ **Documentation**: Self-documenting through class docstrings and metadata  
✅ **Production-Ready**: Supports version control and change tracking  
✅ **Development-Friendly**: Quick parameter variations without code changes  
✅ **Extensible**: Room for detector-specific preprocessing, corrections  
✅ **Observable**: Rich validation and diagnostic capabilities  

---

## Questions for Implementation

1. **Should parameter uncertainties be stored per-detector?**
   - Yes, useful for error propagation and statistical analysis

2. **Should we support detector transformations (flip, crop)?**
   - Yes, some detectors require pre-processing before pattern matching

3. **Should beamline presets include energy ranges?**
   - Yes, some X-ray optics restrict available energies per beamline

4. **Should we auto-detect detector hardware from images?**
   - Future enhancement; TIFF headers sometimes contain specs

5. **Should detector history be versioned?**
   - Recommended using Git, but JSON schema can include version numbers

---

## References & Further Reading

- **LaueTools Documentation**: Parameter meanings in the source code
- **ESRF BM32**: Real-world example of multi-detector beamline
- **Detector Calibration Methods**: Powder diffraction, Rietveld refinement
- **Quaternion Rotations**: Understanding beta/gamma angles in detector frame

---

## Next Steps

1. **Review this design** with project stakeholders
2. **Prioritize Phase 1 implementation** based on immediate needs
3. **Create base `DetectorConfig` class** to unblock dependent work
4. **Begin catalog creation** with known beamlines
5. **Plan calibration tool development** in parallel

