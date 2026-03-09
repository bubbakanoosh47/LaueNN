"""
Detector Configuration Management for LaueNN

Provides DetectorConfig class for encapsulating and validating detector
parameters (geometric calibration + hardware specifications).

Author: LaueNN Development Team
Version: 1.0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
import warnings
import json
from pathlib import Path


@dataclass
class ValidationResult:
    """Result of detector parameter validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        lines = []
        if self.is_valid:
            lines.append("✓ Detector configuration is VALID")
        else:
            lines.append("✗ Detector configuration is INVALID")
        
        if self.errors:
            lines.append("\nErrors:")
            for err in self.errors:
                lines.append(f"  • {err}")
        
        if self.warnings:
            lines.append("\nWarnings:")
            for warn in self.warnings:
                lines.append(f"  • {warn}")
        
        return "\n".join(lines)
    
    def explain(self) -> str:
        """Alias for __str__"""
        return str(self)


class DetectorConfig:
    """
    Unified detector configuration with validation and utilities.
    
    Encapsulates:
    - Geometric calibration (distance, center, angles)
    - Hardware specifications (dimensions, pixel size)
    - Beamline geometry mode
    - Metadata (calibration date, method, uncertainties)
    
    Provides:
    - Validation of physical constraints
    - Serialization/deserialization to/from dicts and JSON
    - Utilities for parameter manipulation
    - Human-readable descriptions
    
    Example:
        >>> detector = DetectorConfig(
        ...     name="BM32_sCMOS_2025",
        ...     distance=79.553,
        ...     xcen=979.32,
        ...     ycen=932.31,
        ...     angle_beta=0.37,
        ...     angle_gamma=0.447,
        ...     pixelsize=0.0734,
        ...     dim1=2018,
        ...     dim2=2016,
        ...     geometry="Z>0"
        ... )
        >>> if detector.is_physically_plausible():
        ...     params = detector.detectorparameters
        ...     print(detector.describe())
    """
    
    # Class constants for validation
    GEOMETRY_MODES = {"Z>0", "X>0", "X<0"}
    DISTANCE_MIN = 30.0  # mm
    DISTANCE_MAX = 300.0  # mm
    ANGLE_MAX = 0.5  # radians (~28 degrees)
    PIXELSIZE_MIN = 0.005  # mm (very small pixel)
    PIXELSIZE_MAX = 0.5  # mm (very large pixel)
    
    def __init__(
        self,
        name: str,
        distance: float,
        xcen: float,
        ycen: float,
        angle_beta: float,
        angle_gamma: float,
        pixelsize: float,
        dim1: int,
        dim2: int,
        geometry: str = "Z>0",
        hardware_label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a DetectorConfig.
        
        Args:
            name: Unique identifier for this configuration
            distance: Sample-to-detector distance (mm)
            xcen: Detector X center coordinate (pixels)
            ycen: Detector Y center coordinate (pixels)
            angle_beta: Detector rotation around vertical axis (radians)
            angle_gamma: Detector rotation around beam axis (radians)
            pixelsize: Physical pixel size (mm/pixel)
            dim1: Detector width (pixels)
            dim2: Detector height (pixels)
            geometry: Laue geometry mode: "Z>0", "X>0", or "X<0"
            hardware_label: Reference to hardware type (e.g., "sCMOS")
            metadata: Optional dict with calibration info, uncertainties, etc.
        """
        self.name = name
        self.distance = float(distance)
        self.xcen = float(xcen)
        self.ycen = float(ycen)
        self.angle_beta = float(angle_beta)
        self.angle_gamma = float(angle_gamma)
        self.pixelsize = float(pixelsize)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.geometry = str(geometry)
        self.hardware_label = hardware_label
        self.metadata = metadata or {}
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def detectorparameters(self) -> List[float]:
        """
        Return detector parameters as list for LaueTools compatibility.
        
        Format: [distance, xcen, ycen, angle_beta, angle_gamma]
        
        This is the standard 5-element array used throughout LaueTools
        and LaueNN generation/simulation functions.
        
        Returns:
            List of calibration parameters
        """
        return [self.distance, self.xcen, self.ycen, self.angle_beta, self.angle_gamma]
    
    @property
    def detector_dict(self) -> Dict[str, Any]:
        """
        Return detector as dictionary suitable for CCDcalib structures.
        
        Format matches LaueTools internal representations:
        {
            "dd": distance,
            "xcen": center_x,
            "ycen": center_y,
            "xbet": angle_beta,
            "xgam": angle_gamma,
            "pixelsize": pixelsize,
            "framedim": (dim1, dim2),
            "CCDLabel": hardware_label,
            "kf_direction": geometry
        }
        
        Returns:
            Dictionary of calibration parameters
        """
        return {
            "dd": self.distance,
            "xcen": self.xcen,
            "ycen": self.ycen,
            "xbet": self.angle_beta,
            "xgam": self.angle_gamma,
            "pixelsize": self.pixelsize,
            "framedim": (self.dim1, self.dim2),
            "CCDLabel": self.hardware_label or "user_defined",
            "kf_direction": self.geometry,
        }
    
    @property
    def detector_diameter_mm(self) -> float:
        """
        Calculate detector diameter in mm (diagonal distance from edges).
        
        Returns:
            Diagonal size of detector in mm
        """
        diag_pixels = (self.dim1**2 + self.dim2**2) ** 0.5
        return diag_pixels * self.pixelsize
    
    @property
    def center_offset_mm(self) -> Tuple[float, float]:
        """
        Calculate offset of beam center from physical center (mm).
        
        Returns:
            Tuple of (dx_mm, dy_mm) from geometric center
        """
        center_x_pixels = self.dim1 / 2.0
        center_y_pixels = self.dim2 / 2.0
        dx = (self.xcen - center_x_pixels) * self.pixelsize
        dy = (self.ycen - center_y_pixels) * self.pixelsize
        return (dx, dy)
    
    @property
    def scattering_angle_coverage_degrees(self) -> float:
        """
        Estimate maximum scattering angle visible (degrees).
        
        Approximation based on distance and detector size.
        
        Returns:
            Maximum scattering angle in degrees
        """
        import math
        # Use corner-to-center distance
        half_diag = self.detector_diameter_mm / 2.0
        angle_rad = math.atan(half_diag / self.distance)
        return math.degrees(angle_rad)
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    def validate(self) -> ValidationResult:
        """
        Validate detector parameters against physical constraints.
        
        Checks:
        - Distance within reasonable range
        - Center coordinates within detector bounds
        - Angles within typical range
        - Pixel size within expected range
        - Geometry mode is valid
        - Dimensions are positive
        
        Returns:
            ValidationResult with is_valid flag and error/warning lists
        """
        errors = []
        warnings = []
        
        # Check distance
        if self.distance <= 0:
            errors.append(f"Distance must be positive, got {self.distance}")
        if self.distance < self.DISTANCE_MIN:
            errors.append(
                f"Distance {self.distance} mm is unusually small "
                f"(typical minimum: {self.DISTANCE_MIN} mm)"
            )
        if self.distance > self.DISTANCE_MAX:
            errors.append(
                f"Distance {self.distance} mm is unusually large "
                f"(typical maximum: {self.DISTANCE_MAX} mm)"
            )
        
        # Check center coordinates
        if not (0 <= self.xcen < self.dim1):
            errors.append(
                f"X center {self.xcen} is outside detector bounds [0, {self.dim1})"
            )
        if not (0 <= self.ycen < self.dim2):
            errors.append(
                f"Y center {self.ycen} is outside detector bounds [0, {self.dim2})"
            )
        
        # Warn if center too close to edges
        edge_margin = 0.05
        if self.xcen < edge_margin * self.dim1:
            warnings.append(
                f"X center {self.xcen} is very close to left edge (< 5% margin)"
            )
        if self.xcen > (1 - edge_margin) * self.dim1:
            warnings.append(
                f"X center {self.xcen} is very close to right edge (> 95%)"
            )
        if self.ycen < edge_margin * self.dim2:
            warnings.append(
                f"Y center {self.ycen} is very close to top edge (< 5% margin)"
            )
        if self.ycen > (1 - edge_margin) * self.dim2:
            warnings.append(
                f"Y center {self.ycen} is very close to bottom edge (> 95%)"
            )
        
        # Check angles
        if abs(self.angle_beta) > self.ANGLE_MAX:
            errors.append(
                f"Beta angle {self.angle_beta:.4f} rad exceeds typical range "
                f"(|angle| < {self.ANGLE_MAX} rad)"
            )
        if abs(self.angle_gamma) > self.ANGLE_MAX:
            errors.append(
                f"Gamma angle {self.angle_gamma:.4f} rad exceeds typical range "
                f"(|angle| < {self.ANGLE_MAX} rad)"
            )
        
        # Check pixel size
        if self.pixelsize <= 0:
            errors.append(f"Pixel size must be positive, got {self.pixelsize}")
        if self.pixelsize < self.PIXELSIZE_MIN:
            errors.append(
                f"Pixel size {self.pixelsize} mm is unusually small "
                f"(typical minimum: {self.PIXELSIZE_MIN} mm)"
            )
        if self.pixelsize > self.PIXELSIZE_MAX:
            errors.append(
                f"Pixel size {self.pixelsize} mm is unusually large "
                f"(typical maximum: {self.PIXELSIZE_MAX} mm)"
            )
        
        # Check dimensions
        if self.dim1 <= 0:
            errors.append(f"Width (dim1) must be positive, got {self.dim1}")
        if self.dim2 <= 0:
            errors.append(f"Height (dim2) must be positive, got {self.dim2}")
        
        # Check geometry
        if self.geometry not in self.GEOMETRY_MODES:
            errors.append(
                f"Geometry '{self.geometry}' not in valid modes: {self.GEOMETRY_MODES}"
            )
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
    
    def is_physically_plausible(self) -> bool:
        """
        Quick sanity check: are parameters physically reasonable?
        
        Returns:
            True if validation has no errors (ignores warnings)
        """
        return self.validate().is_valid
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize detector configuration to dictionary.
        
        Format suitable for JSON export.
        
        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "hardware_label": self.hardware_label,
            "geometry": self.geometry,
            "calibration": {
                "distance": self.distance,
                "xcen": self.xcen,
                "ycen": self.ycen,
                "angle_beta": self.angle_beta,
                "angle_gamma": self.angle_gamma,
            },
            "hardware": {
                "pixelsize": self.pixelsize,
                "dim1": self.dim1,
                "dim2": self.dim2,
            },
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> DetectorConfig:
        """
        Deserialize detector configuration from dictionary.
        
        Args:
            config_dict: Dictionary with structure from to_dict()
        
        Returns:
            DetectorConfig instance
        
        Raises:
            KeyError: If required fields are missing
            ValueError: If parameter types are invalid
        """
        # Handle nested structure
        if "calibration" in config_dict:
            calib = config_dict["calibration"]
            hardware = config_dict.get("hardware", {})
        else:
            # Flat structure for convenience
            calib = config_dict
            hardware = config_dict
        
        return cls(
            name=config_dict.get("name", "unnamed"),
            distance=float(calib.get("distance", calib.get("dd", 70.0))),
            xcen=float(calib.get("xcen", hardware.get("dim1", 1000) / 2)),
            ycen=float(calib.get("ycen", hardware.get("dim2", 1000) / 2)),
            angle_beta=float(calib.get("angle_beta", calib.get("xbet", 0.0))),
            angle_gamma=float(calib.get("angle_gamma", calib.get("xgam", 0.0))),
            pixelsize=float(hardware.get("pixelsize", 0.1)),
            dim1=int(hardware.get("dim1", 2018)),
            dim2=int(hardware.get("dim2", 2016)),
            geometry=config_dict.get("geometry", "Z>0"),
            hardware_label=config_dict.get("hardware_label"),
            metadata=config_dict.get("metadata", {}),
        )
    
    def to_json(self, filepath: Optional[str] = None, indent: int = 2) -> str:
        """
        Serialize to JSON string or write to file.
        
        Args:
            filepath: Optional path to write JSON. If None, returns string.
            indent: JSON indentation (default 2 spaces)
        
        Returns:
            JSON string if filepath is None, else empty string after writing
        """
        json_str = json.dumps(self.to_dict(), indent=indent)
        
        if filepath:
            Path(filepath).write_text(json_str)
            return ""
        return json_str
    
    @classmethod
    def from_json(cls, json_source: str) -> DetectorConfig:
        """
        Deserialize from JSON string or file.
        
        Args:
            json_source: JSON string or filepath to JSON file
        
        Returns:
            DetectorConfig instance
        """
        # Try to load as file first
        try:
            if Path(json_source).exists():
                json_data = json.loads(Path(json_source).read_text())
                return cls.from_dict(json_data)
        except (OSError, json.JSONDecodeError):
            pass
        
        # Try to parse as JSON string
        json_data = json.loads(json_source)
        return cls.from_dict(json_data)
    
    # =========================================================================
    # MANIPULATION
    # =========================================================================
    
    def apply_parameter_variation(self, **kwargs) -> DetectorConfig:
        """
        Create a modified copy with changed parameters.
        
        Useful for parameter studies and sensitivit analysis.
        
        Args:
            **kwargs: Parameter names and new values (e.g., distance=75.0)
        
        Returns:
            New DetectorConfig with specified modifications
        
        Example:
            >>> base = DetectorConfig(...)
            >>> variant = base.apply_parameter_variation(distance=75.0, xcen=1000)
        """
        # Create dict from current configuration
        config_dict = self.to_dict()
        
        # Apply modifications
        for key, value in kwargs.items():
            if key == "name":
                config_dict["name"] = value
            elif key in ("distance", "xcen", "ycen", "angle_beta", "angle_gamma"):
                config_dict["calibration"][key] = value
            elif key in ("pixelsize", "dim1", "dim2"):
                config_dict["hardware"][key] = value
            elif key in ("geometry", "hardware_label"):
                config_dict[key] = value
            else:
                warnings.warn(
                    f"Unknown parameter: {key}. Skipping.",
                    UserWarning
                )
        
        return DetectorConfig.from_dict(config_dict)
    
    def distance_from_pixels_mm(self, pixel_dist: float) -> float:
        """
        Convert pixel distance to physical distance at detector.
        
        Args:
            pixel_dist: Distance in pixels
        
        Returns:
            Physical distance in mm
        """
        return pixel_dist * self.pixelsize
    
    def pixels_from_distance_mm(self, mm_dist: float) -> float:
        """
        Convert physical distance to pixel distance at detector.
        
        Args:
            mm_dist: Physical distance in mm
        
        Returns:
            Distance in pixels
        """
        return mm_dist / self.pixelsize
    
    # =========================================================================
    # DESCRIPTION & UTILITIES
    # =========================================================================
    
    def describe(self, verbose: bool = True) -> str:
        """
        Generate human-readable description of detector configuration.
        
        Args:
            verbose: If True, include additional derived metrics
        
        Returns:
            Formatted string description
        """
        lines = []
        lines.append(f"{'='*70}")
        lines.append(f"Detector Configuration: {self.name}")
        lines.append(f"{'='*70}")
        
        # Geometry
        lines.append(f"\nGeometry: {self.geometry}")
        if self.hardware_label:
            lines.append(f"Hardware: {self.hardware_label}")
        
        # Calibration parameters
        lines.append(f"\nCalibration Parameters:")
        lines.append(f"  Distance       : {self.distance:10.3f} mm")
        lines.append(f"  X Center       : {self.xcen:10.2f} pixels")
        lines.append(f"  Y Center       : {self.ycen:10.2f} pixels")
        lines.append(f"  Beta Angle     : {self.angle_beta:10.6f} rad ({self.angle_beta*180/3.14159:.4f}°)")
        lines.append(f"  Gamma Angle    : {self.angle_gamma:10.6f} rad ({self.angle_gamma*180/3.14159:.4f}°)")
        
        # Hardware specifications
        lines.append(f"\nHardware Specifications:")
        lines.append(f"  Dimensions     : {self.dim1} × {self.dim2} pixels")
        lines.append(f"  Pixel Size     : {self.pixelsize:.4f} mm/pixel")
        lines.append(f"  Detector Size  : {self.detector_diameter_mm:.1f} mm (diagonal)")
        
        # Derived metrics
        if verbose:
            lines.append(f"\nDerived Metrics:")
            dx, dy = self.center_offset_mm
            lines.append(f"  Center Offset  : ({dx:+.2f}, {dy:+.2f}) mm")
            lines.append(f"  Scattering Angle Coverage: {self.scattering_angle_coverage_degrees:.1f}°")
        
        # Validation
        validation = self.validate()
        lines.append(f"\nValidation: {'✓ VALID' if validation.is_valid else '✗ INVALID'}")
        if validation.warnings:
            lines.append(f"  Warnings: {len(validation.warnings)}")
        
        # Metadata
        if self.metadata:
            lines.append(f"\nMetadata:")
            for key, value in self.metadata.items():
                lines.append(f"  {key}: {value}")
        
        lines.append(f"{'='*70}")
        return "\n".join(lines)
    
    def __str__(self) -> str:
        """String representation"""
        return self.describe(verbose=False)
    
    def __repr__(self) -> str:
        """Developer-friendly representation"""
        return (
            f"DetectorConfig(name='{self.name}', distance={self.distance}, "
            f"xcen={self.xcen}, ycen={self.ycen}, pixelsize={self.pixelsize}, "
            f"dim1={self.dim1}, dim2={self.dim2})"
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_detector_from_lauetools_legacy(
    name: str,
    detectorparameters: List[float],
    pixelsize: float,
    dim1: int,
    dim2: int,
    geometry: str = "Z>0",
    hardware_label: Optional[str] = None,
) -> DetectorConfig:
    """
    Create DetectorConfig from legacy LaueTools parameter format.
    
    Convenience function for migrating existing code.
    
    Args:
        name: Name for this detector
        detectorparameters: [distance, xcen, ycen, angle_beta, angle_gamma]
        pixelsize: Pixel size in mm
        dim1: Width in pixels
        dim2: Height in pixels
        geometry: Geometry mode (default "Z>0")
        hardware_label: Optional hardware identifier
    
    Returns:
        DetectorConfig instance
    
    Example:
        >>> detector = create_detector_from_lauetools_legacy(
        ...     name="BM32_2025",
        ...     detectorparameters=[79.553, 979.32, 932.31, 0.37, 0.447],
        ...     pixelsize=0.0734,
        ...     dim1=2018,
        ...     dim2=2016
        ... )
    """
    distance, xcen, ycen, angle_beta, angle_gamma = detectorparameters
    return DetectorConfig(
        name=name,
        distance=distance,
        xcen=xcen,
        ycen=ycen,
        angle_beta=angle_beta,
        angle_gamma=angle_gamma,
        pixelsize=pixelsize,
        dim1=dim1,
        dim2=dim2,
        geometry=geometry,
        hardware_label=hardware_label,
    )


if __name__ == "__main__":
    # Example usage
    print("DetectorConfig Module - Example Usage\n")
    
    # Create a detector
    detector = DetectorConfig(
        name="BM32_sCMOS_2025",
        distance=79.553,
        xcen=979.32,
        ycen=932.31,
        angle_beta=0.37,
        angle_gamma=0.447,
        pixelsize=0.0734,
        dim1=2018,
        dim2=2016,
        geometry="Z>0",
        hardware_label="sCMOS",
        metadata={
            "calibration_date": "2025-01-15",
            "calibration_method": "powder_diffraction",
            "notes": "Annual recalibration"
        }
    )
    
    # Describe it
    print(detector.describe())
    
    # Validate it
    print("\nValidation Result:")
    print(detector.validate())
    
    # Create a variation
    print("\nCreating variation with shorter distance...")
    variant = detector.apply_parameter_variation(distance=70.0,name="BM32_test_shorter")
    print(variant)
    
    # Serialize
    print("\nJSON Representation:")
    print(detector.to_json())
