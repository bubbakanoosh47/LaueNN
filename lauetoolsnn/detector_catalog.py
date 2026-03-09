"""
Detector Catalog Management for LaueNN

Provides DetectorCatalog class for managing collections of detector configurations,
supporting built-in hardware definitions, beamline-specific catalogs, and detector variants.

Author: LaueNN Development Team
Version: 1.0
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import warnings

from lauetoolsnn.detector_config import DetectorConfig


class DetectorCatalog:
    """
    Registry of detector configurations for LaueNN.
    
    Manages:
    - Built-in hardware specifications (from hardware_definitions.json)
    - Beamline-specific calibrations (from beamlines/{name}/calibrations.json)
    - Detector variants (parameter variations for testing)
    
    Supports:
    - Loading from JSON files
    - Creating variations of base configurations
    - Querying available detectors
    - Setting up beamline defaults
    
    Example:
        >>> catalog = DetectorCatalog.load_default()
        >>> detector = catalog.get_detector("BM32_sCMOS_2025")
        >>> print(detector.describe())
        
        >>> variant = catalog.create_detector_variant(
        ...     "BM32_sCMOS_2025",
        ...     distance=75.0
        ... )
    """
    
    # Built-in hardware definitions (populated from detector_config.json)
    HARDWARE_DEFINITIONS = {
        "sCMOS": {
            "dim1": 2018,
            "dim2": 2016,
            "pixelsize": 0.0734,
            "saturation": 65535,
            "bit_depth": 16,
            "description": "sCMOS camera (binned 2×2 standard)",
            "geometry": "Z>0",
        },
        "sCMOS_1x1": {
            "dim1": 4036,
            "dim2": 4032,
            "pixelsize": 0.0367,
            "saturation": 65535,
            "bit_depth": 16,
            "description": "sCMOS camera (unbinned 1×1)",
            "geometry": "Z>0",
        },
        "MARCCD165": {
            "dim1": 2048,
            "dim2": 2048,
            "pixelsize": 0.079142,
            "saturation": 65535,
            "bit_depth": 16,
            "description": "MAR Research 165 mm detector",
            "geometry": "Z>0",
        },
        "VHR": {
            "dim1": 2594,
            "dim2": 3764,
            "pixelsize": 0.031,
            "saturation": 10000,
            "bit_depth": 16,
            "description": "VHR detector (Diamond)",
            "geometry": "Z>0",
        },
        "EIGER_1M": {
            "dim1": 1065,
            "dim2": 1030,
            "pixelsize": 0.075,
            "saturation": 4294967295,
            "bit_depth": 32,
            "description": "EIGER 1M detector",
            "geometry": "Z>0",
        },
        "EIGER_4M": {
            "dim1": 2167,
            "dim2": 2070,
            "pixelsize": 0.075,
            "saturation": 4294967295,
            "bit_depth": 32,
            "description": "EIGER 4M detector",
            "geometry": "Z>0",
        },
    }
    
    # Default calibration values for development/testing
    DEFAULT_CALIBRATIONS = {
        "development": {
            "name": "development",
            "hardware_label": "sCMOS",
            "geometry": "Z>0",
            "distance": 79.553,
            "xcen": 979.32,
            "ycen": 932.31,
            "angle_beta": 0.37,
            "angle_gamma": 0.447,
        },
        "BM32_setup_2025_01": {
            "name": "BM32_setup_2025_01",
            "hardware_label": "sCMOS",
            "geometry": "Z>0",
            "distance": 79.553,
            "xcen": 979.32,
            "ycen": 932.31,
            "angle_beta": 0.37,
            "angle_gamma": 0.447,
            "calibration_date": "2025-01-15",
            "calibration_method": "powder_diffraction",
        },
    }
    
    def __init__(self, name: str = "default"):
        """
        Initialize an empty DetectorCatalog.
        
        Args:
            name: Name of this catalog (for reference)
        """
        self.name = name
        self.detectors: Dict[str, DetectorConfig] = {}
        self.metadata: Dict[str, Any] = {
            "catalog_name": name,
            "detector_count": 0,
        }
        
        # Register built-in detectors
        self._register_builtin_detectors()
    
    def _register_builtin_detectors(self) -> None:
        """Register default development detectors"""
        for name, calib_data in self.DEFAULT_CALIBRATIONS.items():
            hardware_label = calib_data.get("hardware_label", "sCMOS")
            hw_spec = self.HARDWARE_DEFINITIONS.get(hardware_label, {})
            
            detector = DetectorConfig(
                name=calib_data.get("name", name),
                distance=calib_data.get("distance", 79.553),
                xcen=calib_data.get("xcen", 979.32),
                ycen=calib_data.get("ycen", 932.31),
                angle_beta=calib_data.get("angle_beta", 0.37),
                angle_gamma=calib_data.get("angle_gamma", 0.447),
                pixelsize=hw_spec.get("pixelsize", 0.0734),
                dim1=hw_spec.get("dim1", 2018),
                dim2=hw_spec.get("dim2", 2016),
                geometry=calib_data.get("geometry", "Z>0"),
                hardware_label=hardware_label,
                metadata={
                    k: v
                    for k, v in calib_data.items()
                    if k not in (
                        "name",
                        "distance",
                        "xcen",
                        "ycen",
                        "angle_beta",
                        "angle_gamma",
                        "hardware_label",
                        "geometry",
                    )
                },
            )
            self.detectors[name] = detector
        
        self.metadata["detector_count"] = len(self.detectors)
    
    # =========================================================================
    # LOADING & SAVING
    # =========================================================================
    
    @classmethod
    def load_default(cls) -> DetectorCatalog:
        """
        Create a catalog with default built-in detectors.
        
        Returns:
            DetectorCatalog with development defaults
        """
        return cls(name="default")
    
    @classmethod
    def load_from_file(
        cls,
        filepath: str,
        name: Optional[str] = None,
    ) -> DetectorCatalog:
        """
        Load a catalog from a JSON file.
        
        File format:
        {
            "catalog_metadata": { "name": "...", "created": "..." },
            "detectors": {
                "detector_name": {
                    "name": "detector_name",
                    "hardware_label": "sCMOS",
                    ...all DetectorConfig fields...
                },
                ...
            }
        }
        
        Args:
            filepath: Path to JSON catalog file
            name: Optional catalog name (defaults to filename)
        
        Returns:
            DetectorCatalog populated from file
        
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is invalid JSON
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Catalog file not found: {filepath}")
        
        data = json.loads(path.read_text())
        catalog_name = name or data.get("catalog_metadata", {}).get("name", path.stem)
        
        catalog = cls(name=catalog_name)
        catalog.detectors.clear()  # Clear built-in defaults
        
        # Load detector configs
        for det_name, det_data in data.get("detectors", {}).items():
            detector = DetectorConfig.from_dict(det_data)
            catalog.detectors[det_name] = detector
        
        # Load catalog-level metadata
        if "catalog_metadata" in data:
            catalog.metadata.update(data["catalog_metadata"])
        
        catalog.metadata["detector_count"] = len(catalog.detectors)
        return catalog
    
    def save_to_file(self, filepath: str, include_metadata: bool = True) -> None:
        """
        Save catalog to a JSON file.
        
        Args:
            filepath: Path where to save JSON
            include_metadata: If True, include catalog metadata in output
        """
        output = {}
        
        if include_metadata:
            output["catalog_metadata"] = self.metadata
        
        output["detectors"] = {
            name: detector.to_dict()
            for name, detector in self.detectors.items()
        }
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2))
    
    # =========================================================================
    # QUERY & RETRIEVAL
    # =========================================================================
    
    def get_detector(self, name: str) -> DetectorConfig:
        """
        Retrieve a detector configuration by name.
        
        Args:
            name: Detector name
        
        Returns:
            DetectorConfig instance
        
        Raises:
            KeyError: If detector not found
        """
        if name not in self.detectors:
            available = ", ".join(sorted(self.detectors.keys()))
            raise KeyError(
                f"Detector '{name}' not found in catalog.\n"
                f"Available detectors: {available}"
            )
        return self.detectors[name]
    
    def list_detectors(self) -> List[str]:
        """
        List all available detector configurations.
        
        Returns:
            Sorted list of detector names
        """
        return sorted(self.detectors.keys())
    
    def get_detectors_by_hardware(self, hardware_label: str) -> List[str]:
        """
        Find all detectors using a specific hardware.
        
        Args:
            hardware_label: Hardware type (e.g., "sCMOS")
        
        Returns:
            List of detector names using this hardware
        """
        return [
            name
            for name, detector in self.detectors.items()
            if detector.hardware_label == hardware_label
        ]
    
    def get_detectors_by_geometry(self, geometry: str) -> List[str]:
        """
        Find all detectors with a specific geometry mode.
        
        Args:
            geometry: Geometry mode ("Z>0", "X>0", "X<0")
        
        Returns:
            List of detector names with this geometry
        """
        return [
            name
            for name, detector in self.detectors.items()
            if detector.geometry == geometry
        ]
    
    def describe_all(self) -> str:
        """
        Generate description of all detectors in catalog.
        
        Returns:
            Formatted string listing all detectors
        """
        lines = [f"Detector Catalog: {self.name}", f"{'='*70}"]
        lines.append(f"Total detectors: {len(self.detectors)}\n")
        
        for name in sorted(self.detectors.keys()):
            detector = self.detectors[name]
            lines.append(f"  • {name}")
            lines.append(f"      Hardware: {detector.hardware_label}")
            lines.append(f"      Distance: {detector.distance:.2f} mm")
            lines.append(
                f"      Center: ({detector.xcen:.1f}, {detector.ycen:.1f}) pixels"
            )
            lines.append(f"      Geometry: {detector.geometry}")
        
        return "\n".join(lines)
    
    # =========================================================================
    # REGISTRATION & MANAGEMENT
    # =========================================================================
    
    def register_detector(
        self,
        detector: DetectorConfig,
        allow_overwrite: bool = False,
    ) -> None:
        """
        Add a detector configuration to the catalog.
        
        Args:
            detector: DetectorConfig instance
            allow_overwrite: If True, overwrite if name exists; else raise error
        
        Raises:
            ValueError: If detector name already exists and allow_overwrite=False
        """
        if detector.name in self.detectors and not allow_overwrite:
            raise ValueError(
                f"Detector '{detector.name}' already exists in catalog. "
                f"Use allow_overwrite=True to replace it."
            )
        
        self.detectors[detector.name] = detector
        self.metadata["detector_count"] = len(self.detectors)
    
    def remove_detector(self, name: str) -> None:
        """
        Remove a detector from the catalog.
        
        Args:
            name: Detector name
        
        Raises:
            KeyError: If detector not found
        """
        if name not in self.detectors:
            raise KeyError(f"Detector '{name}' not in catalog")
        
        del self.detectors[name]
        self.metadata["detector_count"] = len(self.detectors)
    
    # =========================================================================
    # VARIANTS
    # =========================================================================
    
    def create_detector_variant(
        self,
        base_name: str,
        variant_name: Optional[str] = None,
        **modifications,
    ) -> DetectorConfig:
        """
        Create a variant of an existing detector configuration.
        
        Useful for parameter studies and sensitivity analysis.
        
        Args:
            base_name: Name of base detector in catalog
            variant_name: Name for the variant (defaults to base_name + suffix)
            **modifications: Parameter changes (e.g., distance=75.0, xcen=1000)
        
        Returns:
            New DetectorConfig with modifications applied
        
        Raises:
            KeyError: If base detector not found
        
        Example:
            >>> catalog = DetectorCatalog.load_default()
            >>> variant = catalog.create_detector_variant(
            ...     "BM32_sCMOS_2025",
            ...     variant_name="test_closer",
            ...     distance=70.0
            ... )
        """
        base_detector = self.get_detector(base_name)
        
        # Generate variant name if not provided
        if variant_name is None:
            mod_str = "_".join(
                f"{k}_{v}".replace(".", "p") for k, v in modifications.items()
            )
            variant_name = f"{base_name}_{mod_str}"
        
        variant = base_detector.apply_parameter_variation(
            name=variant_name, **modifications
        )
        return variant
    
    def register_variant(
        self,
        base_name: str,
        variant_name: Optional[str] = None,
        **modifications,
    ) -> DetectorConfig:
        """
        Create a variant and register it in the catalog.
        
        Args:
            base_name: Name of base detector
            variant_name: Name for variant (auto-generated if None)
            **modifications: Parameter changes
        
        Returns:
            The registered DetectorConfig variant
        """
        variant = self.create_detector_variant(
            base_name, variant_name=variant_name, **modifications
        )
        self.register_detector(variant, allow_overwrite=True)
        return variant
    
    # =========================================================================
    # BEAMLINE MANAGEMENT
    # =========================================================================
    
    def get_beamline_detectors(self, beamline_id: str) -> List[str]:
        """
        Get all detector configurations for a specific beamline.
        
        Args:
            beamline_id: Beamline identifier (e.g., "BM32")
        
        Returns:
            List of detector names matching beamline
        """
        return [
            name
            for name in self.detectors.keys()
            if beamline_id.upper() in name.upper()
        ]
    
    def get_primary_detector(self, beamline_id: str) -> Optional[DetectorConfig]:
        """
        Get primary (production) detector for a beamline.
        
        Assumes primary detector name contains beamline_id + "2025" or latest year.
        
        Args:
            beamline_id: Beamline identifier
        
        Returns:
            DetectorConfig or None if not found
        """
        candidates = self.get_beamline_detectors(beamline_id)
        
        if not candidates:
            return None
        
        # Prefer recent calibration (contains year)
        recent = [c for c in candidates if "202" in c]
        if recent:
            return self.get_detector(sorted(recent)[-1])
        
        return self.get_detector(candidates[0])
    
    # =========================================================================
    # UTILITY
    # =========================================================================
    
    def validate_all(self) -> Dict[str, bool]:
        """
        Validate all detectors in catalog.
        
        Returns:
            Dict mapping detector names to validation results (True=valid)
        """
        return {
            name: detector.is_physically_plausible()
            for name, detector in self.detectors.items()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics about catalog.
        
        Returns:
            Dictionary with counts and ranges
        """
        distances = [d.distance for d in self.detectors.values()]
        pixelsizes = [d.pixelsize for d in self.detectors.values()]
        
        return {
            "total_detectors": len(self.detectors),
            "distance_range_mm": (min(distances), max(distances)),
            "pixelsize_range_mm": (min(pixelsizes), max(pixelsizes)),
            "geometries": set(d.geometry for d in self.detectors.values()),
            "hardware_types": set(d.hardware_label for d in self.detectors.values()),
            "valid_count": sum(self.validate_all().values()),
        }
    
    def __str__(self) -> str:
        """String representation"""
        return self.describe_all()
    
    def __repr__(self) -> str:
        """Developer-friendly representation"""
        return f"DetectorCatalog(name='{self.name}', detectors={len(self.detectors)})"
    
    def __getitem__(self, name: str) -> DetectorConfig:
        """Allow dict-like access: catalog[detector_name]"""
        return self.get_detector(name)
    
    def __len__(self) -> int:
        """Return number of detectors in catalog"""
        return len(self.detectors)
    
    def __iter__(self):
        """Iterate over detector names"""
        return iter(sorted(self.detectors.keys()))


if __name__ == "__main__":
    # Example usage
    print("DetectorCatalog Module - Example Usage\n")
    
    # Create default catalog
    catalog = DetectorCatalog.load_default()
    print(catalog)
    print()
    
    # Get a specific detector
    detector = catalog.get_detector("BM32_setup_2025_01")
    print(f"Retrieved: {detector.name}")
    print(f"Geometry: {detector.geometry}")
    print()
    
    # Create a variant
    print("Creating variant with shorter distance...")
    variant = catalog.create_detector_variant(
        "BM32_setup_2025_01",
        variant_name="test_shorter",
        distance=70.0,
        xcen=985.0,
    )
    print(f"Variant: {variant.name}")
    print(f"Distance: {variant.distance} mm (was {detector.distance})")
    print(f"X center: {variant.xcen} (was {detector.xcen})")
    print()
    
    # Register variant in catalog
    catalog.register_variant(
        "BM32_setup_2025_01",
        variant_name="production_test",
        distance=75.0,
    )
    print("Registered variant in catalog")
    print(f"Now have {len(catalog)} detectors:\n")
    print(", ".join(catalog.list_detectors()))
    print()
    
    # Statistics
    print("Catalog Statistics:")
    stats = catalog.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
