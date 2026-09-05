"""Detection stage: vehicles (with tracking) and license plates."""

from __future__ import annotations

from pelakx.detect import plate
from pelakx.detect.base import BaseDetector, DetectorUnavailable
from pelakx.detect.plate import OnnxPlateDetector, UltralyticsPlateDetector
from pelakx.detect.vehicle import (
    COCO_VEHICLE_IDS,
    DEFAULT_VEHICLE_CLASSES,
    VehicleDetector,
)

build_plate_detector = plate.build

__all__ = [
    "COCO_VEHICLE_IDS",
    "DEFAULT_VEHICLE_CLASSES",
    "BaseDetector",
    "DetectorUnavailable",
    "OnnxPlateDetector",
    "UltralyticsPlateDetector",
    "VehicleDetector",
    "build_plate_detector",
]
